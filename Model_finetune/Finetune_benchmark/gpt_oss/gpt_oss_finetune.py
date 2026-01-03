import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from huggingface_hub import login
import bitsandbytes as bnb

# Login to Hugging Face
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)


def find_all_linear_names(model):
    """Find all linear layer names for LoRA targeting."""
    lora_module_names = set()
    
    # Try to find Linear4bit layers (for BnB quantized models)
    try:
        cls = bnb.nn.Linear4bit
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    except:
        pass
    
    # If no Linear4bit found, look for regular Linear layers
    if not lora_module_names:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """Print the number of trainable parameters."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def format_medqa_example(example):
    """Format MedQA-USMLE example for instruction fine-tuning."""
    question = example['question']
    options = example['options']
    correct_answer = example['answer_idx']  # Already a letter (A, B, C, D)
    
    prompt = f"""Given your background as a doctor, answer the following multiple choice question by responding with only the letter (A, B, C, or D) of the correct answer.

Question: {question}

A. {options.get('A', '')}
B. {options.get('B', '')}
C. {options.get('C', '')}
D. {options.get('D', '')}

Answer: {correct_answer}"""
    
    return {"text": prompt}


def main(args):
    # Set cache directory
    os.environ["HF_HOME"] = args.cache_dir
    
    print(f"Loading model: {args.model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Try to load model - check if it's pre-quantized
    try:
        if args.no_quantize:
            # Load without quantization (for pre-quantized models)
            print("Loading model without additional quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                device_map="auto",
                cache_dir=args.cache_dir,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        else:
            # Quantization config for QLoRA
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                cache_dir=args.cache_dir,
                trust_remote_code=True,
            )
    except ValueError as e:
        if "already quantized" in str(e).lower() or "Mxfp4Config" in str(e):
            print(f"Model is pre-quantized, loading without BitsAndBytes...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                device_map="auto",
                cache_dir=args.cache_dir,
                trust_remote_code=True,
            )
        else:
            raise e
    
    model = prepare_model_for_kbit_training(model)
    
    # Find all linear layers for LoRA
    target_modules = find_all_linear_names(model)
    print(f"Target modules for LoRA: {target_modules}")
    
    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    
    # Load and format dataset
    print("Loading MedQA-USMLE dataset...")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    
    # MedQA only has train and test splits, use train for training
    # Split train into train/eval
    train_test_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    # Format the dataset
    train_dataset = train_dataset.map(format_medqa_example, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_medqa_example, remove_columns=eval_dataset.column_names)
    
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        learning_rate=args.learning_rate,
        bf16=True,
        save_strategy="epoch",
        save_total_limit=20,
        logging_steps=10,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        warmup_ratio=0.01,
        ddp_find_unused_parameters=False,
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.run_name,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        #dataset_text_field="text",
        #max_seq_length=args.max_seq_length,
        #packing=False,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLM on MedQA-USMLE dataset")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the base model")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--cache_dir", type=str, 
                        default="xx",
                        help="Cache directory for models")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    
    # Quantization
    parser.add_argument("--no_quantize", action="store_true",
                        help="Don't apply BitsAndBytes quantization (for pre-quantized models)")
    
    # Dataset arguments
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Max training samples (for debugging)")
    parser.add_argument("--max_eval_samples", type=int, default=500,
                        help="Max eval samples")
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--run_name", type=str, default="medqa_finetune",
                        help="Run name for logging")
    
    args = parser.parse_args()
    main(args)


