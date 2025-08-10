from datasets import load_dataset
import torch,os
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets
from utils import find_all_linear_names, print_trainable_parameters
from accelerate import Accelerator
accelerator = Accelerator()
device_index = Accelerator().process_index
device_map = {"": device_index}
import os

token = '/xx' # replace with your tokens
output_dir = "/xx" # replace with your directory
model_name = "meta-llama/Meta-Llama-3-8B"
os.environ["WANDB_PROJECT"] = output_dir.split('/')[-1]

# local_rank = int(os.getenv("LOCAL_RANK", 0))
# device_map = {"": f"cuda:{local_rank}"}
# torch.cuda.set_device(local_rank)


#NER
train_dataset_medqa = load_dataset("YBXL/medqa-finetuned-dataset", split="train",
                                   cache_dir="/xx", token=token) # replace with your directory

train_dataset_medmcqa = load_dataset("YBXL/medmcqa-finetuned-dataset", split="train",
                                     cache_dir="/xx", token=token) # replace with your directory

# Concatenate
combined_dataset = concatenate_datasets([train_dataset_medqa, train_dataset_medmcqa])
combined_dataset = combined_dataset.rename_column("conversations", "dialog")

combined_dataset.to_json("combined_data.jsonl", orient="records", lines=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map=device_map,
    cache_dir="/xx", # replace with your directory
    token=token,
    attn_implementation="eager",  # <-- Add this line
    trust_remote_code=True        # <-- Required for LLaMA 3 family
)


base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=find_all_linear_names(base_model),
    #target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = get_peft_model(base_model, peft_config)
print_trainable_parameters(base_model)
#base_model = accelerator.prepare(base_model)

def formatting_prompts_func(example):
    user = example["dialog"][0]["content"].strip()
    assistant = example["dialog"][1]["content"].strip()
    return f"User: {user}\nAssistant: {assistant}"




# Parameters for training arguments details => https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L158
training_args = TrainingArguments(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    gradient_checkpointing = True,
    max_grad_norm= 0.3,
    num_train_epochs=3, 
    learning_rate=1e-5,
    bf16=True,
    save_strategy="epoch",
    save_total_limit=20,
    logging_steps=10,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    weight_decay=0.00001,
    warmup_ratio=0.01,
    ddp_find_unused_parameters=False,
    #generation_max_length=1500
    #load_best_model_at_end=True,
    #metric_for_best_model='eval_loss'
)

trainer = SFTTrainer(
    base_model,
    train_dataset=combined_dataset,
    processing_class=tokenizer,
    #max_seq_length=1500,
    formatting_func=formatting_prompts_func,
    args=training_args,
    #callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)


#trainer.train(resume_from_checkpoint=True) 
trainer.train() 
trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

