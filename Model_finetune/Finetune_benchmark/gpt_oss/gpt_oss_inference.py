import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os

# Retrieve the Hugging Face token from the environment
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    raise ValueError("Hugging Face token is not set. Please set the HUGGINGFACE_HUB_TOKEN environment variable.")


def load_model(model_path, cache_dir):
    """Load the model and tokenizer."""
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    
    return model, tokenizer


@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_new_tokens=1024):
    """Generate a response for a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def main(args):
    # Set cache directory
    os.environ["HF_HOME"] = args.cache_dir
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.cache_dir)
    
    # Load the dataset
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} entries...")
    all_results = []
    
    for i, entry in enumerate(tqdm(data, desc="Processing entries")):
        prompt = entry["prompt"]
        input_text = entry["input"]
        combined_text = f"{prompt}\n\n{input_text}"
        
        try:
            generated_text = generate_response(model, tokenizer, combined_text, args.max_tokens)
            entry["gpt_oss_response"] = generated_text
        except Exception as e:
            print(f"Error processing entry {i}: {e}")
            entry["gpt_oss_response"] = ""
        
        all_results.append(entry)
        
        # Save results periodically
        if (i + 1) % 100 == 0:
            with open(args.output_file, 'w', encoding='utf-8') as f_out:
                json.dump(all_results, f_out, ensure_ascii=False, indent=4)
    
    # Final save
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        json.dump(all_results, f_out, ensure_ascii=False, indent=4)
    
    print(f"Inference completed and results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with GPT-OSS model.')
    parser.add_argument('--model_path', type=str, default="openai/gpt-oss-20b",
                        help='Path to the model')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input JSON file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the output JSON file')
    parser.add_argument('--max_tokens', type=int, default=1024,
                        help='Maximum tokens to generate')
    parser.add_argument('--cache_dir', type=str,
                        default='/gpfs/radev/home/yf329/scratch/hf_models',
                        help='Cache directory for models')
    
    args = parser.parse_args()
    main(args)

