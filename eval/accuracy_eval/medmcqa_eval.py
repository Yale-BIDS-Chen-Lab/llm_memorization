import json
import argparse
import re
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from huggingface_hub import login
from datasets import load_dataset
import os

# Retrieve the Hugging Face token from the environment
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: Hugging Face token is not set. Some models may not be accessible.")


def load_model(model_path):
    """Load the model and tokenizer."""
    print(f"Loading tokenizer from {model_path}...")
    
    # Some models (like PMC_LLAMA) need LlamaTokenizer directly
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir='xx',
            trust_remote_code=True
        )
        # Check if tokenizer loaded correctly
        if not hasattr(tokenizer, 'pad_token'):
            raise ValueError("Invalid tokenizer returned")
    except Exception as e:
        print(f"AutoTokenizer failed ({e}), trying LlamaTokenizer...")
        tokenizer = LlamaTokenizer.from_pretrained(
            model_path,
            cache_dir='xx',
            trust_remote_code=True
        )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir='xx',
        trust_remote_code=True
    )
    model.eval()
    
    return model, tokenizer


def format_question(question, opa, opb, opc, opd):
    """Format the question with options A, B, C, D."""
    prompt = f"""Given your background as a doctor, answer the following multiple choice question by responding with only the letter (A, B, C, or D) of the correct answer.
Question: {question}

A. {opa}
B. {opb}
C. {opc}
D. {opd}

Answer:"""
    return prompt


def extract_answer(response):
    """Extract the answer letter from the model response."""
    response = response.strip().upper()
    
    # Try to find the first letter A, B, C, or D in the response
    # First check if response starts with the letter
    if response and response[0] in ['A', 'B', 'C', 'D']:
        return response[0]
    
    # Search for patterns like "A.", "A)", "A:", or just "A"
    match = re.search(r'\b([A-D])[.\):\s]', response)
    if match:
        return match.group(1)
    
    # Search for standalone letter
    match = re.search(r'\b([A-D])\b', response)
    if match:
        return match.group(1)
    
    # If nothing found, return the first character if it exists
    if response:
        return response[0]
    
    return None


def cop_to_letter(cop):
    """Convert cop (0, 1, 2, 3) to letter (A, B, C, D).
    MedMCQA uses 0-indexed values: 0=A, 1=B, 2=C, 3=D
    """
    mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    return mapping.get(cop, None)


@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_new_tokens=10):
    """Generate a response for a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Greedy decoding
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new_tokens=10):
    """Generate responses for a batch of prompts."""
    # Tokenize with padding
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=2048
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Greedy decoding
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(outputs)
    
    # Decode responses
    responses = []
    for i, output in enumerate(outputs):
        # Get only the new tokens (after the input)
        input_len = inputs['input_ids'][i].shape[0]
        response = tokenizer.decode(output[input_len:], skip_special_tokens=True)
        responses.append(response.strip())
    
    return responses


def main(args):
    # Load MedMCQA dataset
    print("Loading MedMCQA dataset...")
    dataset = load_dataset("openlifescienceai/medmcqa", split=args.split)
    
    # Optionally limit the number of samples
    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    
    print(f"Loaded {len(dataset)} samples from {args.split} split")
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    # Prepare data
    print("Preparing prompts...")
    prompts = []
    ground_truths = []
    data_records = []
    
    for item in tqdm(dataset, desc="Formatting questions"):
        prompt = format_question(
            item['question'],
            item['opa'],
            item['opb'],
            item['opc'],
            item['opd']
        )
        prompts.append(prompt)
        ground_truths.append(cop_to_letter(item['cop']))
        data_records.append({
            'question': item['question'],
            'opa': item['opa'],
            'opb': item['opb'],
            'opc': item['opc'],
            'opd': item['opd'],
            'correct_option': cop_to_letter(item['cop']),
            'subject_name': item.get('subject_name', ''),
            'topic_name': item.get('topic_name', ''),
        })
    
    # Run inference
    print("Running inference...")
    all_responses = []
    batch_size = args.batch_size
    
    if batch_size > 1:
        # Batch inference
        for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
            batch_prompts = prompts[i:i + batch_size]
            try:
                responses = generate_batch(model, tokenizer, batch_prompts)
                all_responses.extend(responses)
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                # Fall back to single inference
                for prompt in batch_prompts:
                    try:
                        response = generate_response(model, tokenizer, prompt)
                        all_responses.append(response)
                    except Exception as e2:
                        print(f"Error: {e2}")
                        all_responses.append("")
    else:
        # Single inference
        for prompt in tqdm(prompts, desc="Processing samples"):
            try:
                response = generate_response(model, tokenizer, prompt)
                all_responses.append(response)
            except Exception as e:
                print(f"Error: {e}")
                all_responses.append("")
    
    # Calculate accuracy
    print("Calculating accuracy...")
    correct = 0
    total = len(ground_truths)
    
    results = []
    for i, (response, gt) in enumerate(zip(all_responses, ground_truths)):
        predicted = extract_answer(response)
        is_correct = (predicted == gt)
        if is_correct:
            correct += 1
        
        data_records[i]['model_response'] = response
        data_records[i]['predicted_answer'] = predicted
        data_records[i]['is_correct'] = is_correct
        results.append(data_records[i])
    
    accuracy = correct / total * 100
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Results for {args.model_path}")
    print(f"{'='*50}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*50}") 
    
    # Calculate per-subject accuracy if available
    subject_stats = {}
    for record in results:
        subject = record.get('subject_name', 'Unknown')
        if subject not in subject_stats:
            subject_stats[subject] = {'correct': 0, 'total': 0}
        subject_stats[subject]['total'] += 1
        if record['is_correct']:
            subject_stats[subject]['correct'] += 1
    
    if len(subject_stats) > 1:
        print("\nPer-subject accuracy:")
        for subject, stats in sorted(subject_stats.items()):
            subj_acc = stats['correct'] / stats['total'] * 100
            print(f"  {subject}: {subj_acc:.2f}% ({stats['correct']}/{stats['total']})")
    
    # Save results
    output_data = {
        'model': args.model_path,
        'split': args.split,
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'subject_accuracy': {k: v['correct']/v['total']*100 for k, v in subject_stats.items()},
        'results': results
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate LLM on MedMCQA dataset')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model or Hugging Face model ID')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to save the output JSON file')
    parser.add_argument('--split', type=str, default='validation',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to use (default: validation)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (default: 1)')
    
    args = parser.parse_args()
    main(args)
