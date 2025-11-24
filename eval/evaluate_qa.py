import argparse
import os
import json
import torch
from tqdm import tqdm
from transformers import modeling_utils, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from typing import Dict, List, Tuple
import re

# NOTE: This $ROOT_DIR mechanism ensures the same behavior no matter which directory you run this script from.
ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])

import sys
import_dir = os.path.join(ROOT_DIR, 'src', 'finetune', 'meditron')
sys.path.insert(0, import_dir)
from check_medqa_dataset import medqa_example_to_message


def extract_answer_letter(text: str) -> str:
    """
    Extract the answer choice (A, B, C, D) from model output.
    Handles the specific format: "###Answer: OPTION X IS CORRECT."
    Also tries multiple fallback patterns.
    """
    text_upper = text.upper().strip()

    # Pattern 1: "###Answer: OPTION X IS CORRECT" or "OPTION X IS CORRECT" (dataset format)
    match = re.search(r'OPTION\s+([A-D])\s+IS\s+CORRECT', text_upper)
    if match:
        return match.group(1)

    # Pattern 2: "###Answer: X" or "Answer: X"
    match = re.search(r'###?\s*ANSWER\s*:\s*(?:OPTION\s+)?([A-D])\b', text_upper)
    if match:
        return match.group(1)

    # Pattern 3: "The answer is X" or "The correct answer is X"
    match = re.search(r'(?:THE\s+)?(?:CORRECT\s+)?ANSWER\s+IS\s+(?:OPTION\s+)?([A-D])\b', text_upper)
    if match:
        return match.group(1)

    # Pattern 4: "X is correct" or "X is the answer"
    match = re.search(r'\b([A-D])\s+IS\s+(?:THE\s+)?(?:CORRECT|ANSWER)', text_upper)
    if match:
        return match.group(1)

    # Pattern 5: First standalone letter (A, B, C, D)
    match = re.search(r'\b([A-D])\b', text_upper)
    if match:
        return match.group(1)

    return None


def extract_ground_truth_answer(conversation: List[Dict]) -> str:
    """
    Extract ground truth answer from conversation format.
    Looks for the assistant's response which should contain the answer.
    """
    # Find the last assistant message
    for msg in reversed(conversation):
        if msg['role'] == 'assistant':
            answer = extract_answer_letter(msg['content'])
            if answer:
                return answer
    return None


def perform_inference(tokenizer: AutoTokenizer,
                      model: AutoModelForCausalLM,
                      input_text: str,
                      max_input_length: int = 1024,
                      max_new_tokens: int = 1024,
                      device: torch.device = 'cpu') -> str:
    """
    Perform inference on the model.
    """
    # Encode the input text
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=max_input_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate output
    with torch.no_grad():
        output_tokens = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode output
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # Remove input from output
    generated_text = output_text[len(input_text):].strip()

    return generated_text


def format_conversation_to_prompt(conversation: List[Dict], tokenizer: AutoTokenizer,
                                  include_answer: bool = False) -> str:
    """
    Format conversation into prompt string.
    If include_answer=False, excludes the assistant's answer (for inference).
    """
    # Filter messages based on whether we want the answer
    if not include_answer:
        messages = [msg for msg in conversation if msg['role'] != 'assistant']
    else:
        messages = conversation

    # Use the chat template if available
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=not include_answer)

    # Fallback to manual formatting
    prompt = ""
    for message in messages:
        if message['role'] == 'user':
            prompt += f"[INST] {message['content']} [/INST]"
        elif message['role'] == 'assistant' and include_answer:
            prompt += f" {message['content']} </s>"

    if not include_answer:
        prompt += " "  # Add space for generation

    return prompt


def evaluate_model(model: AutoModelForCausalLM,
                   tokenizer: AutoTokenizer,
                   test_dataset,
                   device: torch.device,
                   max_input_length: int,
                   max_new_tokens: int,
                   model_name: str) -> Tuple[List[Dict], Dict]:
    """
    Evaluate model on test dataset and return results and metrics.
    """
    results = []
    correct = 0
    total = 0

    for idx, entry in enumerate(tqdm(test_dataset, desc=f'Evaluating {model_name}')):
        try:
            conversations = medqa_example_to_message(entry)

            # Extract ground truth answer
            ground_truth = extract_ground_truth_answer(conversations)

            if ground_truth is None:
                print(f"Warning: Could not extract ground truth for example {idx}")
                continue

            # Format prompt (without the answer)
            prompt = format_conversation_to_prompt(conversations, tokenizer, include_answer=False)

            # Extract question for display
            question = ""
            for msg in conversations:
                if msg['role'] == 'user':
                    question = msg['content']
                    break

            # Generate prediction
            generated_text = perform_inference(
                tokenizer, model, prompt,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                device=device
            )

            # Extract predicted answer
            predicted_answer = extract_answer_letter(generated_text)

            # Compute accuracy
            is_correct = predicted_answer == ground_truth if predicted_answer else False

            if predicted_answer:
                correct += is_correct
                total += 1
            else:
                print(f"Warning: Could not extract answer from prediction for example {idx}")

            # Store result
            result = {
                'index': idx,
                'question': question[:300] + '...' if len(question) > 300 else question,
                'ground_truth_answer': ground_truth,
                'predicted_answer': predicted_answer,
                'full_response': generated_text[:1000],  # Store more for debugging
                'correct': is_correct,
            }
            results.append(result)

        except torch.cuda.OutOfMemoryError:
            print(f'Skipping example {idx} due to CUDA out of memory')
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f'Error processing example {idx}: {str(e)}')
            continue

    # Aggregate metrics
    metrics = {
        'accuracy': correct / total if total > 0 else 0.0,
        'correct': correct,
        'total': total,
        'num_examples_in_dataset': len(test_dataset),
    }

    # Print sample predictions for verification
    if results:
        print(f"\n--- Sample Predictions (first 3) ---")
        for i, result in enumerate(results[:3]):
            print(f"\nExample {i}:")
            print(f"  Ground Truth: {result['ground_truth_answer']}")
            print(f"  Predicted: {result['predicted_answer']}")
            print(f"  Correct: {result['correct']}")
            print(f"  Response (first 150 chars): {result['full_response'][:150]}...")

    return results, metrics


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hot fix for model_parallel_style bug
    if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
        modeling_utils.ALL_PARALLEL_STYLES = ['tp', 'none', 'colwise', 'rowwise']

    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load test dataset
    print(f"\nLoading test dataset: {args.dataset_name}")
    test_dataset = load_dataset(
        args.dataset_name,
        split=args.test_split,
        cache_dir=args.cache_dir,
        token=args.hf_token,
    )
    print(f"Test dataset size: {len(test_dataset)}")

    # Limit dataset size if specified
    if args.max_examples > 0:
        test_dataset = test_dataset.select(range(min(args.max_examples, len(test_dataset))))
        print(f"Limited to {len(test_dataset)} examples")

    # Load tokenizer
    print(f"\nLoading tokenizer from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token=args.hf_token,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Set chat template if needed (same as training)
    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<s>' if loop.first else '' }}"
            "{{ '[INST] ' + message['content'].strip() + ' [/INST]' if message['role'] == 'user' else message['content'].strip() }}"
            "{{ ' </s>' if loop.last else '' }}"
            "{% endfor %}"
        )

    all_results = {}

    # Evaluate pretrained model
    print(f"\n{'='*60}")
    print("EVALUATING PRETRAINED MODEL")
    print('='*60)

    pretrained_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=args.hf_token,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
    )
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()

    pretrained_results, pretrained_metrics = evaluate_model(
        pretrained_model,
        tokenizer,
        test_dataset,
        device,
        args.max_input_length,
        args.max_new_tokens,
        'Pretrained'
    )

    all_results['pretrained'] = {
        'results': pretrained_results,
        'metrics': pretrained_metrics
    }

    print(f"\nPretrained Model Results:")
    print(f"  Accuracy: {pretrained_metrics['accuracy']:.4f} ({pretrained_metrics['accuracy']*100:.2f}%)")
    print(f"  Correct: {pretrained_metrics['correct']}/{pretrained_metrics['total']}")

    # Free memory if we're loading another model
    if args.peft_model_folder:
        del pretrained_model
        torch.cuda.empty_cache()

    # Evaluate fine-tuned model if checkpoint provided
    if args.peft_model_folder:
        print(f"\n{'='*60}")
        print("EVALUATING FINE-TUNED MODEL")
        print('='*60)

        # Load base model
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            token=args.hf_token,
            torch_dtype=torch.bfloat16,
            cache_dir=args.cache_dir,
        )

        # Load and merge LoRA weights
        print(f"Loading LoRA weights from: {args.peft_model_folder}")
        finetuned_model = PeftModel.from_pretrained(
            finetuned_model,
            args.peft_model_folder,
            torch_dtype=torch.bfloat16
        )
        finetuned_model = finetuned_model.merge_and_unload()
        finetuned_model = finetuned_model.to(device)
        finetuned_model.eval()

        finetuned_results, finetuned_metrics = evaluate_model(
            finetuned_model,
            tokenizer,
            test_dataset,
            device,
            args.max_input_length,
            args.max_new_tokens,
            'Fine-tuned'
        )

        all_results['finetuned'] = {
            'results': finetuned_results,
            'metrics': finetuned_metrics
        }

        print(f"\nFine-tuned Model Results:")
        print(f"  Accuracy: {finetuned_metrics['accuracy']:.4f} ({finetuned_metrics['accuracy']*100:.2f}%)")
        print(f"  Correct: {finetuned_metrics['correct']}/{finetuned_metrics['total']}")

        # Calculate improvement (pretrained is always evaluated for comparison)
        accuracy_improvement = finetuned_metrics['accuracy'] - pretrained_metrics['accuracy']

        print(f"\n{'='*60}")
        print("IMPROVEMENT")
        print('='*60)
        print(f"  Accuracy: {accuracy_improvement:+.4f} ({accuracy_improvement*100:+.2f} percentage points)")

        if pretrained_metrics['accuracy'] > 0:
            relative_improvement = (accuracy_improvement / pretrained_metrics['accuracy']) * 100
            print(f"  Relative improvement: {relative_improvement:+.2f}%")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    output_json_path = os.path.join(args.output_dir, args.output_json_name)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Save summary
    summary_path = os.path.join(args.output_dir, args.output_json_name.replace('.json', 'evaluation_summary.txt'))
    with open(summary_path, 'w') as f:
        f.write("Medical QA Evaluation Summary (Multiple Choice)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Split: {args.test_split}\n")
        f.write(f"Base Model: {args.model_name}\n")
        if args.peft_model_folder:
            f.write(f"LoRA Checkpoint: {args.peft_model_folder}\n")
        f.write("\n")

        if 'pretrained' in all_results:
            f.write("Pretrained Model:\n")
            f.write(f"  Accuracy: {all_results['pretrained']['metrics']['accuracy']:.4f} ({all_results['pretrained']['metrics']['accuracy']*100:.2f}%)\n")
            f.write(f"  Correct: {all_results['pretrained']['metrics']['correct']}/{all_results['pretrained']['metrics']['total']}\n\n")

        if 'finetuned' in all_results:
            f.write("Fine-tuned Model:\n")
            f.write(f"  Accuracy: {all_results['finetuned']['metrics']['accuracy']:.4f} ({all_results['finetuned']['metrics']['accuracy']*100:.2f}%)\n")
            f.write(f"  Correct: {all_results['finetuned']['metrics']['correct']}/{all_results['finetuned']['metrics']['total']}\n\n")

        if 'finetuned' in all_results:
            accuracy_improvement = all_results['finetuned']['metrics']['accuracy'] - all_results['pretrained']['metrics']['accuracy']
            f.write("Improvement:\n")
            f.write(f"  Accuracy: {accuracy_improvement:+.4f} ({accuracy_improvement*100:+.2f} percentage points)\n")

            if all_results['pretrained']['metrics']['accuracy'] > 0:
                relative_improvement = (accuracy_improvement / all_results['pretrained']['metrics']['accuracy']) * 100
                f.write(f"  Relative improvement: {relative_improvement:+.2f}%\n")

    print(f"\n{'='*60}")
    print(f"Results saved to:")
    print(f"  {output_json_path}")
    print(f"  {summary_path}")
    print('='*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--hf_token", type=str, default=None, help="HF token if needed for gated/private datasets.")
    parser.add_argument('--model_name', type=str, default='epfl-llm/meditron-7b')
    parser.add_argument('--peft_model_folder', type=str, default=None,
                        help='Path to fine-tuned LoRA weights (if None, only pretrained is evaluated)')
    parser.add_argument('--cache_dir', type=str, default=os.path.join(ROOT_DIR, '.cache', 'huggingface_models'), help='Cache directory for models.')

    # Dataset arguments
    parser.add_argument('--dataset_name', default='GBaker/MedQA-USMLE-4-options')
    parser.add_argument('--test_split', type=str, default='test',
                        help='Dataset split to use for evaluation')
    parser.add_argument('--max_examples', type=int, default=-1,
                        help='Maximum number of examples to evaluate (-1 for all)')

    # Generation arguments
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--max_new_tokens', type=int, default=1024)

    # Output arguments
    parser.add_argument('--output_dir', type=str, default=os.path.join(ROOT_DIR, 'results', 'medqa'))
    parser.add_argument('--output_json_name', type=str, default='meditron7b_medqa.json')

    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    main(args)
