import json
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Initialize the vLLM
llm = LLM(
    model="epfl-llm/meditron-7b",
    tensor_parallel_size=1,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    max_seq_len_to_capture=2048,
)

def perform_inference_batch(inputs, temperature, top_k):
    # Define sampling parameters with fixed top_p=0.9
    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=0.9,
        max_tokens=1024
    )
    # Generate output
    generated = llm.generate(inputs, sampling_params)
    return [output.outputs[0].text.strip() for output in generated]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run inference with Meditron model.'
    )
    parser.add_argument(
        '--input_file', '-i',
        type=str, required=True,
        help='Path to the input JSON file'
    )
    parser.add_argument(
        '--output_file', '-o',
        type=str, required=True,
        help='Path to the output JSON file'
    )
    parser.add_argument(
        '--batch_size', '-b',
        type=int, default=16,
        help='Batch size for processing inputs'
    )
    parser.add_argument(
        '--temperature', '-t',
        type=float, default=0.0,
        help='Sampling temperature (e.g., 0.0 for greedy).'
    )
    parser.add_argument(
        '--top_k', '-k',
        type=int, default=1,
        help='Top-k sampling cutoff.'
    )

    args = parser.parse_args()

    # Load the dataset
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_results = []
    batch_inputs = []

    for i, entry in enumerate(tqdm(data, desc="Processing entries")):
        prompt = entry.get("prompt", "")
        input_text = entry.get("input", "")
        combined_text = f"{prompt}\n\n{input_text}"
        batch_inputs.append(combined_text)

        # Reach batch_size or process the last entry, perform inference
        if len(batch_inputs) == args.batch_size or i == len(data) - 1:
            try:
                generated_texts = perform_inference_batch(
                    batch_inputs,
                    temperature=args.temperature,
                    top_k=args.top_k
                )
                # Write the generated results back to data and append to all_results
                for j, gen in enumerate(generated_texts):
                    idx = i - len(batch_inputs) + 1 + j
                    data[idx]["meditron_response"] = gen
                all_results.extend(
                    data[i - len(batch_inputs) + 1 : i + 1]
                )
            except Exception as e:
                print(f"Skipping batch due to error: {e}")
            finally:
                batch_inputs = []

            # Save after each batch is processed
            with open(args.output_file, 'w', encoding='utf-8') as f_out:
                json.dump(all_results, f_out, ensure_ascii=False, indent=4)

    print("Inference completed and results saved to", args.output_file)
