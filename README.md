# Data Memorization of LLMs in Medicine
This repository provides the related codes, datasets, and models for studying memorization of large language models in medicine.

## Install Meditron Environment
To set up the environment, follow the steps below:
```
conda create -n meditron python=3.10
conda activate meditron
pip install -r requirements.txt
```

## Original datasets
### Meditron Pre-train Datasets 
The following datasets are used for Meditron pre-training: 
- Clinical Guidelines
- Paper Abstracts
- Medical Papers
- Replay dataset
  
You can find these datasets in the [gap-replay directory](https://github.com/epfLLM/meditron/tree/main/gap-replay).

### PMCLLaMA Datasets
For PMCLLaMa, the following datasets are available:
- Pre-training datasets. We provide the detailed procedure for acquiring and processing [pre-training datasets](./PMCLLama/readme.md).
- Fine-tuning datasets. You can find these datasets through this [link](https://huggingface.co/datasets/axiong/pmc_llama_instructions). 

## Dataset Processing
We split each instance into two parts, let the first part  (top-𝑙 tokens) be the input and the second part be the groundtruth. 
Please refer to 
use two main strategies for sampling the text data:
   - Script: [guidelines_process_token.py](./Meditron/guidelines_process_token.py)

## Model Finetune
For model finetune, we include model fine-tuning over both benchmarks and clinical notes. 
Please refer to [Model_finetune/Finetune_benchmark/finetune.py](Model_finetune/Finetune_benchmark/finetune.py) and [Model_finetune/Finetune_clinical_notes/run.sh](Model_finetune/Finetune_clinical_notes/run.sh). 

## Model Inference
We leverage  [vLLM](https://github.com/vllm-project/vllm) to speed up the inference. For details, please refer to [vllm_inference.py](./Meditron/vllm_inference.py).

## Evaluation results:
Detailed evaluation results for all experiments can be found in this [Google spreadsheet](https://docs.google.com/spreadsheets/d/1cbOuZKMctm0PAj3LCwNYm2mJBz-tFvfkHrGIHNxxGow/edit?usp=sharing).

## Evaluation metrics:
After running model inference, the generated responses are stored in a dedicated response folder. We then apply various evaluation metrics using the script [eval_all.py](./eval/full_eval/eval_all.py), which includes:
1. Top-n Consecutive Token Match
2. ROUGE Score
3. BLEU Score
4. BERTScore
5. BARTScore


### Valid Length Evaluation
We add two kinds of evaluation here: evaluating first-K tokens and first-K sentences.

1. First-K tokens comparison
    - Script: [eval_valid_all.py](./eval/valid_length_eval/eval_valid_all.py)
    - Evaluates the similarity between model output and GT for the first K tokens.

2. First-K sentences comparison
    - Script: [eval_valid_firstsen.py](./eval/valid_sent_eval/eval_valid_firstsen.py)
    - Evaluates the similarity for the first K sentences of the text.

### Partial memorization metric
The partial memorization function measures how closely a model's output matches the groundtruth at the token level within the first 100 tokens, focusing on exact matches at corresponding positions. Please refer to [partial_memorization_eval](./partial_memorization_eval.py).


## License

This repository is provided under the [MIT License](./LICENSE) (or whichever license applies). Please refer to the `LICENSE` file for details.







