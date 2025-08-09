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
- Pre-training datasets. The model used for pretraining is from this [link](https://huggingface.co/chaoyi-wu/PMC_LLAMA_7B).
To obtain the PMC open source articles we firstly need to download the csv file that include all the PMCid of the articles. 
```
wget ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv
```
Then we randomly select select the size PMCid and extract setences from the abstract and body using the pmc_extract_articles.py
```
python pmc_extract_articles.py --csv_path ./oa_file_list.csv --download_dir ./downloads --extract_dir ./extracted_articles --num_samples 11000 --random_seed 42 --word_limit 2000 --intermediate_json ./pmc_intermediate.json
```
This command downloads and processes 11,000 sample articles, saves to the ./downloads directory, extracts them to the ./extracted_articles directory, and stores the extracted text content in the ./pmc_intermediate.json file.

- Fine-tuning datasets. You can find these datasets through this [link](https://huggingface.co/datasets/axiong/pmc_llama_instructions). 

## Dataset Processing
We split each instance into two parts, let the first part  (top-𝑙 tokens) be the input and the second part be the groundtruth.  
- For example, for the clinical guidelines dataset, run:
```
python guidelines_process_token.py --sample_size 4000 --input_length 50 --output_length 500 --seed 42 --output_file dataset/4k_sample_50.json
```
This generates a 4,000 sampled instances `4k_sample_50.json.json`. The `sample_size` denotes total training samples, `input_length` and `output_length` denote the input length and the output length.

## Model Finetune
For model finetune, we include model fine-tuning over both benchmarks and clinical notes. 
Please refer to [Model_finetune/Finetune_benchmark/](Model_finetune/Finetune_benchmark/) and [Model_finetune/Finetune_clinical_notes/](Model_finetune/Finetune_clinical_notes/). 

## Model Inference
We leverage  [vLLM](https://github.com/vllm-project/vllm) to speed up the inference. For example, to perform inference of the fine-tuned PMCLLaMA model, please refer to [Model_finetune/Finetune_benchmark/PMCLLaMA/inference_vllm.py](./Model_finetune/Finetune_benchmark/PMCLLaMA/inference_vllm.py).

## Memorization Evaluations:
After performing model inference, the generated responses are stored in a response folder. We then apply various evaluation metrics using the script [eval_all.py](./eval/full_eval/eval_all.py), which includes:
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

2. Specified tokens comparison
    - Script: [eval_valid_firstsen.py](./eval/valid_sent_eval/eval_valid_firstsen.py)
    - Evaluates the similarity for the first K sentences of the text.

### Partial memorization metric
The partial memorization function measures how closely a model's output matches the groundtruth at the token level within the first 100 tokens, focusing on exact matches at corresponding positions. Please refer to [partial_memorization_eval](./partial_memorization_eval.py).


## License

This repository is provided under the [MIT License](./LICENSE) (or whichever license applies). Please refer to the `LICENSE` file for details.















