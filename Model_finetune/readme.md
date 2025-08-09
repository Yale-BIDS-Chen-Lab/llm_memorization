# Fine-tuning model inference
For medqa dataset, we employ [fine-tuned_meditron70b_medqa](https://huggingface.co/hippocrates/fine-tuned_meditron70b_medqa) for inference. Since there is no tokenizer files in the Huggingface repository, we need to manually copy the meditron tokenizer or llama2 tokenizer files into the local huggingface hub.
```
# Huggingface login
huggingface-cli login

# Download the fine-tuned model
huggingface-cli download hippocrates/fine-tuned_meditron70b_medqa

# Manually copy the tokenizer files into the huggingface hub.

# Model inference
run finetune/finetune_inference.py --input_file /gpfs/home/yy923/LLM_privacy/Meditron/dataset/medqa/1k_sample_50.json --output_file /gpfs/home/yy923/LLM_privacy/Meditron/output/vllm/tokenized/medqa/1k_sample_50.json --batch_size 256
```
Processed medqa dataset can be found [onedrive](https://yaleedu-my.sharepoint.com/:f:/r/personal/anran_li_yale_edu/Documents/Medical%20LLMs%20Memorization/Experiment%20results/Finetuning/MedQA/MedQA_datasets?csf=1&web=1&e=tVg94i)
