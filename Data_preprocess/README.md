## Dataset preprocess
Split each instance into two parts and let the first part  (top-𝑙 tokens) be the input and the second part be the groundtruth.

- For example, for the clinical guidelines dataset, run:
```
python guidelines_process_token.py --sample_size 4000 --input_length 50 --output_length 500 --seed 42 --output_file dataset/4k_sample_50.json
```
This generates a 4,000 sampled instances `4k_sample_50.json.json`. The `sample_size` denotes total training samples, `input_length` and `output_length` denote the input length and the output length.

