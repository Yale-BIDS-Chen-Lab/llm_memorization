## Dataset preprocess
Split each instance into two parts and let the first part  (top-𝑙 tokens) be the input and the second part be the groundtruth.

- To process the clinical guidelines dataset, run:
```
python guidelines_process_token.py --sample_size 4000 --input_length 50 --output_length 500 --seed 42 --output_file dataset/4k_sample_50.json
```
This generates a 4,000 sampled dataset `4k_sample_50.json.json` with first half as input question and second half as groundtruth. The `sample_size` stands for total training samples, `input_length` and `output_length` stand for the word length we want to separate.
Same process for replay, abstract, and full text dataset.

- For pubmedqa:
To process the PubMedQA dataset, run:
```
python pqa_process.py 
```
This generates two 1k sampled datasets:
- `context_question_dataset.json`: using context as input and question as the ground truth.
- `question_split_dataset.json`: using the first half of the question as input and the second half as ground truth.
