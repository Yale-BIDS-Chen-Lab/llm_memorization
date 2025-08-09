## Pre-training model inference
The pmc_article_completion.py can be used to generate the pmc model answers and the pmc_article_completion_batch.py using the batch size as input which can speed up the inference.
```
python pmc_article_completion_batch.py --input_file ./pmc_article_50token.json --output_file response/pmc_response_50token.json --batch_size 8
```
