# ✍️ PatentWriter: A Benchmarking Study for Patent Drafting with LLMs


:boom: We generate abstracts of patents by utilizing the first claim
of each corresponding patent as an input for CPC subclasses A61, H04, and G06. All the patents used in this paper are granted by the United States Patent and Trademark Office [USPTO](https://www.uspto.gov)  and obtained  from [PatentsView](https://patentsview.org/download/data-download-tables).


## Data
🔄 Processed data can be found  [here](https://github.com/hhshomee/pwriter/tree/main/data). All the processed data has the following information as columns: patent_id, patent_abstract, patent_title, claim_text, cpc_subclass.

📄 Sample generated data can be found [here](https://github.com/hhshomee/pwriter/tree/main/results).

## Patent Drafting 

We generate abstracts and claims of patents using models such as GPT-3.5, GPT-4o,GPT-4.1,Llama2, Llama3, and DeepSeek.

To run the code:

``python main.py --input data/A61.csv --output_prefix results/A61_GPT4.1_abstracts --limit 8000  --part abs --llm_type openai --llm_model gpt-4.1``

## Patent Tasks

Code for patent tasks can be found in the folder tasks/


## Evaluation

To run the evaluation code:

``python evaluation.py --input results/A61_GPT4_abs.csv --output_prefix results/evaluation/eval_A61_GPT4_abs.csv --part abs`` 




