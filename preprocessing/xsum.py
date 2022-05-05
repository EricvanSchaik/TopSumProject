import pandas as pd
from transformers import pipeline
# import os
from datasets import load_dataset
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# cache_directory = 'D:\\Users\\EricvanSchaik\\.cache\\'

# os.environ["TRANSFORMERS_CACHE"] = cache_directory
# os.environ["HD_DATASETS_CACHE"] = cache_directory


model_name = 'google/pegasus-xsum'
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)
pegasus_pipeline = pipeline(task='summarization', model=pegasus_model, tokenizer=pegasus_tokenizer)
dataset = load_dataset('xsum', split='train')
xsum_df = pd.DataFrame(dataset[:1000])
xsum_df.rename(columns={'document': 'text'}, inplace=True)
xsum_df_text = pd.DataFrame(xsum_df['text'])
xsum_df_text.to_json('./my_datasets/xsum_text.json', orient='records', lines=True)
xsum_df_summaries = pd.DataFrame(xsum_df['summary'])
xsum_df_summaries.rename(columns={'summary': 'text'}, inplace=True)
xsum_df_summaries.to_json('./my_datasets/xsum_summaries.json', orient='records', lines=True)