import pandas as pd
from transformers import pipeline
# import os
from datasets import load_dataset
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from helpers.serialization import df_to_json

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
df_to_json(xsum_df_text[:400], './my_datasets/xsum_text.json')
##TODO generate summaries with pegasus, don't take the reference summaries
xsum_df_summaries = pd.DataFrame(xsum_df['summary'])
xsum_df_summaries.rename(columns={'summary': 'text'}, inplace=True)
df_to_json(xsum_df_summaries[400:], './my_datasets/pegasus_summaries.json')