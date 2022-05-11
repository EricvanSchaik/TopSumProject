import pandas as pd
from helpers.serialization import df_read_json, df_to_json
import transformers
import os

def summarize_yelp():
    yelp_df = df_read_json(os.path.join(os.getcwd(), 'my_datasets', 'yelp_text.json'))
    pegasus = transformers.pipeline(task='summarization', model='google/pegasus-xsum')
    yelp_df_sample = (yelp_df['text'][:2]).to_list()
    print(pegasus(inputs=yelp_df_sample))