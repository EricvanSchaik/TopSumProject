import pandas as pd
from helpers.serialization import df_read_json, df_to_json
import transformers

def summarize_yelp():
    yelp_df = df_read_json('my_datasets\yelp_text.json')
    print(yelp_df)
    pegasus = transformers.pipeline(task='sentiment-analysis')
    yelp_df_sample = (yelp_df['text'][:2])
    print((yelp_df_sample))
    print(pegasus(inputs=[yelp_df_sample[0]]))