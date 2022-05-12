import pandas as pd
from helpers.serialization import df_read_json, df_to_json
import transformers
import os

def summarize_yelp():
    yelp_df = df_read_json(os.path.join(os.getcwd(), 'my_datasets', 'yelp_text2.json'))
    pegasus = transformers.pipeline(task='summarization', model='google/pegasus-xsum')
    yelp_df_sample = pd.DataFrame(yelp_df['text'][:100])
    df_split = pd.DataFrame(yelp_df_sample['text'].apply(lambda x: x.split()))
    df_split = df_split[df_split['text'].map(len) > 64]
    df_split = df_split[df_split['text'].map(len) < 512]
    yelp_df_sample = yelp_df_sample.filter(items=df_split.index, axis=0)
    pegasus_df = pd.DataFrame.from_dict(pegasus(inputs=yelp_df_sample['text'].to_list()))
    pegasus_df.rename(columns={'summary_text': 'text'}, inplace=True)
    df_to_json(pegasus_df, os.path.join(os.getcwd(), 'my_datasets', 'pegasus_on_yelp_summaries.json'))