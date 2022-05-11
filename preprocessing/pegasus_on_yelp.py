import pandas as pd
from helpers.serialization import df_read_json, df_to_json
import transformers

yelp_df = df_read_json('my_datasets\yelp_text.json')
pegasus = transformers.pipeline('google/pegasus-xsum')
