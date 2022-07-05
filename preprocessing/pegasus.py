import pandas as pd
import os
from datasets import load_dataset
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from helpers.serialization import df_to_json

cache_directory = 'D:\\Users\\EricvanSchaik\\.cache\\'

os.environ["TRANSFORMERS_CACHE"] = cache_directory
os.environ["HD_DATASETS_CACHE"] = cache_directory

def save_xsum():
    dataset = load_dataset('xsum', split='train')
    xsum_df = pd.DataFrame(dataset[:1000])
    xsum_df.rename(columns={'document': 'text'}, inplace=True)
    xsum_df_text = pd.DataFrame(xsum_df['text'])
    df_to_json(xsum_df_text[:400], './my_datasets/xsum_text.json')


def generate_summaries():
    model_name = 'sshleifer/distill-pegasus-xsum-16-4'
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    tokenizer = PegasusTokenizer.from_pretrained(model_name)

    data = load_dataset('xsum', split='validation[:600]')
    
    summaries = list()
    
    for document in data['document']:
        inputs = tokenizer(document, truncation=True, max_length=1024, return_tensors='pt')
        summary_ids = model.generate(inputs['input_ids'])
        summaries.append([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0])
    df_to_json(pd.DataFrame(data={'text': summaries}), './my_datasets/pegasus_summaries.json')