import pandas as pd
from transformers import pipeline
import os
from datasets import load_dataset
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

cache_directory = 'D:\\Users\\EricvanSchaik\\.cache\\'

os.environ["TRANSFORMERS_CACHE"] = cache_directory
os.environ["HD_DATASETS_CACHE"] = cache_directory

model_name = 'google/pegasus-xsum'
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name, cache_dir=cache_directory)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_directory)
pegasus_pipeline = pipeline(task='summarization', model=pegasus_model, tokenizer=pegasus_tokenizer)
dataset = load_dataset('xsum', cache_dir=cache_directory)
print(dataset.info)
print(dataset[:3])