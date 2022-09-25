import os
import sys

from torch.utils.data import DataLoader
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from src.helpers.serialization import df_read_json


def run_model(input_string: str, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    return output


def test_summaries(path) -> dict:
    model_name = "allenai/t5-small-next-word-generator-qoogle"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    summaries = df_read_json(path)
    correct = 0
    total = 0
    for index, summary in enumerate(summaries['text']):
        processed = ''
        for word in summary.split():
            processed += word + ' '
            next_word = run_model(processed, model=model, tokenizer=tokenizer)[0]
            current_index = len(processed)
            if summary[current_index:current_index+len(next_word)] == next_word:
                correct += 1
            total += 1
        print('predicted summary ' + str(index))
    return {'correct': correct, 'total': total, 'ratio': correct/total}