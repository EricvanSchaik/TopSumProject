from datasets import load_dataset
from helpers.serialization import df_to_json
import pandas as pd

def generate_wikitext():
    dataset = load_dataset('wikitext', 'wikitext-103-v1', split='train')
    wiki = dataset.shuffle()[:1000]
    wiki_df = pd.DataFrame(wiki)
    wiki_df = wiki_df[wiki_df['text'] != ""]
    wiki_df = wiki_df.apply(lambda x: x.str.strip())
    wiki_df = wiki_df[~wiki_df.text.str.startswith('=')]
    df_to_json(wiki_df, '../../data/wikitext.json')