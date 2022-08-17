from readline import append_history_file
from transformers import pipeline
import pandas as pd
from src.helpers.serialization import df_to_json
import os

def summarize_amazon():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", truncation=True)
    amazon_df = pd.read_csv('./data/amazon_sorted/most_populair_products.csv')
    amazon_df = amazon_df.drop(columns='index').dropna().reset_index()
    product_ids = amazon_df['product_id'].unique()
    review_summaries = list()
    for id in product_ids:
        product_reviews = amazon_df[amazon_df['product_id'] == id]['review_body']
        product_reviews.reset_index(drop=True, inplace=True)
        subsummaries = ''
        for i in range(10):
            full_text = ''
            for j in range(100):
                full_text += '\n' + product_reviews[i*100+j]
            subsummaries += '\n' + summarizer(full_text)[0]['summary_text']
        review_summaries.append(summarizer(subsummaries)[0]['summary_text'])
    results_df = pd.DataFrame.from_dict(summarizer(inputs=review_summaries))
    results_df.rename(columns={'summary_text': 'text'}, inplace=True)
    df_to_json(results_df, os.path.join(os.getcwd(), 'data', 'distilbart_on_amazon_summaries.json'))
        

    
    # review1 = "Sennheiser has a great reputation for headphones which really adds to my disappointment for this particular model.    Good points:    1) very convenient way to recharge headphones   2) replaceable batteries   3) good sound quality    Bad points:   1) annoying beeps - when you turn it on/off & when you're not receiving an audio source.  Is this a design fad?        Everything from microwave ovens to cars have annoying indicator beeps or honks these days.  At least allow the (intelligent)       user to turn down and turn off the beeps.   2) can't turn the volume all the way down - you still hear sound at the lowest volume level.  So when I'm watching TV and      want to cut the volume for commercials, I have to turn off the headset (BEEP!) to completely cut off the sound.  And when      the program comes back, I have to turn back on the headset (BEEP!) to listen again.  What a moronic design!!!    The bad (and unforgivable) points outweigh the good for me.      In contrast, I have great IR headphones from Pioneer (SE-DHP800) that don't have the beep and volume problems.  The only drawback  is that I have to remain in line of sight of the transmitter since they're not RF."
    # review2 = "A little static sometimes but works well."
    # print(summarizer(inputs=[review1, review2]))
    # summaries = pd.DataFrame.from_dict(summarizer(amazon_dataset['review_body'][:100]))
    # df_to_json(summaries, os.path.join(os.getcwd(), 'data', 'distilbart_on_amazon_summaries.json'))