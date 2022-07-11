from multiprocessing.spawn import prepare
from preprocessing import amazon_sample
from huggingface_summarizer import summarize_text

if __name__ == '__main__':
    amazon_sample.sort_amazon()
