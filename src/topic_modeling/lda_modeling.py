from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from gensim import corpora


def model_topics(corpus, num_topics) -> gensim.models.ldamodel.LdaModel:

    # stop = set(stopwords.words('english'))
    # exclude = set(string.punctuation)

    # lemma = WordNetLemmatizer()

    # clean_corpus = []

    # for doc in corpus:
    #     try:
    #     # convert text into lower case + split into words
    #         stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    #     except:
    #         return ''
        
    #     # remove any stop words present
    #     punc_free = ''.join(ch for ch in stop_free if ch not in exclude)  
        
    #     # remove punctuations + normalize the text
    #     normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())  
    #     clean_corpus.append(normalized.split())

    clean_corpus = [clean(doc).split() for doc in corpus]

    dict_ = corpora.Dictionary(clean_corpus)
    doc_term_matrix = [dict_.doc2bow(i) for i in clean_corpus]
    Lda = gensim.models.ldamodel.LdaModel
    result = Lda(doc_term_matrix, num_topics=num_topics, id2word = dict_, passes=1, random_state=0, eval_every=None)
    return result

def clean(doc):

    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)

    lemma = WordNetLemmatizer()
    
    try:
    # convert text into lower case + split into words
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    except:
        return ''
    
    # remove any stop words present
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)  
    
    # remove punctuations + normalize the text
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())  
    return normalized