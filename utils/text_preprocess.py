from nltk.stem import SnowballStemmer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bs4 import BeautifulSoup

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import wordnet

# def get_wordnet_pos(treebank_tag):
#
#     if treebank_tag.startswith('J'):
#         return wordnet.ADJ
#     elif treebank_tag.startswith('V'):
#         return wordnet.VERB
#     elif treebank_tag.startswith('N'):
#         return wordnet.NOUN
#     elif treebank_tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return wordnet.NOUN

# tokenize words
# remove stop words
# stemming
stop_words = set(stopwords.words('english'))
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = SnowballStemmer('english')
        self.tokenizer = RegexpTokenizer(r'\b[a-z]{2,}\b')
    def __call__(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = BeautifulSoup(text, "html.parser").get_text()
        tokens = [self.wnl.stem(x) for x in self.tokenizer.tokenize(text) if x not in stop_words]
        return tokens