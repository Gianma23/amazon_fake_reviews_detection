from nltk.stem import SnowballStemmer
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup

nltk.download('stopwords')
nltk.download('punkt')

# tokenize words
# remove stop words
# stemming
class Parser(object):
    def __call__(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = BeautifulSoup(text, "html.parser").get_text()
        return text

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