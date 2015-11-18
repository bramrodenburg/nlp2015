import sys
import nltk
from nltk.corpus import stopwords
import numpy as np
from nltk.stem.porter import *
from bs4 import BeautifulSoup
import codecs
import pickle
from timeit import default_timer as timer

# The global number of topics
K_GL = 10


def check_doc_word_matrix(mat, revs, w):
    print revs[0][0]
    t = np.squeeze(np.asarray(mat[0, :].nonzero()))
    print t
    for idx in t:
        print w[idx]


def preprocessing(inFile):

    xdoc = codecs.open(inFile, 'r', encoding='utf8', errors='ignore') # , errors='replace'
    stops = stopwords.words("english")
    reviews = extractallsentences(xdoc, stops)

    w = []
    start = timer()
    for rev in reviews:
        for sentences in rev:
            for ww in sentences:
                if ww not in rev:
                    w.append(ww)
    end = timer()
    print "create bag of words %s" % (end - start)
    return reviews, w


def extractallsentences(xdoc, stops):

    stemmer = PorterStemmer()
    reviews = []
    punct = ['#','/','[',']','}','--',',','-/','+','-','((','))']

    # parser = ET.XMLParser(encoding="utf-8")
    # tree = ET.parse(xdoc, parser=parser)
    tree = BeautifulSoup(xdoc, "xml")

    # assuming file has <data> ... </data> as root tag, so must be added
    # because that was not included in the original files
    numOfReviews = 0
    start = timer()
    reviewsentences_cleaned = []

    for review in tree.find_all("review_text"):
        # for each review text: 1) split in sentences
        #                               2) get rid of punctuations
        #                               3) tokenize to words
        #                               4) remove stop words
        reviewsentences = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(review.text)]

        for sentence in reviewsentences:
            # get rid off stop words
            sentence__clean = [stemmer.stem(word.lower().strip().strip(",.;:?!-#*[]()")) for word in sentence if (word not in stops and word not in punct)]
            reviewsentences_cleaned.append(sentence__clean)

        reviews.append(reviewsentences_cleaned)

        numOfReviews += 1

    end = timer()

    print "review in tree.find_all %s" % (end - start)
    print "# of reviews %s" % numOfReviews
    return reviews


def word_indices(vec):
    """
    taken from gist.github.com/mblondel/542786

    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in vec.nonzero()[0]:
        for i in xrange(int(vec[idx])):
            yield idx


class LDAModel(object):

    def __init__(self, doc_word_m):

        # number of docs/reviews, total vocabulary size of corpus
        self.n_docs, self.vocab_size = doc_word_m.shape
        # number of times document m and topic k co-occur
        self.nmk = np.zeros((self.n_docs, K_GL))
        # number of times topic k and word w co-occur
        self.nkw = np.zeros((K_GL, self.vocab_size))
        # number of words in each document m
        self.nm = np.zeros(self.n_docs)
        # number of words assigned to topic k
        self.nk = np.zeros(K_GL)
        # dictionary: key is tuple of (docID, wordIdx), value is equal to topic
        self.topics = {}

    def initialize(self, doc_word_m):

        for m in xrange(self.n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, wd in enumerate(word_indices(doc_word_m[m, :])):
                # choose an arbitrary topic as first topic for word i
                k = np.random.randint(K_GL)
                self.nmk[m, k] += 1
                self.nm[m] += 1
                self.nkw[k, wd] += 1
                self.nk[k] += 1
                self.topics[(m, i)] = k # assign topic to word


def create_doc_word_matrix(docs, words):

    # vector that holds for each doc the word usage
    dw = np.zeros(len(words))
    docs_words_m = np.zeros((len(docs), len(words)))

    # loop over reviews
    for m, r in enumerate(docs):
        # loop over sentences in review
        for s in r:
            # loop over word in sentence
            for wd in s:
                idx = words.index(wd)
                dw[idx] += 1
        docs_words_m[m, :] = dw[:]

    return docs_words_m


if __name__ == '__main__':

    """
    parameters:
        (1) preprocess files: "True" or "False" (no boolean but string)
        (2) directory path for input & output files
    """

    if len(sys.argv) == 1:
        preprocess = "False"
        dir_path = 'F:/temp/'
    else:
        preprocess = sys.argv[1]
        dir_path = sys.argv[2]

    inFile = dir_path + "dvdReviews.xml"
    pickelfile = dir_path + "dvd_reviews_limited.pkl"

    # inFile = sys.argv[2] + "dvd.xml" huge file

    if preprocess == 'True':
        print "Preprocssing...."
        reviews, w = preprocessing(inFile)
        print "Save objects to file %s" % pickelfile
        with open(pickelfile, 'wb') as f:
            pickle.dump(reviews, f)
            pickle.dump(w, f)
    else:
        with open(pickelfile, 'rb') as f:
            print "Loading objects from file...."
            reviews = pickle.load(f)
            print "Number of reviews %s" % (len(reviews))
            w = pickle.load(f)
            print "Bag of words %s" % (len(w))

    start = timer()
    doc_words = create_doc_word_matrix(reviews, w)
    end = timer()
    print "create_doc_word_matrix %s" % (end - start)
    # check_doc_word_matrix(doc_words, reviews, w)

    # create LDAModel object and initialize counters for Gibbs sampling
    lda = LDAModel(doc_words)
    lda.initialize(doc_words)
    print lda.nkw[0, :]
