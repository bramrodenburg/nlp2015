
import nltk
from nltk.corpus import stopwords
import numpy as np
from nltk.stem.porter import *
from bs4 import BeautifulSoup
import codecs
from timeit import default_timer as timer
import re

my_stop_words = ['nt', 'one', 'the', 've', 'get', 're', 'mr', 'que', 'this', 'two', 'got', 'll', 'they', 'you']

def preprocessing(inFile, use_stem=False):

    start = timer()
    print "Preprocssing...."
    xdoc = codecs.open(inFile, 'r',  errors='ignore')  # , errors='replace'
    stops = stopwords.words("english")
    reviews, bag_of_w = extractallsentences(xdoc, stops, use_stem)
    bag_of_w = list(set(bag_of_w))
    print "Length bag of words %d" % len(bag_of_w)
    end = timer()
    print "preprocessing time %s" % (end - start)

    start = timer()
    bag_of_w = filter(None, bag_of_w)
    dic_bag_of_w = dict([(bag_of_w[i], i) for i in range(len(bag_of_w))])
    doc_words, docs_sentence_words = create_doc_word_matrix(reviews, dic_bag_of_w)
    end = timer()
    print "create_doc_word_matrix time %s" % (end - start)

    return reviews, dic_bag_of_w, bag_of_w, doc_words, docs_sentence_words


def extractallsentences(xdoc, stops, use_stem=False):

    stemmer = PorterStemmer()
    reviews = []

    # add some words to stop words
    stops.append('one')
    stops.append('two')
    stops.append('get')
    stops.append('got')
    tree = BeautifulSoup(xdoc, "xml")

    # assuming file has <data> ... </data> as root tag, so must be added
    # because that was not included in the original files
    numOfReviews = 0
    start = timer()
    bag_of_w = []

    for review in tree.find_all("review_text"):
        reviewsentences_cleaned = []
        # for each review text: 1) split in sentences
        #                               2) get rid of punctuations
        #                               3) tokenize to words
        #                               4) remove stop words
        reviewsentences = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(review.text)]

        for sentence in reviewsentences:
            # get rid off stop words
            if use_stem:
                sentence__clean = [stemmer.stem(re.sub(r'[^\w\s]', '', word.lower())) for word in sentence if (re.sub(r'[^\w\s]', '', word.lower().strip()) not in stops and len(re.sub(r'[^\w\s]', '', word.lower().strip())) > 2)]
            else:
                sentence__clean = [re.sub(r'[^\w\s]', '', word.lower().strip()) for word in sentence if (re.sub(r'[^\w\s]', '', word.lower().strip()) not in stops and len(re.sub(r'[^\w\s]', '', word.lower().strip())) > 2)]
            sentence__clean = filter(None, sentence__clean)
            for ww in sentence__clean:
                bag_of_w.append(ww)
            if len(sentence__clean) > 0:
                reviewsentences_cleaned.append(sentence__clean)

        reviews.append(reviewsentences_cleaned)
        numOfReviews += 1
        # print "Number of reviews %d" % numOfReviews
    end = timer()

    print "review in tree.find_all %s" % (end - start)
    print "# of reviews %s" % numOfReviews
    return reviews, bag_of_w


def create_doc_word_matrix(docs, words):

    # vector that holds for each doc the word counts
    dw = np.zeros(len(words))
    # doc_words is a matrix of size "num of docs" X "num of words corpus"
    docs_words_m = np.zeros((len(docs), len(words)))
    docs_sentence_words = []

    print ("Creating doc word matrix...")
    # loop over reviews
    for m, r in enumerate(docs):
        # loop over sentences in review
        doc_sent = []
        for s_idx, s in enumerate(r):
            # loop over word in sentence
            # there were still '' empty symbols in sentences although I thought I filtered them already
            # above...strange
            s = filter(None, s)
            sent_words = []
            for wd in s:
                idx = words[wd]
                docs_words_m[m, idx] += 1
                sent_words.append(idx)
            doc_sent.append(sent_words)
        docs_sentence_words.append(doc_sent)

    # print docs_sentence_words
    print("Finished creating doc word matrix.")

    return docs_words_m, docs_sentence_words

