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

    start = timer()
    print "Preprocssing...."
    xdoc = codecs.open(inFile, 'r', encoding='utf8', errors='ignore') # , errors='replace'
    stops = stopwords.words("english")
    reviews, bag_of_w = extractallsentences(xdoc, stops)

    bag_of_w = list(set(bag_of_w))
    end = timer()
    print "preprocessing time %s" % (end - start)

    start = timer()
    bag_of_w = filter(None, bag_of_w)
    doc_words = create_doc_word_matrix(reviews, bag_of_w)
    end = timer()
    print "create_doc_word_matrix time %s" % (end - start)

    return reviews, bag_of_w, doc_words


def extractallsentences(xdoc, stops):

    stemmer = PorterStemmer()
    reviews = []
    punct = ['#', '/', '[', ']', '}', '--', ',', '-/', '+', '-', '((', '))', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # parser = ET.XMLParser(encoding="utf-8")
    # tree = ET.parse(xdoc, parser=parser)
    tree = BeautifulSoup(xdoc, "xml")

    # assuming file has <data> ... </data> as root tag, so must be added
    # because that was not included in the original files
    numOfReviews = 0
    start = timer()
    reviewsentences_cleaned = []
    bag_of_w = []

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
            sentence__clean = filter(None, sentence__clean)
            for ww in sentence__clean:
                bag_of_w.append(ww)

        reviews.append(reviewsentences_cleaned)

        numOfReviews += 1

    end = timer()

    print "review in tree.find_all %s" % (end - start)
    print "# of reviews %s, words in bag %s" % (numOfReviews, len(bag_of_w))
    return reviews, bag_of_w


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

    def __init__(self, all_words, doc_word_m, num_of_gl_topics, alpha, beta):

        # number of docs/reviews, total vocabulary size of corpus
        self.n_docs, self.vocab_size = doc_word_m.shape
        self.num_of_topics = num_of_gl_topics

        # for the time being we assume synchronous Dirichlet distributions
        self.alpha = np.empty(self.num_of_topics)
        self.alpha.fill(alpha)
        self.beta = np.empty(self.vocab_size)
        self.beta.fill(beta)

        # number of times document m and topic k co-occur
        self.ndk = np.zeros((self.n_docs, self.num_of_topics))
        # number of times topic k and word w co-occur
        self.nkw = np.zeros((self.num_of_topics, self.vocab_size))
        # number of words in each document m
        self.nd = np.zeros(self.n_docs)
        # number of words assigned to topic k
        self.nk = np.zeros(self.num_of_topics)
        # dictionary: key is tuple of (docID, wordIdx), value is equal to topic
        self.doc_w_topics_assgn = {}
        # matrix that holds per document the counts for the words (dims: #of_docs X #words_in_bag)
        self.doc_word_counts = doc_word_m
        # the bag of words
        self.bag_of_words = all_words
        # phi = p(w|k) distribution, matrix of num of topics X num of words in corpus
        self.phi_dist = np.zeros((self.num_of_topics, self.vocab_size))
        # theta distribution = p(k|d), matrix of num of docs X num of topics
        self.theta_dist = np.zeros((self.n_docs, self.num_of_topics))

    def initialize(self, doc_word_m):

        for d in xrange(self.n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            # for each document, take the doc/word counter and use that to
            # create a long vector that contains each word token (so a word can appear more than once
            # the index "i" indicates the i-th word in the document
            for i, wd in enumerate(word_indices(doc_word_m[d, :])):
                # choose an arbitrary topic as first topic for word i
                k = np.random.randint(self.num_of_topics)
                self.ndk[d, k] += 1
                self.nd[d] += 1
                self.nkw[k, wd] += 1
                self.nk[k] += 1
                self.doc_w_topics_assgn[(d, i)] = k  # assign topic to word in document!

    def build_topic_multinomial_dist(self, d, wd):
        # will return for each topic the current probability for a word-document being assigned to that topic
        # use this vector of "num_of_topics" to construct the multinomial distribution from which we will sample
        # a new topic for the "current word"
        p_k = np.empty(self.num_of_topics)

        for tp in range(self.num_of_topics):
            # reference to formula in Ivan's slides
            # self.nd =C(i)=C(d) is # of words in document
            # sum(self.alpha) = K*alpha
            # sum(self.beta) = V*beta
            how_likeli_topic_k_in_d = (self.alpha[tp] + self.ndk[d, tp]) / (self.nd[d] + sum(self.alpha))
            how_well_w_fits_topic = (self.beta[wd] + self.nkw[tp, wd]) / (sum(self.beta) + self.nk[tp])
            p_k[tp] = how_well_w_fits_topic * how_likeli_topic_k_in_d
            if self.nk[tp] < 0 or self.nkw[tp, wd] < 0 or self.ndk[d, tp] < 0 or self.nd[d] < 0:
                print self.nk[tp], self.nkw[tp, wd], self.ndk[d, tp], self.nd[d]

        # still need to normalize because this equation is just proportional to p_k
        p_k = p_k * 1/sum(p_k)
        return p_k

    def run_gibbs_sampling(self, max_iterations=2):

        # how long do we iter before we stop?
        for gibbs_iteration in range(max_iterations):
            # loop over documents/reviews
            print "Iteration %s" % (gibbs_iteration + 1)
            for d in xrange(self.n_docs):
                print "Document %s" % (d + 1)
                # again the same trick as in _initialize method above
                # i is the running index for all words in the document/review
                # wd is the index of the word taken from the document/word matrix
                # if doc contains 3 times the word "book" and twice the word "science" the result is:
                # 0 book, 1 book, 2 book, 3 science, 4 science....
                for i, wd in enumerate(word_indices(self.doc_word_counts[d, :])):

                    # in fact we don't need the word
                    word = self.bag_of_words[wd]
                    # choose the topic for the word we assigned in the initialization
                    k = self.doc_w_topics_assgn[(d, i)]
                    # lower all necessary counts for this topic
                    self.ndk[d, k] -= 1
                    self.nkw[k, wd] -= 1
                    self.nk[k] -= 1
                    p_k = self.build_topic_multinomial_dist(d, wd)
                    # sample a new topic from the "new" distribution p_k (p_k is a num_of_topics dimensional vector)
                    k = np.nonzero(np.random.multinomial(1, p_k))[0][0]
                    # increase counters according to the new sampled topic
                    self.ndk[d, k] += 1
                    self.nkw[k, wd] += 1
                    self.nk[k] += 1
                    self.doc_w_topics_assgn[(d, i)] = k

    # the method could return the last four counters, the one we need in order to construct our
    # distributions phi_k(w) = p(w|k) and theta_d(k) = p(k|d)
    # but we can work with the complete LDA-object instead (see below)

    def build_phi_matrix(self):
        # phi is a matrix of dimension: (num of topics X num of words in corpus)
        # and contains for each topic/word combination the probability for a word belonging to that topic k

        for k in range(self.num_of_topics):
            self.phi_dist[k, :] = self.nkw[k, :] * 1/np.sum(self.nkw[k, :])
            # print "phi row sum to %s" % np.sum(self.phi_dist[k, :])

    def build_theta_matrix(self):
        # theta is a matrix that holds the topic probabilities for a certain document p(k|d)
        for d in xrange(self.n_docs):
            self.theta_dist[d, :] = self.ndk[d, :] * 1/np.sum(self.ndk[d, :])
            # print "theta row sum to %s" % np.sum(self.theta_dist[d, :])


def create_doc_word_matrix(docs, words):

    # vector that holds for each doc the word counts
    dw = np.zeros(len(words))
    # doc_words is a matrix of size "num of docs" X "num of words corpus"
    docs_words_m = np.zeros((len(docs), len(words)))
    print ("Creating doc word matrix...")
    # loop over reviews
    for m, r in enumerate(docs):
        # loop over sentences in review
        for s in r:
            # loop over word in sentence
            # there were still '' empty symbols in sentences although I thought I filtered them already
            # above...strange
            s = filter(None, s)
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

    # inFile = dir_path + "dvd.xml"
    inFile = dir_path + "dvdReviews.xml"
    inFile = dir_path + "example.xml"
    pickelfile = dir_path + "example.pkl"
    # pickelfile = dir_path + "dvd_reviews_limited.pkl"

    # inFile = sys.argv[2] + "dvd.xml" huge file

    if preprocess == 'True':

        reviews, w, doc_words = preprocessing(inFile)
        print "Save objects to file %s" % pickelfile
        start = timer()
        with open(pickelfile, 'wb') as f:
            pickle.dump(reviews, f)
            pickle.dump(w, f)
            print "# of words in bag %s %s" % doc_words.shape
            pickle.dump(doc_words, f)
        end = timer()
        print "save objects %s" % (end - start)
    else:
        with open(pickelfile, 'rb') as f:
            print "Loading objects from file...."
            reviews = pickle.load(f)
            w = pickle.load(f)
            doc_words = pickle.load(f)
            print "# of docs %s" % (len(reviews))
    # check_doc_word_matrix(doc_words, reviews, w)

    # create LDAModel object and initialize counters for Gibbs sampling
    lda = LDAModel(w, doc_words, K_GL, 0.15, 0.1)
    # initialize counters
    start = timer()
    print "LDA initialize..."
    lda.initialize(doc_words)
    end = timer()
    print "LDA initialize time %s" % (end - start)
    # run Gibbs sampling, parameter is number of times we run Gibbs
    start = timer()
    num_of_iterations = 5
    print "Gibbs sampling for %s" % num_of_iterations, " iterations..."
    lda.run_gibbs_sampling(num_of_iterations)
    end = timer()
    print "Gibbs sampling time %s" % (end - start)
    # build the phi matrix
    lda.build_phi_matrix()
    lda.build_theta_matrix()
    # print lda.theta_dist
