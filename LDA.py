import sys
from preprocessing import *
import pickle
from timeit import default_timer as timer
import h5py

# The global number of topics
K_GL = 20
N_GIBBS_SAMPLING_ITERATIONS = 50
ALPHA = 0.05 #50. / K_GL
BETA = 0.01 #200. / 31440

def check_doc_word_matrix(mat, revs, w):
    print revs[0][0]
    t = np.squeeze(np.asarray(mat[0, :].nonzero()))
    print t
    for idx in t:
        print w[idx]


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
        self.sum_alpha = sum(self.alpha)
        self.sum_beta = sum(self.beta)

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
                k = np.random.randint(self.num_of_topics, size=1)[0]
                self.ndk[d, k] += 1
                self.nd[d] += 1
                self.nkw[k, wd] += 1 
                self.nk[k] += 1
                self.doc_w_topics_assgn[(d, i)] = k  # assign topic to word in document!
                # print "d=%d, i=%d, wd=%d, word=%s" % (d, i, wd, self.bag_of_words.values()[wd])

    def build_topic_multinomial_dist(self, d, wd):
        # will return for each topic the current probability for a word-document being assigned to that topic
        # use this vector of "num_of_topics" to construct the multinomial distribution from which we will sample
        # a new topic for the "current word"
        hltkid = (self.alpha + self.ndk[d, :]) / (self.nd[d] + self.num_of_topics*self.sum_alpha)
        hwwft = (self.beta[wd] + self.nkw[:, wd]) / (self.vocab_size*self.sum_beta + self.nk)
        p_k = hltkid*hwwft
        pk = p_k/sum(p_k)
        return pk

    def run_gibbs_sampling(self, max_iterations=2):
        # print "Total number of documents: %d" % self.n_docs
        # how long do we iter before we stop?
        for gibbs_iteration in range(max_iterations):
            # loop over documents/reviews
            print "Iteration %s" % (gibbs_iteration + 1)
            for d in xrange(self.n_docs):
                # again the same trick as in _initialize method above
                # i is the running index for all words in the document/review
                # wd is the index of the word taken from the document/word matrix
                # if doc contains 3 times the word "book" and twice the word "science" the result is:
                # 0 book, 1 book, 2 book, 3 science, 4 science....
                # print "Length of doc_word_counts: (%d, ,)" % self.doc_word_counts[d, :].shape
                # print "Number of words: %d" % np.sum(self.doc_word_counts[d, :])
                for i, wd in enumerate(word_indices(self.doc_word_counts[d, :])):
                    # in fact we don't need the word
                    # word = self.bag_of_words.values()[wd]
                    # choose the topic for the word we assigned in the initialization
                    k = self.doc_w_topics_assgn[(d, i)]
                    # lower all necessary counts for this topic
                    self.ndk[d, k] -= 1
                    self.nkw[k, wd] -= 1
                    self.nk[k] -= 1
                    self.nd[d] -= 1
                    # print "d=%d, wd=%d, k=%d, word=%s" % (d, wd, k, self.bag_of_words.keys()[wd])
                    p_k = self.build_topic_multinomial_dist(d, wd)
                    # sample a new topic from the "new" distribution p_k (p_k is a num_of_topics dimensional vector)
                    k = np.random.choice(self.num_of_topics, size=1, p=p_k)[0]
                    # increase counters according to the new sampled topic
                    self.ndk[d, k] += 1
                    self.nkw[k, wd] += 1
                    self.nk[k] += 1
                    self.nd[d] += 1
                    self.doc_w_topics_assgn[(d, i)] = k

    # the method could return the last four counters, the one we need in order to construct our
    # distributions phi_k(w) = p(w|k) and theta_d(k) = p(k|d)
    # but we can work with the complete LDA-object instead (see below)

    def build_phi_matrix(self):
        # phi is a matrix of dimension: (num of topics X num of words in corpus)
        # and contains for each topic/word combination the probability for a word belonging to that topic k

        nkw_aug = self.nkw + self.beta
        for k in range(self.num_of_topics):
            if np.sum(np.sum(nkw_aug[k, :] != 0)):
                self.phi_dist[k, :] = nkw_aug[k, :] * 1/np.sum(nkw_aug[k, :])
                # print "phi row sum to %s" % np.sum(self.phi_dist[k, :])

    def build_theta_matrix(self):
        # theta is a matrix that holds the topic probabilities for a certain document p(k|d)
        for d in xrange(self.n_docs):
            if np.sum(self.ndk[d, :] != 0):
                self.theta_dist[d, :] = self.ndk[d, :] * 1/np.sum(self.ndk[d, :])
                # print "theta row sum to %s" % np.sum(self.theta_dist[d, :])

    def store_results(self, mem_file):
        # store the theta and phi matrix
        h5f = h5py.File(mem_file, 'w')
        h5f.create_dataset('theta_dist', data=self.theta_dist)
        h5f.create_dataset('phi_dist', data=self.phi_dist)
        h5f.close()
        '''
        with open(mem_file, 'wb') as f:
            pickle.dump(self.theta_dist, f)
            pickle.dump(self.phi_dist, f)
        f.close()
        '''


if __name__ == '__main__':

    """
    parameters:
        (1) preprocess files: "True" or "False" (no boolean but string)
        (2) directory path for input & output files
    """

    if len(sys.argv) == 1:
        preprocess = "True"
        dir_path = 'F:/temp/topics/'
        # dir_path = "/Users/jesse/Desktop/nlp1_project/src/"
    else:
        preprocess = sys.argv[1]
        dir_path = sys.argv[2]

    #inFile = dir_path + "dvd.xml"
    #inFile = dir_path + "dvdReviews.xml"
    # inFile = dir_path + "example.xml"
    inFile = dir_path + "all.review"
    pickelfile = dir_path + "dvd_reviews_limited.pkl"
    h5_file = dir_path + "data.h5"
    # pickelfile = dir_path + "example.pkl"
    # pickelfile = dir_path + "dvd_reviews.pkl"
    mem_file_results = dir_path + "lda_results.h5"

    # inFile = sys.argv[2] + "dvd.xml" huge file

    if preprocess == 'True':
        reviews, w, doc_words = preprocessing(inFile)
        print "Save objects to file %s" % pickelfile
        start = timer()
        with open(pickelfile, 'wb') as f:
            pickle.dump(reviews, f)
            pickle.dump(w, f)
            print "Number of reviews : %d" % len(reviews)
            print "# of words in bag %s %s" % doc_words.shape
            #pickle.dump(doc_words, f)
        end = timer()
        h5f = h5py.File(h5_file, 'w')
        h5f.create_dataset('doc_words', data=doc_words)
        h5f.close()
        print "Saved objects to file in %s seconds." % (end - start)
    else:
        with open(pickelfile, 'rb') as f:
            print "Loading objects from file...."
            reviews = pickle.load(f)
            w = pickle.load(f)
            #doc_words = pickle.load(f)
            print "# of docs %s" % (len(reviews))
            print "vocabulary size %d" % (len(w))
        h5f = h5py.File(h5_file, 'r')
        doc_words = h5f['doc_words'][:]
        h5f.close()
    # check_doc_word_matrix(doc_words, reviews, w)
    # create LDAModel object and initialize counters for Gibbs sampling
    lda = LDAModel(w, doc_words, K_GL, ALPHA, BETA)
    # initialize counters
    start = timer()
    print "LDA initialize..."
    lda.initialize(doc_words)
    end = timer()
    print "LDA initialize time %s" % (end - start)
    # run Gibbs sampling, parameter is number of times we run Gibbs
    start = timer()
    num_of_iterations = N_GIBBS_SAMPLING_ITERATIONS
    print "Gibbs sampling for %s" % num_of_iterations, " iterations..."
    lda.run_gibbs_sampling(num_of_iterations)
    end = timer()
    print "Gibbs sampling time %s" % (end - start)
    # build the phi matrix
    lda.build_phi_matrix()
    lda.build_theta_matrix()
    # print lda.phi_dist
    for i in range(K_GL):
    	print "Sum of Topic %d is %.4f" % (i, np.sum(lda.phi_dist[i, :]))
    lda.store_results(mem_file_results)
