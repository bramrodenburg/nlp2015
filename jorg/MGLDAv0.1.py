import sys
from preprocessv12 import *
import pickle
from timeit import default_timer as timer

# The global number of topics
K_GL = 10
K_LOC = 5
N_GIBBS_SAMPLING_ITERATIONS = 2

def sample_r():

    if np.random.randint(2) == 0:
        return "gl"
    else:
        return "loc"


def count_sent_docs(revs):
    # count number of sentences per document
    docs_sent_len = np.zeros(len(revs))
    for r, doc in enumerate(revs):
        docs_sent_len[r] = len(doc)

    return docs_sent_len


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

    def __init__(self, all_words, doc_sentences_words, doc_s_count, num_of_gl_topics, num_of_loc_topics,
                 alpha_gl, alpha_loc, beta_gl, beta_loc, gamma, alpha_mix_gl, alpha_mix_loc):

        # number of sliding windows per sentence
        self.n_windows = 3

        # number of docs/reviews, total vocabulary size of corpus
        self.n_docs, self.num_of_max_sentences, self.vocab_size = doc_sentences_words.shape
        self.num_of_gl_topics = num_of_gl_topics
        self.num_of_loc_topics = num_of_loc_topics
        self.doc_s_count = doc_s_count

        # for the time being we assume synchronous Dirichlet distributions
        self.alpha_gl = alpha_gl
        self.alpha_loc = alpha_loc
        self.beta_gl = beta_gl
        self.beta_loc = beta_loc
        self.gamma = gamma
        #
        self.alpha_mix_gl = alpha_mix_gl
        self.alpha_mix_loc = alpha_mix_loc

        # number of times document m and global topic k co-occur
        self.ndk_gl = np.zeros((self.n_docs, self.num_of_gl_topics))
        # number of times document m and local topic k co-occur
        self.ndk_loc = np.zeros((self.n_docs, self.num_of_loc_topics))
        # a little redundant but for convenience (we could calculate these by summing over ndk_gl & ndk_loc
        self.nd_gl = np.zeros(self.n_docs)
        self.nd_loc = np.zeros(self.n_docs)
        # number of times word w co-occur with global topic k
        self.nkw_gl = np.zeros((self.num_of_gl_topics, self.vocab_size))
        # number of times word w co-occur with local topic k
        self.nkw_loc = np.zeros((self.num_of_loc_topics, self.vocab_size))

        # number of words assigned to topic k
        self.nk_gl = np.zeros(self.num_of_gl_topics)
        self.nk_loc = np.zeros(self.num_of_loc_topics)

        # length of sentence s in document m
        self.nds = np.zeros((self.n_docs, self.num_of_max_sentences))
        # number of times a word from sentence s is assigned to window v. Each document has
        # number of sentences + 2 windows.
        self.ndsv = np.zeros((self.n_docs, self.num_of_max_sentences, self.n_windows))

        # number of times a word in document d is assigned to window v
        self.ndv = np.zeros((self.n_docs, self.num_of_max_sentences + 2))
        # number times a global topic was assigned to document d and window v
        self.ndv_gl = np.zeros((self.n_docs, self.num_of_max_sentences + 2))
        # number times a local topic was assigned to document d and window v
        self.ndv_loc = np.zeros((self.n_docs, self.num_of_max_sentences + 2))
        # number of local topics in document d and window v assigned to local topic k
        self.ndvk_loc = np.zeros((self.n_docs, self.num_of_max_sentences + 2, self.num_of_loc_topics))

        # dictionary: key is tuple of (docID, wordIdx), value is equal to topic
        self.doc_w_topics_assgn = {}  # k_di
        # assignment of word at position i in document d to window v
        self.doc_w_window_assgn = {}  # v_di
        # assignment of word at position i in document d to global OR local topic
        self.doc_w_gl_loc_assgn = {}  # r_di: gl = global topic; loc = local topic

        # matrix that holds per document the counts for the words (dims: #of_docs X #words_in_bag)
        self.doc_sentences_words = doc_sentences_words
        # the bag of words
        self.bag_of_words = all_words
        # phi = p(w|k) distribution, matrix of num of topics X num of words in corpus
        self.phi_dist = np.zeros((self.num_of_gl_topics, self.vocab_size))
        # theta distribution = p(k|d), matrix of num of docs X num of topics
        self.theta_dist = np.zeros((self.n_docs, self.num_of_gl_topics))

    def initialize(self):

        for d in xrange(self.n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            # for each document, take the doc/word counter and use that to
            # create a long vector that contains each word token (so a word can appear more than once
            # the index "i" indicates the i-th word in the document
            start = timer()
            for s in xrange(int(self.doc_s_count[d])):

                for i, wd in enumerate(word_indices(self.doc_sentences_words[d, s, :])):
                    # choose one of the three windows that can be associated with the specific sentence
                    # a number between 0-2 (currently with 3 possible windows per sentence
                    # s + 0/1/2 e.g. document sentence 1 can belong to window {1,2,3} and
                    # sentence 2 can belong to window {2,3,4} etc.
                    v = np.random.randint(self.n_windows)
                    self.doc_w_window_assgn[(d, i)] = v
                    # choose whether for this word we sample from global or local topics
                    # 0 = global topic, 1 = local topic
                    r = sample_r()
                    self.doc_w_gl_loc_assgn[(d, i)] = r
                    # number of times a word from document d is assigned to window v
                    self.ndv[d, s+v] += 1
                    self.ndsv[d, s, v] += 1
                    self.nds[d, s] += 1

                    if r == "gl":
                        # global topic assignment
                        k = np.random.randint(self.num_of_gl_topics)
                        self.nkw_gl[k, wd] += 1
                        self.ndk_gl[d, k] += 1
                        self.nd_gl[d] += 1
                        self.ndv_gl[d, s+v] += 1
                        self.nk_gl[k] += 1

                    else:
                        # local topic assignment
                        k = np.random.randint(self.num_of_loc_topics)
                        self.nkw_loc[k, wd] += 1
                        self.ndk_loc[d, k] += 1
                        self.ndv_loc[d, s+v] += 1
                        self.ndvk_loc[d, s+v, k] += 1
                        self.nd_loc[d] += 1
                        self.nk_loc[k] += 1

                    self.doc_w_topics_assgn[(d, i)] = k  # assign topic to word in document!

            end = timer()

    def lower_counts(self, d, s, k, v, r, wd):

        self.ndv[d, s+v] -= 1
        self.ndsv[d, s, v] -= 1
        self.nds[d, s] -= 1

        if r == "gl":
            self.nkw_gl[k, wd] -= 1
            self.ndk_gl[d, k] -= 1
            self.nd_gl[d] -= 1
            self.ndv_gl[d, s+v] -= 1
            self.nk_gl[k] -= 1
        else:
            self.nkw_loc[k, wd] -= 1
            self.ndk_loc[d, k] -= 1
            self.ndv_loc[d, s+v] -= 1
            self.ndvk_loc[d, s+v, k] -= 1
            self.nd_loc[d] -= 1
            self.nk_loc[k] -= 1
            self.ndsv[d, s, v] -= 1

    def increase_counts(self, d, s, k, v, r, wd):

        self.ndv[d, s+v] += 1
        self.ndsv[d, s, v] += 1
        self.nds[d, s] += 1

        if r == "gl":
            self.nkw_gl[k, wd] += 1
            self.ndk_gl[d, k] += 1
            self.nd_gl[d] += 1
            self.ndv_gl[d, s+v] += 1
            self.nk_gl[k] -= 1
        else:
            self.nkw_loc[k, wd] += 1
            self.ndk_loc[d, k] += 1
            self.ndv_loc[d, s+v] += 1
            self.ndvk_loc[d, s+v, k] += 1
            self.nd_loc[d] += 1
            self.nk_loc[k] += 1
            self.ndsv[d, s, v] += 1

    def sample_k_v_gl_loc(self, d, s, wd, v):

        # sampling topic new_z for t
        p_v_r_k = []
        label_v_r_k = []
        # for the number of sliding windows / Ivan uses 3
        for v_idx in range(self.n_windows):
            # for the global topics
            for k_idx in range(self.num_of_gl_topics):
                label = [v_idx, "gl", k_idx]
                label_v_r_k.append(label)
                # sampling eq as gl
                # term1 = float(self.n_gl_z_w[z_t][word] + self.beta_gl) / (self.n_gl_z[z_t] + self.W*self.beta_gl)
                part1 = float(self.nkw_gl[k_idx, wd] + self.beta_gl) / (self.nk_gl[k_idx] + (self.vocab_size * self.beta_gl))
                # term2 = (self.ndsv[d, s, v_idx] + self.gamma) / (self.n_d_s[m][s] + self.T*self.gamma)
                part2 = float(self.ndsv[d, s, v_idx] + self.gamma) / (self.nds[d, s] + (self.n_windows * self.gamma))
                # term3 = float(self.n_d_v_gl[m][s+v_t] + self.alpha_mix_gl) / (self.n_d_v[m][s+v_idx]
                # + self.alpha_mix_gl + self.alpha_mix_loc)
                part3 = float(self.ndv_gl[d, s+v_idx]  + self.alpha_mix_gl) / (self.ndv[d, s+v_idx] + self.alpha_mix_gl + self.alpha_mix_loc)
                # term4 = float(self.n_d_gl_z[m][z_t] + self.alpha_gl) / (self.n_d_gl[m] + self.K_gl*self.alpha_gl)
                part4 = float(self.ndk_gl[d, k_idx] + self.alpha_gl) / (self.nd_gl[d] + (self.num_of_gl_topics*self.alpha_gl))
                score = part1 * part2 * part3 * part4
                if score < 0.0:
                    print "global: ", part1, part2, part3, part3
                    print "0 > %f" % score
                p_v_r_k.append(score)
            # for local topics
            for k_idx in range(self.num_of_loc_topics):
                label = [v_idx, "loc", k_idx]
                label_v_r_k.append(label)
                # sampling eq as loc
                # term1 = float(self.n_loc_z_w[z_t][word] + self.beta_loc) / (self.n_loc_z[z_t] + self.W*self.beta_loc)
                part1 = float(self.nkw_loc[k_idx, wd] + self.beta_loc) / (self.nk_loc[k_idx] + (self.vocab_size * self.beta_loc))
                part2 = float(self.ndsv[d, s, v_idx] + self.gamma) / (self.nds[d, s] + (self.n_windows * self.gamma))
                # term3 = float(self.n_d_v_loc[m][s+v_t] + self.alpha_mix_loc) / (self.n_d_v[m][s+v_t]
                # + self.alpha_mix_gl + self.alpha_mix_loc)
                part3 = float(self.ndv_loc[d, s+v] + self.alpha_mix_loc) / (self.ndv[d, s+v_idx] + self.alpha_mix_gl + self.alpha_mix_loc)
                # term4 = float(self.n_d_v_loc_z[m][s+v_t][z_t] + self.alpha_loc) / (self.n_d_v_loc[m][s+v_t]
                # + self.K_loc * self.alpha_loc)
                part4 = float(self.ndvk_loc[d, s+v_idx, k_idx] + self.alpha_loc) / (self.ndv_loc[d, s+v_idx]
                            + (self.num_of_loc_topics * self.alpha_loc))
                score = part1 * part2 * part3 * part4
                if score < 0.0:
                    print "local: ", part1, part2, part3, part3
                    print self.nkw_loc[k_idx, wd], self.beta_loc, self.nk_loc[k_idx], self.vocab_size, self.beta_loc
                    print "0 > %f" % score
                p_v_r_k.append(score)

        np_p_v_r_k = np.array(p_v_r_k)
        num = np_p_v_r_k / np_p_v_r_k.sum()
        # print num
        # print num.shape, num
        new_p_v_r_k_idx = np.random.multinomial(1, num).argmax()
        print new_p_v_r_k_idx
        new_v, new_r, new_k = label_v_r_k[new_p_v_r_k_idx]
        return new_v, new_r, new_k

    def run_gibbs_sampling(self, max_iterations=2):

        for gibbs_iteration in range(max_iterations):
            print "Iteration %s" % (gibbs_iteration + 1)
            for d in xrange(self.n_docs):
                print "Document %s" % (d + 1)
                # i is a number between 0 and doc_length-1
                # w is a number between 0 and vocab_size-1
                # for each document, take the doc/word counter and use that to
                # create a long vector that contains each word token (so a word can appear more than once
                # the index "i" indicates the i-th word in the document
                start = timer()
                for s in xrange(int(self.doc_s_count[d])):
                    n = 0
                    for i, wd in enumerate(word_indices(self.doc_sentences_words[d, s, :])):
                        n += 1
                        k = self.doc_w_topics_assgn[(d, i)]
                        v = self.doc_w_window_assgn[(d, i)]
                        r = self.doc_w_gl_loc_assgn[(d, i)]
                        # lower all counts
                        self.lower_counts(d, s, k, v, r, wd)
                        v_new, r_new, k_new = self.sample_k_v_gl_loc(d, s, wd, v)
                        self.increase_counts(d, s, k_new, v_new, r_new, wd)
                        self.doc_w_topics_assgn[(d, i)] = k_new
                        self.doc_w_window_assgn[(d, i)] = v_new
                        self.doc_w_gl_loc_assgn[(d, i)] = r_new
                    print "Number of iteratoins: %d" % n


if __name__ == '__main__':

    """
    parameters:
        (1) preprocess files: "True" or "False" (no boolean but string)
        (2) directory path for input & output files
    """

    if len(sys.argv) == 1:
        preprocess = "False"
        dir_path = 'F:/temp/topics/'
        # dir_path = "/Users/jesse/Desktop/nlp1_project/src/"
    else:
        preprocess = sys.argv[1]
        dir_path = sys.argv[2]

    # inFile = dir_path + "dvd.xml"
    # inFile = dir_path + "dvdReviews.xml"
    # inFile = dir_path + "example.xml"
    # inFile = dir_path + "all.review"
    inFile = dir_path + "example_tiny.xml"
    # mem_file_results = dir_path + "lda_results.mem"
    # mem_file_results = dir_path + "example.mem"
    s_object_file = dir_path + "example_tiny_objects.mem"

    # inFile = sys.argv[2] + "dvd.xml" huge file

    if preprocess == 'True':

        reviews, w, doc_words, docs_sentence_words = preprocessing(inFile)

        print "Save objects to file %s" % s_object_file
        start = timer()
        obj_saved = {'reviews': reviews, 'bag_of_words': w, 'doc_words': doc_words,
                     'doc_sentence_words': docs_sentence_words}
        with open(s_object_file, 'wb') as fs:
            np.savez_compressed(fs, **obj_saved)
        end = timer()
        print "Saved objects to file in %s seconds." % (end - start)
    else:
        with open(s_object_file, 'rb') as fs:
            npz_docs = np.load(fs)
            for obj_id in npz_docs:
                print obj_id
                if obj_id == 'reviews':
                    reviews = npz_docs[obj_id]
                elif obj_id == 'bag_of_words':
                    w = npz_docs[obj_id]
                elif obj_id == 'doc_words':
                    doc_words = npz_docs[obj_id]
                elif obj_id == 'doc_sentence_words':
                    docs_sentence_words = npz_docs[obj_id]
            print "sum of first row in docs-sentence-word %d" % np.sum(docs_sentence_words[0, 2, :])
    # check_doc_word_matrix(doc_words, reviews, w)

    # last parameter is the max number of sentences for corpus
    doc_sentence_count = count_sent_docs(reviews)
    print doc_sentence_count
    # create LDAModel object and initialize counters for Gibbs sampling
    lda = LDAModel(w, docs_sentence_words, doc_sentence_count, K_GL, K_LOC, 0.1, 0.1, 0.1, 0.1,
                   0.1, 0.1, 0.2)
    # initialize counters
    start = timer()
    print "LDA initialize..."
    lda.initialize()
    print docs_sentence_words.shape
    print lda.n_docs,  np.sum(lda.nk_gl), np.sum(lda.nk_loc)
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
    # lda.build_phi_matrix()
    # lda.build_theta_matrix()
    # print lda.phi_dist
    #  print np.sum(lda.phi_dist[1, :])
    # lda.store_results(mem_file_results)

