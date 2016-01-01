import sys
from preprocess_movie import *
from preprocessingv14 import *
from handle_pp_objects import *
from scipy.special import gammaln

# The global number of topics
K_GL = 20
K_LOC = 7
N_GIBBS_SAMPLING_ITERATIONS = 500

global product


def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)


def sample_r():
    """
    determine whether we're going to assign a local or global topic to a word
    :return:
    """
    if np.random.randint(2) == 0:
        return "gl"
    else:
        return "loc"


def count_sent_docs(revs):
    """
    count the number of sentences per document/review
    :param revs: corpus
    :return: vector with number of sentences per doc
    """
    docs_sent_len = np.zeros(len(revs))
    max_number_s = 0
    for r, doc in enumerate(revs):
        docs_sent_len[r] = len(doc)
        if max_number_s < docs_sent_len[r]:
            max_number_s = docs_sent_len[r]

    return docs_sent_len, max_number_s


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

    def __init__(self, l_bag_of_words, doc_sentences_words, doc_s_count, max_number_s, num_of_gl_topics, num_of_loc_topics,
                 alpha_gl, alpha_loc, beta_gl, beta_loc, gamma, alpha_mix_gl, alpha_mix_loc, dir_out):

        self.dir_out = dir_out
        # number of sentences covered by a sliding window. Ivan uses 3 in his paper
        self.n_windows = 3
        # number of docs/reviews, max number of sentences/review in corpus, total vocabulary size of corpus
        self.n_docs = doc_s_count.shape[0]
        self.num_of_max_sentences = max_number_s
        self.vocab_size = len(l_bag_of_words)
        self.num_of_gl_topics = num_of_gl_topics
        self.num_of_loc_topics = num_of_loc_topics
        # vector with number of sentences for each document
        self.doc_s_count = doc_s_count
        print "Num of docs ", self.n_docs, " Vocabulary size ", self.vocab_size

        # for the time being we assume synchronous Dirichlet distributions
        # parameter for Dirichlet prior dist. from which we sample our global/local topics
        self.alpha_gl = alpha_gl
        self.alpha_loc = alpha_loc
        # parameter for Dirichlet prior dist. from which we sample K_GL/K_LOC word/topic distributions
        self.beta_gl = beta_gl
        self.beta_loc = beta_loc
        # parameter for Dirichlet dist. that samples the window covering the sentence
        self.gamma = gamma
        # parameter of Beta distribution from which we sample whether a word will be assigned to a global/local
        # topic. non-symmetrical, so we can regulate whether we prefer global or local topics
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

        # number of words assigned to topic global/local topics
        self.nk_gl = np.zeros(self.num_of_gl_topics)
        self.nk_loc = np.zeros(self.num_of_loc_topics)

        # length of sentence s in document m: here I don't really understand why this is a counter
        # you would imagine that this is constant for the document, right?
        self.nds = np.zeros((self.n_docs, self.num_of_max_sentences))
        # number of times a word from sentence s is assigned to window v. So this matrix isn't that
        # beautiful because of the num_of_max_sentences dimension. So our matrix is very sparse because
        # especially if the number of sentences/doc varies very much in the corpus. But couldn't come up
        # with a better solution. At least I changed the last dimension to the number of sentences a
        # window covers, that will be 3. Be aware that we will often have to add to the "window" number
        # the running index of the sentence.
        self.ndsv = np.zeros((self.n_docs, self.num_of_max_sentences, self.n_windows))

        # number of times a word in document d is assigned to window v
        self.ndv = np.zeros((self.n_docs, self.num_of_max_sentences + 2))
        # number of times a global topic was assigned to document d and window v
        self.ndv_gl = np.zeros((self.n_docs, self.num_of_max_sentences + 2))
        # number of times a local topic was assigned to document d and window v
        self.ndv_loc = np.zeros((self.n_docs, self.num_of_max_sentences + 2))
        # number of local topics in document d and window v assigned to local topic k
        self.ndvk_loc = np.zeros((self.n_docs, self.num_of_max_sentences + 2, self.num_of_loc_topics))

        # dictionary: key is tuple of (docID, sentenceID, wordIdx), value is equal to topic
        self.doc_w_topics_assgn = {}  # k_di
        # assignment of word at position i in document d, sentence s, to window v
        self.doc_w_window_assgn = {}  # v_di
        # assignment of word at position i in document d, sentence s, to global OR local topic
        self.doc_w_gl_loc_assgn = {}  # r_di: gl = global topic; loc = local topic

        # matrix that holds per document the counts for the words (dims: #of_docs X #words_in_bag)
        self.doc_sentences_words = doc_sentences_words
        # the bag of words
        self.bag_of_words = l_bag_of_words
        # phi = p(w|k) distribution, matrix of num of topics X num of words in corpus
        self.phi_dist = np.zeros((self.num_of_gl_topics, self.vocab_size))
        # theta distribution = p(k|d), matrix of num of docs X num of topics
        self.theta_dist = np.zeros((self.n_docs, self.num_of_gl_topics))

        # distribution over words for global topics
        self.phi_dist_gl = np.zeros((self.num_of_gl_topics, self.vocab_size))
        # distribution over words for local topics
        self.phi_dist_loc = np.zeros((self.num_of_loc_topics, self.vocab_size))

        # version 0.3: adding accumulators for nkw_gl & nkw_loc after advice from Wilker
        self.acc_nkw_gl = np.zeros((self.num_of_gl_topics, self.vocab_size))
        self.acc_nkw_loc = np.zeros((self.num_of_loc_topics, self.vocab_size))
        # build separate phi matrix based an accumulators
        self.acc_phi_dist_gl = np.zeros((self.num_of_gl_topics, self.vocab_size))
        self.acc_phi_dist_loc = np.zeros((self.num_of_loc_topics, self.vocab_size))

    def initialize(self):

        for d in xrange(self.n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            # for each document, take the doc/word counter and use that to
            # create a long vector that contains each word token (so a word can appear more than once
            # the index "i" indicates the i-th word in the document
            start = timer()
            # print "doc-sen-count %d" % self.doc_s_count[d]
            for s in xrange(int(self.doc_s_count[d])):
                # print "sentence %d" % (s+1), np.sum(self.doc_sentences_words[d, s, :]),
                # len(self.doc_sentences_words[d, s, :])
                for i, wd in enumerate(self.doc_sentences_words[d][s]):
                    # print i, self.bag_of_words.keys()[wd]
                    # choose one of the three windows that can be associated with the specific sentence
                    # a number between 0-2 (currently with 3 possible windows per sentence
                    # s + 0/1/2 e.g. document sentence 1 can belong to window {1,2,3} and
                    # sentence 2 can belong to window {2,3,4} etc.
                    v = np.random.randint(self.n_windows)
                    self.doc_w_window_assgn[(d, s, i)] = v
                    # choose whether for this word we sample from global or local topics
                    # 0 = global topic, 1 = local topic
                    r = sample_r()
                    self.doc_w_gl_loc_assgn[(d, s, i)] = r
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
                        # print "(d,s,i) (%d,%d,%d) k %d v %d gl word %d %s " % (d, s, i, k, v, wd,
                        # self.bag_of_words.keys()[wd]), self.nkw_gl[k, wd]
                    else:
                        # local topic assignment
                        k = np.random.randint(self.num_of_loc_topics)
                        self.nkw_loc[k, wd] += 1
                        self.ndk_loc[d, k] += 1
                        self.ndv_loc[d, s+v] += 1
                        self.ndvk_loc[d, s+v, k] += 1
                        self.nd_loc[d] += 1
                        self.nk_loc[k] += 1
                        # print "(d,s,i) (%d,%d,%d) k %d v %d loc word %d %s " % (d, s, i, k, v, wd,
                        # self.bag_of_words.keys()[wd]), self.nkw_loc[k, wd]

                    self.doc_w_topics_assgn[(d, s, i)] = k  # assign topic to word in document!

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
            # print "i-is %d lower gl word %d %s %d" % (i, wd, self.bag_of_words.keys()[wd], k), self.nkw_gl[k, wd]
        else:
            self.nkw_loc[k, wd] -= 1
            self.ndk_loc[d, k] -= 1
            self.ndv_loc[d, s+v] -= 1
            self.ndvk_loc[d, s+v, k] -= 1
            self.nd_loc[d] -= 1
            self.nk_loc[k] -= 1
            # print "i-is %s lower loc word %d %s %d" % (i, wd, self.bag_of_words.keys()[wd], k), self.nkw_loc[k, wd]

    def increase_counts(self, d, s, k, v, r, wd):

        self.ndv[d, s+v] += 1
        self.ndsv[d, s, v] += 1
        self.nds[d, s] += 1

        if r == "gl":
            self.nkw_gl[k, wd] += 1
            self.ndk_gl[d, k] += 1
            self.nd_gl[d] += 1
            self.ndv_gl[d, s+v] += 1
            self.nk_gl[k] += 1
        else:
            self.nkw_loc[k, wd] += 1
            self.ndk_loc[d, k] += 1
            self.ndv_loc[d, s+v] += 1
            self.ndvk_loc[d, s+v, k] += 1
            self.nd_loc[d] += 1
            self.nk_loc[k] += 1

    def sample_k_v_gl_loc(self, d, s, wd):

        # sampling topic new_z for t
        p_v_r_k = []
        label_v_r_k = []
        # start = timer()
        # for the number of sliding windows / Ivan uses 3
        for v_idx in xrange(self.n_windows):
            # for the global topics
            # term2 only depends on v_idx, so only calculate ones
            part2 = float(self.ndsv[d, s, v_idx] + self.gamma) / (self.nds[d, s] + (self.n_windows * self.gamma))
            # term3 depends on v_idx and on global/local counter. so calculate once here and once before
            # the local loop
            part3 = float(self.ndv_gl[d, s+v_idx] + self.alpha_mix_gl) / (self.ndv[d, s+v_idx] + self.alpha_mix_gl + self.alpha_mix_loc)
            for k_idx in xrange(self.num_of_gl_topics):
                label = [v_idx, "gl", k_idx]
                label_v_r_k.append(label)
                part1 = float(self.nkw_gl[k_idx, wd] + self.beta_gl) / (self.nk_gl[k_idx] + (self.vocab_size * self.beta_gl))

                part4 = float(self.ndk_gl[d, k_idx] + self.alpha_gl) / (self.nd_gl[d] + (self.num_of_gl_topics * self.alpha_gl))
                score = part1 * part2 * part3 * part4
                p_v_r_k.append(score)
            # for local topics
            part3 = float(self.ndv_loc[d, s+v_idx] + self.alpha_mix_loc) / (self.ndv[d, s+v_idx] + self.alpha_mix_gl + self.alpha_mix_loc)
            for k_idx in xrange(self.num_of_loc_topics):
                label = [v_idx, "loc", k_idx]
                label_v_r_k.append(label)
                part1 = float(self.nkw_loc[k_idx, wd] + self.beta_loc) / (self.nk_loc[k_idx] + (self.vocab_size * self.beta_loc))
                part4 = float(self.ndvk_loc[d, s+v_idx, k_idx] + self.alpha_loc) / (self.ndv_loc[d, s+v_idx] + (self.num_of_loc_topics * self.alpha_loc))
                score = part1 * part2 * part3 * part4
                p_v_r_k.append(score)

        # end = timer()
        # print "build distr. for 1 word %s" % (end - start)
        # start = timer()
        np_p_v_r_k = np.array(p_v_r_k)
        # end = timer()
        # print "np_p_v_r_k = np.array(p_v_r_k) %s" % (end - start)
        # num = np_p_v_r_k / np_p_v_r_k.sum()
        # start = timer()
        new_p_v_r_k_idx = np.random.multinomial(1, np_p_v_r_k / np_p_v_r_k.sum()).argmax()
        # end = timer()
        # print "multinomial time %s" % (end - start)
        new_v, new_r, new_k = label_v_r_k[new_p_v_r_k_idx]
        return new_v, new_r, new_k, np_p_v_r_k

    def calc_loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        likeli_gl = 0
        likeli_loc = 0

        # p(w|r=gl, k)
        for k in xrange(self.num_of_gl_topics):
            likeli_gl += log_multi_beta(self.nkw_gl[k, :]+self.beta_gl)
            likeli_gl -= log_multi_beta(self.beta_gl, self.vocab_size)

        # p(w|r=loc, k)
        for k in xrange(self.num_of_loc_topics):
            likeli_loc += log_multi_beta(self.nkw_loc[k, :]+self.beta_loc)
            likeli_loc -= log_multi_beta(self.beta_loc, self.vocab_size)

        # p(z|r, v)
        for m in xrange(self.n_docs):
            likeli_gl += log_multi_beta(self.ndk_gl[m, :]+self.alpha_gl)
            likeli_gl -= log_multi_beta(self.alpha_gl, self.num_of_gl_topics)

            likeli_loc += log_multi_beta(self.ndk_loc[m, :]+self.alpha_loc)
            likeli_loc -= log_multi_beta(self.alpha_loc, self.num_of_loc_topics)

        # p(v|d,s)
        for d in xrange(self.n_docs):
            likeli_gl += log_multi_beta(self.ndv_gl[d, :] + self.alpha_mix_gl)
            likeli_gl -= log_multi_beta(self.alpha_mix_gl, self.alpha_mix_gl + self.alpha_mix_loc)
            likeli_loc += log_multi_beta(self.ndv_loc[d, :] + self.alpha_mix_loc)
            likeli_loc -= log_multi_beta(self.alpha_mix_loc, self.alpha_mix_gl + self.alpha_mix_loc)
            for s in xrange(int(self.doc_s_count[d])):
                likeli_gl += log_multi_beta(self.ndsv[d, s, :] + self.gamma)
                likeli_gl -= log_multi_beta(self.gamma, self.n_windows)
                likeli_loc += log_multi_beta(self.ndsv[d, s, :] + self.gamma)
                likeli_loc -= log_multi_beta(self.gamma, self.n_windows)

        return likeli_gl, likeli_loc

    def print_counts(self):

        # print "self.ndv ", self.ndv
        # print "self.ndsv ", self.ndsv
        # print "self.nds ", self.nds
        print "self.nkw_gl ", self.nkw_gl
        print "self.ndk_gl ", self.ndk_gl
        print "self.nd_gl ", self.nd_gl
        print "self.ndv_gl ", self.ndv_gl
        print "self.nk_gl ", self.nk_gl

    def run_gibbs_sampling(self, max_iterations=2):
        global product
        global reviews

        for gibbs_iteration in range(max_iterations):
            gibbs_counter = 0
            start = timer()
            print "Iteration %s" % (gibbs_iteration + 1)
            for d in xrange(self.n_docs):
                # print "Document %s" % (d + 1)
                # i is a number between 0 and doc_length-1
                # w is a number between 0 and vocab_size-1
                # for each document, take the doc/word counter and use that to
                # create a long vector that contains each word token (so a word can appear more than once
                # the index "i" indicates the i-th word in the document

                for s in xrange(int(self.doc_s_count[d])):
                    n = 0
                    for i, wd in enumerate(self.doc_sentences_words[d][s]):
                        n += 1

                        k = self.doc_w_topics_assgn[(d, s, i)]
                        v = self.doc_w_window_assgn[(d, s, i)]
                        r = self.doc_w_gl_loc_assgn[(d, s, i)]
                        # print self.bag_of_words[wd], r,v,k
                        # lower all counts
                        self.lower_counts(d, s, k, v, r, wd)
                        # start = timer()
                        v_new, r_new, k_new, joined_p_v_r_k = self.sample_k_v_gl_loc(d, s, wd)
                        # end = timer()
                        # print "multinomial sampling time %s" % (end - start)
                        self.increase_counts(d, s, k_new, v_new, r_new, wd)
                        self.doc_w_topics_assgn[(d, s, i)] = k_new
                        self.doc_w_window_assgn[(d, s, i)] = v_new
                        self.doc_w_gl_loc_assgn[(d, s, i)] = r_new

                    gibbs_counter += n
            # skip the first iteration
            # if gibbs_iteration > 0:
            #    self.gibbs_iter_postprocessing(gibbs_iteration)
            end = timer()
            # log_gl, log_loc = self.calc_loglikelihood()
            # print "Lap time iteration %s / took %d samples / ll_gl %f ll_loc %f" % ((end - start), gibbs_counter, log_gl, log_loc)
            if gibbs_iteration > 1 and (gibbs_iteration % 20 == 0 or gibbs_iteration+1 == N_GIBBS_SAMPLING_ITERATIONS):
                self.store_acc_phi_matrices(self.dir_out + str(gibbs_iteration) + "_phi_" + product + ".mem")

        self.build_phi_matrix_gl()
        self.build_phi_matrix_loc()

    def debug_phis(self):
        nkwgl_aug = self.nkw_gl + self.beta_gl
        nkwgl_aug /= np.sum(nkwgl_aug, axis=1)[:, np.newaxis]
        print np.sum(nkwgl_aug, axis=1)
        nkwgl_aug = self.nkw_loc + self.beta_loc
        nkwgl_aug /= np.sum(nkwgl_aug, axis=1)[:, np.newaxis]
        print np.sum(nkwgl_aug, axis=1)

    def gibbs_iter_postprocessing(self, i_gibbs):
        global N_GIBBS_SAMPLING_ITERATIONS

        self.acc_nkw_gl = self.nkw_gl + self.beta_gl
        self.acc_nkw_loc = self.nkw_loc + self.beta_loc
        if i_gibbs % 10 == 0 or i_gibbs+1 == N_GIBBS_SAMPLING_ITERATIONS:
            self.build_acc_phi_matrix_gl(i_gibbs)
            self.build_acc_phi_matrix_loc(i_gibbs)
            self.store_acc_phi_matrices(self.dir_out + str(i_gibbs) + "_phi_accu.mem")

    def build_phi_matrix_gl(self):
        # phi is a matrix of dimension: (num of topics X num of words in corpus)
        # and contains for each topic/word combination the probability for a word belonging to that topic k

        nkw_aug = self.nkw_gl + self.beta_gl
        for k in range(self.num_of_gl_topics):
            self.phi_dist_gl[k, :] = nkw_aug[k, :] * 1/np.sum(nkw_aug[k, :])

    def build_phi_matrix_loc(self):
        # phi is a matrix of dimension: (num of topics X num of words in corpus)
        # and contains for each topic/word combination the probability for a word belonging to that topic k

        nkw_aug = self.nkw_loc + self.beta_loc
        for k in range(self.num_of_loc_topics):
            self.phi_dist_loc[k, :] = nkw_aug[k, :] * 1/np.sum(nkw_aug[k, :])

    def build_acc_phi_matrix_gl(self, i_gibbs):

        nkw_aug = self.acc_nkw_gl * 1/i_gibbs
        for k in range(self.num_of_gl_topics):
            self.acc_phi_dist_gl[k, :] = nkw_aug[k, :] * 1/np.sum(nkw_aug[k, :])

    def build_acc_phi_matrix_loc(self, i_gibbs):

        nkw_aug = self.acc_nkw_loc * 1/i_gibbs
        for k in range(self.num_of_loc_topics):
            self.acc_phi_dist_loc[k, :] = nkw_aug[k, :] * 1/np.sum(nkw_aug[k, :])

    def store_acc_phi_matrices(self, save_file):

        h5f = h5py.File(save_file, 'w')
        h5f.create_dataset('acc_phi_dist_gl', data=self.acc_phi_dist_gl)
        h5f.create_dataset('acc_phi_dist_loc', data=self.acc_phi_dist_loc)
        h5f.close()

    def store_counters(self, save_file):
        h5f = h5py.File(save_file, 'w')
        h5f.create_dataset('nkw_gl', data=self.nkw_gl)
        h5f.create_dataset('nkw_loc', data=self.nkw_loc)
        h5f.create_dataset('ndv', data=self.ndv)
        h5f.create_dataset('ndsv', data=self.ndsv)
        h5f.create_dataset('nds', data=self.nds)
        h5f.create_dataset('ndk_gl', data=self.ndk_gl)
        h5f.create_dataset('nd_gl', data=self.nd_gl)
        h5f.create_dataset('ndv_gl', data=self.ndv_gl)
        h5f.create_dataset('nk_gl', data=self.nk_gl)
        h5f.create_dataset('ndk_loc', data=self.ndk_loc)
        h5f.create_dataset('ndv_loc', data=self.ndv_loc)
        h5f.create_dataset('ndvk_loc', data=self.ndvk_loc)
        h5f.create_dataset('nd_loc', data=self.nd_loc)
        h5f.create_dataset('nk_loc', data=self.nk_loc)

        h5f.close()

    def store_results(self, save_file):
        # store the theta and phi matrix
        h5f = h5py.File(save_file, 'w')
        h5f.create_dataset('phi_global', data=self.phi_dist_gl)
        h5f.create_dataset('phi_local', data=self.phi_dist_loc)
        h5f.close()

if __name__ == '__main__':

    """
    parameters:
        (1) preprocess files: "True" or "False" (no boolean but string)
        (2) directory path for input & output files
    """
    global reviews

    if len(sys.argv) == 1:
        preprocess = "True"
        # dir_path = 'F:/temp/topics/D - data/movie/test/'
        product = "electronics"
        dir_path = 'S:/Workspace/data/sports/'
    else:
        preprocess = sys.argv[1]
        dir_path = sys.argv[2]

    mem_file_results = dir_path + "mglda_" + product + "_" + str(N_GIBBS_SAMPLING_ITERATIONS) + ".mem"

    if preprocess == 'True':
        if product != "movie":
            inFile = dir_path +  "all.review.xml"
            reviews, d_vocab, l_bag_of_words, m_doc_words, m_docs_sentence_words = preprocessing(inFile)
        else:
            reviews, d_vocab, l_bag_of_words, m_doc_words, m_docs_sentence_words = buildvocab(dir_path, 3, '0')
        save_objects(dir_path, product, reviews, d_vocab, l_bag_of_words, m_doc_words, m_docs_sentence_words)
    else:
        reviews, d_vocab, l_bag_of_words, m_doc_words, m_docs_sentence_words = load_objects(dir_path, product)

    # check_doc_word_matrix(doc_words, reviews, w)
    # last parameter is the max number of sentences for corpus
    doc_sentence_count, max_number_s = count_sent_docs(reviews)
    # create LDAModel object and initialize counters for Gibbs sampling
    lda = LDAModel(l_bag_of_words, m_docs_sentence_words, doc_sentence_count, max_number_s, K_GL, K_LOC, 0.005, 0.005, 0.005, 0.005,
                   0.005, 0.005, 0.005, dir_path)
    # initialize counters
    start = timer()
    print "LDA initialize..."
    lda.initialize()
    # lda.print_counts()
    end = timer()
    print "LDA initialize time %s" % (end - start)
    # run Gibbs sampling, parameter is number of times we run Gibbs
    start = timer()
    num_of_iterations = N_GIBBS_SAMPLING_ITERATIONS
    print "Gibbs sampling for %s" % num_of_iterations, " iterations..."
    lda.run_gibbs_sampling(num_of_iterations)
    end = timer()
    print "Gibbs sampling time %s" % (end - start)
    lda.store_results(mem_file_results)

    mem_file_counter = dir_path + product + "_counters_" + str(N_GIBBS_SAMPLING_ITERATIONS) + ".mem"
    lda.store_counters(mem_file_counter)

    # print w['newsweek']
    # print np.sum(doc_words[:, w['newsweek']])
    # print w['unseen']
    # print np.sum(doc_words[:, w['unseen']])
    # print w['unseen']
    # print np.sum(doc_words[:, w['unseen']])
    # print w['magazine']
    # print np.sum(doc_words[:, w['magazine']])

    # magazine magazim  newsweekli newsweek



