import h5py
import pickle
import numpy as np
from handle_pp_objects import *


def load_bag_of_words(infile):

    with open(infile, 'rb') as f:
        w = pickle.load(f)
        return w


def load_nkw_counters(infile):

        h5f = h5py.File(infile, 'r')
        nkw_gl = h5f['nkw_gl'][:]
        nkw_loc = h5f['nkw_loc'][:]
        h5f.close()
        return nkw_gl, nkw_loc


def show_topics(phi, topic_type, l_vocab):

    for topic in range(phi.shape[0]):
        ind = sorted(xrange(len(phi[topic, :])), key=lambda x: phi[topic, x], reverse=True)
        print "================ " + topic_type + " %d ================" % (topic + 1)
        print [l_vocab[word] for word in ind[0: 20]]


def show_results(dir_path, product, num_of_iters):

    mem_file_results = dir_path + "mglda_" + product+ "_" + num_of_iters + ".mem"
    # picklefile = dir_path + "movie_dic_words.pkl"
    # h5_file = dir_path + "movie_dw_dsw.h5"
    mem_file_counter = dir_path + product + "_counters_" + num_of_iters + ".mem"

    lda_results = h5py.File(mem_file_results, 'r')
    phi_global = lda_results['phi_global'][:]
    phi_local = lda_results['phi_local'][:]

    lda_results.close()

    reviews, d_vocab, l_bag_of_words, m_doc_words, m_docs_sentence_words = load_objects(dir_path, product)

    nkw_gl, nkw_loc = load_nkw_counters(mem_file_counter)

    # d_vocab = load_bag_of_words(picklefile)

    show_topics(phi_global, "Global", l_bag_of_words)

    for k in range(phi_global.shape[0]):
        print "================ Global Topic %d ==========================" % (k + 1)
        for word in np.argsort(-phi_global[k])[:20]:
            print "%s: %f" % (l_bag_of_words[word], phi_global[k, word])
            # print "%s: %f" % (d_vocab.keys()[word], phi_global[k, word])

    show_topics(phi_local, "Local", l_bag_of_words)

    for k in range(phi_local.shape[0]):
        print "================ Local Topic %d ==========================" % (k + 1)
        for word in np.argsort(-phi_local[k])[:20]:
            print "%s: %f" % (l_bag_of_words[word], phi_local[k, word])


def check_counters(infile1, infile2):

    nkw_gl, nkw_loc = load_nkw_counters(infile1)
    w = load_bag_of_words(infile2)

    m_idx = w['magazin']
    print nkw_gl[:, m_idx]


def load_accumulator_results(infile_acc, bag_w_file):
    h5f = h5py.File(infile_acc, 'r')
    acc_phi_dist_gl = h5f['acc_phi_dist_gl'][:]
    acc_phi_dist_loc = h5f['acc_phi_dist_loc'][:]
    h5f.close()
    with open(bag_w_file, 'rb') as f:
        w = pickle.load(f)

    show_topics(acc_phi_dist_gl, "Global", w)
    for k in range(acc_phi_dist_gl.shape[0]):
        print "***** Global Topic %d" % (k + 1)
        for word in np.argsort(-acc_phi_dist_gl[k])[:20]:
            print "%s: %f" % (w.keys()[word], acc_phi_dist_gl[k, word])

    show_topics(acc_phi_dist_loc, "Local", w)
    for k in range(acc_phi_dist_loc.shape[0]):
        print "***** Local Topic %d" % (k + 1)
        for word in np.argsort(-acc_phi_dist_loc[k])[:20]:
            print "%s: %f" % (w.keys()[word], acc_phi_dist_loc[k, word])

if __name__ == '__main__':

    # show_results(dir_path)
    # check_counters(dir_path + "magazine_counters.mem", dir_path + "magazines.pkl")
    dir_path = 'F:/temp/topics/R -results/26/'
    show_results(dir_path, "software", "500")
    # load_accumulator_results(dir_path + "300_phi_accu.mem", dir_path + "magazines.pkl")

