import pickle
from timeit import default_timer as timer
import h5py
import numpy as np


file1_suffix = "_dic_words.pkl"
file2_suffix = "_rev_bow.savez"
file3_suffix = "_dw_dsw.h5f"


def save_objects(path, product, l_reviews, d_vocab, l_bag_of_words, m_doc_words, m_docs_sentence_words):

    print "Save objects to files in directory %s" % path
    start = timer()
    picklefile = path + product + file1_suffix
    with open(picklefile, 'wb') as f:
        pickle.dump(d_vocab, f)

    numpy_file = path + product + file2_suffix
    obj_saved = {'l_reviews': l_reviews, 'l_bag_of_words': l_bag_of_words,
                 'm_docs_sentence_words': m_docs_sentence_words}
    with open(numpy_file, 'wb') as fs:
        np.savez_compressed(fs, **obj_saved)

    h5_file = path + product + file3_suffix
    h5f = h5py.File(h5_file, 'w')
    h5f.create_dataset('doc_words', data=m_doc_words)
    h5f.close()
    end = timer()
    print "Saved objects to file in %s seconds." % (end - start)


def load_objects(path, product):
    print "Load objects from directory %s" % path

    picklefile = path + product + file1_suffix
    with open(picklefile, 'rb') as f:
        d_vocab = pickle.load(f)

    numpy_file = path + product + file2_suffix
    with open(numpy_file, 'rb') as fs:
        npz_docs = np.load(fs)
        for obj_id in npz_docs:
            print obj_id
            if obj_id == 'l_reviews':
                l_reviews = npz_docs[obj_id]
            elif obj_id == 'l_bag_of_words':
                l_bag_of_words = npz_docs[obj_id]
            elif obj_id == 'm_docs_sentence_words':
                m_docs_sentence_words = npz_docs[obj_id]

    h5_file = path + product + file3_suffix
    h5f = h5py.File(h5_file, 'r')
    m_doc_words = h5f['doc_words'][:]
    h5f.close()

    return l_reviews, d_vocab, l_bag_of_words, m_doc_words, m_docs_sentence_words
