from os import listdir
from os.path import isfile, join
import numpy as np
import nltk
import string
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import EnglishStemmer


vocab = dict()
bag_of_words = []


def getStemmer(stem_response):

    if stem_response == '0':
        # print "Using no stemmer..."
        stemmer = None
    elif stem_response == '1':
        # print "Using PorterStemmer..."
        stemmer = PorterStemmer()
    elif stem_response == '2':
        stemmer = LancasterStemmer()
    elif stem_response == '3':
        stemmer = EnglishStemmer()
    return stemmer


def build_stop_words(path):
    global stop_words_dict
    with open(path + 'stopwords.txt') as f:
        stopwords = f.read().lower().split()
    stop_words_dict = dict(zip(stopwords, stopwords))


def count_docs_sentences(revs):
    """
    count the number of sentences per document/review
    :param revs: corpus
    :return: vector with number of sentences per doc
    """
    docs_sent_len = np.zeros(len(revs))
    for r, doc in enumerate(revs):
        docs_sent_len[r] = len(doc)

    return docs_sent_len


def get_file_names(path):
    return [path + "/" + f for f in listdir(path) if isfile(join(path, f))]


def remove_punct(in_str):
    import re
    return re.sub("[^\w]", " ", in_str)


def stem_tokens(tokens, stemmer):
    return [stemmer.stem(token) for token in tokens]


def remove_stop_and_short_words(token_words, min_word_size=2):
    global stop_words_dict
    clean_words = []
    for word in token_words:
        if len(word) >= min_word_size and word not in stop_words_dict:
            clean_words.append(word)
    return clean_words


def file_to_tokens(file_name, min_word_size, stemmer=None):
    global bag_of_words

    with open(file_name) as f:
        # print "\nProcessing file %s" % file_name
        file_string = f.read()
        file_string = filter(lambda x: x in string.printable, file_string)
    review = []
    file_string = file_string.rstrip('\n')
    for sent in nltk.sent_tokenize(file_string):
        tokens = remove_punct(sent).split()
        tokens = remove_stop_and_short_words(tokens, min_word_size)
        if stemmer:
            tokens = stem_tokens(tokens, stemmer)
        for token in tokens:
            if token not in bag_of_words:
                bag_of_words.append(token)
        if len(tokens) > 0:
            review.append(tokens)

    return review


def buildvocab(path, min_word_size, stem_ind, debug=False):
    print "\nBuilding the dictionary..."
    global vocab
    global bag_of_words
    global stop_words_dict
    reviews = []
    build_stop_words(path)
    #
    all_file_names = get_file_names(path + 'pos') + get_file_names(path + 'neg')
    # Build the vocab
    for file_name in all_file_names:
        # the below call to file_to_tokens does not need to stem, just leave
        # off the stemmer arg (Porter, Lancaster, English)
        stemmer = getStemmer(stem_ind)
        reviews.append(file_to_tokens(file_name, min_word_size, stemmer))

    # fill dictionary that will be used when we create the co-occurrence matrices
    vocab = dict([(bag_of_words[i], i) for i in range(len(bag_of_words))])
    doc_words, docs_sentence_words = create_doc_word_matrix(reviews)

    # look at dictionary in debug mode
    if debug:
        words_per_page = 20
        vocab_debug_printer(words_per_page)
    print "Processed Reviews: %d, Vocab-size: %d" % (len(reviews), len(vocab))
    print "Done."
    return reviews, vocab, bag_of_words, doc_words, docs_sentence_words


def vocab_debug_printer(words_per_page):
    count = 0
    for word in vocab:
        print "\nVocabulary list is {} words long.".format(len(vocab))
        if count % words_per_page == 0:
            resp = raw_input(
                '''please press enter to see next {}, or press 's' to skip printing vocabulary
                '''.format(words_per_page))
            if resp == 's':
                break
        else:
            print word + " {}".format(vocab[word])
        count += 1
    print "\nWARNING: debug mode produces inaccurate dictionary build times!"


def create_doc_word_matrix(docs):
    global vocab

    # vector that holds for each doc the word counts
    dw = np.zeros(len(vocab))
    # doc_words is a matrix of size "num of docs" X "num of words corpus"
    docs_words_m = np.zeros((len(docs), len(vocab)))
    docs_sentence_words = []

    print ("\nCreating doc word matrix...")
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
                idx = vocab[wd]
                docs_words_m[m, idx] += 1
                sent_words.append(idx)
            doc_sent.append(sent_words)
        docs_sentence_words.append(doc_sent)

    # print docs_sentence_words
    print("\nFinished creating doc word matrix.")

    return docs_words_m, docs_sentence_words
