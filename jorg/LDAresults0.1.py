import sys
import pickle

file_path_matrices = sys.argv[1]
file_path_corpus = sys.argv[2]
number_of_topics = int(sys.argv[3])

print file_path_matrices
print file_path_corpus

with open(file_path_matrices, 'rb') as f:
	theta_mat = pickle.load(f)
	phi_mat = pickle.load(f)
f.close()

with open(file_path_corpus, 'rb') as f:
	pickle.load(f)
	w = pickle.load(f)
        #doc_words = pickle.load(f)
f.close()

for topic in range(number_of_topics):
	ind = sorted(xrange(len(phi_mat[topic, :])), key=lambda x:phi_mat[topic, x], reverse=True)
        print "Topic %d" % topic
        print phi_mat[topic, ind]
        print "----------------------------"
	print [w.keys()[word] for word in ind[0:20]]
