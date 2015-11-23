import numpy as np

# D: Total number of documents
# K: Total number of classes
# n_iterations: Total number of Gibbs sampling iterations
'''
vocabulary: np.array containing all words with corresponding word counts; Size of vocabulary is denoted by V
n_iterations: number of iterations for the Gibbs sampling algorithm
K: Total number of topics
D: Total number of documents
alpha: np.array containing K values 
beta: np.array containg V values
'''
def lda_gibs_sampling(vocabulary, n_iterations, K, D, alpha, beta):
	# Setup parameters
	(N, 1) = vocabulary.shape # N = number of words in vocabulary
	z = np.random.randint(0, K, size=N) # Initialize topic assignments to something random
	n_dk = np.zeros((D, K)) 
	n_kw = np.zeros((K, N))
	n_k = np.zeros((K, 1))

	# Repeat the sampling process 'n_iterations' times
	for n in range(n_iterations):
		# Repeat for every word in vocabulary
		for i in range(N):
			word = vocabulary[i]
			topic = z[i]
			
			n_dk[d, topic] -= 1; n_kw[word, topic] -= 1; n_k[topic] -= 1
a
			p_zk = np.zeros((K, 1))
			for k in range(K):
				p_zk[k] = (n_dk[d, k]+alpha[k]) * (n_kw[word, k]+beta[word])/(n_k[k]+beta*W)

			topic = np.random.choice(range(K), p_zk)
			z[i] = topic
			n_dk[d, topic] += 1; n_kw[word, topic] += 1; n_k[topic] += 1
