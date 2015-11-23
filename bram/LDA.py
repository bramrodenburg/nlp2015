import numpy as np

'''
vocabulary: np.array containing all words with corresponding word counts; Size of vocabulary is denoted by V
n_iterations: number of iterations for the Gibbs sampling algorithm
K: Total number of topics
D: Total number of documents
alpha: np.array containing K values 
beta: np.array containg V values
'''
def lda_gibs_sampling(vocabulary, n_iterations, K, D, alpha, beta):
	(V, ) = vocabulary.shape # V = number of words in vocabulary
	z = np.random.randint(0, K, size=V) # Initialize topic assignments to something random
	n_dk = np.zeros((D, K)) # Topic counts for each document
	n_kw = np.zeros((K, V)) # Word counts for each topic
	n_k = np.zeros((K, 1)) # Topic counts

	# Repeat the sampling process 'n_iterations' times
	for n in range(n_iterations):
		# Repeat for each document
		for d in range(D):
			# Repeat for every word in vocabulary
			for i in range(V):
				word = vocabulary[i]
				topic = z[i]
			
				n_dk[d, topic] -= 1; n_kw[word, topic] -= 1; n_k[topic] -= 1
			
				p_zk = np.zeros((K, 1))
				for k in range(K):
					p_zk[k] = (n_dk[d, k]+alpha[k]) * (n_kw[word, k]+beta[word])/(n_k[k]+beta*W)

				topic = np.random.choice(range(K), p_zk)
				z[i] = topic
				n_dk[d, topic] += 1; n_kw[word, topic] += 1; n_k[topic] += 1
