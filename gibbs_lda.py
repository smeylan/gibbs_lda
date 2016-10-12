from __future__ import print_function
import pandas, numpy as np, time, scipy, sys, sklearn, numba, pdb, json, hashlib, itertools, cStringIO, pickle, os, nltk, pathos
from numba import jit
from IPython.display import display, clear_output

class LDA():
	def __init__(self, V,D, vocab_map, params):
		'''Initialize the LDA model'''
		
		self.V = V 
		self.D = D		
		self.vocab_map = vocab_map
		self.params = params		

		if self.params['lda_k'] > 255:
			raise ValueError('Limit to 255 topics to allow for unsigned 8-bit representations of topic assignments')

		self.Z = np.random.randint(0,self.params['lda_k'], len(self.V)).astype('uint8')

		#setup the sampling	table			
		self.ZZ = None		

		print('Initialized LDA model with '+str(len(self.V))+ ' tokens, '+str(len(np.unique(self.V)))+' types and '+str(len(np.unique(self.D)))+' documents')
		print('LDA model k: '+str(self.params['lda_k'])+'; alpha: '+str(self.params['lda_alpha'])+'; beta: '+str(self.params['lda_beta']))
		return(None)

	def update(self, iterations, sample, thinning):	

		if sample:
			if (iterations % thinning) != 0:
				raise ValueError('Iterations must be a multiple of thinning')

		# build book-keeping variables
		#k*types; built from Z and V; aggregate Z according to V
		topicTypeCounts = np.vstack([augmentedBincount(self.Z[self.V == i], self.params['lda_k']).astype('uint32') for i in range(np.max(self.V)+1)]).T

		#k*1, from the row marginals of topicTypeCounts
		topicCounts = topicTypeCounts.sum(1)

		#k*d; built from Z and D
		topicDocCounts =  np.vstack([augmentedBincount(self.Z[self.D == i], self.params['lda_k']).astype('uint32') for i in range(np.max(self.D)+1)]).T

		#d*1, column marginals from topicDocCounts
		docCounts = topicDocCounts.sum(0)
		
		ZZ, LL, topicTypeCounts, topicDocCounts = fast_lda(iterations,self.Z,self.D,self.V,topicTypeCounts,topicCounts,topicDocCounts,docCounts, self.params['lda_beta'], self.params['lda_alpha'], self.params['lda_k'], sample=sample, thinning=thinning)

		self.Z = ZZ[-1,:] #new Z is the final line of ZZ, regardless of the number of samples in ZZ		
		

		if sample:
			self.ZZ = ZZ
		else:
			# do a sanity check on the new Z against topicTypeCounts and topicDocCounts to confirm that the bookkeeping is consistent. Thinning means that the final sample in ZZ is not consistent with the bookkeeping variables.				
			if not (topicTypeCounts == np.vstack([augmentedBincount(self.Z[self.V == i], self.params['lda_k']).astype('uint32') for i in range(np.max(self.V)+1)]).T).all():			
				raise ValueError("Bookkeeping variable topicTypeCounts is inconsistent with final topic assignments")

			if not (topicDocCounts ==  np.vstack([augmentedBincount(self.Z[self.D == i], self.params['lda_k']).astype('uint32') for i in range(np.max(self.D)+1)]).T).all():
				raise ValueError("Bookkeeping variable topicDocCounts is inconsistent with final topic assignments")

		if hasattr(self, 'LL'):
			self.LL = np.hstack([self.LL, LL]) 
		else:
			self.LL = LL	
			
		rewriteLine('LDA: updated all token -> topic assignments')			
		return(self)
	
	def burnIn(self, iterations, thinning):
		''' Convenience wrapper for update for burning in the model''' 
		print('Burning in the LDA model for '+str(iterations)+' samples')
		self.update(iterations,sample=False, thinning=thinning)
		return(self)
	
	def sample(self, iterations, thinning):
		''' Convenience wrapper for update for sampling from the model''' 
		print('Sampling from the LDA model for '+str(iterations)+' samples with thinning of '+str(thinning))
		self.update(iterations,sample=True, thinning=thinning)
		return(self)

	def getTopicDistributionsForDocuments(self):
		''' Get a list of topics in each document, ordered by the relative importance to the document '''
		numDocs = len(np.unique(self.D))				
		self.countsPerToken = np.apply_along_axis(augmentedBincount, 0, self.ZZ, self.params['lda_k']).astype('uint8')

		#aggregate each row by the V indices so that we have an estimate for each word
		countsPerTopicPerDocument = np.matrix([[np.sum(self.countsPerToken[topic,np.where(self.D == i)]) for i in range(numDocs)] for topic in range(self.params['lda_k'])])
		normed_matrix = np.round(sklearn.preprocessing.normalize(countsPerTopicPerDocument*1., axis=0, norm='l1').T, decimals = 8)

		self.TopicDistributionsForDocuments = normed_matrix
		return(normed_matrix) 		
		
	def inspectTopics(self):
		''' Get a Pandas DF of the types in each topic, ordered by their importance to the topic '''
		self.countsPerToken = np.apply_along_axis(augmentedBincount, 0, self.ZZ, self.params['lda_k'])

		#aggregate each row by the V indices so that we have an estimate for each word
		#this is hella slow
		countsPerTopicPerWord = np.matrix([[np.sum(self.countsPerToken[topic,np.where(self.V == i)]) for i in range(len(np.unique(self.V)))] for topic in range(self.params['lda_k'])])		
		
		topicCollection = []
		for i in range(self.params['lda_k']):
			order = np.squeeze(np.asarray(np.fliplr(countsPerTopicPerWord[i,:].argsort())))
			#order of the vocabulary
			word = np.array(self.vocab_map)[order]
			count = np.squeeze(np.asarray(countsPerTopicPerWord[i,:]))[order]
			normalizedCount = np.round(count / (sum(count) * 1.), decimals=8)
			df = pandas.DataFrame({'order': range(len(order)), 'count': count, 'normalizedCount':normalizedCount})
			df['word'] =  word
			df['topic'] = i
			topicCollection.append(df)
		combined_df = pandas.concat(topicCollection)	
		combined_df.sort_values(by=['topic', 'normalizedCount'])		

		self.WordDistributionsForTopics = combined_df
		return(combined_df)

	def cache(self):				
		print('Cacheing LDA samples...')		
		cache_path = self.get_cache_signature() 		
		# pickle the state of the object
		pickle.dump({'ZZ': self.ZZ, 'Z': self.Z, 'LL': self.LL}, open( cache_path, "wb" ))		


	def load_cache(self):			
		print('Loading cached LDA samples...')
		cache_path = self.get_cache_signature() 		
		cache = pickle.load(open( cache_path, "rb" ))

		self.ZZ = cache['ZZ']
		self.Z = cache['Z']
		self.LL = cache['LL'] 

	def cache_exists(self):
		return(os.path.exists(self.get_cache_signature()))		


	def get_cache_signature(self) :	
		'''check if this set of parameters has been run before. Only checks uniqueness of params, not V or D'''

		# jsonify the LDA parameters
		lda_keys  = [x for x in self.params.keys() if x.startswith('lda')]
		lda_params = { new_key: self.params[new_key] for new_key in lda_keys }

		run_params = json.dumps(lda_params)				
		# hash the resulting string

		file_hash = hashlib.sha256(run_params).hexdigest()
		file_path = os.path.join(self.params['cache_dir'],file_hash+'.pickle')			
		return(file_path)
		
	def getDistribution(self, word):
		'''return the normalized distribution over topics'''
		if word is not None:
			typeIndex  = np.argwhere(np.array(self.vocab_map)== word)[0][0]
			wordIndices = np.array(self.V == typeIndex)
		else:
			wordIndices = np.ones(np.shape(self.V)[0]).astype(bool)
		wordMatrix = self.countsPerToken[:,wordIndices].astype(np.float)
		wc = wordMatrix.sum(1)
		wc = wc/np.sum(wc)
		return(wc)    

	def getN(self, word):
		'''return the number of tokens for that word in the model'''
		typeIndex = np.argwhere(np.array(self.vocab_map) == word)[0][0]
		return(len(np.argwhere(np.array(self.V == typeIndex))))

	def getHighestProbTopic(self, word):
		'''return the index of the highest probability topic'''
		return(np.argmax(self.getDistribution(word)))    
    	
	def getDivergence(self):
		self.wordDF = pandas.DataFrame({'word':self.vocab_map})
		baseline = self.getDistribution(None)
		self.wordDF['kl'] = [scipy.stats.entropy(pk = self.getDistribution(x), qk=baseline) for x in self.wordDF['word']]
		self.wordDF['n'] = [self.getN(x) for x in self.wordDF['word']]
		self.wordDF['log_n'] = np.log(self.wordDF['n'])
		self.wordDF['primary_k'] = [self.getHighestProbTopic(x) for x in self.wordDF['word']]     	
		

@jit(nopython=True)
def fast_lda(iterations,Z,D,V,topicTypeCounts,topicCounts,topicDocCounts,docCounts, lda_beta, lda_alpha, lda_k, sample, thinning):	
	# this compiles down to C with Numba (@jit(nopython=True)), so it uses a restricted set of numpy functions

	#precompute constants
	denom_beta = np.float(len(Z)) * lda_beta	
	denom_k_alpha = np.float(lda_k) * lda_alpha
	
	#initialize return variable	
	likelihoodSampleRate = 100
	LL = np.zeros((iterations/likelihoodSampleRate), dtype=np.float32)
	if sample:
		ZZ = np.zeros((iterations/thinning, len(Z)),dtype=np.uint8)		
	else:
		ZZ = np.zeros((1, len(Z)), dtype=np.uint8)	

			

	for i in range(iterations):		
		L = np.zeros(len(Z), dtype=np.float32)	
		for w in range(len(Z)):
			
			currentTopic = Z[w]
			currentDocument = D[w]
			currentWord = V[w]

			#update the bookkeeping variables
			topicTypeCounts[currentTopic, currentWord] -= 1
			topicCounts[currentTopic] -= 1
			topicDocCounts[currentTopic, currentDocument] -= 1
			docCounts[currentDocument] -= 1

			topicP = np.zeros(lda_k)
			for proposedTopic in range(lda_k):
				#print('reached topic proposals')
				# see Griffiths and Steyvers (2004) Equation 5 for the Gibbs Sampler

				#topicTypeCounts[proposedTopic, currentWord]: n^(w_i)_(-i,j), number of times this type is assigned to topic j, exlcuding the current token 

				#topicCounts[proposedTopic]: n^(.)_-i,j, total number of tokens assigned to topic j, excluding current token

				#topicDocCounts[proposedTopic, currentDocument]: n^(d_i)_(-i,j), number of tokens in document i assigned to topic j, not including the current
				
				#docCounts[proposedTopic] : n^(d_i)_(-i), number of tokens in document i, excluding the current token
				num = (topicTypeCounts[proposedTopic, currentWord] + lda_beta) * (topicDocCounts[proposedTopic, currentDocument] + lda_alpha)						
				den = (topicCounts[proposedTopic] + denom_beta) * (docCounts[currentDocument] + denom_k_alpha)
				topicP[proposedTopic] = num / den

			#print('reached assignment')				
			normalizedP = topicP / np.sum(topicP) # normalize										
			rs = np.random.random_sample()			
			cumulativeTopicProb = np.cumsum(normalizedP)
			newTopic = np.uint8(0)			
			while rs > cumulativeTopicProb[newTopic]: 
				newTopic += np.uint8(1)			

			Z[w] = newTopic
			L[w] = np.log(normalizedP[newTopic])

			#update all of the bookkeeping variables
			topicTypeCounts[newTopic, currentWord] += 1
			topicCounts[newTopic] += 1
			topicDocCounts[newTopic, currentDocument] += 1
			docCounts[currentDocument] += 1			

		#store Z depending on thinning and sampling
		if sample and (i % thinning == 0):						
			ZZ[i/thinning,:] = Z			
		else:
			ZZ[0,:] = Z			
		if i % likelihoodSampleRate == 0:
			LL[i/likelihoodSampleRate] = np.sum(L)	

	
	return(ZZ, LL, topicTypeCounts, topicDocCounts)			

def augmentedBincount(vector, max):
	''' Return the counts for values 0...max (i.e. padded with 0's if n < max have 0 counts)'''
	rv = np.zeros(max)
	bc = np.bincount(vector)
	rv[0:len(bc)] = bc
	return rv	

def rewriteLine(str):
	''' Clear buffer and re-write to stdout to avoid too many print statements when in a notebook '''
	# this is a variant of print that rewrites the line each time
	clear_output(wait=True)
	print(str)
	sys.stdout.flush()	


def cleanInput(V, D, K, C, vocab_map, fwList, tolower=True, lemmatize=False, tokenLimit=2, docLimit=2, childLimit=None, corpusLimit=None):
	
	def n_unique(x):
		return(len(np.unique(x)))
		
	corpDF = pandas.DataFrame([vocab_map[x] for x in V])
	corpDF.columns = ['word']
	corpDF.loc[:,'D'] = D
	corpDF.loc[:,'C'] = C
	corpDF.loc[:,'K'] = K
	
	if tolower:
		# if true, all words converted to lower case
		corpDF.loc[:,'word'] = [x.lower() for x in corpDF['word']]

	if fwList is not None:
		#if set, remove function words from the dataset
		functionWords = pandas.read_csv(fwList)	
		fwl = functionWords['word'].tolist()
		corpDF.loc[:,'fw'] = [str(x) in fwl  for x in corpDF['word']] 
		corpDF = corpDF.loc[~corpDF['fw']]	

	if lemmatize:
		#if true, all words lemmatized using the WordNet lemmatizer in NLTK
		wnl = nltk.stem.WordNetLemmatizer()
		corpDF.loc[:,'word'] = [wnl.lemmatize(x) for x in corpDF['word']]
	
	if tokenLimit is not None:
		#if set, discard word types with fewer than tokenLimit instances in the corpus
		tokenCounts = corpDF.groupby(corpDF['word'])['D'].agg(len).reset_index()
		tokenCounts.columns = ['word','count']	
		passingTokenFilter = tokenCounts[tokenCounts['count'] > tokenLimit]['word'].tolist()
		corpDF = corpDF.loc[corpDF['word'].isin(passingTokenFilter)]	
	

	if docLimit is not None:	
		#if set, remove words appearing in fewer than docLimit documents
		
		docCounts = corpDF.groupby(corpDF['word'])['D'].agg(n_unique).reset_index()
		docCounts.columns = ['word','count']	
		passingDocFilter = docCounts[docCounts['count'] > docLimit]['word'].tolist()
		corpDF = corpDF.loc[corpDF['word'].isin(passingDocFilter),:]				
	
	if childLimit is not None:
		#if set, remove words appeaning less than n times with m *children*. This is much more stringent than the docLimit approach
		
		childTokenCounts = corpDF.groupby([corpDF['K'], corpDF['word']])['D'].size().reset_index()
		childTokenCounts.columns = ['K','word','count']
		
		childTokenCounts = childTokenCounts[childTokenCounts['count'] > childLimit['tokenThreshold']]
		numberOfChildrenPassingFilter = childTokenCounts.groupby(childTokenCounts['word'])['K'].agg(n_unique).reset_index()
		numberOfChildrenPassingFilter = numberOfChildrenPassingFilter[ numberOfChildrenPassingFilter['K'] > childLimit['childThreshold']]

		corpDF = corpDF.loc[corpDF['word'].isin(numberOfChildrenPassingFilter.word),:]
		
	if corpusLimit is not None:
		#if set, remove words appeaning less than n times with m *corpora*. This is much more stringent than the docLimit approach
		
		corpusTokenCounts = corpDF.groupby([corpDF['C'], corpDF['word']])['D'].size().reset_index()
		corpusTokenCounts.columns = ['C','word','count']
		
		corpusTokenCounts = corpusTokenCounts[corpusTokenCounts['count'] > corpusLimit['tokenThreshold']]
		numberOfCorporaPassingFilter = corpusTokenCounts.groupby(corpusTokenCounts['word'])['C'].agg(n_unique).reset_index()
		numberOfCorporaPassingFilter = numberOfCorporaPassingFilter[numberOfCorporaPassingFilter['C'] > corpusLimit['corpusThreshold']]

		corpDF = corpDF.loc[corpDF['word'].isin(numberOfCorporaPassingFilter.word),:]	

	vocab_map_new = np.unique(corpDF['word'])
	reverse_vocab_map = dict(zip(vocab_map_new.tolist(), range(len(vocab_map_new))))

	corpDF['V'] = [reverse_vocab_map[x] for x in corpDF['word']]

	V_new = np.array(corpDF['V']).astype('uint16') 
	D_new = np.array(corpDF['D']).astype('uint16') 	
	C_new = np.array(corpDF['C']).astype('uint8') 	
	K_new = np.array(corpDF['K']).astype('uint8') 	

	return(V_new, D_new, K_new, C_new, vocab_map_new.tolist())	

def filterAcrossChildren(V, D, K, C, vocab_map, n):
    '''remove types with more than n% of their probability mass from a single child. 
    This helps to identify things like names, places, and recording conventions'''
    corpDF = pandas.DataFrame({'V':V,'D':D,'K':K,'C':C, 'order':range(len(V))})    
    print('original dims: '+'*'.join([str(x) for x in corpDF.shape]))
    
    if (n > 1 or n < 0):
        raise ValueError('n must be between 0 and 1')
    
    child_vocab_frequency = np.zeros([len(np.unique(K)), len(np.unique(V))])
    print(child_vocab_frequency.shape)
    for i in range(len(K)):
        child_vocab_frequency[K[i], V[i]] += 1 
    
    # find words with more than n% of the probability mass belonging to a single child (n=90)?
    child_vocab_probs = child_vocab_frequency/child_vocab_frequency.sum(axis=0)[None,:]
    child_vocab_probs_max = child_vocab_probs.max(axis=0)

    cvp_df = pandas.DataFrame({'index':range(len(vocab_map)),'word':vocab_map,'prob':child_vocab_probs_max})
    to_remove = cvp_df[cvp_df.prob > n]
    to_keep = cvp_df[cvp_df.prob <= n]
    print('Removed '+str(to_remove.shape[0])+' words: '+' '.join(to_remove.word))
    
    corpDF_short = corpDF.merge(to_keep[['index','prob','word']], left_on='V', right_on='index').sort_values(by='order')
    print('filtered dims: '+'*'.join([str(x) for x in corpDF_short.shape]))
    
    vocab_map_new = np.unique(corpDF_short['word'])
    reverse_vocab_map = dict(zip(vocab_map_new.tolist(), range(len(vocab_map_new))))

    corpDF_short['V'] = [reverse_vocab_map[x] for x in corpDF_short['word']]
    V_new = np.array(corpDF_short['V']).astype('uint16') 
    D_new = np.array(corpDF_short['D']).astype('uint16') 	
    C_new = np.array(corpDF_short['C']).astype('uint8') 	
    K_new = np.array(corpDF_short['K']).astype('uint8') 	
    
    return(V_new, D_new, K_new, C_new, vocab_map_new.tolist())	


class ParallelLDA():
	def __init__(self, V,D, vocab_map, params, instances):

		print('Initializing parallel instances of LDA model...')
		try:	
						
			self.V = V
			self.D = D
			self.vocab_map = vocab_map
			self.params = params
			self.instances = instances	
				
		except ValueError:
			# return the parameters that generated the error
			print('Failed to initialize parallelzid LDA models.')
			return({'params': params, 'error': ValueError})	    

		self.pool = pathos.multiprocessing.ProcessingPool(self.instances)	
		self.lda_set = self.pool.map(LDA, [self.V for i in range(self.instances)], [self.D for i in range(self.instances)], [self.vocab_map for i in range(self.instances)], [self.params for i in range(self.instances)])			

	def burnIn(self, samples):			
		self.lda_set = self.pool.map(lambda x,y: x.burnIn(iterations=y, thinning=0), self.lda_set, [samples for i in range(self.instances)])	
		
	def sample(self, samples, thinning):			
		self.lda_set = self.pool.map(lambda x,y,z: x.sample(iterations=y, thinning=z), self.lda_set, [samples for i in range(self.instances)], [thinning for i in range(self.instances)])


	def get_cache_signature(self) :	
		'''check if this set of parameters has been run before. Only checks uniqueness of params, not V or D'''

		raise NotImplementedError
		#is there a reasonable way to run this?


		# jsonify the LDA parameters
		# lda_keys  = [x for x in self.params.keys() if x.startswith('lda')]
		# lda_params = { new_key: self.params[new_key] for new_key in lda_keys }

		# run_params = json.dumps(lda_params)				
		# # hash the resulting string

		# file_hash = hashlib.sha256(run_params).hexdigest()
		# file_path = os.path.join(self.params['cache_dir'],file_hash+'.pickle')			
		# return(file_path)



