import numpy as np
import gensim
from nltk.corpus import stopwords

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        #self.dim = len(word2vec.items().next())
        self.dim=320

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


stoplist=stopwords.words('english')
with open("text.txt", "rb") as lines:
    w2v=[line for line in lines]
print(len(w2v))
print("Start")
X=[[word for word in line.split() if word not in stoplist] for line in w2v]
print(X[0])
print(len(X))
model = gensim.models.Word2Vec(X, size=200 ,min_count=5)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))

print("Embedding")
#Embeddingvectorizer
Z=MeanEmbeddingVectorizer(w2v)
Z1=Z.transform(X)
np.savetxt("Word2vecEmbed.csv",Z1,fmt="%.2f",delimiter=",")



