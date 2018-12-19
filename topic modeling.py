
from gensim import corpora,models,simailarities
corpus=corpora.BleiCorpus('/home/pycharmprojects/data/ap/ap.dat','/home/pycharmprojects/data/ap/vocab.txt')
model=models.ldamodel.LdaModel(corpus,num_topics=100,id2word=corpus.id2word)
topics=[model[c] for c in corpus]
print(topics[0])
