import sklearn.datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
newsgroups_train = fetch_20newsgroups(subset='train')
from pprint import pprint
groups=['comp.graphics']
train_data= sklearn.datasets.load_files('20news-bydate-train',categories=groups,encoding='Utf-8',decode_error='ignore')
print(len(train_data.filenames))
vectorizer = CountVectorizer(min_df=1)
vectorized=vectorizer.fit_transform(train_data)
print(vectorized.shape)
num_samples,num_features=vectorized.shape
print("#samples: %d , #features: %d"%(num_samples,num_features))
num_clusters = 2
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2, init='random', n_init=1,verbose=1)
km.fit(vectorized)
print(km.labels_)
print(km.labels_.shape)
print(km.cluster_centers_)
new_post =["Disk drive problems. Hi, I have a problem with my hard disk.After 1 year it is working only " \
          "sporadically now.I tried to format it, but now it doesn't boot any more.Any ideas? Thanks"]
new_post_vec=vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]
similar_indices = (km.labels_==new_post_label).nonzero()[0]
