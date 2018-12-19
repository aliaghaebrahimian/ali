import os
posts = [open(os.path.join("SDR", f)).read() for f in os.listdir("SDR")]
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1,stop_words='english')
x_train=vectorizer.fit_transform(posts)

num_samples , num_features = x_train.shape
print("#samples:%d,#features:%d" % (num_samples , num_features))
new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])
import scipy as sp
def dist_raw(v1,v2):
    v1_normalized=v1/sp.linalg.norm(v1.toarray())
    v2_normalized=v2/sp.linalg.norm(v2.toarray())
    delta=v1_normalized-v2_normalized
    return sp.linalg.norm(delta.toarray())
import sys
best_doc = None
best_dist = sys.maxsize
best_i= None
for i in range(0,num_samples):
    post=posts[i]
    if post==new_post:
        continue
    post_vec=x_train.getrow(i)
    d=dist_raw(post_vec,new_post_vec)
    print("===post%i with dist=%.2f:%s"%(i,d,post))
    if d<best_dist:
        best_dist=d
        best_i=i
print("best post is %i with dist=%.2f" %(best_i,best_dist))
import nltk.stem
english_stemmer= nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def bild_analyzer(self):
        analyzer=super(StemmedCountVectorizer,self).build_analyzer()
        return lambda doc : (english_stemmer.stem(w) for w in analyzer(doc))
edctorizer=StemmedCountVectorizer(min_df=1, stop_words='english')
from sklearn.feature_extraction.text import TfidfVectorizer
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer=super(TfidfVectorizer,self).build_analyzer()
        return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))
vectorizer=StemmedTfidfVectorizer(min_df=1,stop_words='english')
