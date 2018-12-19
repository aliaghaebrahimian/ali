import keras as ks
from keras.datasets import imdb
import numpy as np
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)
k=0
j=0
for i in y_train:
    if i==0:
        j+=1
    else:
        k+=1
pc=(j/len(y_train))
print("number of neg class:",j,"number of pos class:",k)
print("probability of neg class:",pc)

voc_size = 0
for row in x_train:
    row.sort()
    s=list(set(row))
    while voc_size<=len(s):
        voc_size=voc_size+len(s)
print("voc_size:"voc_size)

for row in x_train:
    for l in row:
        m=list(row)
        w_f=m.count(l)
        print("word:",l,':',"word frequency:",w_f)
