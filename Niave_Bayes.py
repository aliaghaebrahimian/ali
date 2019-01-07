from collections import Counter
import keras as ks
from keras.datasets import imdb
import numpy as np
import collections
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)
N = 0
P = 0
for M in y_train:                                                #comput the probability of each class
    if M==0:
        N+=1
P_negetive=(N/len(y_train))
P_positive = 1-P_negetive
print("probability of neg class:",P_negetive)

print("probability of positive",P_positive)

NegClass=[]
PosClass=[]
my_list = []
for i in range(0,25000):                                                    #mapping lables to each document
    x_train[i].append(y_train[i])
    tmp = x_train[i]
    my_list.append(tmp)

tmp_list_negclass = []
tmp_list_posclass = []

for j in range(len(my_list)):
    vec=my_list[j]
    if vec[-1] == 0:
        tmp_list_negclass.append(vec)
    else:
        tmp_list_posclass.append(vec)

my_list_negclass=[]
my_list_posclass=[]

for k in range(len(tmp_list_negclass)):
    vec1=tmp_list_negclass[k]
    vec1.pop()
    my_list_negclass.append(vec1)
    my_list_negclass.sort()

for l in range(len(tmp_list_posclass)):
    vec2=tmp_list_posclass[l]
    vec2.pop()
    my_list_posclass.append(vec2)
    my_list_posclass.sort()

print("my_list_negclass:",my_list_negclass)
print("my_list_posclass:",my_list_posclass)



new_list_negclass = []
new_list_posclass = []
for m in my_list_negclass:               ##for every item in list my_list_negsclass compute the lenght of each list and append them into the new list
    new_list_negclass.append(len(m))
for n in my_list_posclass:
    new_list_posclass.append(len(n))



total_word_in_negclass =sum(new_list_negclass)        #compute the total word in each class
total_word_in_posclass =sum(new_list_posclass)

print("total_word_in_negclass:",total_word_in_negclass)
print("total_word_in_posclass:",total_word_in_posclass)

voc_size_negclass = []
voc_size_posclass = []

for j in my_list_negclass:
    b= list(set(j))
    voc_size_negclass.append(len(b))

for h in my_list_posclass:    #for every item in list my_list_posclass sort and remove the frequent elements and put the lenght each item in new list after all
    c=list(set(h))              # return the sum all elements in this list
    voc_size_posclass.append(len(h))
d=sum(voc_size_negclass)
e=sum(voc_size_posclass)
print("voc_size_negclass:",d)
print("voc_size_posclass:",e)


voc_size = (d + e)


print("voc_size:",voc_size)  # vocabulary size

final_list_negclass = [x for xs in my_list_negclass for x in xs]    # merging all items into one list to compute word frequency in each class
final_list_posclass = [x for xs in my_list_posclass for x in xs]
print("final_list_negclass:",final_list_negclass)
print("final_list_posclass:",final_list_posclass)

word_frequency_negclass = Counter(final_list_negclass)
word_frequency_posclass = Counter(final_list_posclass)



print("word_frequency_negclass:",word_frequency_negclass)
print("word_frequency_posclass:",word_frequency_posclass)

values_wordfreq_negclass = []
values_wordfreq_posclass = []
for z in word_frequency_negclass.values():    # collecting word frequency for each word in each class
    values_wordfreq_negclass.append(z)

for w in word_frequency_posclass.values():
    values_wordfreq_posclass.append(w)


print("values_wordfreq_negclass:",values_wordfreq_negclass)
print("values_wordfreq_posclass:",values_wordfreq_posclass)

liklihood_probability_negclass = []
liklihood_probability_posclass = []

for items in values_wordfreq_negclass:                          # compute the maximum likelyhood probability and put into the separate list for each class
    probability_negclass=((items+1)/(voc_size+total_word_in_negclass))
    liklihood_probability_negclass.append(probability_negclass)


for item in values_wordfreq_negclass:
    probability_posclass = ((item + 1) / (voc_size + total_word_in_posclass))
    liklihood_probability_posclass.append(probability_posclass)

print("liklihood_probability_negclass:",liklihood_probability_negclass)
print("liklihood_probability_posclass:",liklihood_probability_posclass)




print(sum(liklihood_probability_negclass))
print(sum(liklihood_probability_posclass))
