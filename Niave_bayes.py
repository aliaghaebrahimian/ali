from collections import Counter
import keras as ks
from keras.datasets import imdb
import numpy as np
import collections
import math
import itertools
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


word_frequency_negclass = Counter(final_list_negclass)
word_frequency_posclass = Counter(final_list_posclass)


for x in word_frequency_negclass:
    word_frequency_negclass[x] =math.log10((word_frequency_negclass[x]+1)/(voc_size+total_word_in_negclass))

for xz in word_frequency_posclass:
    word_frequency_posclass[xz] =math.log10((word_frequency_posclass[xz]+1)/(voc_size+total_word_in_posclass))

logpro_testfornegclass = []
logpro_testforposclass = []

for ki in range(len(x_test)):
    vec1 = x_test[ki]
    logpro_testfornegclass.append(vec1)   #take two copy from  x_test

for kj in range(len(x_test)):
    vec2 = x_test[kj]
    logpro_testforposclass.append(vec2)


logpro_fornegclass = []
for p in range(len(logpro_testfornegclass)):     #compute x_test log probability for each statment in x_test
    temp = []
    for t in logpro_testfornegclass[p]:
        t= word_frequency_negclass[t]
        temp.append(t)
    logpro_fornegclass.append(temp)

logpro_forposclass=[]
for q in range(len(logpro_testforposclass)):
    temp1=[]
    for g in logpro_testforposclass[q]:
        g=word_frequency_posclass[g]
        temp1.append(g)
    logpro_forposclass.append(temp1)

sumlogprofornegclass=[]   #compute summation of elements in each lists and append them to one list
for kn in range(len(logpro_fornegclass)):
    logprovec1=logpro_fornegclass[kn]
    sumlogprofornegclass.append(sum(logprovec1))

sumlogproforposclass=[]
for km in range(len(logpro_forposclass)):
    logprovec2=logpro_forposclass[km]
    sumlogproforposclass.append(sum(logprovec2))

abssumlogprofornegclass=[abs(x) for x in sumlogprofornegclass]          #compute the absolute value for each element in each list
abssumlogproforposclass=[abs(n) for n in sumlogproforposclass]

comparisonlist = []
for elem in range(len(abssumlogprofornegclass)) and range(len(abssumlogproforposclass)): #Compare two resault list element by element and specify wich one is posetive and wiche one is negative
    if abssumlogprofornegclass[elem]>abssumlogproforposclass[elem]:
        comparisonlist.append(1)
    else:
        comparisonlist.append(0)

def Getaccuracy(comparisonlist,y_test):
    correct=0
    for i in range(len(comparisonlist)) and range(len(y_test)):
        if comparisonlist[i] == y_test[i]:
            correct += 1
    accuracy = (correct/float(len(y_test)))*100.0
    return accuracy

Accuracy = Getaccuracy(comparisonlist,y_test)
print("Accuracy:",Accuracy)


def Predictedsentiment(comparisonlist):   #specify each sentiment of each statment
    for i in comparisonlist:
        if i==1:
            print("predictedsentiment--->>","Positive")
        else:
            print("predictedsentiment--->>","Negative")


Sentimentofeachstatment = Predictedsentiment(comparisonlist)
