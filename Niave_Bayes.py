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
    else:
        P+=1
    pc=(N/len(y_train))
probabilityclass = pc
print("number of neg class:",N,"number of pos class:",P)
print("probability of neg class:",probabilityclass)

NegClass=[]
PosClass=[]
list=[]
for i in range(0,25000):                                                    #mapping lables to each document
    x_train[i].append(y_train[i])
    tmp = x_train[i]
    list.append(tmp)

def separatByClass(list):                                                 # separate each document based on their lables
    separated={}
    for j in range(0,25000):
        vec=list[j]
        if (vec[-1] not in separated):
            separated[vec[-1]]=[]
            separated[vec[-1]].append(vec)
    return separated


separated=separatByClass(list)
NegClass=separated.get(0)                                                    #return indices in class Neg
PosClass =separated.get(1)                                                    #return indices in class Pos

for k in NegClass:                                                    #comput the total word occurrence in each class
    x=collections.Counter(k)
    total_index_in_class0=sum(x.values())
    print(x)
    print("total_index_in_class0:",total_index_in_class0)
for l in PosClass:
    b=collections.Counter(l)
    total_index_in_class1=sum(b.values())
    print(b)
    print("total_index_in_class1:",total_index_in_class1)

voc_size=(total_index_in_class0+total_index_in_class1)                              #computeing voc_size

for keys in x,b:     #separate occurrence of each indices in each class
    nkn =x.values()
    nkp = b.values()
print(nkn)
print(nkp)

Niave_bayeslist0=[]
Niave_bayeslist1=[]
for v in nkn:                                                      #saving the Niave_Bayes probability in two separated list
    Niave_bayesclass0 = pc*(v+1)/(voc_size+total_index_in_class0)
    Niave_bayeslist0.append(Niave_bayesclass0)
print('Niave_Bayeslist0:',Niave_bayeslist0)
for V in nkp:
    Niave_bayesclass1 = pc*(V+1)/(voc_size+total_index_in_class1)
    Niave_bayeslist1.append(Niave_bayesclass1)
print('Niave_Bayeslist1:',Niave_bayeslist0)
print(sum(Niave_bayeslist0))
print(sum(Niave_bayeslist1))

if sum(Niave_bayeslist1)>sum(Niave_bayeslist0):                                              #Estimate the sentiment
    print("sentiment is POSITIVE")
else:
    print("sentiment is NEGETIVE")



listtest=[]
for a in range(len(x_test)):
    x_test[a].append(y_test[a])
    tmptest=x_test[a]
    listtest.append(tmptest)

def getaccuracy(listtest,separated):              #Getting accuracy
    correct = 0
    for g in range(len(listtest)):
        if listtest[g][-1]==separated[g]:
            correct+=1
    return (correct/float(len(listtest)))*100
Accuracy = getaccuracy(listtest,separated)
print("Accuracy:",Accuracy)
