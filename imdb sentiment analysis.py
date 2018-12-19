import keras as ks
from keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)
dataset = [[1,20,1],[1,21,10],[0,22,1],[0,21,4],[1,51,3]]
def sepratebyclass(dataset):
    seprated = {}
    for i in range(len(x_train)):
        vector =x_train[i]
        if vector[0] not in seprated:
            seprated[vector[0]] = []
        seprated[vector[0]].append(vector)
    return seprated
seprated = sepratebyclass(x_train)


import math
def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg=mean(numbers)
    variance=sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(x_train):
    summaries =[(mean(attribute),stdev(attribute))for attribute in zip(*x_train)]
    del summaries[0]
    return summaries

def summarizebyclass(x_train):
    seprated = sepratebyclass(x_train)
    summaries = {}
    for classvalue, instances in seprated.items():
        summaries[classvalue] = summarize(instances)
    return summaries
summary = summarizebyclass(x_train)
