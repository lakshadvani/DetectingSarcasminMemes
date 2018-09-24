import numpy as np
from sklearn.cluster import KMeans
import csv
from collections import defaultdict
#from matplotlib import pyplot
import matplotlib.pyplot as pyplot
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import cluster
import numpy.random as npr
columns = defaultdict(list) # each value in each column is appended to a list

with open('test.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value
            columns[k].append(v) # append the value into the appropriate list
                                 # based on column name k
#X = np.array([])

Y = np.array([])

X = []
dd = []
scr= []
#d = np.array([])
m = 0
for i in (columns['level_summary.subject']):
    for j,k in zip(columns['level_summary.problems.nright'],columns['level_summary.problems.ntotal']):
    #print(i)

        if i == 'fractions':
            b = 1
        elif i == 'decimals':
            b = 2
        elif i == 'ratios':
            b = 3
        elif i == 'rational_numbers':
            b = 4
        elif i == 'review':
            b = 5
        elif i == '':
            b = 0

        #print(int(j),int(k))
        score = int(j)/int(k)
        #print(score)
        #print("score",score)
        dd.append(int(b))
        scr.append(score)
        d = [int(b),score]
        X.append(d)

    #print(m)
    #print(d)
    #X.append(d)
    Y = np.asarray(X)


#print(H)

print("------------------")
print(Y)
kmeans = KMeans(n_clusters=3).fit(Y)
#labels = kmeans.labels_

centroids = kmeans.cluster_centers_

#v = kmeans.predict([0, 2])
print(kmeans.cluster_centers_)
Z = kmeans.predict([[4, 0.55]])
#labels = kmeans.predict([[1,0.5]])
#print(labels)
labels = kmeans.labels_
k = 5
print(labels)


#Glue back to originaal data
#df_tr = labels

for i in range(len(Y)):
    # select only data observations with cluster label == i
    # plot the data observations
    pyplot.plot(Y[:,0],Y[:,1],'o')
    # plot the centroids
    lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
    # make the centroid x's bigger
    pyplot.setp(lines,ms=15.0)
    pyplot.setp(lines,mew=2.0)
pyplot.show()
