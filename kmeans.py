import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
import csv
from sklearn import svm
from collections import defaultdict
#from matplotlib import pyplot
from scipy.spatial.distance import cdist

import matplotlib.pyplot as pyplot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import seaborn as sns

from sklearn import cluster
import numpy.random as npr
from sklearn import cluster
import numpy.random as npr
columns = defaultdict(list) # each value in each column is appended to a list

with open('sampleData.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}

        for (k,v) in row.items(): # go over each column name and value
            columns[k].append(v) # append the value into the appropriate list
                                 # based on column name k
#X = np.array([])
#print(columns)
#print(columns['time_spent'])
X = []
Y = []
#print(set(columns['student.section_id']))
data = pd.read_csv('sampleData.csv')
print(max(data['time_spent']))
print(min(data['time_spent']))
print(np.mean(data['diff']))
for i in (columns['diff']):
    if i == '':
        b = 0
    else:
        b = i
    X.append(b)

for i in columns['time_spent']:
    if i == '':
        b = 0
    else:
        Y.append(i)

f1 = data['diff'].values
print(type(f1))
f2 = data['time_spent'].values
p = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)
#print(Y)
#print(len(Y))
df = pd.DataFrame({
    'x': X,
    'y': Y
})
#print(df)

kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
#print(kmeans.labels_)
#print(kmeans.cluster_centers_)

for i in kmeans.cluster_centers_:
    print(i)

#print(len( kmeans.cluster_centers_))
c1 =  kmeans.cluster_centers_[0]
c2 =  kmeans.cluster_centers_[1]


plt.scatter(f1, f2, c='#050505', s=7)
plt.scatter(c1,c2, marker='*', s=200, c='g')
plt.show()
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(df)
    kmeanModel.fit(df)
    distortions.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
