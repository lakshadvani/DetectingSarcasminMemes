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
        #print(i)
        b = ((int(i)/1000)/60)
    Y.append(b)

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


#plt.scatter(f1,f2)
#plt.show()
