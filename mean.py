import pandas as pd
import numpy as np

df = pd.read_csv('sampleData.csv')
j = df['time_spent']
f = (df['diff'])
d = []
print(len(j))
print(j[2])
for i in range(len(f)):
    if f[i] > float(0.45):
        d.append(j[i])


print(sum(j) / float(len(j)))
