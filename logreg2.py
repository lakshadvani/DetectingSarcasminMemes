import pandas as pd
import numpy as np
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn import datasets
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from bokeh.plotting import figure, show, output_notebook
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
data = pd.read_csv('test3.csv',sep='\s*,\s*',error_bad_lines=True,nrows = 25000, delimiter=',', encoding="utf-8-sig")

data = data.dropna()
print(data.shape)
#print(list(data.columns))
#print(data.head)
b = data['correct'].value_counts()
#print(b)
#sns.countplot(x = 'correct',data = data, palette = 'hls')
#plt.show()
#print(data.groupby('correct').mean())
for column in data.columns:
    if data[column].dtype == type(object):
        le = pre.LabelEncoder()
        data[column] = le.fit_transform(data[column])


cat_vars = ['student.student_id','level_summary.t_elapsed','level_summary.problems.nwrong', 'level_summary.problems.nretry_right', 'level_summary.problems.nretry_wrong', 'level_summary.mastery.mean','qual_id']
#target = 'correct'
#print(cat_vars)
#train_x, test_x, train_y, test_y = train_test_split(data[training_features], data[target], train_size=0.7)

#print(test_x.size)

for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_vars = ['student.student_id','level_summary.t_elapsed','level_summary.problems.nwrong', 'level_summary.problems.nretry_right', 'level_summary.problems.nretry_wrong', 'level_summary.mastery.mean','qual_id']
data_vars=data.columns.values.tolist()
#print(data_vars)
to_keep=[i for i in data_vars if i not in cat_vars]
#print(to_keep)
data_final=data[to_keep]
#print(data_final)

data_final_vars=data_final.columns.values.tolist()
y=['correct']
X=[i for i in data_final_vars if i not in y]
#print(X)

logreg = LogisticRegression()
#rfe = RFE(logreg, 18)
#rfe = rfe.fit(data_final[X], data_final[y] )
#rint(rfe.support_)
#print(rfe.ranking_)
#d = rfe.ranking_.tolist()


#ind = [i for i, x in enumerate(d) if x == 1]
#print(ind)

#for i in ind:
#    print(data_final.columns.values[i])


cols = ['level_summary.stars.earned','time_spent_8312','time_spent_13442','time_spent_19069','time_spent_30469','time_spent_49918','qual_id_4','qual_id_5','qual_id_6','qual_id_15','qual_id_24','qual_id_29','qual_id_30','qual_id_32','qual_id_33','qual_id_34','qual_id_77','qual_id_80']


X=data_final['level_summary.blank_slate_mastery.mean']
y=data_final['correct']
print(type(X))
#print(y)
print(X.shape)
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
#print(result.summary())
X = X.values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression(max_iter = 10000,penalty = 'l2',solver ='saga',tol=0.000001)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))





admitted = []
rejected = []
diff1 = []
diff2 = []
time1 = []
time2 = []
print(data['correct'])

l = list(data['correct'])
m = list(data['diff'])

n = list(data['time_spent'])

if 0 in l:
    print("lol")
lll = [i for i,x in enumerate(l) if x==1]

ll = [i for i,x in enumerate(l) if x==0]
print(ll)
# [14, 20, 23, 33, 37, 66]

#ll.remove(52)
print(m[45])

k = []
for i in range(len(n)):
    if n[i] > 66000:
        k.append(i)

ll = ll+k
lll = [x for x in lll if x not in k]
'''
11
17
28
29
30
44
49
50
51
52
53
54
55
103
104
105
122
131
139
140
142
146
191
193
194
220
239
251
293
298
323
337
358
390
395
401
418
468
472
552
560
561
569
570
571
576
774
840
846
847
850
854
858
875
880
890
895
896
931
938

'''





for i in lll:
    admitted.append(l[i])
    diff1.append(m[i])
    time1.append(n[i])


for i in ll:
    rejected.append(l[i])
    diff2.append(m[i])
    time2.append(n[i])

print((diff2))

plt.figure(figsize=(6, 6))
plt.scatter(diff1, time1, c='b', marker='+', label='correct')
plt.scatter(diff2, time2, c='y', marker='o', label='incorrect')

plt.xlabel('diff');
plt.ylabel('time');


from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

X = np.column_stack((data['diff'], data['time_spent']))
Y = data['correct']

classifier = OneVsRestClassifier(LogisticRegression(penalty='l2')).fit(X, Y)
print('Coefficents: ', classifier.coef_)
print('Intercept" ', classifier.intercept_)

coef = classifier.coef_
intercept = classifier.intercept_
print(coef)
# see the coutour approach for a more general solution
ex1 = np.linspace(0, 100, 100)
ex2 = -(7*coef[:, 0] * ex1 + intercept[:, 0]) / coef[:,1]*0.3

plt.plot(ex1, ex2, color='r', label='decision boundary');
plt.legend();
plt.show()










plt.ylabel('Correct')
plt.xlabel('Difficulty')
plt.xticks(np.arange(-1, 1))
plt.yticks([0, 0.5, 1])
plt.ylim(-.25, 1.25)
plt.xlim(-4, 10)
plt.legend(('Logistic Regression Model', 'Linear Regression Model'),
           loc="lower right", fontsize='small')
plt.title("Logistic Regression")
plt.show()


p = figure(height=450, width=450)
p.xaxis.axis_label = "Score"
p.yaxis.axis_label = "Score"
l = list(data['correct'])
ll = list(data['qual_id'])
lll = list(data['diff'])
llll = list(data['time_spent'])


incorrect = []
correct = []


for i in range(len(l)):
    #print(i)
    k = [lll[i],llll[i]]
    if l[i] == 0:
        incorrect.append(k)
    elif l[i] == 1:
        correct.append(k)

print(type(incorrect))
incorrect = np.array(incorrect)
correct = np.array(correct)
plt.scatter(incorrect[:,0],incorrect[:,1],color = 'red')
plt.scatter(correct[:,0],correct[:,1],color = "blue")


plt.show()
predicted = cross_validation.cross_val_predict(LogisticRegression(), X_train, y_train, cv=10)

print(metrics.accuracy_score(y_train, predicted))

print(metrics.classification_report(y_train, predicted))
