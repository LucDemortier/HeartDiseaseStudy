'''
This program selects features from the Cleveland heart disease study on the UC Irvine
Machine Learning Repository.  It does this by maximizing the accuracy of a K-Nearest Neighbors
classifier.  The initial features are the following:
 1. age: continuous
 2. sex: categorical, 2 values {0: female, 1: male}
 3. cp (chest pain type): categorical, 4 values
    {1: typical angina, 2: atypical angina, 3: non-angina, 4: asymptomatic angina}
 4. restbp (resting blood pressure on admission to hospital): continuous (mmHg)
 5. chol (serum cholesterol level): continuous (mg/dl)
 6. fbs (fasting blood sugar): categorical, 2 values {0: <= 120 mg/dl, 1: > 120 mg/dl}
 7. restecg (resting electrocardiography): categorical, 3 values
    {0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy}
 8. thalach (maximum heart rate achieved): continuous
 9. exang (exercise induced angina): categorical, 2 values {0: no, 1: yes}
10. oldpeak (ST depression induced by exercise relative to rest): continuous
11. slope (slope of peak exercise ST segment): categorical, 3 values 
    {1: upsloping, 2: flat, 3: downsloping}
12. ca (number of major vessels colored by fluoroscopy): discrete (0,1,2,3)
13. thal: categorical, 3 values {3: normal, 6: fixed defect, 7: reversible defect}
14. num (diagnosis of heart disease): categorical, 5 values 
    {0: less than 50% narrowing in any major vessel, 1-4: more than 50% narrowing in 1-4 vessels}
    
The actual number of feature variables (after converting categorical variables
to dummy ones) is: 
1 (age) + 1 (sex) + 3 (cp) + 1 (restbp) + 1 (chol) + 1 (fbs) + 2 (restecg) + 
1 (thalach) + 1 (exang) + 1 (oldpeak) + 2 (slope) + 1 (ca) + 2 (thal) = 18

The response variable (num) is categorical with 5 values, but we don't have
enough data to predict all the categories. Therefore we'll replace num with:
14. hd (heart disease): categorical, 2 values {0: no, 1: yes}
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
import itertools
from pprint import pprint

columns = ["age", "sex", "cp", "restbp", "chol", "fbs", "restecg", 
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
df      = pd.read_table("data/heart_disease_all14.csv", sep=',', header=None, names=columns)

# Convert categorical variables with more than two values into dummy variables
dummies = pd.get_dummies(df["cp"],prefix="cp")
df      = df.join(dummies)
del df["cp"]
del df["cp_4.0"]
df      = df.rename(columns = {"cp_1.0":"cp_1","cp_2.0":"cp_2","cp_3.0":"cp_3"})

dummies = pd.get_dummies(df["restecg"],prefix="recg")
df      = df.join(dummies)
del df["restecg"]
del df["recg_0.0"]
df      = df.rename(columns = {"recg_1.0":"recg_1","recg_2.0":"recg_2"})

dummies = pd.get_dummies(df["slope"],prefix="slope")
df      = df.join(dummies)
del df["slope"]
del df["slope_2.0"]
df      = df.rename(columns = {"slope_1.0":"slope_1","slope_3.0":"slope_3"})

dummies = pd.get_dummies(df["thal"],prefix="thal")
df      = df.join(dummies)
del df["thal"]
del df["thal_3.0"]
df      = df.rename(columns = {"thal_6.0":"thal_6","thal_7.0":"thal_7"})

# Replace response variable
df.replace(to_replace=[1,2,3,4],value=1,inplace=True)
df      = df.rename(columns = {"num":"hd"})

print '\nNumber of records read in: %i\n' % len(df.index)
print df.head()

# Convert dataframe into lists for use by classifiers
yall = df["hd"]
del df["hd"]
Xall = df.values

'''
Optimize K Nearest Neighbors Classifier
'''

best_score = []
best_std   = []
best_comb  = []
best_kval  = []
kval_max   = 1
nfeatures  = 18
iterable   = range(nfeatures)
for s in xrange(len(iterable)+1):
    for comb in itertools.combinations(iterable, s):
        if len(comb) > 0:
            Xsel = []
            for patient in Xall:
                Xsel.append([patient[ind] for ind in comb])
            for kval in range(1,kval_max+1):
                model      = KNeighborsClassifier(n_neighbors=kval)
                this_scores = cross_val_score(model, Xsel, y=yall, cv=3 )
                score_mean  = np.mean(this_scores)
                score_std   = np.std(this_scores)
                if len(best_score) > 0: 
                    if score_mean > best_score[0]:
                        best_score = []
                        best_std   = []
                        best_comb  = []
                        best_kval  = []
                        best_score.append(score_mean)
                        best_std.append(score_std)
                        best_comb.append(comb)
                        best_kval.append(kval)
                    elif score_mean == best_score[0]:
                        best_score.append(score_mean)
                        best_std.append(score_std)
                        best_comb.append(comb)
                        best_kval.append(kval)
                else:
                    best_score.append(score_mean)
                    best_std.append(score_std)
                    best_comb.append(comb)
                    best_kval.append(kval)

# Print out results
output   = open('results/knn_results.txt', 'w')
num_ties = len(best_score)
for ind in range(num_ties):
    print 'For k=%i, comb=%s, kNN Accuracy = %f +/- %f' \
            % (best_kval[ind],best_comb[ind],best_score[ind],best_std[ind])
    print >> output, 'For k=%i, comb=%s, kNN Accuracy = %f +/- %f' \
            % (best_kval[ind],best_comb[ind],best_score[ind],best_std[ind])
output.close()
