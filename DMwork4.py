import random

import pandas as pd
import numpy as np
from pyod.utils.example import visualize
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
# Import models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix

from sklearn.preprocessing import MinMaxScaler
def roc(y_test,yProb,title):
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, yProb)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.plot(false_positive_rate, true_positive_rate, color='blue', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1])
    plt.axis('tight')
    plt.ylabel(u'真阳性率', FontProperties='MicroSoft YaHei')
    plt.xlabel(u'假阳性率', FontProperties='MicroSoft YaHei')
    plt.show()



df = pd.read_csv("C:\\Users\\木子\\Desktop\\课程\\数据挖掘\\互评作业4\\wine\\benchmarks\\wine_benchmark_0005"+".csv")
print(df)




#df[['V2','V1','V3','V4','V5','V6','V7','diff.score']] = scaler.fit_transform(df[['V2','V1','V3','V4','V5','V6','V7','diff.score']])
df[['fixed.acidity','volatile.acidity','citric.acid','residual.sugar','chlorides','free.sulfur.dioxide','total.sulfur.dioxide','density','pH','sulphates','alcohol']].head()
#print(df['ground.truth'])
class_mapping = {'nominal':0, 'anomaly':1}

y_test = df['ground.truth'].map(class_mapping)

X1 = df['fixed.acidity'].values.reshape(-1,1)
X2 = df['citric.acid'].values.reshape(-1,1)
X3 = df['volatile.acidity'].values.reshape(-1,1)
X4 = df['residual.sugar'].values.reshape(-1,1)
X5 = df['free.sulfur.dioxide'].values.reshape(-1,1)
X6 = df['total.sulfur.dioxide'].values.reshape(-1,1)
X7 = df['pH'].values.reshape(-1,1)
X8 = df['alcohol'].values.reshape(-1,1)

X = np.concatenate((X1,X2,X3,X4,X5,X6,X7,X8),axis=1)


random_state = np.random.RandomState(42)
outliers_fraction = 0.1
# Define seven outlier detection tools to be compared
classifiers = {
        'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
        'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state),
        'Feature Bagging':FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state=random_state),
        'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
        'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),
        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
        'Average KNN': KNN(method='mean',contamination=outliers_fraction)
}

xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))

for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(X)

    scores_pred = clf.decision_function(X) * -1

    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(X)
    #print("y_pred:",y_pred)

    roc(y_test = y_test,yProb=y_pred,title=clf_name)
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)
    plt.figure(figsize=(10, 10))

    print('准确率: {}'.format(accuracy_score(y_test, y_pred)))



'''
if __name__ == '__main__':
    filename = "C:\\Users\\木子\\Desktop\\课程\\数据挖掘\\互评作业4\\abalone\\benchmarks\\abalone_benchmark_0001.csv"
    reader = csv.reader(filename)
    rows = [row for row in reader]'''

