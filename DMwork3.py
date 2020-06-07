import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def pie(column:list,columnName):
    columnDict = {}
    for item in column:
        if item == "":
            if ("Missing value" in columnDict):
                columnDict["Missing value"] += 1
            else:
                columnDict["Missing value"] = 1
        elif item in columnDict:
            columnDict[item] += 1
        else:
            columnDict[item] = 1
    # 递减排序
    columnDict = dict(sorted(columnDict.items(), key=lambda item: item[1], reverse=True))
    print(columnDict)
    plt.figure(figsize=(6,6))
    label=[x for x in columnDict][0:12]
    values=[columnDict[x] for x in columnDict][0:12]
    plt.pie(values,labels=label,autopct='%1.1f%%')#绘制饼图
    plt.title( columnName+ '-' +u'饼状图', FontProperties='MicroSoft YaHei')
    plt.savefig("pie/"+columnName+".png")
    plt.show()

def boxPlots(column: list,columnName):
    #绘制盒图
    column = sorted(column)
    minValue = column[0]
    maxValue = column[-1]
    length = len(column)
    Q1 = column[int(length * 0.25)]
    Q2 = column[int(length * 0.75)]

    if length % 2 == 0:
        median = (column[int(length / 2)] + column[int(length / 2) - 1]) / 2
    else:
        median = column[int(length / 2)]
    all_data = [minValue, Q1, median, Q2, maxValue]
    plt.figure(figsize=(8, 6))
    print(all_data)
    plt.boxplot(all_data,notch=False, vert=True, sym ='o',patch_artist=True)
    plt.xlabel(columnName)
    plt.title(u'盒图', FontProperties='MicroSoft YaHei')
    plt.savefig("BoxplotPic/" + columnName + ".png")
    plt.show()

def Histogram(column: list,columnName):
    #绘制直方图
    #统计属性每个值的个数
    columnDict = {}
    for item in column:
        if item == "":
            if("Missing value" in columnDict):
                columnDict["Missing value"] += 1
            else:
                columnDict["Missing value"] = 1
        elif item in columnDict:
            columnDict[item] += 1
        else:
            columnDict[item] = 1
    #递减排序
    #columnDict = dict(sorted(columnDict.items(), key=lambda item: item[1],reverse=True))
    #columnDict = sorted(columnDict.keys(), reverse=False)
    xValue = []
    yValue = []
    for key in columnDict:
        xValue.append(str(key).replace(" ","\n"))
        yValue.append(columnDict[key])

    plt.figure(figsize=(15, 10))
    plt.bar(xValue , yValue , label=columnName)
    plt.legend()
    plt.xlabel(columnName)
    plt.title(u'直方图', FontProperties='MicroSoft YaHei')
    plt.savefig("histPic/" + columnName+".png")
    plt.show()

def price(rows):
    #计算酒店的每个月份的平均价格
    #adr:27列 adult:9列 children:10列 arrival_date_month：4列
    adr = [row[27] for row in rows]
    adults =  [row[9] for row in rows]
    children =  [row[10] for row in rows]
    try:
        avgPrice = [float(adr[i]) / (float(adults[i]) + float(children[i] ) + 1) for i in range(len(adr))]
    except ValueError:
        avgPrice = [float(adr[i]) / (float(adults[i]) + 0 + 1) for i in range(len(adr))]
    month = [row[4] for row in rows]

    monthAndPric = {}
    monthAndCount = {}
    for i in range(len(adr)):
        if month[i] not in monthAndPric:
            monthAndPric[month[i]] = avgPrice[i]
            monthAndCount[month[i]] = 1
        else:
            monthAndPric[month[i]] += avgPrice[i]
            monthAndCount[month[i]] += 1
    for key in monthAndPric:
        monthAndPric[key] = monthAndPric[key] / monthAndCount[key]
    x = []
    y = []
    for key in monthAndPric:
        x.append(key)
        y.append(monthAndPric[key])
    print(monthAndPric)
    plt.plot(x, y)
    plt.title('Resort Hotel')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    input()

def percMv(x, y):
    perc = y.isnull().sum() / len(x) * 100
    return perc

def forecast():
    filename = "C:\\Users\\木子\\Desktop\\课程\\数据挖掘\\作业\\hotel_bookings.csv"
    df = pd.read_csv(filename)
    #print(df.shape)
    # print("空值个数:", df.isnull().sum(), sep='\n')
    #print('缺失值和比例:\nCompany: {}\nAgent: {}\nCountry: {}'.format(percMv(df, df['company']),percMv(df, df['agent']),percMv(df, df['country'])))
    data = df.copy()
    #print(data.dtypes)
    #预处理

    data['hotel'] = data['hotel'].map({'Resort Hotel': 0, 'City Hotel': 1})
    data['arrival_date_month'] = data['arrival_date_month'].map({'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7,
                                                                 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12})
    #特征化
    data = feature(data)
    data = data.drop(columns=['adults', 'babies', 'children', 'deposit_type', 'reservation_status_date'])
    #删除country中的空行
    indices = data.loc[pd.isna(data["country"]), :].index
    data = data.drop(data.index[indices])
    #根据相关性删除一些列
    data = data.drop(columns=['arrival_date_week_number', 'stays_in_weekend_nights', 'arrival_date_month', 'agent','company','totalCustomer'],axis=1)
    df1 = data.copy()
    df1 = pd.get_dummies(data=df1, columns=['meal','market_segment', 'distribution_channel','reserved_room_type', 'assigned_room_type','customer_type', 'reservation_status'])
    lableEncoder = LabelEncoder()
    df1['country'] = lableEncoder.fit_transform(df1['country'])
    #print("空值个数:", df1.isnull().sum(), sep='\n')
    #print(df1)
    #pd.DataFrame(df1).to_csv("df1.csv")
    #df1 = df1.drop("hotel",axis=1)

    df1 = df1.drop(columns=['reservation_status_Canceled', 'reservation_status_Check-Out', 'reservation_status_No-Show'], axis=1)
    y = df1["is_canceled"]
    X = df1.drop(["is_canceled"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
    #加载线性回归模型
    LogRModel = LogisticRegression(solver="liblinear",max_iter=100)
    LogRModel.fit(X_train,y_train)
    alg_model = LogRModel.fit(X_train, y_train)
    yProb = LogRModel.predict_proba(X_test)[:, 1]
    yPred = alg_model.predict(X_test)

    print('准确率: {}\n\n误差矩阵:\n {}'.format(accuracy_score(y_test, yPred), confusion_matrix(y_test, yPred)))

    cv_scores = cross_val_score(LogRModel, X, y, cv=8, scoring='accuracy')
    print('Mean Score of CV: ', cv_scores.mean())

    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, yProb)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.figure(figsize=(10, 10))
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, color='blue', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1])
    plt.axis('tight')
    plt.ylabel(u'真阳性率', FontProperties='MicroSoft YaHei')
    plt.xlabel(u'假阳性率', FontProperties='MicroSoft YaHei')
    plt.show()







def family(data):
    if data['adults'] > 0 and (data['children'] > 0 or data['babies'] > 0):
        return 1
    return 0


def deposit(data):
    if data['deposit_type'] == 'No Deposit' or data['deposit_type'] == 'Refundable':
        return 0
    return 1

def feature(data):
    data["totalCustomer"] = data["adults"] + data["children"] + data["babies"]
    data["isFamily"] = data.apply(family, axis = 1)
    data["totalNights"] = data["stays_in_weekend_nights"]+data["stays_in_week_nights"]
    return data





if __name__ == '__main__':
    forecast()
    input()
    filename = "C:\\Users\\木子\\Desktop\\课程\\数据挖掘\\作业\\hotel_bookings.csv"
    csvfile = open(filename, "r", encoding="utf-8")