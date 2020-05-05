import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame


def minDistance(rows):
    #计算相似度
    countrys = [row[1] for row in rows]
    countrys.pop(0)
    points = [row[4] for row in rows]
    points.pop(0)
    points = list(map(float, points))
    provinces = [row[6] for row in rows]
    provinces.pop(0)
    varieties = [row[9] for row in rows]
    varieties.pop(0)
    prices = [row[5] for row in rows]
    prices.pop(0)
    for i in range(len(prices)):
        if(i % 1500 == 0):
            print(i/150930)
        if prices[i] == "":
            country = countrys[i]
            point = points[i]
            province = provinces[i]
            variety = varieties[i]

            sumPrice = 0
            disNum = 0
            distance = 4
            for j in range(len(prices)):
                if(prices[j] != ""):
                    infoA = [countrys[j],provinces[j],varieties[j]]
                    infoB = [country,province,variety]
                    #print(points[j])
                    newDistance = len(list(set(infoA).difference(set(infoB)))) + ((20.0 - abs((points[j]) - (point)))/20.0)
                    if newDistance == 4:
                        pass
                    elif newDistance < distance:
                        sumPrice = float(prices[j])
                        disNum = 1
                    elif newDistance == distance:
                        sumPrice += float(prices[j])
                        disNum += 1
                    else:
                        pass
            #更新空值

            if(sumPrice == 0):
                prices[i] = 24.0
            else:
                prices[i] = sumPrice / disNum

    #价格刷新
    prices = list(map(float, prices))
    with open("test2.txt","w") as f:
        f.write(str(prices))

def padding(rows):
    #填充数据
    points = [row[4] for row in rows]
    price = [row[5] for row in rows]
    points.pop(0)
    price.pop(0)
    points = list(map(float,points))
    #price = list[map(float,price)]
    with open("priceDict.csv","r") as f:
        tempDict = f.read()
    for i in range(len(price)):
        if price[i] == "":
            price[i] = tempDict[points[i]]
    price = list(map(float,price))
    print(price)
    Histogram(price[0:400],"price")
    input()

def dataAnalyse(rows):
    for k in [4]:
        points = [row[k] for row in rows]
        prize = [row[5] for row in rows]
        points.pop(0)
        prize.pop(0)
        x = []
        y = []
        for i in range(len(points)):
            if points[i] and prize[i] != "":
                try:
                    x.append(float(points[i]))
                except ValueError:
                    x.append(points[i])
                y.append(float(prize[i]))
        plt.figure(figsize=(5, 3))
        plt.xlabel("point")
        plt.ylabel("prize")
        plt.title(u'平均值-散点图', FontProperties='MicroSoft YaHei')
        plt.scatter(x, y, marker='o', s=5)
        plt.savefig("otherPic/1-point-prize-scatter.png")
        plt.show()
        ''' 
        plt.figure(figsize=(15, 10))
        plt.plot(newx, newy)
        plt.xlabel("point")
        plt.ylabel("prize")
        plt.title(u'折线图', FontProperties='MicroSoft YaHei')
        plt.savefig("otherPic/1-point-prize-plot.png")
        plt.show()
        '''

def saveDictAsCSV(data:dict,filename):
    content = []
    for key in data:
        content.append({filename:key,"value":data[key]})
    pd.DataFrame(data = content).to_csv("datacsv/" + filename + ".csv")

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
    plt.figure(figsize=(6,6))
    label=[x for x in columnDict][0:12]
    values=[columnDict[x] for x in columnDict][0:12]
    #colors = ["lightblue","lightgrey","lightgreen","orange","lightpink"]
    plt.pie(values,labels=label,autopct='%1.1f%%')#绘制饼图
    plt.title( columnName+ '-' +u'饼状图', FontProperties='MicroSoft YaHei')
    plt.savefig("pie/"+columnName+".png")
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
    columnDict = dict(sorted(columnDict.items(), key=lambda item: item[1],reverse=True))
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

if __name__ == '__main__':
    filename = "C:\\Users\\木子\\Desktop\\课程\\数据挖掘\\wine-reviews\\winemag-data_first150k.csv"
    csvfile = open(filename, "r",encoding="utf-8")
    #with open(filename, "r",encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    #'','country', 'description', 'designation', 'points', 'price', 'province', 'region_1', 'region_2', 'variety', 'winery'
    #标称属性：country，province,region_1,region_2,variety,
    #数值属性：points,price
    #描述（description）和名称（designation）和酒厂（winery）
    rows = [row for row in reader]
    #dataAnalyse(rows)
    #minDistance(rows)

    #标称属性 直方图
    locs = [1,6,7,8,9]
    #locs = [5]
    for loc in locs:
        print("loc",loc)
        column = [row[loc] for row in rows]
        columnName = column[0]
        column.pop(0)
        pie(column,columnName)
        #Histogram(column,columnName)
        #boxPlots(column,columnName)