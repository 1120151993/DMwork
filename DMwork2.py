import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

#对数据集进行处理，转换成适合进行关联规则挖掘的形式
def preprocess(filename):
    with open(filename, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        # '','country', 'description', 'designation', 'points', 'price', 'province', 'region_1', 'region_2', 'variety', 'winery'
        # 标称属性：country 1，province 6,region_1 7,region_2 8,variety 9,
        # 数值属性：points 4,price 5
        # 描述（description）和名称（designation）和酒厂（winery）
        rows = [row for row in reader]
        rows.pop(0)
        for i in rows:
            i.pop(3) #删去
            i.pop(2)
            i.pop(0)
        return rows

def ALLconf(dict,first,second):
    return dict[first+second] / max(dict[first],dict[second])

def Lift(dict,first,second):
    return dict[first+second] / (dict[first] * dict[second])

def Apriori(rows):
    minsup = 0.1
    curItemlist = oneItem(rows,minsup)
    frequentList = []
    frequentList.append(curItemlist)
    flag = True
    count = 1
    while flag:
        count += 1
        allnextItemlist = nextItem(curItemlist,count)
        nextItemlist = []
        print("nextItem list length")
        print(len(allnextItemlist))
        print("nextItem list:")
        print(allnextItemlist[0:10])
        flag = False
        templist = []
        for item in allnextItemlist:
            if isFrequent(rows,item,minsup):
                flag = True
                templist.append(item)
                nextItemlist.append(item)
        if templist != []:
            frequentList.append(templist)
            print("frequent list:")
            print(frequentList)
        curItemlist = nextItemlist
    return frequentList

def nextItem(curItemlist,count):
    nextItemlist = []
    if count == 2:
        for i in range(len(curItemlist)):
            for j in range(len(curItemlist)):
                if j > i:
                    nextItemlist.append([curItemlist[i],curItemlist[j]])
    else:
        for i in range(len(curItemlist)):
            for j in range(len(curItemlist)):
                if j > i and canConnect(curItemlist[i],curItemlist[j]):
                    temp = curItemlist[i].copy()
                    temp.append(curItemlist[j][-1])
                    nextItemlist.append(temp)
    return nextItemlist

def canConnect(listA,listB):
    for i in range(len(listA)-1):
        if listA[i] != listB[i]:
            return False;
    if listA[-1] != listB[-1]:
        return True
    else:
        return False

def isFrequent(rows,items,minsup):
    count = 0
    for row in rows:
        add = 1
        for item in items:
            if item not in row:
                add = 0
                break
        count += add
    if count > len(rows) * minsup:
        with open("saveData.txt","a",encoding="utf-8") as file:
            file.write("[{" + str(items)+" " + str(round(count/len(rows),4))+"}]\n")
        return True
    else:
        return False

def oneItem(rows,minsup):
    oneItems = []
    oneItemsDict = []
    length = len(rows)
    #print(length)
    for i in range(len(rows[0])):
        column = [row[i] for row in rows]
        columnDict = {}
        for item in column:
            if item == "":
                pass
            elif item in columnDict:
                columnDict[item] += 1
            else:
                columnDict[item] = 1
        #columnDict = dict(sorted(columnDict.items(), key=lambda item: item[1], reverse=True))
        keys = [key for key in columnDict]
        for key in keys:
            if(columnDict[key] < length * minsup):
                columnDict.pop(key)
            else:
                columnDict[key] = round(columnDict[key] / len(rows),4)
        if columnDict != {}:
            oneItemsDict.append(columnDict)
            oneItems += [key for key in columnDict]
    with open("saveData.txt","a",encoding="utf-8") as file:
        file.write(str(oneItemsDict)+"\n")
    return oneItems

if __name__ == '__main__':
    filename = "C:\\Users\\木子\\Desktop\\课程\\数据挖掘\\作业\\wine-reviews\\winemag-data_first150k.csv"
    rows = preprocess(filename)
    length = len(rows)
    print(rows[0])
    Apriori(rows)
