# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:32:29 2021

@author: MATH
"""

import cvxopt
import numpy as np
import random
import pandas as pd
from cvxopt import matrix
import os
import openpyxl
import random
from numpy import linalg as LA
import copy
import math
import time
import pickle
from sklearn.feature_selection import SelectKBest, chi2, f_classif, VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

def initial_chromosome(num_chromosome,length_chromosome):
    chromosomes=[]
    for i in range(num_chromosome):
        chromosome=[]
        for j in range(length_chromosome):
            chromosome.append(random.randint(0, 1)) # random features
        chromosome.append(1) #pick up label
        chromosomes.append(chromosome)
    return chromosomes

def mergeFeatureLabel(x,y):
    x=list(x)
    for i in range(len(y)):
        x[i]=list(x[i])
        x[i].append(y[i])
    return x
def irisNewTarget(iris):
    y=[]
    for i in range(len(iris.target)):
        y.append(iris.target[i]+1)
    return y

def separateFeatureLabel(data):
    x=copy.deepcopy(data)
    y=[]
    for i in range(len(x)):
        y.append(int(x[i].pop()))
    return np.array(x),y
def mergeFeatureLabel(x,y):
    x=list(x)
    for i in range(len(y)):
        x[i]=list(x[i])
        x[i].append(y[i])
    return x

def create_M_position(m): # m is the dimensioal of x
    startPosition=[0] 
    for l in range(1,m):
        temp=startPosition[l-1]+m-l+1
        startPosition.append(temp)
    #print('startPosition:',startPosition)
    
    k=1 # while index
    MPosition=[]
    while k <= m:
        MPosition.append([])
        for i in range(1,k):
            temp=(k-1)+(i-1)*(-1)
            temp2=startPosition[i-1]+temp
            MPosition[k-1].append(temp2)
        #print('k:',k)
        if k==m:
            MPosition[k-1].append(startPosition[k-1])
        else:
            for i in range(startPosition[k]-startPosition[k-1]):
                temp=startPosition[k-1]+i
                MPosition[k-1].append(temp)
        k+=1
    return MPosition
    
def createM(x,MPosition): #for one point
    m=len(MPosition)
    M=np.zeros((m,int((m*m+m)/2)))
    for i in range(m):
        for j in range(len(MPosition[i])):
            M[i][MPosition[i][j]]=x[j]
    return M

def createH(M):
    n=len(M)
    H=np.block([
        [M,np.eye(n)]
    ])
    return H

def createG(x): #for all points
    m=len(x[0]) #dimension
    MPosition=create_M_position(m)
    sumInnerProduct=np.zeros((int((m*m+3*m)/2),int((m*m+3*m)/2)))
    for i in range(len(x)): #for each point
        M=createM(x[i],MPosition)
        H=createH(M)
        innerProductH=H.T.dot(H)
        sumInnerProduct+=innerProductH
        #print('innerProductH:',innerProductH)
    G=sumInnerProduct
    return G


def createS(x): # for all points
    m=len(x[0])
    S=[]
    for k in range(len(x)):
        #S.append([])
        for i in range(m):
            for j in range(i,m):
                if i == j:
                    temp=x[k][i]*x[k][j]/2
                else:
                    temp=x[k][i]*x[k][j]
                S.append(temp)
        for i in range(m):
            S.append(x[k][i])
    return S

def create_s(x): #for one point
    m=len(x)
    s=[]
    for i in range(m):
        for j in range(i,m):
            if i == j:
                temp=x[i]*x[j]/2
            else:
                temp=x[i]*x[j]
            s.append(temp)
    for i in range(m):
            s.append(x[i])
    return s

def matrixP(x):
    n=len(x)
    len_c=1
    G=createG(x)
    len_block=len(G)
    P=matrix(2*np.block([ #1/2
        [G                          , np.zeros((len_block,len_c)), np.zeros((len_block,n)), np.zeros((len_block,n))],
        [np.zeros((len_c,len_block)), np.zeros((len_c,len_c))    , np.zeros((len_c,n))    , np.zeros((len_c,n))    ],
        [np.zeros((n,len_block))    , np.zeros((n,len_c))        , np.zeros((n,n))        , np.zeros((n,n))        ],
        [np.zeros((n,len_block))    , np.zeros((n,len_c))        , np.zeros((n,n))        , np.zeros((n,n))        ]
    ]))
    return P

def matrixq(penaltyC,x):
    n=len(x)
    m=len(x[0])
    len_c=1
    num_row=1
    len_w=int((np.power(m,2)+m)/2)
    q=matrix(penaltyC*np.block([
        [np.zeros((num_row,len_w)), np.zeros((num_row,m)), np.zeros((num_row,len_c)), np.ones((num_row,n)), np.zeros((num_row,n))]
    ]))
    
    return q.T

def matrixb(x):
    n=len(x)
    num_col=1
    b=matrix(np.block([
        np.ones((n,num_col))
    ]))
    return b

def matrixG(x):
    n=len(x)
    m=len(x[0])
    len_c=1
    len_w=int((np.power(m,2)+m)/2)
    G=matrix(np.block([ 
        [np.zeros((n,len_w)), np.zeros((n,m)), np.zeros((n,len_c)), -np.eye(n)     , np.zeros((n,n))],
        [np.zeros((n,len_w)), np.zeros((n,m)), np.zeros((n,len_c)), np.zeros((n,n)), -np.eye(n)]
    ]))
    return G

def matrixh(x):
    n=len(x)
    num_col=1
    h=matrix(np.zeros((2*n,num_col)))
    return h

def create_ys(x,y): #for all points
    block=[]
    for i in range(len(x)):
        s_temp=create_s(x[i]) #for one point
        block.append(y[i]*np.array(s_temp)) #s_temp is a list
    return np.array(block)

def creat_y(y):
    y_list=[]
    for i in range(len(y)):
        y_list.append([])
        y_list[i].append(y[i])
    return np.array(y_list)

def matrixA(x,y):
    temp_ys=create_ys(x,y)
    temp_y=creat_y(y)
    n=len(x)
    A=matrix(np.block([
        temp_ys,temp_y,np.eye(n),-np.eye(n)
    ]))
    return A

def cvxoptSVM(penaltyC,x,label):
    q=matrixq(penaltyC,x)
    P=matrixP(x)
    b=matrixb(x)
    G=matrixG(x)
    h=matrixh(x)
    A=matrixA(x,label)
    sv=cvxopt.solvers.qp(P,q,G,h,A,b,kktsolver='ldl', options={'kktreg':1e-9})
    n=len(x)
    m=len(x[0])
    len_w=(np.power(m,2)+m)/2
    w_list=[]
    b_list=[]
    xi_list=[]
    eta_list=[]
    for i in range(len(sv['x'])):
        if i<len_w:
            w_list.append(sv['x'][i])
        elif len_w<=i<len_w+m:
            b_list.append(sv['x'][i])
        elif len_w+m<=i<len_w+m+1:
            c=sv['x'][i]
        elif len_w+m+1<=i<len_w+m+1+n:
            xi_list.append(sv['x'][i])
        else:
            eta_list.append(sv['x'][i])
    return w_list,b_list,c,xi_list,eta_list



def createSymetricW(w_list,m): #m is diensional of data point
    W=np.zeros((m,m))
    k=0
    for i in range(m):
        for j in range(i,m):
            W[i][j]=w_list[k]
            if i!=j:
                W[j][i]=W[i][j]/2
                W[i][j]/=2
            k+=1
    return W

def computeCenter(W,b,c):
    eigenvalue, eigenvectorP=np.linalg.eig(W)
    b_bar=b.T.dot(eigenvectorP.T)
    center=[]
    #print('eigenvalue:',eigenvalue)
    for i in range(len(eigenvalue)):
        temp=-b_bar[i]/(2*eigenvalue[i]+1)  # / 0
        center.append(temp)
    tempCenter=np.array(center) #x'
    Center=eigenvectorP.T.dot(tempCenter) #x
    return Center

def fvalueFun(W,b,c,x, Center):
    value=(np.transpose(x).dot(W.dot(x)))/2+np.transpose(b).dot(x)+c
    normValue=LA.norm(W.dot(x)+b)
    rho=value/normValue
    
    d_c=decisionValueFun(x,Center)
    return rho, value, normValue, d_c

def decisionValueFun(x,center):
    return np.linalg.norm(x-center)

def objectFucntion(W,b,c,x): #for one point with given m & W belong to m*m array, b belong to m*1 array
    #print('w shape', W.shape)
    value=x.T.dot(W.dot(x))/2+b.T.dot(x)+c #functional value
    decisionValue=value/np.linalg.norm(W.dot(x)+b) #relative geometric distance between f(x)=0 and data point
    center=computeCenter(W,b,c)
    distanceOfCenter=decisionValueFun(x,center)
    return decisionValue, distanceOfCenter

def objectFucntionOAO(W,b,c,x): #for one point with given m & W belong to m*m array, b belong to m*1 array
    #print('w shape', W.shape)
    value=x.T.dot(W.dot(x))/2+b.T.dot(x)+c #functional value
    return value



def SQSSVM(x,label,penaltyC):
    w_list,b_list,c,xi_list,eta_list=cvxoptSVM(penaltyC,x,label)
    m=len(x[0])
    b=np.array(b_list)
    W=createSymetricW(w_list,m)
    xi=np.array(xi_list)
    eat=np.array(eta_list)
    return W,b,c,xi,eat


def one_hot_encoding(row,value,numCategory):
    for i in range(numCategory):
        if i!=value-1:
            row.append(0)
        else:
            row.append(1)
    return row




def dataCalssIndex(data,seedNumber,numClass):
    random.seed(seedNumber)
    shuffleData=copy.deepcopy(data)
    random.shuffle(shuffleData)
    indexClass=[]
    for j in range(numClass):
            indexClass.append([])
    for i in range(len(data)):
        for j in range(numClass):
            if shuffleData[i][-1]==j+1:
                indexClass[j].append(i)
                continue
    return shuffleData, indexClass
        
def separateHoldoutData(indexClass,shuffleData,percentage,numClass):
    trainData=[]
    testData=[]
    for i in range(numClass):
        numTrainData=int(len(indexClass[i])*percentage/100)
        #print('numTrainData:',numTrainData)
        for j in range(numTrainData):
            trainData.append(shuffleData[indexClass[i][0]])
            
            indexClass[i].pop(0)
        for j in range(len(indexClass[i])):
            testData.append(shuffleData[indexClass[i][j]])
    
    return trainData,testData


    


def createMthModelData(trainingData,label): #one against all, label means the mth label as +1, others label as -1
    y=[]
    mthTrainingData=copy.deepcopy(trainingData)
    for i in range(len(mthTrainingData)):
        if mthTrainingData[i][-1]==label:
            mthTrainingData[i].pop()
            y.append(1)
        else:
            y.append(-1)
            mthTrainingData[i].pop()
    return np.array(mthTrainingData),np.array(y)

def createMthModelData_OAO(miltiClassData,label1,label2): #one against one, label means the mth label as +1, (m+1)th label as -1, others drop
    mthBinaryDataTemp=copy.deepcopy(miltiClassData)
    mthBinaryData=[]
    for i in range(len(mthBinaryDataTemp)):
        if mthBinaryDataTemp[i][-1]==label1:
            #print('trainLabel:',mthTrainingDataTemp[i][-1])
            mthBinaryDataTemp[i][-1]=1
            mthBinaryData.append(mthBinaryDataTemp[i])
            
        elif mthBinaryDataTemp[i][-1]==label2:
            #print('trainLabel:',mthTrainingDataTemp[i][-1])
            mthBinaryDataTemp[i][-1]=-1
            mthBinaryData.append(mthBinaryDataTemp[i])
            
    return mthBinaryData

def SQSSVM_multiclass_for_graduate_analysis(numClass,penaltyC,trainingData):
    W_list=[]
    b_list=[]
    c_list=[]
    xi_list=[]
    eta_list=[]
    for i in range(1,numClass+1):
        mthTrainingData,y=createMthModelData(trainingData,i) # multi label convert to binary label 
        W,b,c,xi,eta=SQSSVM(mthTrainingData,y,penaltyC)
        W_list.append(W)
        b_list.append(b)
        c_list.append(c)
        xi_list.append(xi)
        eta_list.append(eta)
    return W_list,b_list,c_list,xi_list,eta_list

def createCheckLabelList(numClass):
    checkLabelList=[]
    for i in range(1,numClass):
        for j in range(i+1,numClass+1):
            checkLabelList.append([i,j])
    return checkLabelList

def SQSSVM__multiclass_OAO_for_graduate_analysis(numClass,penaltyC,trainingData):
    W_list=[]
    b_list=[]
    c_list=[]
    xi_list=[]
    eta_list=[]
    checkLabelList=[]
    for i in range(1,numClass):
        for j in range(i+1,numClass+1):
            checkLabelList.append([i,j])
            mthTrainingData,y=createMthModelData_OAO(trainingData,i,j)
            W,b,c,xi,eta=SQSSVM(mthTrainingData,y,penaltyC)
            W_list.append(W)
            b_list.append(b)
            c_list.append(c)
            xi_list.append(xi)
            eta_list.append(eta)
    return W_list,b_list,c_list,xi_list,eta_list,checkLabelList

def predictDataSalaryLabel(x,W_list,b_list,c_list): #for one point five model decision value OAA
    numClass=len(W_list)
    decisionValueList=[]
    distanceOfCenterList=[]
    for i in range(numClass):
        decisionValue, distanceOfCenter=objectFucntion(W_list[i],b_list[i],c_list[i],x)
        decisionValueList.append(decisionValue)
        distanceOfCenterList.append(distanceOfCenter)
     # relative geometric
    Label=decisionValueList.index(max(decisionValueList))+1
    #print('label:',Label)
    return Label



def predictDataSalaryLabelOAO(x,W_list,b_list,c_list,numClass,checkLabelList): #for one point ten model decision value OAO 
    vote_list=[]
    for i in range(numClass):
        vote_list.append(0)  #given 0 into each label
    decisionValueList=[]
    for i in range(len(W_list)):
        decisionValue=objectFucntionOAO(W_list[i],b_list[i],c_list[i],x)
        decisionValueList.append(decisionValue)
        
    #print(decisionValueList) 
    # functional value  vote
        
    for i in range(len(decisionValueList)):
        if decisionValueList[i] >0:
            vote_list[checkLabelList[i][0]-1]+=1
        else:
            vote_list[checkLabelList[i][1]-1]+=1
    #print('decisionValueList:',decisionValueList)
    #print('vote:', vote_list)
    label=vote_list.index(max(vote_list))+1
    #print('predicte label:', label)
    #print('--------------------------------')
    return label
    
   # print('decisionValueList:',decisionValueList)
    
def createConfuseArrayOAO(testingData,W_list,b_list,c_list,numClass,checkLabelList):
    confuseArray=np.zeros((numClass,numClass))
    X_list=copy.deepcopy(testingData)
    X=np.array(X_list)
    count=0
    for i in range(len(X)):
        #print('CorrectLabel:',X[i][-1])
        Label=predictDataSalaryLabelOAO(X[i][:-1],W_list,b_list,c_list,numClass,checkLabelList)
       
        confuseArray[int(Label-1)][int(X[i][-1]-1)]+=1
        if X[i][-1]-1 == Label-1:
            count+=1
    accurcy=float(count/len(testingData))
    return confuseArray,accurcy
        

def compute_error(data,W_list,b_list,c_list,decisionValueType):
    X_list=copy.deepcopy(data)
    X=np.array(X_list)
    error_index=[]
    for i in range(len(X)):
        Label=predictDataSalaryLabel(X[i][:-1],W_list,b_list,c_list,decisionValueType)
        if X[i][-1]!=Label:
            error_index.append(i)
    error_rate=float(len(error_index)/len(data))
    return error_rate
        
def createConfuseArray(testingData,W_list,b_list,c_list):
    numClass=len(W_list)
    confuseArray=np.zeros((numClass,numClass))
    X_list=copy.deepcopy(testingData)
    X=np.array(X_list)
    count=0
    for i in range(len(X)):
        Label=predictDataSalaryLabel(X[i][:-1],W_list,b_list,c_list)
        confuseArray[int(Label-1)][int(X[i][-1]-1)]+=1
        if X[i][-1]-1 == Label-1:
            count+=1
    accurcy=float(count/len(testingData))
    return confuseArray,accurcy

#20210914 revise
def saveObjectVariables(W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcy,filename,dirname,subdirname):
    os.chdir('C:\\Users\\MATH\\graduate_20210914\\'+dirname+'\\'+subdirname)
    f = open(filename, 'wb')
    # W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcyTest, accuracyTrain
    pickle.dump([W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcy], f)
    f.close()

def saveObjectVariablesKbest(W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcy,filename,dirname,subdirname, k):
    os.chdir('C:\\Users\\MATH\\graduate_20210914\\'+dirname+'\\'+subdirname)
    f = open(filename, 'wb')
    # W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcyTest, accuracyTrain
    pickle.dump([W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcy, k], f)
    f.close()
    
def saveRecord(record, filename):
    os.chdir('C:\\Users\\MATH\\graduate_20210914\\GA_traindata')
    f = open(filename, 'wb')
    # W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcyTest, accuracyTrain
    pickle.dump(record, f)
    f.close()
    
def loadObjectVariables(filename,dirname,subdirname):
    os.chdir('C:\\Users\\MATH\\graduate_20210914\\'+dirname+'\\'+subdirname)
    f = open(filename, 'rb')
    W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcyTest, accuracyTrain = pickle.load(f)
    f.close()
    return W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcy

def loadChromosome(filename):
    os.chdir('C:\\Users\\MATH\\graduate_20210914')
    f = open(filename, 'rb')
    record = pickle.load(f)
    f.close()
    return record

def loadObjectVariablesKbest(filename,dirname,subdirname):
    os.chdir('C:\\Users\\MATH\\graduate_20210914\\'+dirname+'\\'+subdirname)
    f = open(filename, 'rb')
    W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcyTest, accuracyTrain, k = pickle.load(f)
    f.close()
    return W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcy, k


def loadDataFromPCKL(filename):
    os.chdir('C:\\Users\\MATH\\graduate_20210914')
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def loadRecord(filename):
    os.chdir('C:\\Users\\MATH\\graduate_20210914\\GA_traindata')
    f = open(filename, 'rb')
    record = pickle.load(f)
    f.close()
    return record

def countLabelSavePositions(numClass,Data):
    count=[]
    position=[]
    for i in range(numClass):
        count.append(0)
        position.append([])
    for i in range(len(Data)):
        for j in range(numClass):
            if Data[i][-1]==j+1:
                count[j]+=1
                position[j].append(i)
    return count,position

def shuffleData(dataPosition1,seedNumber):
    random.seed(seedNumber)
    shuffleDataPosition=copy.deepcopy(dataPosition1)
    for i in range(len(shuffleDataPosition)):
        random.shuffle(shuffleDataPosition[i])
    return shuffleDataPosition


def seperateK_FoldData(shuffleDataPosition,Data,numK,count):
    kFoldData=[]
    count1=copy.deepcopy(count)
    for i in range(numK):
        kFoldData.append([])
        if i!=numK-1:
            for j in range(len(count)):
                for k in range(math.floor(count[j]/numK)):
                    kFoldData[i].append(Data[shuffleDataPosition[j][0]]) #get first because pop
                    count1[j]-=1
                    shuffleDataPosition[j].pop(0)
        elif i==numK-1:
            for j in range(len(count1)):
                for k in range(count1[j]):
                    kFoldData[i].append(Data[shuffleDataPosition[j][k]])
    return kFoldData
            

def getTrainTestDataFromKFoldData(kFoldData,Kth):
    trainingData=[]
    testingData=[]
    for i in range(len(kFoldData)):
        if i!=Kth:
            trainingData+=kFoldData[i]
        else:
            testingData=kFoldData[i]
    return trainingData,testingData

def filterFeature(data,p):
    sel = VarianceThreshold(threshold=(p * (1 - p)))
    new_data=sel.fit_transform(data)
    filter_list=sel.get_support()
    return new_data,filter_list

def minMaxData(x):
    scaler = MinMaxScaler()
    scaler.fit(x)
    scaled_x=scaler.transform(x)
    return scaled_x

def selectKBestFunction(k,x,y):
    kbest=SelectKBest(chi2,k)
    new_x=kbest.fit_transform(x,y)
    p_value_list=kbest.pvalues_
    return new_x, p_value_list

def num_K(p_value_list):
    count=0
    for p in p_value_list:
        if p<0.05:
            count+=1
    return count

def OAA_binaryLabel(data,label): #one against all, label means the mth label as +1, others label as -1
    binaryData=copy.deepcopy(data)
    for i in range(len(binaryData)):
        if binaryData[i][-1]==label:
            binaryData[i][-1]=1
        else:
            binaryData[i][-1]=-1
    return binaryData

def binaryLabelPredict(x,W,b,c):
    value = x.T.dot(W.dot(x))/2+b.T.dot(x)+c #functional value
    if value > 0:
        return 1
    else:
        return -1

def binaryConfusearray(testData,W,b,c):
    numClass=2
    tempList=[1,-1]
    confuseArray=np.zeros((numClass,numClass))
    X_list=copy.deepcopy(testData)
    X=np.array(X_list)
    count=0
    for i in range(len(X)):
        Label=binaryLabelPredict(X[i][:-1],W,b,c)
        confuseArray[tempList.index(Label)][tempList.index(X[i][-1])]+=1
        if X[i][-1] == Label:
            count+=1
    accuracy=float(count/len(testData))
    return confuseArray,accuracy

def spliteBinaryData(binaryData,seedNumber,splitePercentage_p):
    random.seed(seedNumber)
    positiveData=[]
    negativeData=[]
    trainData=[]
    testData=[]
    for i in range(len(binaryData)):
        if binaryData[i][-1]==1:
            positiveData.append(binaryData[i])
        else:
            negativeData.append(binaryData[i])
    random.shuffle(positiveData)
    random.shuffle(negativeData)
    numPositiveTrain=math.floor(len(positiveData)*splitePercentage_p/100)
    numNegativeTrain=math.floor(len(negativeData)*splitePercentage_p/100)
    
    for i in range(len(positiveData)):
        if i < numPositiveTrain:
            trainData.append(positiveData[i])
        else:
            testData.append(positiveData[i])
    for i in range(len(negativeData)):
        if i < numNegativeTrain:
            trainData.append(negativeData[i])
        else:
            testData.append(negativeData[i])
    return trainData, testData

def Holdout(binaryData, splitePercentage_p,penaltyC): #binary data input SQSSVM and report model and accuracy
    seedNumber=random.randint(1,100)
    
    trainData, testData=spliteBinaryData(binaryData,seedNumber,splitePercentage_p)
    trainData_x, trainData_y=separateFeatureLabel(trainData)
    W,b,c,xi,eta=SQSSVM(trainData_x,trainData_y,penaltyC) 
    
    confuseArray,accurcy=binaryConfusearray(testData,W,b,c)
    return W,b,c,xi,eta,accurcy

def OAA_Holdout(multiclassData, numClass, splitePercentage_p, penaltyC, numIteration):
    best_W_list=[]
    best_b_list=[]
    best_c_list=[]
    best_xi_list=[]
    best_eta_list=[]
    startTestTime = time.time()
    for j in range(numClass):
        binaryData=OAA_binaryLabel(multiclassData,j+1)
        W_list=[]
        b_list=[]
        c_list=[]
        xi_list=[]
        eta_list=[]
        accuracyList=[]
        for i in range(numIteration):
            W,b,c,xi,eta,accuracy=Holdout(binaryData, splitePercentage_p,penaltyC)
            accuracyList.append(accuracy)
            W_list.append(W)
            b_list.append(b)
            c_list.append(c)
            xi_list.append(xi)
            eta_list.append(eta)
        bestIndex=accuracyList.index(max(accuracyList))
        best_W_list.append(W_list[bestIndex])
        best_b_list.append(b_list[bestIndex])
        best_c_list.append(c_list[bestIndex])
        best_xi_list.append(xi_list[bestIndex])
        best_eta_list.append(eta_list[bestIndex])
    endTestTime = time.time()
    timeTrain=endTestTime-startTestTime
    startTestTime = time.time()
    confuseArray,accuracy=createConfuseArray(multiclassData,best_W_list,best_b_list,best_c_list)
    endTestTime = time.time()
    timePredict=endTestTime-startTestTime
    return best_W_list, best_b_list, best_c_list, best_xi_list, best_eta_list, accuracy, timeTrain, timePredict

def OAO_Holdout(multiclassData, numClass, splitePercentage_p, penaltyC, numIteration):
    checkLabelList=[]
    best_W_list=[]
    best_b_list=[]
    best_c_list=[]
    best_xi_list=[]
    best_eta_list=[]
    startTestTime = time.time()
    for i in range(1,numClass):
        for j in range(i+1,numClass+1):
            checkLabelList.append([i,j])
            binaryData=createMthModelData_OAO(multiclassData,i,j)
            W_list=[]
            b_list=[]
            c_list=[]
            xi_list=[]
            eta_list=[]
            accuracyList=[]
            for i in range(numIteration):
                W,b,c,xi,eta,accuracy=Holdout(binaryData, splitePercentage_p,penaltyC)
                accuracyList.append(accuracy)
                W_list.append(W)
                b_list.append(b)
                c_list.append(c)
                xi_list.append(xi)
                eta_list.append(eta)
            bestIndex=accuracyList.index(max(accuracyList))
            best_W_list.append(W_list[bestIndex])
            best_b_list.append(b_list[bestIndex])
            best_c_list.append(c_list[bestIndex])
            best_xi_list.append(xi_list[bestIndex])
            best_eta_list.append(eta_list[bestIndex])
    endTestTime = time.time()
    timeTrain=endTestTime-startTestTime
    startTestTime = time.time()
    confuseArray,accuracy=createConfuseArrayOAO(multiclassData,best_W_list,best_b_list,best_c_list,numClass,checkLabelList)
    endTestTime = time.time()
    timePredict=endTestTime-startTestTime
    return best_W_list, best_b_list, best_c_list, best_xi_list, best_eta_list, accuracy, timeTrain, timePredict

def separateKset(shuffleBinaryData,numK):
    K_fold=[]
    positiveData=[]
    negativeData=[]
    #print('shuffleBinaryData:',shuffleBinaryData)
    for i in range(len(shuffleBinaryData)):
        if shuffleBinaryData[i][-1]==1:
            positiveData.append(shuffleBinaryData[i])
        else:
            negativeData.append(shuffleBinaryData[i])
    numPositiveTrain=int(len(positiveData)/numK)
    numNegativeTrain=int(len(negativeData)/numK)
    for i in range(numK):
        setList=[]
        if i != numK:
            for j in range(numPositiveTrain):
                setList.append(positiveData[0])
                positiveData.pop(0)
            for j in range(numNegativeTrain):
                setList.append(negativeData[0])
                negativeData.pop(0)
        else:
            for j in range(len(shuffleBinaryData)-(numK-1)*numPositiveTrain):
                setList.append(positiveData[0])
                positiveData.pop(0)
            for j in range(len(shuffleBinaryData)-(numK-1)*numNegativeTrain):
                setList.append(negativeData[0])
                negativeData.pop(0)
        K_fold.append(setList)
    return K_fold

def KFold(binaryData, numK,penaltyC): #binary data input SQSSVM and report model and accuracy
    seedNumber=random.randint(1,100)
    random.seed(seedNumber)
    random.shuffle(binaryData)
    K_fold=separateKset(binaryData,numK)
    W_list=[]
    b_list=[]
    c_list=[]
    xi_list=[]
    eta_list=[]
    accuracyList=[]
    for Kth in range(numK):
        trainData, testData=getTrainTestDataFromKFoldData(K_fold,Kth)
        trainData_x, trainData_y=separateFeatureLabel(trainData)
        W,b,c,xi,eta=SQSSVM(trainData_x,trainData_y,penaltyC) 
        confuseArray,accuracy=binaryConfusearray(testData,W,b,c)
        accuracyList.append(accuracy)
        W_list.append(W)
        b_list.append(b)
        c_list.append(c)
        xi_list.append(xi)
        eta_list.append(eta)
    bestIndex=accuracyList.index(max(accuracyList))
    return W_list[bestIndex],b_list[bestIndex],c_list[bestIndex],xi_list[bestIndex],eta_list[bestIndex],accuracyList[bestIndex]

def N_KFold(numIteration,binaryData, numK,penaltyC):
    W_list=[]
    b_list=[]
    c_list=[]
    xi_list=[]
    eta_list=[]
    accuracyList=[]
    for i in range(numIteration):
        W,b,c,xi,eta,accuracy=KFold(binaryData, numK,penaltyC)
        accuracyList.append(accuracy)
        W_list.append(W)
        b_list.append(b)
        c_list.append(c)
        xi_list.append(xi)
        eta_list.append(eta)
    bestIndex=accuracyList.index(max(accuracyList))
    return W_list[bestIndex],b_list[bestIndex],c_list[bestIndex],xi_list[bestIndex],eta_list[bestIndex],accuracyList[bestIndex]

def OAA_N_KFold(multiclassData, numClass,numIteration, numK,penaltyC):
    W_list=[]
    b_list=[]
    c_list=[]
    xi_list=[]
    eta_list=[]
    startTestTime = time.time()
    for j in range(numClass):
        binaryData=OAA_binaryLabel(multiclassData,j+1)
        W,b,c,xi,eta,accuracy=N_KFold(numIteration,binaryData, numK,penaltyC)
        W_list.append(W)
        b_list.append(b)
        c_list.append(c)
        xi_list.append(xi)
        eta_list.append(eta)
    endTestTime = time.time()
    timeTrain=endTestTime-startTestTime
    startTestTime = time.time()
    confuseArray,accuracy1=createConfuseArray(multiclassData,W_list,b_list,c_list)
    endTestTime = time.time()
    timePredict=endTestTime-startTestTime
    return W_list, b_list, c_list, xi_list, eta_list, accuracy1, timeTrain, timePredict

def OAO_N_KFold(multiclassData, numClass,numIteration, numK,penaltyC):
    checkLabelList=[]
    best_W_list=[]
    best_b_list=[]
    best_c_list=[]
    best_xi_list=[]
    best_eta_list=[]
    startTestTime = time.time()
    for i in range(1,numClass):
        for j in range(i+1,numClass+1):
            checkLabelList.append([i,j])
            binaryData=createMthModelData_OAO(multiclassData,i,j)
            W,b,c,xi,eta,accuracy=N_KFold(numIteration,binaryData, numK,penaltyC)
            best_W_list.appedn(W)
            best_b_list.append(b)
            best_c_list.append(c)
            best_xi_list.append(xi)
            best_eta_list.appedn(eta)
    endTestTime = time.time()
    timeTrain=endTestTime-startTestTime
    startTestTime = time.time()
    confuseArray,accuracy1=createConfuseArrayOAO(multiclassData,best_W_list,best_b_list,best_c_list,numClass,checkLabelList)
    endTestTime = time.time()
    timePredict=endTestTime-startTestTime
    return best_W_list, best_b_list, best_c_list, best_xi_list, best_eta_list, accuracy1, timeTrain, timePredict

def selectDataByColumns(data,selectList):
    newData=[]
    for i in range(len(data)):
        row=[]
        for j in range(len(data[i])):
            if selectList[j]==1:
                row.append(data[i][j])
        newData.append(row)
    return newData

def getOneHotEncodingData(selectedData,selectedNum_oneHotEncode):
    selectEncodingData=[]
    for i in range(1,len(selectedData)): #no header
        row=[]
        for j in range(len(selectedData[i])):
            if selectedNum_oneHotEncode[0][j]!=0:
                row=one_hot_encoding(row,selectedData[i][j],selectedNum_oneHotEncode[0][j])
            else:
                row.append(selectedData[i][j])
        selectEncodingData.append(row)
    return selectEncodingData

def one_hot_encoding(row,value,numCategory):
    for i in range(numCategory):
        if i!=value-1:
            row.append(0)
        else:
            row.append(1)
    return row

def fitnessValue(data,chromosome):
    num_oneHotEncode=[[4,7,40,11,8,5,12,16,7,5,5,5,0,2,0,0,2,12,16,2,5,5,5,0,2,0,0,2,0]] #獨熱編碼數量 given
    selectedData=selectDataByColumns(data,chromosome)
    selectedNum_oneHotEncode=selectDataByColumns(num_oneHotEncode,chromosome)
    selectEncodingData=getOneHotEncodingData(selectedData,selectedNum_oneHotEncode)
    p=0.1
    splitePercentage_p=70
    numIteration=2
    numClass=5
    dirname='GA_traindata' #here
    penaltyC=100
    filename='model'
    W_list,b_list,c_list,xi_list,eta_list,accuracy,timeTrain,timePredict=OAA_Holdout(selectEncodingData, numClass, splitePercentage_p, penaltyC, numIteration)
    #filename1='GA_select'+str(len(selectedNum_oneHotEncode[0]))+'_accuracy'+str(accuracy)+'_variables.pckl' 
    #saveObjectVariables(W_list,b_list,c_list,xi_list,eta_list,timeTrain,timePredict,accuracy,filename1,dirname,'type1')
    return accuracy

def selection(population, numSelection, fitnessValueList):
    sumFitnessValue=0
    for i in range(len(fitnessValueList)):
        sumFitnessValue+=fitnessValueList[i]
    probabilityFitness=[]
    for i in range(len(fitnessValueList)):
        probabilityFitness.append(fitnessValueList[i]/sumFitnessValue)
    Index=[]
    for i in range(len(probabilityFitness)):
        Index.append(i)
    indexSelect=np.random.choice(Index,size=numSelection, p=probabilityFitness)
    
    parent=[]
    for i in range(numSelection):
        parent.append(population[indexSelect[i]])
        print('fitnessValue:',fitnessValueList[indexSelect[i]])
    return parent

def initialChromosomeType2(data ,numChromosome, lengthChromosome):
    numFeature=len(data[0])-1 #delete label 
    chromosomes=[]
    Data=[]
    for i in range(numChromosome):
        tempdata=[]
        chromosome=[]
        selectedColumn=np.random.choice(numFeature, lengthChromosome, replace=False)
        selectedColumn=np.append(selectedColumn,numFeature) # add label
        for k in range(len(data)):
            row=[]
            for j in selectedColumn:
                row.append(data[k][j])
            tempdata.append(row)
        Data.append(tempdata)
        for j in range(len(data[0])):
            if j in selectedColumn:
                chromosome.append(1)
            else:
                chromosome.append(0)
        chromosomes.append(chromosome)
    return chromosomes, Data

def fitnessValueType2(data):
    p=0.1
    splitePercentage_p=70
    numIteration=2
    numClass=5
    dirname='GA_traindata' #here
    penaltyC=100
    filename='model'
    W_list,b_list,c_list,xi_list,eta_list,accuracy,timeTrain,timePredict=OAA_Holdout(data, numClass, splitePercentage_p, penaltyC, numIteration)
    #filename1='GA_select'+str(len(selectedNum_oneHotEncode[0]))+'_accuracy'+str(accuracy)+'_variables.pckl' 
    #saveObjectVariables(W_list,b_list,c_list,xi_list,eta_list,timeTrain,timePredict,accuracy,filename1,dirname,'type1')
    return accuracy
def crossOver(parents, pc, numChromosome):
    newParents=[]
    while len(newParents)<numChromosome:
        selectedParents=np.random.choice(len(parents), 2, replace=False)
        parent1=parents[selectedParents[0]]
        parent2=parents[selectedParents[1]]
        splitePoint=np.random.randint(1,len(parent1)-1)
        for position in range(splitePoint,len(parent1)-1):
            temp1=parent1[position]
            parent1[position]=parent2[position]
            parent2[position]=temp1
        if np.random.rand()<pc:
            newParents.append(parent1)
            newParents.append(parent2)
        else:
            newParents.append(parents[selectedParents[0]])
            newParents.append(parents[selectedParents[1]])
    return newParents

def mutation(population,pm):
    for i in range(len(population)):
        for j in range(len(population[i])-1): #beside label
            if np.random.rand()<pm:
                if population[i][j]==1:
                    population[i][j]=0
                else:
                    population[i][j]=1
    return population

def chromosomeToData(data,chromosomes):
    Data=[]
    for i in range(len(chromosomes)):
        tempData=[]
        for j in range(len(data)):
            row=[]
            for k in range(len(data[j])):
                if chromosomes[i][k]==1:
                    row.append(data[j][k])
            tempData.append(row)
        Data.append(tempData)
    return Data

def chromosomeToDataType2(data,chromosome):
    
    tempData=[]
    for j in range(len(data)):
        row=[]
        for k in range(len(data[j])):
            if chromosome[k]==1:
                row.append(data[j][k])
        tempData.append(row)
        
    return tempData
# In[] 


# In[] 
filename='encodeData1.pckl'
data=loadDataFromPCKL(filename)
num_chromosome=30 #number of chromosome
length_chromosome=28 # number of feature 
chromosomes=initial_chromosome(num_chromosome,length_chromosome)
record=[]
for chromosome in chromosomes:
    accuracy=fitnessValue(data,chromosome)
    record.append([chromosome,accuracy])
# In[]  preprossing
p=0.1
filename='encodeData1.pckl'
data=loadDataFromPCKL(filename)
num_chromosome=30 #number of chromosome
length_chromosome=28 # number of feature 
chromosomes=initial_chromosome(num_chromosome,length_chromosome)
record=[]
x,y=separateFeatureLabel(data[1:])
filterData, filter_list=filterFeature(x,p)
scale_x=minMaxData(filterData)
mergeData=mergeFeatureLabel(scale_x,y)
for chromosome in chromosomes:
    accuracy=fitnessValue(mergeData,chromosome)
    record.append([chromosome,accuracy])
# In[] 
filename='chromosome_accuracy.pckl'
record=loadChromosome(filename)
numSelection=15
population=[]
fitnessValueList=[]
for i in range(len(record)):
    population.append(record[i][0])
    fitnessValueList.append(record[i][1])
parent=selection(population, numSelection, fitnessValueList)

# In[] 20211011 全挑->one-hot-encding->前後各挑30ㄍ->run->save record
os.chdir('C:\\Users\\MATH\\graduate_20210914')
filename='GA_encodingData.pckl'
data=loadDataFromPCKL(filename)
filename1='initialChromosomeAccuray130.pckl'
numChromosome=30
lengthChromosome=130

chromosomes,Data=initialChromosomeType2(data ,numChromosome, lengthChromosome)
record=[]
for i in range(len(chromosomes)):
    accuracy=fitnessValueType2(Data[i])
    record.append([chromosomes[i],accuracy])
saveRecord(record, filename1)

# In[]
os.chdir('C:\\Users\\MATH\\graduate_20210914')
filename='GA_encodingData.pckl'
data=loadDataFromPCKL(filename)
filename='initialChromosomeAccuray.pckl'
record=loadRecord(filename)
numChromosome=30
lengthChromosome=30
numSelection=16
pc=0.8 #低於0.8皆會交配
pm=0.05 #低於0.05會突變
iteration=5
fitnessValueList=[]
for i in range(len(record)):
    fitnessValueList.append(record[i][1])
chromosomes=[]
for i in range(len(record)):
    chromosomes.append(record[i][0])
filename2='result_'
for k in range(5,15):
    parents=selection(chromosomes, numSelection, fitnessValueList)
    newParents=crossOver(parents, pc, numChromosome)
    chromosomes=mutation(newParents,pm)
    fitnessValueList=[]
    Data=chromosomeToData(data,chromosomes)
    record=[]
    for i in range(len(chromosomes)):
        accuracy=fitnessValueType2(Data[i])
        fitnessValueList.append(accuracy)
        record.append([chromosomes[i],accuracy])
    filename3=filename2+str(k+1)+'.pckl'
    saveRecord(record, filename3)

# In[] get chromosome
filename='initialChromosomeAccuray.pckl'
record=loadRecord(filename)
accuracy=[]
chromosomes=[]
for j in range(len(record)):
    accuracy.append(record[j][1])
    chromosomes.append(record[j][0])
minChromosome=chromosomes[accuracy.index(min(accuracy))]
#print(min(accuracy))

filename='result_5.pckl'
record=loadRecord(filename)
accuracy=[]
chromosomes=[]
for j in range(len(record)):
    accuracy.append(record[j][1])
    chromosomes.append(record[j][0])
maxChromosome=chromosomes[accuracy.index(max(accuracy))]
#print(max(accuracy))

# In[] 

index=[]
for i in range(len(maxChromosome)):
    if maxChromosome[i] != minChromosome[i]:
        if maxChromosome[i] == 1:
            index.append(i)

record=[]
for i in range(len(index)):
    newChromosome=copy.deepcopy(minChromosome)
    newChromosome[index[i]]=1
    Data=chromosomeToDataType2(data,newChromosome)
    accuracy=fitnessValueType2(Data)
    record.append([index[i],newChromosome,accuracy])
filename2='compare_'   

for i in range(len(index)):
    newChromosome=copy.deepcopy(minChromosome)
    newChromosome[index[i]]=1
    Data=chromosomeToDataType2(data,newChromosome)
    accuracy=fitnessValueType2(Data)
    record.append([indexArray,newChromosome,accuracy])
    IndexArray.append(indexArray)
filename3=filename2+'1.pckl'
saveRecord(record, filename3)

numTest=400

for j in range(1,10):
    record=[]
    IndexArray=[]
    for i in range(numTest):
        newChromosome=copy.deepcopy(minChromosome)
        indexArray=np.random.choice(index, j, replace=False)
        if indexArray not in IndexArray:
            for k in indexArray:
                newChromosome[k]=1
            Data=chromosomeToDataType2(data,newChromosome)
            accuracy=fitnessValueType2(Data)
            record.append([indexArray,newChromosome,accuracy])
            IndexArray.append(indexArray)
    filename3=filename2+str(j)+'.pckl'
    saveRecord(record, filename3)
    
newChromosome=copy.deepcopy(minChromosome)
indexArray=[63,10,164]
for k in indexArray:
    newChromosome[k]=1
Data=chromosomeToDataType2(data,newChromosome)
accuracy=fitnessValueType2(Data)

# In[]
filename='compare_1.pckl'
record=loadRecord(filename)
accuracy=[]
chromosomes=[]
for i in range(len(record)):
    accuracy.append(record[i][2])
    chromosomes.append(record[i][1])
    
temp = sorted(accuracy)

numMinSelection=10
newRecord=[]
for i in range(numMinSelection):
    chromosome=record[accuracy.index(temp[i])][1] #找出染色體
    flag=0
    
    while flag<1:
        position=np.random.choice(index, 1, replace=False)[0]
        print('position:',position)
        if chromosome[position]==0:
            chromosome[position]+=1
            flag=1
    Data=chromosomeToDataType2(data,chromosome)
    newaccuracy=fitnessValueType2(Data)
    newRecord.append([[record[accuracy.index(temp[i])][0],position],chromosome,newaccuracy])    
    
    
