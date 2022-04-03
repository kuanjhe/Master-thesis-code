# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:12:25 2021

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
import copy
import math
import time
import pickle
from sklearn.feature_selection import SelectKBest, chi2, f_classif, VarianceThreshold
from sklearn.preprocessing import MinMaxScaler



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
    for i in range(len(eigenvalue)):
        temp=b_bar[i]/(2*eigenvalue[i])
        center.append(temp)
    return np.array(center)

def decisionValueFun(x,center):
    return np.linalg.norm(x-center)

def objectFucntion(W,b,c,x): #for one point with given m & W belong to m*m array, b belong to m*1 array
    value=x.T.dot(W.dot(x))/2+b.T.dot(x)+c #functional value
    decisionValue=value/np.linalg.norm(W.dot(x)+b) #relative geometric distance between f(x)=0 and data point
    center=computeCenter(W,b,c)
    distanceOfCenter=decisionValueFun(x,center)
    return decisionValue, distanceOfCenter

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
        print('numTrainData:',numTrainData)
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

def createMthModelData_OAO(trainingData,label): #one against one, label means the mth label as +1, (m+1)th label as -1, others drop
    y=[]
    mthTrainingDataTemp=copy.deepcopy(trainingData)
    mthTrainingData=[]
    for i in range(len(mthTrainingDataTemp)):
        if mthTrainingDataTemp[i][-1]==label:
            #print('trainLabel:',mthTrainingDataTemp[i][-1])
            mthTrainingDataTemp[i].pop()
            mthTrainingData.append(mthTrainingDataTemp[i])
            
            #print('mthLabel:+1')
            y.append(1)
        elif mthTrainingDataTemp[i][-1]==label+1:
            #print('trainLabel:',mthTrainingDataTemp[i][-1])
            mthTrainingDataTemp[i].pop()
            mthTrainingData.append(mthTrainingDataTemp[i])
            
            #print('mthLabel: -1')
            y.append(-1)
        
    return np.array(mthTrainingData),np.array(y)

def SQSSVM_multiclass_for_graduate_analysis(numClass,penaltyC,trainingData):
    W_list=[]
    b_list=[]
    c_list=[]
    xi_list=[]
    eta_list=[]
    for i in range(1,numClass+1):
        mthTrainingData,y=createMthModelData(trainingData,i)
        W,b,c,xi,eta=SQSSVM(mthTrainingData,y,penaltyC)
        W_list.append(W)
        b_list.append(b)
        c_list.append(c)
        xi_list.append(xi)
        eta_list.append(eta)
    return W_list,b_list,c_list,xi_list,eta_list

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
            mthTrainingData,y=createMthModelData_OAO(trainingData,i)
            W,b,c,xi,eta=SQSSVM(mthTrainingData,y,penaltyC)
            W_list.append(W)
            b_list.append(b)
            c_list.append(c)
            xi_list.append(xi)
            eta_list.append(eta)
    return W_list,b_list,c_list,xi_list,eta_list,checkLabelList

def predictDataSalaryLabel(x,W_list,b_list,c_list, decisionValueType): #for one point five model decision value OAA
    numClass=len(W_list)
    decisionValueList=[]
    distanceOfCenterList=[]
    for i in range(numClass):
        decisionValue, distanceOfCenter=objectFucntion(W_list[i],b_list[i],c_list[i],x)
        decisionValueList.append(decisionValue)
        distanceOfCenterList.append(distanceOfCenter)
    if decisionValueType==1: # relative geometric
        m=max(decisionValueList)
        for i in range(len(decisionValueList)):
            if decisionValueList[i]==m:
                Label=i+1
    elif decisionValueType==2: #distance from center
        maxi=-float('inf')
        Label=1
        for i in range(len(decisionValueList)):
            if decisionValueList[i]>0:
                if distanceOfCenterList[i]>maxi:
                    Label=i+1
                    maxi=distanceOfCenterList[i]
    #print('label:',Label)
    return Label



def predictDataSalaryLabelOAO(x,W_list,b_list,c_list,numClass,checkLabelList, decisionValueType): #for one point ten model decision value OAO 
    vote_list=[]
    for i in range(numClass):
        vote_list.append(0)  #given 0 into each label
    decisionValueList=[]
    for i in range(len(W_list)):
        decisionValue=objectFucntion(W_list[i],b_list[i],c_list[i],x)
        decisionValueList.append(decisionValue)
    for i in range(len(checkLabelList)):
        if decisionValueList[i]>0:
            vote_list[checkLabelList[i][0]-1]+=1
        else:
            vote_list[checkLabelList[i][1]-1]+=1
    print('vote_list:',vote_list)
    label=vote_list.index(max(vote_list))+1
    print('label:',label)
    return label
    
   # print('decisionValueList:',decisionValueList)
    
def createConfuseArrayOAO(testingData,W_list,b_list,c_list,numClass,checkLabelList):
    confuseArray=np.zeros((numClass,numClass))
    X_list=copy.deepcopy(testingData)
    X=np.array(X_list)
    count=0
    for i in range(len(X)):
        print('CorrectLabel:',X[i][-1])
        Label=predictDataSalaryLabelOAO(X[i][:-1],W_list,b_list,c_list,numClass,checkLabelList)
       
        confuseArray[int(X[i][-1]-1)][int(Label-1)]+=1
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
        
def createConfuseArray(testingData,W_list,b_list,c_list,decisionValueType):
    numClass=len(W_list)
    confuseArray=np.zeros((numClass,numClass))
    X_list=copy.deepcopy(testingData)
    X=np.array(X_list)
    count=0
    for i in range(len(X)):
        Label=predictDataSalaryLabel(X[i][:-1],W_list,b_list,c_list, decisionValueType)
        confuseArray[int(X[i][-1]-1)][int(Label-1)]+=1
        if X[i][-1]-1 == Label-1:
            count+=1
    accurcy=float(count/len(testingData))
    return confuseArray,accurcy

#20210615 revise
def saveObjectVariables(W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcyTest, accuracyTrain,filename,dirname,subdirname):
    os.chdir('C:\\Users\\MATH\\graduate\\'+dirname+'\\'+subdirname)
    f = open(filename, 'wb')
    # W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcyTest, accuracyTrain
    pickle.dump([W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcyTest, accuracyTrain], f)
    f.close()

def saveObjectVariablesKbest(W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcyTest, accuracyTrain,filename,dirname,subdirname, k):
    os.chdir('C:\\Users\\MATH\\graduate\\'+dirname+'\\'+subdirname)
    f = open(filename, 'wb')
    # W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcyTest, accuracyTrain
    pickle.dump([W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcyTest, accuracyTrain, k], f)
    f.close()
    
def loadObjectVariables(filename,dirname):
    os.chdir('C:\\Users\\MATH\\graduate\\'+dirname)
    f = open(filename, 'rb')
    W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcyTest, accuracyTrain = pickle.load(f)
    f.close()
    return W_list, b_list, c_list, xi_list, eta_list, trainTime, predictTime, accuarcyTest, accuracyTrain


def loadDataFromPCKL(filename,dirname):
    os.chdir('C:\\Users\\MATH\\graduate\\'+dirname)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data



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
def separateFeatureLabel(data):
    x=copy.deepcopy(data)
    x=list(x)
    y=[]
    for i in range(len(x)):
        x[i]=list(x[i])
        y.append(x[i].pop())
    return np.array(x),y

def mergeFeatureLabel(x,y):
    x=list(x)
    for i in range(len(y)):
        x[i]=list(x[i])
        x[i].append(y[i])
    return x

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


# In[]  

#load data
filename='trainData2.pckl'
dirname='traindata1'     #dirname
data1=loadDataFromPCKL(filename,dirname)

dirname='traindata2'     #dirname
data2=loadDataFromPCKL(filename,dirname)

dirname='traindata3'     #dirname
data3=loadDataFromPCKL(filename,dirname)

dirname='traindata4'     #dirname
data4=loadDataFromPCKL(filename,dirname)

dirname='traindata5'     #dirname
data5=loadDataFromPCKL(filename,dirname)

dirname='traindata6'     #dirname
data6=loadDataFromPCKL(filename,dirname)



# In[]  20210617 data holdout OAA type1 decisionValueType 1 
decisionValueType1=1
decisionValueType2=2
training_percentage=70
numInteration=10
numClass=5
dirnameList=['traindata1','traindata2','traindata3','traindata4','traindata5','traindata6'] 
Data=[data1,data2,data3,data4,data5,data6]
penaltyC=[1,5,10,20,30]
filename='model'

for i in range(len(Data)):
    print('i:',i)
    
    for j in range(len(penaltyC)):
        print('j:',j)
        
        for k in range(numInteration):
            filename1=filename+str(i+1)+'_C'+str(penaltyC[j])+'_rand'+str(k+1)+'_Type1_variables.pckl'
            filename2=filename+str(i+1)+'_C'+str(penaltyC[j])+'_rand'+str(k+1)+'_Type2_variables.pckl'
            os.chdir('C:\\Users\\MATH\\graduate\\'+dirnameList[i]+'\\type1')
            if not os.path.isfile(filename1):
                print('k:',k)
                
                startTrainTime = time.time()
                shuffleDataList, indexClass=dataCalssIndex(Data[i],k,numClass)
                trainData, testData=separateHoldoutData(indexClass,shuffleDataList,training_percentage,numClass)
                W_list,b_list,c_list,xi_list,eta_list=SQSSVM_multiclass_for_graduate_analysis(numClass,penaltyC[j],trainData)
                endTrainTime = time.time()
                timeTrain=endTrainTime - startTrainTime
                print("訓練時間：%f 秒" % timeTrain)
                
                #decisionType1
                startTestTime = time.time()
                confuseArray_Test_Type1,accuracy_Test_Type1=createConfuseArray(testData,W_list,b_list,c_list,decisionValueType1)
                endTestTime = time.time()
                timePredict=endTestTime-startTestTime
                confuseArray_Train_Type1,accuracy_Train_Type1=createConfuseArray(trainData,W_list,b_list,c_list,decisionValueType1)
                saveObjectVariables(W_list,b_list,c_list,xi_list,eta_list,timeTrain,timePredict,accuracy_Test_Type1,accuracy_Train_Type1,filename1,dirnameList[i],'type1')
                print('confuseArray_Train_Type1:',confuseArray_Train_Type1)
                print('accuracy_Train_Type1:',accuracy_Train_Type1)
                print('confuseArray_Test_Type1:',confuseArray_Test_Type1)
                print('accuracy_Test_Type1:',accuracy_Test_Type1)
                
                
            
# In[]  20210617 data  10-fold OAA type2 decisionvalueType 1
decisionValueType2=2
p=0.1
numRandom=1
numK=10
numClass=5
penaltyC=[1,5,10,20,30,40]
dirnameList=['traindata1','traindata2','traindata3','traindata4','traindata5','traindata6'] #here
Data=[data1,data2,data3,data4,data5,data6]
filename='model'

for i in range(len(Data)):
    count,dataPosition=countLabelSavePositions(numClass,Data[i])
    print('i:',i)
    for j in range(len(penaltyC)):
        print('j:',j)
        for r in range(numRandom):
            print('r:',r)
            shuffleDataPosition=shuffleData(dataPosition,r)
            kFoldData=seperateK_FoldData(shuffleDataPosition,Data[i],numK,count)
            for k in range(numK):
                filename1=filename+str(i+1)+'_C'+str(penaltyC[j])+'_rand'+str(r+1)+'_'+str(k+1)+'_fold_of_10_fold_Type1_variables.pckl'
                filename2=filename+str(i+1)+'_C'+str(penaltyC[j])+'_rand'+str(r+1)+'_'+str(k+1)+'_fold_of_10_fold_Type2_variables.pckl'
                os.chdir('C:\\Users\\MATH\\graduate\\'+dirnameList[i]+'\\type2')
                if not os.path.isfile(filename1):
                    print('k:',k)
                    startTrainTime = time.time()
                    trainData,testData=getTrainTestDataFromKFoldData(kFoldData,k)
                    W_list,b_list,c_list,xi_list,eta_list=SQSSVM_multiclass_for_graduate_analysis(numClass,penaltyC[j],trainData)
                    endTrainTime = time.time()
                    timeTrain=endTrainTime - startTrainTime
                    print("訓練時間：%f 秒" % timeTrain)
                    
                    #decisionType1
                    startTestTime = time.time()
                    confuseArray_Test_Type1,accuracy_Test_Type1=createConfuseArray(testData,W_list,b_list,c_list,decisionValueType1)
                    endTestTime = time.time()
                    timePredict=endTestTime-startTestTime
                    confuseArray_Train_Type1,accuracy_Train_Type1=createConfuseArray(trainData,W_list,b_list,c_list,decisionValueType1)
                    saveObjectVariables(W_list,b_list,c_list,xi_list,eta_list,timeTrain,timePredict,accuracy_Test_Type1,accuracy_Train_Type1,filename1,dirnameList[i],'type2')
                    print('confuseArray_Train_Type1:',confuseArray_Train_Type1)
                    print('accuracy_Train_Type1:',accuracy_Train_Type1)
                    print('confuseArray_Test_Type1:',confuseArray_Test_Type1)
                    print('accuracy_Test_Type1:',accuracy_Test_Type1)
                    
                   
                    
                    

# In[]  20210615 data preprocessing filter holdout OAA type5 decisionValueType 1 

decisionValueType1=1
decisionValueType2=2
p=0.9
training_percentage=70
numInteration=10
numClass=5
dirnameList=['traindata1','traindata2','traindata3','traindata4','traindata5','traindata6'] #here
Data=[data1,data2,data3,data4,data5,data6]
penaltyC=[1,5,10,20,30]
filename='model'

for i in range(len(Data)):
    print('i:',i)
    x,y=separateFeatureLabel(Data[i])
    filterData, filter_list=filterFeature(x,p)
    scale_x=minMaxData(filterData)
    mergeData=mergeFeatureLabel(scale_x,y)
    
    for j in range(len(penaltyC)):
        print('j:',j)
        
        for k in range(numInteration):
            filename1=filename+str(i+1)+'_C'+str(penaltyC[j])+'_rand'+str(k+1)+'_datapreprocessing_Type1_variables.pckl'
            filename2=filename+str(i+1)+'_C'+str(penaltyC[j])+'_rand'+str(k+1)+'_datapreprocessing_Type2_variables.pckl'
            os.chdir('C:\\Users\\MATH\\graduate\\'+dirnameList[i]+'\\type5')
            if not os.path.isfile(filename1):
                print('k:',k)
                
                startTrainTime = time.time()
                shuffleDataList, indexClass=dataCalssIndex(mergeData,k,numClass)
                trainData, testData=separateHoldoutData(indexClass,shuffleDataList,training_percentage,numClass)
                W_list,b_list,c_list,xi_list,eta_list=SQSSVM_multiclass_for_graduate_analysis(numClass,penaltyC[j],trainData)
                endTrainTime = time.time()
                timeTrain=endTrainTime - startTrainTime
                print("訓練時間：%f 秒" % timeTrain)
                
                #decisionType1
                startTestTime = time.time()
                confuseArray_Test_Type1,accuracy_Test_Type1=createConfuseArray(testData,W_list,b_list,c_list,decisionValueType1)
                endTestTime = time.time()
                timePredict=endTestTime-startTestTime
                confuseArray_Train_Type1,accuracy_Train_Type1=createConfuseArray(trainData,W_list,b_list,c_list,decisionValueType1)
                saveObjectVariables(W_list,b_list,c_list,xi_list,eta_list,timeTrain,timePredict,accuracy_Test_Type1,accuracy_Train_Type1,filename1,dirnameList[i],'type5')
                print('confuseArray_Train_Type1:',confuseArray_Train_Type1)
                print('accuracy_Train_Type1:',accuracy_Train_Type1)
                print('confuseArray_Test_Type1:',confuseArray_Test_Type1)
                print('accuracy_Test_Type1:',accuracy_Test_Type1)
                
                
# In[]  20210615 data preprocessing filter 10-fold OAA type6 decisionvalueType 1 
decisionValueType1=1
decisionValueType2=2
p=0.1
numRandom=1
numK=10
numClass=5
penaltyC=[1,5,10,20,30,40]
dirnameList=['traindata1','traindata2','traindata3','traindata4','traindata5','traindata6']
Data=[data1,data2,data3,data4,data5,data6]
filename='model'

for i in range(len(Data)):
    x,y=separateFeatureLabel(Data[i])
    filterData, filter_list=filterFeature(x,p)
    scale_x=minMaxData(filterData)
    mergeData=mergeFeatureLabel(scale_x,y)
    count,dataPosition=countLabelSavePositions(numClass,mergeData)
    print('i:',i)
    for j in range(len(penaltyC)):
        print('j:',j)
        for r in range(numRandom):
            print('r:',r)
            shuffleDataPosition=shuffleData(dataPosition,r)
            kFoldData=seperateK_FoldData(shuffleDataPosition,mergeData,numK,count)
            for k in range(numK):
                filename1=filename+str(i+1)+'_C'+str(penaltyC[j])+'_rand'+str(r+1)+'_'+str(k+1)+'_fold_of_10_fold_datapreprocessing_Type1_variables.pckl'
                filename2=filename+str(i+1)+'_C'+str(penaltyC[j])+'_rand'+str(r+1)+'_'+str(k+1)+'_fold_of_10_fold_datapreprocessing_Type2_variables.pckl'
                os.chdir('C:\\Users\\MATH\\graduate\\'+dirnameList[i]+'\\type6')
                if not os.path.isfile(filename1):
                    print('k:',k)
                    startTrainTime = time.time()
                    trainData,testData=getTrainTestDataFromKFoldData(kFoldData,k)
                    W_list,b_list,c_list,xi_list,eta_list=SQSSVM_multiclass_for_graduate_analysis(numClass,penaltyC[j],trainData)
                    endTrainTime = time.time()
                    timeTrain=endTrainTime - startTrainTime
                    print("訓練時間：%f 秒" % timeTrain)
                    
                    #decisionType1
                    startTestTime = time.time()
                    confuseArray_Test_Type1,accuracy_Test_Type1=createConfuseArray(testData,W_list,b_list,c_list,decisionValueType1)
                    endTestTime = time.time()
                    timePredict=endTestTime-startTestTime
                    confuseArray_Train_Type1,accuracy_Train_Type1=createConfuseArray(trainData,W_list,b_list,c_list,decisionValueType1)
                    saveObjectVariables(W_list,b_list,c_list,xi_list,eta_list,timeTrain,timePredict,accuracy_Test_Type1,accuracy_Train_Type1,filename1,dirnameList[i],'type6')
                    print('confuseArray_Train_Type1:',confuseArray_Train_Type1)
                    print('accuracy_Train_Type1:',accuracy_Train_Type1)
                    print('confuseArray_Test_Type1:',confuseArray_Test_Type1)
                    print('accuracy_Test_Type1:',accuracy_Test_Type1)
                    
                    

# In[]  20210611 data preprocessing filter holdout OAA type7 decisionValueType 1 
training_percentage=70
numInteration=1
numClass=5
dirnameList=['traindata1','traindata2','traindata3','traindata4','traindata5','traindata6'] 
Data=[data1,data2,data3,data4,data5,data6]
penaltyC=[1,5,10,20,30]
filename='model'
p=0.1 #threshold p(1-p)
for i in range(len(Data)):
    print('i:',i)
    x,y=separateFeatureLabel(Data[i])
    filterData, filter_list=filterFeature(x,p)
    scale_x=minMaxData(filterData)
    mergeData=mergeFeatureLabel(scale_x,y)
    for j in range(len(penaltyC)):
        print('j:',j)
        for k in range(numInteration):
            filename1=filename+str(i+1)+'_C'+str(penaltyC[j])+'_rand'+str(k+1)+'_datapreprocessing_Type1_variables.pckl'
            os.chdir('C:\\Users\\MATH\\graduate\\'+dirnameList[i]+'\\type7')
            if not os.path.isfile(filename1):
                print('k:',k)
                
                startTrainTime = time.time()
                shuffleDataList, indexClass=dataCalssIndex(mergeData,k,numClass)
                #print('indexClass:',indexClass)
                trainData, testData=separateHoldoutData(indexClass,shuffleDataList,training_percentage,numClass)
                W_list,b_list,c_list,xi_list,eta_list,checkLabelList=SQSSVM__multiclass_OAO_for_graduate_analysis(numClass,penaltyC[j],trainData)
                endTrainTime = time.time()
                
                timeTrain=endTrainTime-startTrainTime
                print("訓練時間：%f 秒" % timeTrain)
                confuseArrayTest,accurcyTest=createConfuseArrayOAO(testData,W_list,b_list,c_list,numClass,checkLabelList)
                confuseArrayTrain,accurcyTrain=createConfuseArrayOAO(trainData,W_list,b_list,c_list,numClass,checkLabelList)
                
for i in range(len(Data)):
    print('i:',i)
    x,y=separateFeatureLabel(Data[i])
    filterData, filter_list=filterFeature(x,p)
    scale_x=minMaxData(filterData)
    mergeData=mergeFeatureLabel(scale_x,y)
    
    for j in range(len(penaltyC)):
        print('j:',j)
        
        for k in range(numInteration):
            filename1=filename+str(i+1)+'_C'+str(penaltyC[j])+'_rand'+str(k+1)+'_datapreprocessing_variables.pckl'
            os.chdir('C:\\Users\\MATH\\graduate\\'+dirnameList[i]+'\\type5')
            if not os.path.isfile(filename1):
                print('k:',k)
                
                start = time.time()
                shuffleDataList, indexClass=dataCalssIndex(mergeData,k,numClass)
                trainData, testData=separateHoldoutData(indexClass,shuffleDataList,training_percentage,numClass)
                W_list,b_list,c_list,xi_list,eta_list=SQSSVM_multiclass_for_graduate_analysis(numClass,penaltyC[j],trainData)
                end = time.time()
                cpuTime=end - start
                print("執行時間：%f 秒" % cpuTime)
                confuseArray_Test,accuracy_Test=createConfuseArray(testData,W_list,b_list,c_list,decisionValueType)
                confuseArray_Train,accuracy_Train=createConfuseArray(trainData,W_list,b_list,c_list,decisionValueType)
                print('confuseArray_Train:',confuseArray_Train)
                print('accuracy_Train:',accuracy_Train)
                print('confuseArray_Test:',confuseArray_Test)
                print('accuracy_Test:',accuracy_Test)
                
                filename1=filename+str(i+1)+'_C'+str(penaltyC[j])+'_rand'+str(k+1)+'_datapreprocessing_variables.pckl'
                saveObjectVariables(W_list,b_list,c_list,xi_list,eta_list,cpuTime,accuracy_Test,filename1,dirnameList[i],'type5')
                

# In[]  20210616 type9 holdout datapreprocesiing filiter kbest decisionType 1 

decisionValueType1=1
decisionValueType2=2
p=0.9
K=27  #default

training_percentage=70
numInteration=10
numClass=5
dirnameList=['traindata1','traindata2','traindata3','traindata4','traindata5','traindata6'] #here
Data=[data1,data2,data3,data4,data5,data6]
penaltyC=[1,5,10,20,30,40]
filename='model'
K_best_list=[]
K_best_acuuracy_list=[]
for i in range(len(Data)):
    print('i:',i)
    K_best_list.append([])
    K_best_acuuracy_list.append([])
    x,y=separateFeatureLabel(Data[i])
    filterData, filter_list=filterFeature(x,p)
    scale_x=minMaxData(filterData)
    new_x, p_value_list=selectKBestFunction(K,scale_x,y)
    K=num_K(p_value_list)
    new_x, p_value_list=selectKBestFunction(K,scale_x,y)
    mergeData=mergeFeatureLabel(new_x,y)

    for j in range(len(penaltyC)):
        print('j:',j)
        
        for k in range(numInteration):
            filename1=filename+str(i+1)+'_C'+str(penaltyC[j])+'_rand'+str(k+1)+'_datapreprocessing_Kbest_Type1_variables.pckl'
            filename2=filename+str(i+1)+'_C'+str(penaltyC[j])+'_rand'+str(k+1)+'_datapreprocessing_Kbest_Type2_variables.pckl'
            os.chdir('C:\\Users\\MATH\\graduate\\'+dirnameList[i]+'\\type9')
            if not os.path.isfile(filename1):
                print('k:',k)
                
                startTrainTime = time.time()
                shuffleDataList, indexClass=dataCalssIndex(mergeData,k,numClass)
                trainData, testData=separateHoldoutData(indexClass,shuffleDataList,training_percentage,numClass)
                W_list,b_list,c_list,xi_list,eta_list=SQSSVM_multiclass_for_graduate_analysis(numClass,penaltyC[j],trainData)
                endTrainTime = time.time()
                timeTrain=endTrainTime - startTrainTime
                print("訓練時間：%f 秒" % timeTrain)
                
                #decisionType1
                startTestTime = time.time()
                confuseArray_Test_Type1,accuracy_Test_Type1=createConfuseArray(testData,W_list,b_list,c_list,decisionValueType1)
                endTestTime = time.time()
                timePredict=endTestTime-startTestTime
                confuseArray_Train_Type1,accuracy_Train_Type1=createConfuseArray(trainData,W_list,b_list,c_list,decisionValueType1)
                saveObjectVariablesKbest(W_list,b_list,c_list,xi_list,eta_list,timeTrain,timePredict,accuracy_Test_Type1,accuracy_Train_Type1,filename1,dirnameList[i],'type9', K)
                print('confuseArray_Train_Type1:',confuseArray_Train_Type1)
                print('accuracy_Train_Type1:',accuracy_Train_Type1)
                print('confuseArray_Test_Type1:',confuseArray_Test_Type1)
                print('accuracy_Test_Type1:',accuracy_Test_Type1)
                
                
# In[]  20210617 data preprocessing filter 10-fold OAA type10 decisionvalueType 1 
decisionValueType1=1
decisionValueType2=2
p=0.9
K=27  #default

training_percentage=70
numInteration=10
numClass=5
dirnameList=['traindata1','traindata2','traindata3','traindata4','traindata5','traindata6']
Data=[data1,data2,data3,data4,data5,data6]
penaltyC=[1,5,10,20,30,40]
filename='model'
K_best_list=[]
K_best_acuuracy_list=[]
for i in range(len(Data)):
    print('i:',i)
    K_best_list.append([])
    K_best_acuuracy_list.append([])
    x,y=separateFeatureLabel(Data[i])
    filterData, filter_list=filterFeature(x,p)
    scale_x=minMaxData(filterData)
    new_x, p_value_list=selectKBestFunction(K,scale_x,y)
    K=num_K(p_value_list)
    new_x, p_value_list=selectKBestFunction(K,scale_x,y)
    mergeData=mergeFeatureLabel(new_x,y)
    count,dataPosition=countLabelSavePositions(numClass,mergeData)
    print('i:',i)
    for j in range(len(penaltyC)):
        print('j:',j)
        for r in range(numRandom):
            print('r:',r)
            shuffleDataPosition=shuffleData(dataPosition,r)
            kFoldData=seperateK_FoldData(shuffleDataPosition,mergeData,numK,count)
            for k in range(numK):
                filename1=filename+str(i+1)+'_C'+str(penaltyC[j])+'_rand'+str(r+1)+'_'+str(k+1)+'_fold_of_10_fold_datapreprocessing_Kbest_Type1_variables.pckl'
                filename2=filename+str(i+1)+'_C'+str(penaltyC[j])+'_rand'+str(r+1)+'_'+str(k+1)+'_fold_of_10_fold_datapreprocessing_Kbest_Type2_variables.pckl'
                os.chdir('C:\\Users\\MATH\\graduate\\'+dirnameList[i]+'\\type10')
                if not os.path.isfile(filename1):
                    print('k:',k)
                    startTrainTime = time.time()
                    trainData,testData=getTrainTestDataFromKFoldData(kFoldData,k)
                    W_list,b_list,c_list,xi_list,eta_list=SQSSVM_multiclass_for_graduate_analysis(numClass,penaltyC[j],trainData)
                    endTrainTime = time.time()
                    timeTrain=endTrainTime - startTrainTime
                    print("訓練時間：%f 秒" % timeTrain)
                    
                    #decisionType1
                    startTestTime = time.time()
                    confuseArray_Test_Type1,accuracy_Test_Type1=createConfuseArray(testData,W_list,b_list,c_list,decisionValueType1)
                    endTestTime = time.time()
                    timePredict=endTestTime-startTestTime
                    confuseArray_Train_Type1,accuracy_Train_Type1=createConfuseArray(trainData,W_list,b_list,c_list,decisionValueType1)
                    saveObjectVariablesKbest(W_list,b_list,c_list,xi_list,eta_list,timeTrain,timePredict,accuracy_Test_Type1,accuracy_Train_Type1,filename1,dirnameList[i],'type10', K)
                    print('confuseArray_Train_Type1:',confuseArray_Train_Type1)
                    print('accuracy_Train_Type1:',accuracy_Train_Type1)
                    print('confuseArray_Test_Type1:',confuseArray_Test_Type1)
                    print('accuracy_Test_Type1:',accuracy_Test_Type1)
                    
                   

# In[]  2021606 data preprocessing filter  train OAO

def filterFeature(data,p):
    sel = VarianceThreshold(threshold=(p * (1 - p)))
    new_data=sel.fit_transform(data)
    filter_list=sel.get_support()
    return new_data,filter_list

def separateFeatureLabel(data):
    x=copy.deepcopy(data)
    x=list(x)
    y=[]
    for i in range(len(x)):
        x[i]=list(x[i])
        y.append(x[i].pop())
    return np.array(x),y

def mergeFeatureLabel(x,y):
    x=list(x)
    for i in range(len(y)):
        x[i]=list(x[i])
        x[i].append(y[i])
    return x

def minMaxData(x):
    scaler = MinMaxScaler()
    scaler.fit(x)
    scaled_x=scaler.transform(x)
    return scaled_x

training_percentage=70
numInteration=1
numClass=5
dirnameList=['traindata1','traindata2','traindata3','traindata4','traindata5','traindata6'] 
penaltyC=[1,5,10,20,30]
filename='model'
p=0.1 #threshold p(1-p)
for i in range(len(Data)):
    print('i:',i)
    filterData, filter_list=filterFeature(Data[i],p)
    x,y=separateFeatureLabel(filterData)
    scale_x=minMaxData(x)
    mergeData=mergeFeatureLabel(scale_x,y)
    for j in range(len(penaltyC)):
        print('j:',j)
        for k in range(numInteration):
            os.chdir('C:\\Users\\MATH\\graduate\\'+dirnameList[i])
            #if not os.path.isfile(filename1):
            print('k:',k)
            
            start = time.time()
            
            shuffleDataList, indexClass=dataCalssIndex(mergeData,k,numClass)
            #print('indexClass:',indexClass)
            trainData, testData=separateHoldoutData(indexClass,shuffleDataList,training_percentage,numClass)
            W_list,b_list,c_list,xi_list,eta_list,checkLabelList=SQSSVM__multiclass_OAO_for_graduate_analysis(numClass,penaltyC[j],trainData)
            end = time.time()
            print("執行時間：%f 秒" % (end - start))
            cpuTime=end - start
            confuseArrayTest,accurcyTest=createConfuseArrayOAO(testData,W_list,b_list,c_list,numClass,checkLabelList)
            confuseArrayTrain,accurcyTrain=createConfuseArrayOAO(trainData,W_list,b_list,c_list,numClass,checkLabelList)
            
        


# In[] Holdout

training_percentage=70
numInteration=10
numClass=5
dirnameList=['traindata1','traindata2','traindata3','traindata4','traindata5','traindata6'] #here
Data=[data1,data2,data3,data4,data5,data6]
penaltyC=[1,5,10,20,30,40]
filename='model'

for i in range(5,len(Data)):
    print('i:',i)
    
    for j in range(len(penaltyC)):
        print('j:',j)
        
        for k in range(numInteration):
            print('k:',k)
            start = time.time()
            
            shuffleDataList, indexClass=dataCalssIndex(Data[i],k,numClass)
            #print('indexClass:',indexClass)
            trainData, testData=separateHoldoutData(indexClass,shuffleDataList,training_percentage,numClass)
            
            W_list,b_list,c_list,xi_list,eta_list=SQSSVM_multiclass_for_graduate_analysis(numClass,penaltyC[j],trainData)
            end = time.time()
            print("執行時間：%f 秒" % (end - start))
            cpuTime=end - start
            
            confuseArray_Test,accuracy_Test=createConfuseArray(testData,W_list,b_list,c_list)
            confuseArray_Train,accuracy_Train=createConfuseArray(trainData,W_list,b_list,c_list)
            print('confuseArray_Train:',confuseArray_Train)
            print('accuracy_Train:',accuracy_Train)
            print('confuseArray_Test:',confuseArray_Test)
            print('accuracy_Test:',accuracy_Test)
            
            
            filename1=filename+str(i+1)+'_C'+str(penaltyC[j])+'_rand'+str(k+1)+'_variables.pckl'
            saveObjectVariables(W_list,b_list,c_list,xi_list,eta_list,cpuTime,accuracy_Test,filename1,dirnameList[i])
        





# In[] K-fold
    


numRandom=100
numK=10
numClass=5
penaltyC=[30,40]
dirnameList=['traindata1','traindata2','traindata3','traindata4','traindata5','traindata6'] #here
Data=[data1,data2,data3,data4,data5,data6]
filename='model'
Accuracy=[]
Time=[]

for i in range(5,len(Data)):
    count,dataPosition=countLabelSavePositions(numClass,Data[i])
    print('i:',i)
    for j in range(len(penaltyC)):
        print('j:',j)
        for r in range(numRandom):
            print('r:',r)
            shuffleDataPosition=shuffleData(dataPosition,r)
            kFoldData=seperateK_FoldData(shuffleDataPosition,Data[i],numK,count)
            for k in range(numK):
                print('k:',k)
                start = time.time()
                trainingData,testingData=getTrainTestDataFromKFoldData(kFoldData,k)
                W_list,b_list,c_list,xi_list,eta_list=SQSSVM_multiclass_for_graduate_analysis(numClass,penaltyC[j],trainingData)
                end = time.time()
                print("執行時間：%f 秒" % (end - start))
                cpuTime=end - start
                
                confuseArray,accuracy=createConfuseArray(testingData,W_list,b_list,c_list)
                print('accuracy:',accuracy)
                print('confuseArray:',confuseArray)
                
                filename1=filename+str(i+1)+'_C'+str(penaltyC[j])+'_rand'+str(r+1)+'_'+str(k+1)+'_fold_of_10_fold_variables.pckl'
                saveObjectVariables(W_list,b_list,c_list,xi_list,eta_list,cpuTime,accuracy,filename1,dirnameList[i])


# In[] Holdout KBest chi2
def separateFeatureLabel(data):
    x=copy.deepcopy(data)
    y=[]
    for i in range(len(x)):
        y.append(x[i].pop())
    return x,y
def mergeFeatureLabel(x,y):
    x=list(x)
    for i in range(len(y)):
        x[i]=list(x[i])
        x[i].append(y[i])
    return x

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


training_percentage=70
numInteration=5
numClass=5
penaltyC=[20,30]
k_list=[5,10,15,20,30,40,50,60,70,80]
#k_list=[10,20,30,40,50,60,70,80,90]
dirnameList=['traindata1','traindata2','traindata3','traindata4','traindata5','traindata6'] #here
Data=[data1,data2,data3,data4,data5,data6]
filename='model'

for i in range(2,len(Data)):
    print('i:',i)
    
    for j in range(len(penaltyC)):
        print('j:',j)
        
        for item in range(len(k_list)):
            x,y=separateFeatureLabel(Data[i])
            
            x_new=SelectKBest(chi2,k=k_list[item]).fit_transform(x,y) #chi2 & f_classif
            print('dimension of new_x:',len(x_new[0]))
            #print(x_new.shape)
            #x_new=list(x_new)
            #y=list(y)
            mergeData=mergeFeatureLabel(x_new,y)
            
            for k in range(numInteration):
                filename1=filename+str(i+1)+'_C'+str(penaltyC[j])+'_'+str(k_list[item])+'_Best_rand'+str(k+1)+'_variables.pckl'
                #filename1=filename+str(i+1)+'_C'+str(penaltyC[j])+'_'+str(k_list[item])+'_Best_f_classif__rand'+str(k+1)+'_variables.pckl'
                os.chdir('C:\\Users\\MATH\\graduate\\'+dirnameList[i]+'\\kbest')
                if not os.path.isfile(filename1):
                    print('k:',k)
                    start = time.time()
                    
                    
                    
                    shuffleDataList, indexClass=dataCalssIndex(mergeData,k,numClass)
                    #print('indexClass:',indexClass)
                    trainData, testData=separateHoldoutData(indexClass,shuffleDataList,training_percentage,numClass)
                    
                    W_list,b_list,c_list,xi_list,eta_list=SQSSVM_multiclass_for_graduate_analysis(numClass,penaltyC[j],trainData)
                    end = time.time()
                    print("執行時間：%f 秒" % (end - start))
                    cpuTime=end - start
                    
                    confuseArray,accuracy=createConfuseArray(testData,W_list,b_list,c_list)
                    print('confuseArray:',confuseArray)
                    print('accuracy:',accuracy)
                    
                    #filename1=filename+str(i+1)+'_C'+str(penaltyC[j])+'_'+str(k_list[item])+'_Best_f_classif__rand'+str(k+1)+'_variables.pckl'
                    #filename1=filename+str(i+1)+'_C'+str(penaltyC[j])+'_'+str(k_list[item])+'_Best_rand'+str(k+1)+'_variables.pckl'
                    saveObjectVariables(W_list,b_list,c_list,xi_list,eta_list,cpuTime,accuracy,filename1,dirnameList[i])
            


