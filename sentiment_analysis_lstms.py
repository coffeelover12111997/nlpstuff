# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 20:12:26 2017

@author: preetish
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 22:24:03 2017

@author: preetish
"""

#sentiment analysis for kaggle data(umichigan)
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence
from keras.layers import Dense,Dropout
from keras.layers.embeddings import Embedding
from keras.models import Sequential


def indexer(arr):
    indexmatr=[]
    max_len=0
    for i in range(arr.shape[0]):
        temp=[]
        tempmax=0
        for j in range(arr.shape[1]):
            if arr[i,j]==1:
                temp.append(j)
                tempmax+=1
        if tempmax>max_len:
            max_len=tempmax
        indexmatr.append(temp)
    return indexmatr,max_len    
        
                
train=pd.read_csv('train.txt',sep='\\r\\n')
test=pd.read_csv('test.txt',sep='\\r\\n')

train=np.array(train)
test=np.array(test)

trainX=[]
senti=[]

for i in range(train.shape[0]):
    trainX.append(train[i][0])
    senti.append(train[i][0][0])
    
for i in range(len(trainX)):
    trainX[i]=re.sub(r'[^a-zA-z\' ]','',trainX[i])

trainy=np.array(senti).astype(float)

BOW=CountVectorizer()

trainx=BOW.fit_transform(trainX)
trainx=trainx.toarray()
tot_words=trainx.shape[1]
input_seq,max_length=indexer(trainx)

'''testX=BOW.transform(sentence2)
testX=testX.toarray()
test_seq,maxt_length=indexer(testX)'''

trainx=sequence.pad_sequences(input_seq,maxlen=max_length)
#testX=sequence.pad_sequences(test_seq,maxlen=max_length)


from sklearn.model_selection import StratifiedKFold 
skf = StratifiedKFold(n_splits=2)

y=trainy.reshape((1,-1))
y=y[0]
train_index,test_index = skf.split(trainX,y)
    
    X_t = trainX[train_index[0]]
    X_te= trainX[test_index[0]]
    y_t, y_te = y[train_index[0]], y[test_index[0]]
 

model=Sequential()

model.add(Embedding(input_dim=tot_words,output_dim=100,input_length=trainX.shape[1]))
model.add(LSTM(60,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(25))
model.add(Dense(1,activation='sigmoid'))

model.compile('Adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_t,y_t,epochs=5,validation_data=(X_te,y_te),batch_size=64)

'''final=model.predict(testX)

import pandas as pd
submit=pd.DataFrame({
        'Id':range(1,final.shape[0]+1),'Sentiment':final})
final.to_csv('kaggle.csv',index=False)'''



    






