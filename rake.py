# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 20:56:57 2017

@author: preetish
"""

#RAKE(rapid Key word Extraction)
'''implementation of Automatic keyword extraction from individual documents
Stuart Rose, Dave Engel, Nick Cramer
and Wendy Cowley'''

#individual words without stopwords

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import re

#input text and remove all punctuation and special char
Text=text_inp.lower()
Text= re.sub('[^a-zA-Z0-9 ]','',Text)
Text=''.join(Text)
text=Text.split(' ')

#remove excessive white spaces
index=[]
for i in range(len(text)):
    if len(text[i])==0:
        index.append(i)

for i in range(len(index)):
    index[i]-=i

for i in range(len(index)):
    text.pop(index[i])

Text=' '.join(text)  

#remove stop words
remove=stopwords.words('english')
text=[w for w in text if w not in remove]

#try to find the freq of words
bow=CountVectorizer()
text=' '.join(text)
a=[text]
freq=bow.fit_transform(a)

freq1=freq.toarray()[0]
freq2=bow.vocabulary_


#split with stopwords
text1=Text

for i in remove:
    text1=re.sub(' '+i+' ' or ' '+i+'|' or '|'+i+' ' or '|'+i+'|',' | ',text1)
    

text1=text1.split('|')



#compute degree    
deg={}

for i in text.split(' '):
    if i not in list(deg.keys()):
        deg[i]=0
        for j in text1:
            if i in j:
                deg[i]+=1

score={}

key1=list(deg.keys())
key2=list(freq2.keys())

#finding scores of individual words
for i in text.split(' '):
    if i not in list(score.keys()) and i in key1 and i in key2 and freq1[freq2[i]]!=0:
        score[i]=deg[i]/freq1[freq2[i]]


scorefinal={}
key=list(score.keys())

#final scores for words
for i in text1:
    scorefinal[i]=0
    used=[]
    for j in text.split(' '):
        if j in i and j not in used and j in key :
            scorefinal[i]+=score[j]
            used.append(j)



#freq_based_dict
fbd={}
for i in scorefinal.keys():
    fbd[str(scorefinal[i])]=i


topten=sorted(fbd.keys())[::-1]
topten=topten[:10]
#output top ten keywords
for i in topten:
    print(fbd[i])


#adjoining rare keywords(still to be done)

