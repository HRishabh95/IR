import pandas as pd
import numpy as np
#import re
import language_check
from textblob import TextBlob

df=pd.read_csv("Re.csv")
df=df.values
print (df.shape)
df=df[:,2]

emojis=[]
for i in df:
    count=0
    if type(i)!=float:
        if ":)" in i:
            count=count+1
        if ":-*" in i:
            count=count+1
        if ":-P" in i:
            count=count+1
        if ":D" in i:
            count=count+1
        if ";)" in i:
            count=count+1
    emojis.append(count)

print ("emoji done")
url=[]
for i in df:
    count=0
    if type(i)!=float:
        if "<a href" in i:
            count=1
        if ".com" in i:
            count=1
        if "http:" in i:
            count=1
    url.append(count)

img=[]
for i in df:
    count=0
    if type(i)!=float:
        if "<img src" in i:
            count=1
        if "<img" in i:
            count=1
    img.append(count)

tool = language_check.LanguageTool('en-US')
#gram=[]
senti=[]
j=0
print(df[191357])
print(df[191356])
print(df[191355])
print(df[191358])

for i in df:
    print(j)
    j=j+1
    if type(i) != float:
        text=TextBlob(str(i))
        senti.append(text.sentiment.polarity)
        #matches=tool.check(str(i))
        #gram.append(len(matches))
    else:
        senti.append(0)

features=np.zeros((len(df),4))
features[:,0]=emojis[:]
features[:,1]=url[:]
features[:,2]=img[:]
features[:,3]=senti[:]
#features[:,3]=gram[:]
np.savetxt("Textfeatures.csv",features,fmt="%.2f",delimiter=",")
print ("Done")
'''
f=open("text.txt","w")
j=0
for i in df:
    print (j)
    j=j+1
    i = re.sub(r'^https?:\/\/.*[\r\n]*', '', i)
    i = re.sub('<.*?>', '', i)
    f.write(i)
'''