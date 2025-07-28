import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
df=pd.read_csv("D:\\datascience\\task4\\spam2.csv")[['v1','v2']]
df['v1']=df['v1'].map({'ham':0,'spam':1})
x=df['v1']
y=df['v2']
x_train,x_test,y_train,y_test=train_test_split(y,x,test_size=0.2,random_state=42)
vectorizer=TfidfVectorizer()
x_train_vec=vectorizer.fit_transform(x_train)
x_test_vec=vectorizer.transform(x_test)
model=MultinomialNB()
model.fit(x_train_vec,y_train)
y_pred=model.predict(x_test_vec)
print(classification_report(y_pred,y_test))
mes=str(input("enter message:"))
message=[mes]
vecmes=vectorizer.transform(message)
predict=model.predict(vecmes)
if predict[0]== 0:
    print("its not spam")
else:
    print(("its  spam"))