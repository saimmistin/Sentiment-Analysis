#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob, Word
from wordcloud import WordCloud
from matplotlib import pyplot as plt


# In[2]:


# Load and preprocess the data
data = pd.read_csv('/Users/saim/Desktop/AmazonReview.csv')


# In[3]:


data


# In[4]:


#preprocessing and cleaning
data.info()


# In[5]:


#drop null values
data.dropna(inplace=True)


# In[6]:


#there are 5 ratings convert these to positive =1 and negative=0
#1,2,3->negative(i.e 0)
data.loc[data['Sentiment']<=3,'Sentiment'] = 0
 
#4,5->positive(i.e 1)
data.loc[data['Sentiment']>3,'Sentiment'] = 1


# In[30]:


#convert upper case to lower case
data['Review'] = data['Review'].str.lower()


# In[7]:


#remove all punctuations
data['Review'] = data['Review'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))


# In[7]:


#remove numbers
data['Review'] = data['Review'].str.replace('\d', '')


# In[31]:


data['Review']


# In[32]:


sw = stopwords.words('english')


# In[33]:


sw


# In[34]:


#remove stopwords
data['Review'] = data['Review'].apply(lambda x: " ".join([x for x in str(x).split() if x not in sw]))


# In[10]:


data['Review']


# In[36]:


#thanks to to textblob we can extract nouns and adjectives from text. 
nltk.download("punkt")

from textblob import TextBlob, Word, Blobber


# In[37]:


data['Review'].apply(lambda x: TextBlob(x).words).head()


# In[38]:


#lemmatize (normalize words)
data['Review'] = data['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['Review'].head()


# In[39]:


# Perform term frequency analysis and visualization
term_frequancy = data['Review'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

#retrive words which are greater than 2000
term_frequancy.columns = ["words", "term_frequancy"]
term_frequancy_2 = term_frequancy[term_frequancy["term_frequancy"] > 2000].sort_values("term_frequancy", ascending=False)


# In[40]:


term_frequancy_2


# In[41]:


term_frequancy_2 = term_frequancy[term_frequancy["term_frequancy"] > 2000].sort_values("term_frequancy", ascending=False)


fig, ax = plt.subplots(figsize=(16, 8))

term_frequancy_2.plot.bar(x="words", y="term_frequancy", color="blue", ax=ax)

plt.show()


# In[42]:


#make it single text
text = " ".join(i for i in data['Review'])
text[0:10000]


# In[43]:


#create cloudword
from wordcloud import WordCloud 
wordcloud = WordCloud().generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[44]:


data['Review']


# In[45]:


#cloudword of negative reviews
consolidated=' '.join(word for word in data['Review'][data['Sentiment']==0].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()


# In[46]:


#cloudword of positive reviews
consolidated=' '.join(word for word in data['Review'][data['Sentiment']==1].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()


# In[47]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[48]:


cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(data['Review'] ).toarray()


# In[49]:


from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test=train_test_split(X,data['Sentiment'],
                                                test_size=0.25 ,
                                                random_state=42)


# In[50]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
 
model=LogisticRegression()
 
#Model fitting
model.fit(x_train,y_train)
 
#testing the model
pred=model.predict(x_test)
 
#model accuracy
print(accuracy_score(y_test,pred))


# In[52]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test,pred)
 
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,
                                            display_labels = [False, True])
 
cm_display.plot()
plt.show()


# Our confusion matrix worked relatively well

# In[53]:


pip install jupytext


# In[ ]:




