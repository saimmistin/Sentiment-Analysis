#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


data = pd.read_csv('/Users/derya_ak/Desktop/AmazonReview.csv')
data.head()


# In[7]:


data['Review'].head()


# In[4]:


# convert upper case to lower case
data['Review'] = data['Review'].str.lower()


# In[6]:


data['Review'].head()


# In[8]:


#remove all punctuations
import re
def regex(a):
    a = re.sub(r'[^\w\s]', '', str(a))
    return a 


# In[9]:


data['Review'] = data['Review'].apply(lambda x: regex(x))


# In[10]:


data['Review'] = data['Review'].str.replace('\d', '')


# In[11]:


data['Review'].head()


# In[12]:


# find stopwords and remove from text for tokenization
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
sw


# In[13]:


data['Review'] = data['Review'].apply(lambda x: " ".join([x for x in str(x).split() if x not in (sw)]))
data['Review'].head()


# In[19]:


#number of words
number_of_words = pd.Series(' '.join(data['Review']).split()).value_counts()
number_of_words


# In[20]:


#remove words which are only one
remove_word = number_of_words[number_of_words <= 1]
remove_word


# In[21]:


data['Review'] = data['Review'].apply(lambda x: " ".join([x for x in str(x).split() if x not in (remove_word)]))
data['Review'].head()


# In[23]:


#thanks to to textblob we can extract nouns and adjectives from text. 
nltk.download("punkt")

from textblob import TextBlob, Word, Blobber


# In[25]:


data['Review'].apply(lambda x: TextBlob(x).words).head()


# In[28]:


nltk.download('omw-1.4')
nltk.download('wordnet')
get_ipython().system('python -m textblob.download_corpora')


# In[33]:


#lemmatize (normalize words)
data['Review'] = data['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['Review'].head()


# In[34]:


#find out how many times words show up in the text
term_frequancy = data['Review'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
term_frequancy.columns = ["words", "term_frequancy"]


# In[38]:


term_frequancy.sort_values("term_frequancy", ascending = False)


# In[53]:


#visualize most frequent words
from matplotlib import pyplot as plt

term_frequancy_2 = term_frequancy[term_frequancy["term_frequancy"] > 2000].sort_values("term_frequancy", ascending=False)


fig, ax = plt.subplots(figsize=(16, 8))

term_frequancy_2.plot.bar(x="words", y="term_frequancy", color="blue", ax=ax)

plt.show()


# In[56]:


#make it single text
text = " ".join(i for i in data['Review'])
text[0:10000]


# In[58]:


from wordcloud import WordCloud 
wordcloud = WordCloud().generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[71]:


data['Review']


# In[72]:


nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


# In[92]:


sia = SIA()
sia.polarity_scores("It is very nice to see you")


# In[93]:


data["Polarity_Score"] = data['Review'].apply(lambda x: sia.polarity_scores(x)["compound"])


# In[94]:


data.loc[(data["Polarity_Score"] < 0.0) & (data["Sentiment"] > 3.0)].head()


# In[96]:


data["SentimentLabel"] = data["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")


# In[97]:


data["SentimentLabel"].value_counts()


# In[98]:


data.groupby("SentimentLabel")["Sentiment"].mean()

