#%%
import numpy as np 
import pandas as pd

# %%
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  

#%%
# with "_" denotation
corpus=["I come to ice_cream China to travel","com",'I come to ice_cream China to travel'] 

# without "_" denotation
# corpus=["I come to China to travel", 
#     "This is a car polupar in China",          
#     "I love tea and Apple",   
#     "The work is to write some papers in science"] 

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

#%%
tfidf2 = TfidfVectorizer()
result = tfidf2.fit_transform(corpus)

#%%
print(result)

#%%
dd = pd.DataFrame(result.toarray())

#%%
dc = tfidf2.vocabulary_

dc_reverse = dict([value, key] for key, value in dc.items())

#%%
dd = dd.rename(columns = dc_reverse)

dd

#%%
