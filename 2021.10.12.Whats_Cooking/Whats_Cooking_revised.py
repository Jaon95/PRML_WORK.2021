# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

#%% [markdown]
# # 1.What's Cooking? - Exploratory Data Analysis
# This notebook provides a step-by-step analysis and solution to the given problem. It can also serve as a great starting point for learning how to explore, manipulate, transform and learn from textual data. It is divided into three main sections:
# 
# + Exploratory Analysis - as a first step, we explore the main characteristics of the data with the help of Plotly vizualizations;
# 
# + Text Processing - here we apply some basic text processing techniques in order to clean and prepare the data for model development;
# 
# + Feature Engineering & Data Modeling - in this section we extract features from data and build a predictive model of the cuisine. 

# %%
# Data processing
import pandas as pd
import numpy as np
import json
from collections import Counter
import re
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt


# %%
train_df = pd.read_json('E:/Whats_Cooking/train.json') # store as dataframe objects
train_df.info()


# %%
print("The training data consists of {} recipes".format(len(train_df)))
train_df.head()

#%% [markdown]
# We have imported the data as a DataFrame object and the above codes show us the initial look of training samples. We observe that each recipe is a separate row and has:
# 
# + a unique identifier - the 'id' column;
# + the type of cuisine in which this recipe falls - this is our target variable;
# + a list object with ingredients (the recipe) - this will be the main source of explanatory variables in our classification problem.
# 
# Problem statement: Predict the type of cuisine based on given data (ingredients). This is a classification task which requires text processing and analysis.

# %% [markdown]
# Now let's explore a little bit more about the target variable
print("Number of cuisine categories: {}".format(len(train_df.cuisine.unique())))
train_df.cuisine.unique()

#%% [markdown]
# There are 20 different categories (cuisines) which we are going to predict. 
# 
# This means that the problem at hand is a multi-class classification.


# %%
sns.countplot(y=train_df.cuisine,order=train_df.cuisine.value_counts().reset_index()["index"])
plt.title("Cuisine Distribution")
plt.show()


# %%
train_df.cuisine.value_counts()


# %%
print('Maximum Number of Ingredients in a Dish: ',train_df['ingredients'].str.len().max())
print('Minimum Number of Ingredients in a Dish: ',train_df['ingredients'].str.len().min())

#%% [markdown]
# Which are the most common ingredients in the whole training sample? How many unique ingredients can we find in the dataset? 

#%% [markdown]
# # 2. Text Processing
# We will proceed the analysis by performing some simple data processing. The aim is to prepare the data for model development.

# %%
# Prepare the data 
features = [] # list of list containg the recipes
for item in train_df['ingredients']:
    features.append(item)


# %%
ingrCounter = Counter()
features_processed= [] # here we will store the preprocessed training features
for item in features:
    newitem = []
    for ingr in item:
        # 需要注意
        ingr = ingr.lower() # Case Normalization - convert all to lower case 
        # ingr.lower() # Case Normalization - convert all to lower case 
        ingr = re.sub("[^a-zA-Z]", " ", ingr) # Remove punctuation, digits or special characters 
        # ingr = re.sub(r"(salt|water|onions|garlic|olive oil)", "", ingr)
        ingr = re.sub((r'\b(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\b'), ' ', ingr) # Remove different units
        # 区分原料的组合
        ingr = re.sub(r" ", "_", ingr)
        # if delete the salt etc, this epxression must be included.
        if len(ingr) != 0:
            ingrCounter[ingr] += 1
            newitem.append(ingr)
    features_processed.append(newitem)

# print(features_processed)

#%%
train_df["features_processed"] = features_processed

# %%
ingr_df = pd.DataFrame(ingrCounter.most_common(15),columns=['ingredient','count'])
ingr_df


# %%
#f, ax=plt.subplots(figsize=(12,20))
sns.barplot(y=ingr_df['ingredient'].values, x=ingr_df['count'].values,orient='h')
#plt.ylabel('Ingredient', fontsize=12)
#plt.xlabel('Count', fontsize=12)
#plt.xticks(rotation='horizontal')
#plt.yticks(fontsize=12)
plt.title("Ingredient Count")
plt.show()

#%% [markdown]
# It seems that salt is the most commonly used ingredient which is not surprising at all! We also find water, onions, garlic and olive oil - not so surprising also. :) 
# + Salt, water, onions, garlic are such common ingredients that we expect them to have poor predictive power in recognizing the type of cuisine.

#%% [markdown]
# # 3. Feature Engineering & Data Modeling

# %%
train_df['seperated_ingredients'] = train_df['features_processed'].apply(', '.join)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(binary=True).fit(train_df['seperated_ingredients'].values)
X_train_vectorized = vect.transform(train_df['seperated_ingredients'].values)
X_train_vectorized = X_train_vectorized.astype('float')

# X_train_vectorized

# %%
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_transformed = encoder.fit_transform(train_df.cuisine)


# %%
print(X_train_vectorized) # x.shape = (39774, 6679)
y_transformed

#%% [markdown]
# ## Logistic Regression

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X_train_vectorized, y_transformed, train_size=0.9, random_state = 0)

# lr1 = LogisticRegression(C=9, dual=False)
lr1 = LogisticRegression(C=9, dual=False, penalty = "l2", solver="newton-cg", max_iter = 1000, multi_class = "multinomial")
lr1.fit(X_train , y_train)
lr1.score(X_test, y_test)


# 不去掉 salt 等词，且区分原料的组合
# 0.7976370035193565

# 不使用 idf
# 0.7916038210155857

# C = 9
# 0.799396681749623

# C = 8
# 0.7991452991452992

# %%
