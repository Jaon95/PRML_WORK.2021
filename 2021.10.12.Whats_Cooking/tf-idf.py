# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vect  = TfidfVectorizer(binary=True)
str1 = 'you am li'
str2 = 'are am li mei hang'
#%%
vectorized_str = vect.fit_transform([str1])
vectorized_str = vectorized_str.astype(float)

# %%
print(vect.get_feature_names_out())
print(vectorized_str.toarray())


# %%
vectorized_str2  = vect.fit_transform(['am','are','hang','li','mei','you'])

# %%

# %%
