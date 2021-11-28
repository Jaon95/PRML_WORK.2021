# %%
from matplotlib.colors import same_color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% 读取文件，并且查看文件的信息
df = pd.read_json('train.json')
print(df.info())
# %% 查看数据的前面15行，看看情况
print(df.head())

# %% 查看才要品种
print(df.cuisine.unique(),len(df.cuisine.unique()))

# %%
print(df.cuisine.value_counts().sort_values(ascending=False))
# %%
sns.countplot(y=df.cuisine,order=df.cuisine.value_counts().sort_values(ascending=False).index)
plt.title("Cuisine Distribution")
plt.show()
# %% 数据不平衡的情况似乎有点儿严重.....
nd_recipes = []
reci_type_num =  df.cuisine.value_counts().sort_values(ascending=False)
print(reci_type_num.mean(),reci_type_num.std()**(0.5))
# %% 将多余2000行的数据随机取样
reci_type_num_need = reci_type_num[reci_type_num>2000]
df_data = pd.DataFrame(columns=df.columns)
for i in range(len(reci_type_num_need)):
    sample = df[df['cuisine']==reci_type_num_need.index[i]].sample(n=2000,replace=False)
    # print(reci_type_num_need.index[i])
    df_data = df_data.append(sample)
    print(len(df_data))
# %%将少于1000行的数据，补齐到1000行
reci_type_num_need = reci_type_num[reci_type_num<=2000]
for i in range(len(reci_type_num_need)):
    need = 2000-reci_type_num_need[i]
    # print(reci_type_num_need.index[i])
    if need >0:
        sample = df[df['cuisine']==reci_type_num_need.index[i]].sample(n=need,replace=True)
        df_data = df_data.append(sample).append(df[df.cuisine==reci_type_num_need.index[i]])
        print(len(df_data))
# %% 这个时候，一共有20000样本下面开始处理样本的特征
features = []
for item in df_data.ingredients:
    features.append(item)
# %% 下面开始TF分析
from collections import Counter
import re
word_counter = Counter()
new_features = [] # 用来存储新的特征
for items in features:
    new_items = []
    print(items)
    for ingre in items:
        ingre = ingre.lower()
        ingre = re.sub(r'[^a-zA-Z]',' ',ingre)##去掉特殊字符
        ingre = re.sub(r'\b(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\b',' ',ingre)#去掉单位
        ingre = re.sub(r" ", "_", ingre) #剩下的就是原料组合
        if len(ingre) > 0:
            word_counter[ingre] += 1
            new_items.append(ingre)
    new_features.append(new_items) #将处理后的配方加进去
df_data['features_processed'] = new_features
df_word_count = pd.Series(word_counter).reset_index()
df_word_count.columns = ['ingre','counter']
df_word_count = df_word_count.sort_values(by='counter',ascending=False)
sns.barplot(y=df_word_count.head(15)['ingre'],
            x=df_word_count.head(15)['counter'],
            order=df_word_count.head(15).sort_values(by='counter',ascending=False)['ingre'])
plt.title('ingre distribution')
plt.show()
# %%
df_word_count.head(15).sort_values(by='counter',ascending=False)
# %% 取大于5000的配料，将其从配料表中删除
rest_ingre = df_word_count[df_word_count.counter>3000]
new_features_tf = []
for items in new_features:
    new_items = []
    for ingre in items:
        if ingre in list(rest_ingre['ingre']):
                continue
        else:
            new_items.append(ingre)
    new_features_tf.append(new_items)
df_data['features_processed_tf'] = new_features_tf
# %% 找到某些配方中特有的成分,也就是说，只出现过一次的成分
rest_ingre = df_word_count[df_word_count.counter==1]
unique_df_data = pd.DataFrame()
for row in  df_data.itertuples():
    for ingre in getattr(row,'features_processed'):
        if ingre in list(rest_ingre['ingre']):
            unique_df_data = unique_df_data.append(df_data[df_data.id==getattr(row,'id')])
            unique_df_data[unique_df_data['id']==getattr(row,'id')]
print(unique_df_data)
# %%
df_data['seperated_ingredients'] = df_data['ingredients'].apply(','.join)

# %% 进行tfid处理，词频 & 逆文档频率
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(binary=True).fit(df_data['seperated_ingredients'])
X_train_vectorized = vect.transform(df_data['seperated_ingredients'].values)
X_train_vectorized = X_train_vectorized.astype('float')
# 对结果进行编码 ,标签编码
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_transformed = encoder.fit_transform(df_data.cuisine)
# %% 下面划分训练集和测试集
from sklearn import model_selection
x_train,x_test,y_train,y_test = model_selection.train_test_split(X_train_vectorized,y_transformed,random_state=1,test_size=0.3)
# %%下面开始模型训练
from sklearn.linear_model import LogisticRegression
lr1 = LogisticRegression(C=0.8,dual=False)
lr1.fit(x_train , y_train)
lr1.score(x_test, y_test)
# %%
lr1.score(x_train , y_train)
# %% 尝试用所有数据进行评估,对所有数据进行tfidf编码
df['seperated_ingredients'] = df['ingredients'].apply(','.join)
true_sample = df.sample(n=1)
# %%
x_vect = TfidfVectorizer(binary=True).fit(true_sample['seperated_ingredients'])
x = x_vect.transform(true_sample['seperated_ingredients'].values)
x = x.astype(float)
print(x.shape)
print(x_vect.get_feature_names_out())
temp_dic = {}
for i in x_vect.get_feature_names_out():
    temp_dic[i] = x.toarray()[1]
for i in vect.get_feature_names_out():
    if i in x_vect.get_feature_names_out():
        temp_dic[i] = x.toarray()[
# x_all = TfidfVectorizer(binary=True).fit_transform(df['seperated_ingredients'].values)
# x_all = x_all.astype('float')
# encoder_all = LabelEncoder()
# y_all = encoder_all.fit_transform(df['cuisine'])
# %%
temp_dic = {}
for i in vect.get_feature_names_out():
    temp_dic[i] = [111]

temp = pd.DataFrame(temp_dic)
x= temp[0:1].values

# %%
