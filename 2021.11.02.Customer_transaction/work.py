#%%导入必要的包
#导入需要的包
from contextlib import ContextDecorator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import time
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn import metrics
from sklearn import svm

warnings.filterwarnings('ignore')

# %%
# 1、EDA--exploratory data analysis
train = pd.read_csv('train.csv')
train.head()
# %%
#检查是否有缺失值
train.isnull().any().describe()
# %%
#查看数据统计量信息。
train.describe() 
# %%
#检查样本是否平衡
X = train.iloc[:, 2:].values.astype('float64')
Y = train['target'].values
print(X.shape)
ax = sns.countplot(Y, palette='Set1')
for p in ax.patches:
    ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))
#%%
def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(20,30))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.distplot(df1[feature], hist=False,label=label1)
        sns.distplot(df2[feature], hist=False,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();
#%%
#显示每个指标在每种类型下面的分布情况
# t0 = train.loc[train['target'] == 0]
# t1 = train.loc[train['target'] == 1]
# features = train.columns.values[2:102]#将column压平
# plot_feature_distribution(t0, t1, '0', '1', features)
# %%
def show_time(values):
    t = time.strftime('%H:%M:%S',time.gmtime(values))
    return t
# %%
# plt.figure(figsize=(16,6))
# features = train.columns.values[2:202]
# plt.title("Distribution of mean values per row in the train set")
# sns.distplot(train[features].mean(axis=1),color="green", kde=True,bins=120, label='train')
# plt.legend()
# plt.show()
# %%
# 下面是划分训练集和测试集
x_train, x_test, y_train, y_test = model_selection.train_test_split(X,Y,test_size=0.3,random_state=52)
print('data shape of train:',x_train.shape)
print('data shape of test',x_test.shape)
plt.figure()
sns.countplot(y_train)
plt.figure()
sns.countplot(y_test)

# %%
def plot_confusion_mtx(cm,classes,cmap=plt.cm.Blues):
    #创建网格，用于显示数量
    plt.figure()
    plt.imshow(cm,cmap=cmap)

    #分别取对应的数据，填充到对应的位置
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    #设置x，y的坐标;前一个参数是在坐标轴上面产生的点的序列，后一个参数是每个点的label
    plt.xticks(np.arange(len(classes)),classes,rotation = 0)
    plt.yticks(np.arange(len(classes)),classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#%% 定义一个类，方便操作
class ModelFac:
    def __init__(self,model,x_train,y_train):
        self._model = model
        self._x_train = x_train
        self._y_train = y_train
    
    @property
    def model(self):
        return self._model
    
    def fit(self):
        start_time = time.time()
        self._model.fit(self._x_train , self._y_train)
        print('fitting finished!')
        end_time = time.time()
        fit_time = end_time- start_time
        print('time consumed of fitting:',show_time(fit_time))
    
    def score(self,x_test,y_test):
        #下面是评价模型
        start_time = time.time()
        score = self._model.score(x_test, y_test)
        end_time = time.time()
        score_time = end_time - start_time
        print('scoring finished!')
        print('time consumed of scoring:',show_time(score_time))
        print('score is : ',score)
    
    def fit_and_score(self,x_test,y_test):
        self.fit()
        self.score(x_test,y_test)

    def show_confusion_mtx(self,x_test,y_test):
        predicted = self._model.predict(x_test)
        confusion_mtx = metrics.confusion_matrix(predicted, y_test,labels=[1,0])
        print(confusion_mtx)
        plot_confusion_mtx(confusion_mtx,classes=[1,0])
        acc = (confusion_mtx[0,0]+confusion_mtx[1,1])/confusion_mtx.sum()
        tpr = confusion_mtx[0,0]/(confusion_mtx[0,0]+confusion_mtx[0,1]) #也叫查全率(召回率),100个正实例中，找对了多少个，所以要比较大
        fpr = confusion_mtx[1,0]/(confusion_mtx[1,0]+confusion_mtx[1,1]) # 100个负实例当中，有多少个被判正了,所以要比较小
        ppv = confusion_mtx[0,0]/(confusion_mtx[0,0]+confusion_mtx[1,0]) #也叫精确率，也就是100个被判断为正实例当中，有多少个是真正的正实例，所以要比较小
        f1_score = 2*(tpr*ppv)/(tpr+ppv)
        print('acc is {}, tpr is {}, fpr is {}, ppv is {}, f1-score is {}'.format(acc,tpr,fpr,ppv,f1_score))
    
    def show_auc(self,x_test,y_test):
        x_prob = self._model.predict_proba(x_test)[:,1]
        fpr,tpr,thresholds = metrics.roc_curve(y_test,x_prob,pos_label=1)
        plt.title('the roc_curve ')
        plt.plot(fpr,tpr) 
        print('the auc is : ',metrics.roc_auc_score(y_test,x_prob))
        plt.show()
# %%
logit_model = ModelFac(LogisticRegression(C=0.3,dual=False,max_iter=x_train.shape[0]),x_train,y_train)
logit_model.fit_and_score(x_test,y_test)
logit_model.show_confusion_mtx(x_test,y_test)
logit_model.show_auc(x_test, y_test)
# %%
# 下面对数据进行降维
#标准化
std_scal = StandardScaler().fit(x_train)
x_train_std = std_scal.transform(x_train)
#主成分分析，通过求协方差矩阵的特征向量，利用特征向量构成的矩阵对原来的x进行转化-构成顺序按特征值大小逆序排序，转化后的特征数量是和x的特征数量相等的
sklearn_pca = sklearnPCA().fit(x_train_std)

#查看主成分分析后的个成分的方差信息，并且归一化
var_per = sklearn_pca.explained_variance_ratio_
cum_var_per = sklearn_pca.explained_variance_ratio_.cumsum()

l = len(cum_var_per[cum_var_per <= 0.7])
sklearn_pca = sklearnPCA(n_components=l).fit(x_train_std)

# 使用主成分分析的结果，对原特征进行较为
x_train_pca = sklearn_pca.transform(x_train_std)
x_test_pca = sklearn_pca.transform(std_scal.transform(x_test))
print('降维后的数据训练数据 shape：',x_train_pca.shape)
print('降维后的测试数据shape: ',x_test_pca.shape)
# %% 使用降维后的数据，进行训练
logit_model = ModelFac(LogisticRegression(C=0.3,dual=False,max_iter=x_train.shape[0]),x_train,y_train)
logit_model.fit_and_score(x_test,y_test)
logit_model.show_confusion_mtx(x_test,y_test)
logit_model.show_auc(x_test, y_test)
# %%
logit_model = ModelFac(LogisticRegression(C=0.8,dual=False,max_iter=x_train.shape[0]),x_train,y_train)
logit_model.fit_and_score(x_test,y_test)
logit_model.show_confusion_mtx(x_test,y_test)
logit_model.show_auc(x_test, y_test)
# %%
##执行网格搜索
grid_logit = model_selection.GridSearchCV(
    LogisticRegression(random_state=52),
    cv = 5,
    param_grid={
        'C': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], 
        'solver': ('lbfgs', 'sag')
    },
    scoring='roc_auc'
)
grid_logit.fit(x_train_pca,y_train)
# %%
model = ModelFac(grid_logit,x_train_pca,y_train)
model.show_confusion_mtx(x_test_pca,y_test)
model.show_auc(x_test_pca,y_test)
print('test score is : ',model.model.score(x_test_pca,y_test))
# %%
print(grid_logit.best_params_,grid_logit.best_score_)
# %%
# #下面使用svm对数据进行分析
# svm_model = ModelFac(svm.SVC(C=1,kernel='rbf'),x_train_pca,y_train)
# # %%
# svm_model.fit_and_score(x_test_pca,y_test)
# %%
# 下面使用mpl对数据进行训练
from sklearn import neural_network
nn_model = ModelFac(neural_network.MLPClassifier(solver='sgd',activation="relu",hidden_layer_sizes=(100, )),x_train,y_train)
nn_model.fit_and_score(x_train,y_train)
nn_model.show_confusion_mtx(x_test,y_test)
nn_model.show_auc(x_test,y_test)
# %%
nn_model_pca = ModelFac(neural_network.MLPClassifier(solver='sgd',activation="relu",hidden_layer_sizes=(100, )),x_train_pca,y_train)
nn_model_pca.fit_and_score(x_train_pca,y_train)
nn_model_pca.show_confusion_mtx(x_test_pca,y_test)
nn_model_pca.show_auc(x_test_pca,y_test)
#%%
std_scal = StandardScaler().fit(x_train)
x_train_std = std_scal.transform(x_train)
#主成分分析，通过求协方差矩阵的特征向量，利用特征向量构成的矩阵对原来的x进行转化-构成顺序按特征值大小逆序排序，转化后的特征数量是和x的特征数量相等的
sklearn_pca = sklearnPCA().fit(x_train_std)

#查看主成分分析后的个成分的方差信息，并且归一化
var_per = sklearn_pca.explained_variance_ratio_
cum_var_per = sklearn_pca.explained_variance_ratio_.cumsum()

l = len(cum_var_per[cum_var_per <= 0.5])
sklearn_pca = sklearnPCA(n_components=l).fit(x_train_std)

# 使用主成分分析的结果，对原特征进行较为
x_train_pca = sklearn_pca.transform(x_train_std)
x_test_pca = sklearn_pca.transform(std_scal.transform(x_test))
print('降维后的数据训练数据 shape：',x_train_pca.shape)
print('降维后的测试数据shape: ',x_test_pca.shape)
#%%
grid_logit = model_selection.GridSearchCV(
    neural_network.MLPClassifier(),
    cv = 5,
    param_grid={
        'activation': ['logistic', 'tanh', 'relu'], 
        'solver': ('sgd', 'adma'),
        'hidden_layer_sizes':[(80, ),(100, ),(120, ),(140, )]
    },
    scoring='roc_auc'
)
grid_logit.fit(x_train_pca,y_train)
# %%
