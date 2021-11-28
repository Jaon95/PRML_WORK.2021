# %%
# 导入必要的包
from sklearn import neural_network
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

# %%
# %%
train = pd.read_csv("train.csv")
# %%
# sns.countplot(train['label'])
# 我们也可以使用直方图来画
# plt.hist(train['label'],10)
# plt.show()
# fig,ax = plt.subplots()
# n,bins,patches = ax.hist(train['label'],10,edgecolor='w')
sns.countplot(train['label'])
# ax是基于坐标轴来处理数据的，所以有x、y相关的设置属性
# %%
# Check for null and missing values
train.isnull()  # 判断整个矩阵的每个单元格的是否null，若是则返回True
train.isnull().any(axis=1).describe()
# isnull用来判断每个元素是否nan，none，nat类型，若是，则判断为true，any类似于或操作

# %%
# 多分类训练，使用sklearn自带的工具
Y = train['label'][:20000]  # use more number of rows for more training
# use more number of rows for more training
X = train.drop(['label'], axis=1)[:20000]
x_train, x_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.20, random_state=42)
#%%
# %%
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # 设定图片有多少行，多少列，用于后面向每个格子中填充数据
    plt.imshow(cm, cmap=cmap)  # 四舍五入 interpolation='nearest'

    # 设置标题，以及颜色条
    plt.title(title)
    plt.colorbar()

    # 分为设置x方向和y方向有多少个分类，以及每个分类显示的名称
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    # 判断是否需要归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# %%
# 查看示例数据
plt.imshow(x_train.iloc[0].values.reshape(28, 28))
# %%
model = neural_network.MLPClassifier(alpha=1e-5,  # 正则项的惩罚因子
                                     hidden_layer_sizes=(80,),
                                     # The solver for weight optimization.
                                     solver='lbfgs',
                                     # ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
                                     # ‘sgd’ refers to stochastic gradient descent.
                                     # ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
                                     activation='tanh',
                                     # ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
                                     # ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
                                     # ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
                                     # ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)
                                     random_state=18)
model.fit(x_train, y_train)
# %%
predicted = model.predict(x_val)
print("Classification Report:\n %s:" %
      (metrics.classification_report(y_val, predicted)))
print(model.score(x_val, y_val))

#%%
confusion_mtx = confusion_matrix(predicted, y_val)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=range(10))
# %%
def display_errors(error_index, x_errors, predicted_errors, y_true):
    n = 0
    n_rows = 2
    n_cols = 3
    fig, ax = plt.subplots(n_rows, n_cols)
    for row in range(n_rows):
        for col in range(n_cols):
            index = error_index[n]
            ax[row, col].imshow(x_errors.iloc[index].values.reshape(28, 28))
            ax[row, col].set_title("Predicted label :{}\nTrue label :{}".format(
                predicted_errors[index], y_true[index]))

            n += 1
    fig.tight_layout()
# %%
errors = (predicted - y_val != 0)
pred_classes_errors = predicted[errors]  # 预测后的值（预测错误的部分）
image_errors = x_val[errors]  # 对应的像素
true_classes_errors = y_val[errors].values  # 真实的值
error_idx = np.random.randint(low=0, high=len(
    pred_classes_errors), size=6)  # 取六个随机数
display_errors(error_idx, image_errors,
               pred_classes_errors, true_classes_errors)
