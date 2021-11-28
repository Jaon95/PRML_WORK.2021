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


class NetualNetworkModelFactory:

    def __sigmod(self, z):
        z_shape = z.shape
        z = z.ravel()
        temp = []
        for i in range(len(z)):
            if z[i] >= 0:
                s = 1/(1+np.exp(-z[i]))
                temp.append(s)
            else:
                s = np.exp(z[i])/(1+np.exp(z[i]))
                temp.append(s)
        return np.array(temp).reshape(z_shape)

    _active_function = __sigmod

    @property
    def active_function(self):
        return self._active_function

    @active_function.setter
    def active_function(self, value):
        self._active_function = value

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    # 定义结构
    def __network_architecture(self, X, Y):
        n_x = X.shape[0]  # x作为输入时，已经转置，所以这里的n_x = 784

        n_h = 10

        n_y = Y.shape[0]  # y作为输入时，已经转置，所以这里的n_y为1
        # 整个网络的输入节点有n_x(784)个，输出有n_y(1)个
        return (n_x, n_h, n_y)

    # 初始化各个参数
    def __network_parameters(self, n_x, n_h, n_y):
        # 每个节点都要一组w，所有有n_h组w，每个w有n_x个参数,这里就有10组w，每组w有784个参数
        W1 = np.random.randn(n_h, n_x)*0.01
        b1 = np.random.randn(n_h, 1)
        # 每y节点都要一组w，所有有n_y组w，每个w有n_h个参数，这里就是1组w，共10个参数
        W2 = np.random.randn(n_y, n_h)*0.01
        b2 = np.random.randn(n_y, 1)
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    # 运算
    def __forward_propagation(self, X, params):
        # W1:10*784，784*样本个数，10*1 = 10*样本个数
        Z1 = np.dot(params['W1'], X)+params['b1']
        A1 = self._active_function(Z1)  # 10*样本个数，对每个进行计算

        Z2 = np.dot(params['W2'], A1)+params['b2']  # 1*10 10*样本个数,b2:1*1
        A2 = self._active_function(Z2)  # A2: 1*样本个数
        return {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

    # 计算误差
    def __compute_error(self, predict, actual):
        logprobs = np.multiply(np.log(predict), actual) + \
            np.multiply(np.log(1-predict), actual)
        cost = -np.sum(logprobs)/actual.shape[1]  # actual的列的个数，也就是样本数
        return cost

    # 定义误差反馈，并计算参数变化增量
    def __backward_propagation(self, params, activations, X, Y):
        m = X.shape[1]  # 样本个数

        # output layer
        dZ2 = activations['A2'] - Y  # compute the error derivative ：1*样本个数
        # compute the weight derivative 1*样本个数  样本个数*10, 每个样本的A1，结果就是将每个样本的误差*每个样本对应的A1.
        dW2 = np.dot(dZ2, activations['A1'].T) / m
        # compute the bias derivative,计算输出层的总误差，设置为b2的调整量
        db2 = np.sum(dZ2, axis=1, keepdims=True)/m

        # hidden layer
        # 10*1,1*样本数量=10*样本数量，得出每个样本在隐藏层的误差，1-np.power(activations['A1'], 2)
        dZ1 = np.dot(params['W2'].T, dZ2)*(1-np.power(activations['A1'], 2))
        # 10*样本数量 样本数量*训练参数个数 = 10*训练参数个数（也就是输入层的节点数量）然后/m，得到对每个dw1的调整
        dW1 = np.dot(dZ1, X.T)/m
        # compute the bias derivative,计算输出层的总误差，设置为b1的调整量
        db1 = np.sum(dZ1, axis=1, keepdims=True)/m

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    # 更新参数
    def __update_parameters(self, params, derivatives, alpha=1.2):
        # alpha is the model's learning rate

        params['W1'] = params['W1'] - alpha * derivatives['dW1']
        params['b1'] = params['b1'] - alpha * derivatives['db1']
        params['W2'] = params['W2'] - alpha * derivatives['dW2']
        params['b2'] = params['b2'] - alpha * derivatives['db2']
        return params

    # 开始训练
    def fit(self, X, Y, n_h, num_iterations=100):
        n_x = self.__network_architecture(X, Y)[0]  # 获取到了x的节点个数，也就是输入
        n_y = self.__network_architecture(X, Y)[2]  # 获取了y的节点个数，也就是输出

        params = self.__network_parameters(n_x, n_h, n_y)  # 对所有的参数进行初始化
        for i in range(0, num_iterations):
            results = self.__forward_propagation(X, params)
            error = self.__compute_error(results['A2'], Y)
            derivatives = self.__backward_propagation(params, results, X, Y)
            params = self.__update_parameters(params, derivatives)
        self.params = params

    def predict(self, X):
        results = self.__forward_propagation(X, self.params)
        print(results['A2'][0])
        predictions = np.around(results['A2'])
        return predictions

    def score(self, x, y):
        predictions = self.predict(x)
        score = float((np.dot(y, predictions.T) +
                       np.dot(1-y, 1-predictions.T))/float(y.size)*100)
        print('Accuracy: %d' % score + '%')
        return score

    def plot_confusion_matrix(self, cm, classes,
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
# include only the rows having label = 0 or 1 (binary classification)
# 本次训练只训练 分类为0或者1的
X = train[train['label'].isin([0, 1])]
# target variable
Y = train[train['label'].isin([0, 1])]['label']
# remove the label from X
# drop删除指定的数据，axi=1代表按照列删除,若是没有指定，则需要输入行索引，按照行来删除
X = X.drop(['label'], axis=1)


# %%
# 训练集和测试集的划分
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    X, Y, random_state=100, test_size=0.3)
x_train = x_train.T.values
y_train = y_train.values.reshape(1, y_train.size)
x_test = x_test.T.values
y_test = y_test.values.reshape(1, y_test.size)
# %%
print('下面是节点为：100，迭代次数为：10的训练')
model = NetualNetworkModelFactory()
model.fit(x_train, y_train, n_h=100, num_iterations=10)
# %%
print('------训练集------')
model.score(x_train, y_train)
confusion_mtx = confusion_matrix(model.predict(
    x_train).reshape(-1, 1), y_train.reshape(-1, 1))
classes = range(2)
model.plot_confusion_matrix(confusion_mtx, classes=classes)
print('-----测试集------')
model.score(x_test, y_test)
confusion_mtx = confusion_matrix(model.predict(
    x_test).reshape(-1, 1), y_test.reshape(-1, 1))
model.plot_confusion_matrix(confusion_mtx, classes=classes)

# %%
print('下面是节点为：10，迭代次数为：100 的训练')
model = NetualNetworkModelFactory()
model.fit(x_train, y_train, n_h=10, num_iterations=100)
# %%
print('------训练集------')
model.score(x_train, y_train)
confusion_mtx = confusion_matrix(model.predict(
    x_train).reshape(-1, 1), y_train.reshape(-1, 1))
classes = range(2)
model.plot_confusion_matrix(confusion_mtx, classes=classes)
# %%
print('-----测试集------')
model.score(x_test, y_test)
confusion_mtx = confusion_matrix(model.predict(
    x_test).reshape(-1, 1), y_test.reshape(-1, 1))
model.plot_confusion_matrix(confusion_mtx, classes=classes)
# %%
# train_score = []
# test_score = []
# for i in range(10, 20):
#     model = NetualNetworkModelFactory()
#     model.fit(x_train, y_train, n_h=10, num_iterations=i)
#     train_score.append(model.score(x_train, y_train))
#     test_score.append(model.score(x_test, y_test))

# %%
# fig, ax = plt.subplots()
# ax.plot([i for i in range(10, 20)], train_score)
# ax.plot([i for i in range(10, 20)], test_score)
# ax.set_xlabel([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# plt.show()
# %%
# 多分类训练，使用sklearn自带的工具
Y = train['label'][:20000]  # use more number of rows for more training
# use more number of rows for more training
X = train.drop(['label'], axis=1)[:20000]
x_train, x_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.20, random_state=42)
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


confusion_mtx = confusion_matrix(predicted, y_val)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=range(10))
# %%
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
error_idx = np.random.randint(low=0, high=len(pred_classes_errors), size=6)  # 取六个随机数
display_errors(error_idx, image_errors,
               pred_classes_errors, true_classes_errors)
# %%
