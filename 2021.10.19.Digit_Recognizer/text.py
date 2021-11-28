import numpy as np
import sklearn.naive_bayes

model = sklearn.naive_bayes.BernoulliNB()
model.fit(np.array([0,1,1]))
print(model)