#**************************** Data *******************************

train_raw.csv: 从原始大数据集（55M条）train.csv中切分的部分(200K条)数据，作为原始数据集；

train1.csv：利用NYC_Data.ipynb对train_raw进行数据处理，清理了噪声数据后的训练数据集；

test1.csv: 从原始大数据集（55M条）train.csv中切分的部分(10K条)数据，并利用NYC_Data.ipynb进行清理后，作为训练数据集，并将其fare_amount列（表示计程车费用）切分到value1.csv中，作为评估RMSE的数据；

value1.csv: 与test1.csv中key列对应的fare_amount列，表示实际的计程车费用；

#*************************** Program *****************************

NYC_Data.ipynb: Data Exploration program, remove noise in raw data;

NYC_LinearRegression.ipynb: Train simple Linear Regression Model with training set(train1.csv),  predict on test set(test1.csv), and compute RMSE between predition and groundtruth(value1.csv).
