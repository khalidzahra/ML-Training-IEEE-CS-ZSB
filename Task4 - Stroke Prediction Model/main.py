import numpy as np
import os
import math
import csv 

DATASET_DIR = "./data/healthcare-dataset-stroke-data.csv"
TRAIN_AMOUNT = 2500
TEST_AMOUNT = 2500


def readData(path, start, end):
    features = []
    Y = []
    with open(path) as data:
        csv_reader = csv.reader(data, delimiter=',')
        rowCount = 0
        for row in csv_reader:
            if rowCount == 0:
                rowCount += 1
                continue
            if (rowCount >= start and rowCount <= end):
                feature = []
                feature.append(1)
                feature.append(0 if row[1] == "Male" else 1)
                feature.append(float(row[2]))
                feature.append(float(row[3]))
                feature.append(float(row[4]))
                feature.append(0 if row[5] == "No" else 1)
                feature.append(0 if row[6] == "Private" else 1 if row[6] == "Self-employed" else 2)
                feature.append(0 if row[7] == "Urban" else 1)
                feature.append(float(row[8]))
                feature.append(0 if row[9] == "N/A" else float(row[9]))
                feature.append(0 if row[10] == "never smoked" else 1 if row[6] == "Unknown" else 2)
                Y.append(float(row[11]))
                features.append(feature)
            rowCount += 1
    return features, Y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hypothesis(theta, x):
    return sigmoid(np.matmul(x, theta.T))


def costFunction(theta, x, y, m):
    hyp = hypothesis(theta, x)
    J = (-(1 / m)) * (np.matmul(y, np.log(hyp).T) + np.matmul((1 - y), np.log(1 - hyp).T))
    grad = np.divide(np.sum(np.multiply(x, np.expand_dims((hyp - y), axis=-1))), m)
    return J, grad


def train(theta, x, y, m, repetitions = 1000, alpha = 0.0003):
    for i in range(repetitions):
        J, grad = costFunction(theta, x, y, m)
        theta = theta - (alpha) * grad
    return theta

def predict(theta, x):
    return np.where(hypothesis(theta, x) > 0.5, 1, 0)


theta = [0] * 11
x_train, y_train = readData(DATASET_DIR, 0, TRAIN_AMOUNT)

print("Initial cost: " + str(costFunction(np.array(theta), np.array(x_train), np.array(y_train), len(y_train))[0]))

theta = train(np.array(theta), np.array(x_train), np.array(y_train), len(y_train))

x_test, y_test = readData(DATASET_DIR, 0, 5011)

predictions = predict(theta, np.array(x_test))

correctCount = 0
for i in range(len(predictions)):
    if predictions[i] == y_test[i]:
         correctCount += 1

print("Final cost: " + str(costFunction(np.array(theta), np.array(x_test), np.array(y_test), len(y_test))[0]))


print("Accuracy: " + str((correctCount * 100 / len(predictions))) + "%")