import csv
import matplotlib.pyplot as plt


def hypothesis_function(theta, x):
    return [theta[0] + theta[1] * feature for feature in x]


def compute_cost(theta, x, y):
    m = len(x)
    hypothesis = hypothesis_function(theta, x)
    error_squared = sum([pow(prediction - result, 2) for prediction, result in zip(hypothesis, y)])
    return error_squared / (2 * m)


def gradient_descent(theta, x, y, alpha=0.03, repeat=50):
    m = len(x)
    cost_hist = []
    for _ in range(repeat):
        hypothesis = hypothesis_function(theta, x)
        error = [prediction - result for prediction, result in zip(hypothesis, y)]
        newtheta = [theta[0] - ((alpha / m) * sum(error)),
                    theta[1] - ((alpha / m) * sum([curr_x * err for curr_x, err in zip(x, error)]))]
        theta = newtheta
        cost_hist.append(compute_cost(theta, x, y))
    return [theta, cost_hist]


with open('./data/dataset.csv') as file:
    data = csv.reader(file, delimiter=',')
    y = []
    x = []
    theta = [0, 0]
    index = 0
    for row in data:
        if index == 0:
            index += 1
            continue
        x.append(float(row[0]))
        y.append(float(row[1]))
    theta, cost_hist = gradient_descent(theta, x, y, 0.0003)
    # hypothesis best fit plot
    hypFig, hypAx = plt.subplots()
    hypAx.set_title("Hypothesis best fit through data set")
    hypAx.set_ylabel("Result")
    hypAx.set_xlabel("Feature")
    hypAx.scatter(x, y)
    hypAx.plot(x, hypothesis_function(theta, x))
    # cost function plot
    costFig, costAx = plt.subplots()
    costAx.plot([i for i in range(50)], cost_hist)
    costAx.set_title("Cost function variance with number of repetitions")
    costAx.set_ylabel("Cost")
    costAx.set_xlabel("Number of Repetitions")
    plt.show()
    print(theta)
