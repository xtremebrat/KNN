import csv
import random
import math
import operator

# handle the data. read the iris.data file and convert it to csv & display it
"""
with open('data/iris.data', 'rb') as csvfile:
    lines = csv.reader(csvfile)
    for now in lines:
        print ', '.join(now)
"""

# split test & training dataset, convert string values to integer


def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

# Display the train and test data
"""
trainingSet = []
testSet = []
loadDataset('data/iris.data', 0.66, trainingSet, testSet)
print "\nTrain: " + repr(len(trainingSet))
print "Test: " + repr(len(testSet))
"""

# calculate the similarity in order to make predictions. Use euclidean distance measure


def euclideanDistance(inst1, inst2, length):
    distance = 0
    for x in range(length):
        distance += pow((inst1[x] - inst2[x]), 2)
    return math.sqrt(distance)

# test euclideanDistance with some sample data
"""
data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']
distance = euclideanDistance(data1, data2, 3)
print "\nDistance: " + repr(distance)
"""

# find k most similar instances (neighbors)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# test the getNeighbors function
"""
trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
testInstance = [5, 5, 5]
k = 1 # 1 neighbor
neighbors = getNeighbors(trainSet, testInstance, 1)
print "\nNeighbors Are: ",
print (neighbors)
"""

# devise predicted response based on neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
        sortedVotes = sorted(classVotes.iteritems(), key = operator.itemgetter(1))
        return sortedVotes[0][0]

# test the response
"""
neighbors = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
response = getResponse(neighbors)
print "\nResponse Is: " + repr(response)
"""

# classification accuracy, ratio of total correct predictions out of all predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100

# test the accuracy
"""
testSet = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print "\nAccuracy Is: " + repr(accuracy)
"""

# main


def main():
    # prepare the data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('data/iris.data', split, trainingSet, testSet)
    print "\nTrain Set: " + repr(len(trainingSet))
    print "Test Set: " + repr(len(testSet)) + "\n"

    #generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print "Accuracy: " + repr(accuracy) + "%"

main()
