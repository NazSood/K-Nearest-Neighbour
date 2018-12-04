'''
Author: Nishant Sood

The following program does K nearest neighbour computation
 for the data inputs for the ages old classicIris dataset.

 Interpreted with python 3.x.x
'''

import math
import csv
import random
import operator

'''Load dataset and split it into training & test data based on random function'''
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename) as ifile:
        lines = csv.reader(ifile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


'''Calculating euclidean distance and returning the root'''
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


'''Method designated for calculating nearby data  point occurances'''
def getNeighbours(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for x in range(k):
        neighbours.append(distances[x][0])
    return neighbours

# get which class to which datapoints belong to
def getResponse(neighbours):
    classVotes = {}
    for x in range(len(neighbours)):
        response = neighbours[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=False)
    return sortedVotes[0][0]

#Calculate Accuracy of the prediction done by model
def getAccuracy(testSet=[], predictions=[]):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    ''' Get accuracy method check
    testSet = [[1,1,1,'Iris-setosa'], [2,2,2,'Iris-setosa'], [3,3,3,'b']]
    predictions = ['Iris-setosa', 'a', 'a']
    accuracy =  getAccuracy(testSet, predictions)
    print(accuracy)
    '''
    #define training and test set/lists
    trainingSet = []
    testSet = []
    split = 0.60 # splitting the data into 60% training and 40%(remaining) for testing
    loadDataset('iris.txt', split, trainingSet, testSet)
    print('Train Set: ' + repr(len(trainingSet)))
    print('Test Set: ' + repr(len(testSet)))
    # generate predictions
    predictions = []
    k = 3 # number of nearby neighbours to look for in order to judge a class for an input
    for x in range(len(testSet)):
        neighbours = getNeighbours(trainingSet, testSet[x], k)
        result = getResponse(neighbours)
        predictions.append(result)
        #print(predictions)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

    #predictions = ['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-virginica', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-versicolor', 'Iris-virginica', 'Iris-virginica', 'Iris-versicolor', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica']
    accuracy = getAccuracy(testSet, predictions) # calculate accuracy in % of correct ones as to predicted wrongly
    print('Accuracy: ' + repr(accuracy) + '%')

main()
