#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "PreprocessData.h"

double* perceptronLearning(double learning_rate, int max_epochs, struct Student *trainingSet, int trainingSetSize);
double testModelWeights(double* weights, struct Student *testingSet, int testingSetSize);



#endif