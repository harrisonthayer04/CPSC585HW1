#include "Perceptron.h"
#include "PreprocessData.h"
#include <time.h>

double *perceptronLearning(double learning_rate, int max_epochs,
                           struct Student *trainingSet, int trainingSetSize) 
{
    // Initialize all weights to 0
    double *weights = (double *)malloc(38 * sizeof(double));
    for (int i = 0; i < 38; i++) {
        weights[i] = 0.0;  // Set all weights to 0
    }

    // Save initial weights
    FILE *fp = fopen("weight_evolution.txt", "w");
    if (fp == NULL) {
        printf("Error opening weight_evolution.txt for writing\n");
        return NULL;
    }

    // Save weights with bias term
    for (int i = 0; i < 38; i++) {
        fprintf(fp, "%.6f", weights[i]);
        if (i < 37) fprintf(fp, ",");
    }
    fprintf(fp, "\n");

    // Training loop
    for (int epoch = 0; epoch < max_epochs; epoch++) {
        int errors = 0;
        
        for (int j = 0; j < trainingSetSize; j++) {
            double weightedSum = 0.0;
            for (int k = 0; k < 38; k++) {
                weightedSum += weights[k] * trainingSet[j].studentData[k];
            }

            // Perceptron prediction
            int prediction = (weightedSum >= 0.0) ? 1 : 0;

            // Error = (true_target - predicted_label)
            double error = trainingSet[j].target - prediction;

            // Update weights if there's an error
            if (error != 0.0) {
                errors++;
                for (int k = 0; k < 38; k++) {
                    weights[k] += learning_rate * error * trainingSet[j].studentData[k];
                }
            }
        }

        // Save weights after each epoch with bias term
        for (int i = 0; i < 38; i++) {
            fprintf(fp, "%.6f", weights[i]);
            if (i < 37) fprintf(fp, ",");
        }
        fprintf(fp, "\n");

        // Optional: print progress
        if (epoch % 100 == 0) {
            printf("Epoch %d, errors: %d\n", epoch, errors);
        }

        // Early stopping if converged
        if (errors == 0) {
            printf("Converged at epoch %d\n", epoch);
            break;
        }
    }

    fclose(fp);
    return weights;
}


double testModelWeights(double *weights, struct Student *testingSet,
                        int testingSetSize) 
{
    int correctPredictions = 0;

    for (int i = 0; i < testingSetSize; i++) {
        double weightedSum = 0.0;
        for (int k = 0; k < 38; k++) {
            weightedSum += weights[k] * testingSet[i].studentData[k];
        }
        
        int prediction = (weightedSum >= 0.0) ? 1 : 0;
        
        if (prediction == (int)testingSet[i].target) {
            correctPredictions++;
        }
    }

    double accuracy = ((double)correctPredictions) / testingSetSize;
    return accuracy;
}