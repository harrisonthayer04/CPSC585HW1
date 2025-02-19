#include <stdio.h>
#include <stdlib.h>

#include "Perceptron.h"
#include "PreprocessData.h"

void debugPrint(struct Student *trainingSet) {
  printf("Printing 5 samples from training set:\n");
  for (int i = 0; i < 5; i++) {
    printf("Sample %i: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, "
           "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, "
           "%f, %f, %f, %f, %f, %f, %f, %f, Target: %i\n",
           i, trainingSet[i].studentData[0], trainingSet[i].studentData[1],
           trainingSet[i].studentData[2], trainingSet[i].studentData[3],
           trainingSet[i].studentData[4], trainingSet[i].studentData[5],
           trainingSet[i].studentData[6], trainingSet[i].studentData[7],
           trainingSet[i].studentData[8], trainingSet[i].studentData[9],
           trainingSet[i].studentData[10], trainingSet[i].studentData[11],
           trainingSet[i].studentData[12], trainingSet[i].studentData[13],
           trainingSet[i].studentData[14], trainingSet[i].studentData[15],
           trainingSet[i].studentData[16], trainingSet[i].studentData[17],
           trainingSet[i].studentData[18], trainingSet[i].studentData[19],
           trainingSet[i].studentData[20], trainingSet[i].studentData[21],
           trainingSet[i].studentData[22], trainingSet[i].studentData[23],
           trainingSet[i].studentData[24], trainingSet[i].studentData[25],
           trainingSet[i].studentData[26], trainingSet[i].studentData[27],
           trainingSet[i].studentData[28], trainingSet[i].studentData[29],
           trainingSet[i].studentData[30], trainingSet[i].studentData[31],
           trainingSet[i].studentData[32], trainingSet[i].studentData[33],
           trainingSet[i].studentData[34], trainingSet[i].studentData[35],
           trainingSet[i].studentData[36], trainingSet[i].studentData[37],
           trainingSet[i].target);
  }
}

void printWeights(double *weights) {
  for (int i = 0; i < 38; i++) {
    printf("Weight %i: %f", i, weights[i]);
  }
}

int main() {

  struct Student *students = loadDataSetFromCSV("data.csv");

  int newNumberOfStudents = 0;
  students = removeEnrolledStudents(students, 4225, &newNumberOfStudents);
  students = scaleDataSet(students, newNumberOfStudents);

  printf("Size of dataset: %i\n", newNumberOfStudents);
  printf("Splitting into testing and training sets\n");
  int lastOfTrainingSetIndex = (int)(newNumberOfStudents * .8);

  struct Student *trainingSet = malloc(lastOfTrainingSetIndex * sizeof(struct Student));
  struct Student *testingSet = malloc((newNumberOfStudents - lastOfTrainingSetIndex) * 
                                    sizeof(struct Student));

  int training_count = 0;
  for (int i = 0; i < lastOfTrainingSetIndex; i++) {
    memcpy(&trainingSet[i], &students[i], sizeof(struct Student));
    training_count++;
  }

  int testing_count = 0;
  for (int i = lastOfTrainingSetIndex; i < newNumberOfStudents; i++) {
    memcpy(&testingSet[testing_count], &students[i], sizeof(struct Student));
    testing_count++;
  }

  printf("Size of training set: %d\n", training_count);
  printf("Size of test set: %d\n", testing_count);

  // debugPrint(trainingSet);

  // After creating training set
  int graduates = 0;
  int dropouts = 0;
  double max_val = -1000;
  double min_val = 1000;

  // Check class distribution and data ranges in training set
  for (int i = 0; i < training_count; i++) {
    if (trainingSet[i].target == 1) graduates++;
    if (trainingSet[i].target == 0) dropouts++;
    
    for (int j = 0; j < 38; j++) {
      if (trainingSet[i].studentData[j] > max_val) max_val = trainingSet[i].studentData[j];
      if (trainingSet[i].studentData[j] < min_val) min_val = trainingSet[i].studentData[j];
    }
  }

  printf("\nTraining set statistics:\n");
  printf("Graduates: %d (%.2f%%)\n", graduates, (float)graduates/training_count * 100);
  printf("Dropouts: %d (%.2f%%)\n", dropouts, (float)dropouts/training_count * 100);
  printf("Data range: [%f, %f]\n", min_val, max_val);

  // Save training data to file
  FILE *training_file = fopen("training_data.txt", "w");
  if (training_file == NULL) {
    printf("Error opening training_data.txt for writing\n");
    return 1;
  }

  // Write training data (including bias term)
  for (int i = 0; i < training_count; i++) {
    // Write all features including bias
    for (int j = 0; j < 38; j++) {
        fprintf(training_file, "%.6f", trainingSet[i].studentData[j]);
        if (j < 37) fprintf(training_file, ",");
    }
    // Write target
    fprintf(training_file, ",%d\n", trainingSet[i].target);
  }

  fclose(training_file);

  // Save test data to file
  FILE *test_file = fopen("test_data.txt", "w");
  if (test_file == NULL) {
    printf("Error opening test_data.txt for writing\n");
    return 1;
  }

  // Write test data
  for (int i = 0; i < testing_count; i++) {
    // Write features
    for (int j = 0; j < 38; j++) {
        fprintf(test_file, "%.6f", testingSet[i].studentData[j]);
        if (j < 37) fprintf(test_file, ",");
    }
    // Write target
    fprintf(test_file, ",%d\n", testingSet[i].target);
  }

  fclose(test_file);

  double *weights = perceptronLearning(0.01, 1000, trainingSet, training_count);
  // printWeights(weights);

  double trainingAccuracy = testModelWeights(weights, trainingSet, training_count);
  double testingAccuracy = testModelWeights(weights, testingSet, testing_count);
  printf("Training accuracy: %f\n", trainingAccuracy);
  printf("Testing accuracy: %f\n", testingAccuracy);

  free(students);
  free(trainingSet);
  free(testingSet);
  return 0;
}