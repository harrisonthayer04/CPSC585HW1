#ifndef PREPROCESS_DATA_H
#define PREPROCESS_DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE_MAX_SIZE 2048    
#define MAX_NUM_STUDENTS 5000

struct Student{
    double studentData[38];
    int target;
};

struct Student* loadDataSetFromCSV(char* csvFilePath);
void printDataSet(struct Student* students, int numStudents);
struct Student* removeEnrolledStudents(struct Student* students, int numStudents, int* newNumberOfStudents);
struct Student* scaleDataSet(struct Student* students, int numStudents);

#endif