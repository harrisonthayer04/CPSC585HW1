#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "PreprocessData.h"

struct Student *loadDataSetFromCSV(char *csvFilePath) {
  struct Student *students =
      (struct Student *)malloc(MAX_NUM_STUDENTS * sizeof(struct Student));
  FILE *file = fopen(csvFilePath, "r+");
  if (file == NULL) {
    printf("Opening file failed. Try again.\n");
  }
  char line[LINE_MAX_SIZE];
  int studentIndex = 0;
  while (fgets(line, LINE_MAX_SIZE, file)) {
    char *token;
    int columnIndex = 0;

    // Set first feature as 1 (bias term)
    students[studentIndex].studentData[0] = 1.0;

    token = strtok(line, ";");

    while (token != NULL) {
      // Shift all features one position to the right
      if (columnIndex < 36) {
        students[studentIndex].studentData[columnIndex + 1] = atof(token);
      } else if (columnIndex == 36) {
        token[strcspn(token, "\r\n")] = '\0';
        if (strcmp(token, "Graduate") == 0) {
          students[studentIndex].target = 1;
        } else if (strcmp(token, "Dropout") == 0) {
          students[studentIndex].target = 0;
        } else if (strcmp(token, "Enrolled") == 0) {
          students[studentIndex].target = 2;
        }
      }
      token = strtok(NULL, ";");
      columnIndex++;
    }
    studentIndex++;
  }
  fclose(file);
  printf("Loaded %d students from CSV\n", studentIndex);
  return students;
}

struct Student *removeEnrolledStudents(struct Student *students,
                                       int numStudents,
                                       int *newNumberOfStudents) {
    // Allocate with correct size
    struct Student *newStudentsList = malloc(numStudents * sizeof(struct Student));
    *newNumberOfStudents = 0;
    
    // Debug counters
    int graduates = 0;
    int dropouts = 0;
    int enrolled = 0;
    
    for (int i = 0; i < numStudents; i++) {
        if (students[i].target == 2) {
            enrolled++;
            continue;
        }
        // Copy the student data
        memcpy(&newStudentsList[*newNumberOfStudents], &students[i], sizeof(struct Student));
        
        // Count class distribution
        if (students[i].target == 1) graduates++;
        if (students[i].target == 0) dropouts++;
        
        (*newNumberOfStudents)++;
    }
    
    printf("\nOriginal data statistics:\n");
    printf("Graduates: %d\n", graduates);
    printf("Dropouts: %d\n", dropouts);
    printf("Enrolled: %d\n", enrolled);
    printf("Total processed: %d\n", graduates + dropouts + enrolled);
    
    return newStudentsList;
}

void printDataSet(struct Student students[], int numStudents) {
  for (int i = 0; i < numStudents; i++) {
    printf("Student %d:\n", i);
    printf("  Data: ");
    // Print the 35 data values for the student
    for (int j = 0; j < 35; j++) {
      printf("%f ", students[i].studentData[j]);
    }
    printf("\n  Target: %d\n\n", students[i].target);
  }
}

struct Student* scaleDataSet(struct Student *students, int numStudents) {

  for (int feature = 0; feature < 37; feature++) {
    double min = students[0].studentData[feature];
    double max = students[0].studentData[feature];

    for (int i = 1; i < numStudents; i++) {
      if (students[i].studentData[feature] < min) {
        min = students[i].studentData[feature];
      }
      if (students[i].studentData[feature] > max) {
        max = students[i].studentData[feature];
      }
    }
    if (max == min) {
      continue;
    }
            for (int i = 0; i < numStudents; i++) {
            students[i].studentData[feature] = 
                (students[i].studentData[feature] - min) / (max - min);
        }
  }
  return students;
}
