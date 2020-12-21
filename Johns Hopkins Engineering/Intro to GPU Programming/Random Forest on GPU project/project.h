#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef PROJECT_HELPER
#define PROJECT_HELPER

/* Contains the host helper functions for the Random Forest algorithm */

//Used to select the math operations
enum MathOperations { addition, substraction, multiplication, modulus };
static const char *MathOperationsNames[] = {
		"addition", "substraction", "multiplication", "modulus",
};

//Used to select the CUDA memory type
enum MemoryTypeDevice { global, shared, registerdevice };
static const char *MemoryTypeDeviceNames[] = {
		"global", "shared", "register"
};

// Calculate the total number of correctly and incorrectly the parameter predicted the outcome
void calculateNodeValue() {}

#endif

