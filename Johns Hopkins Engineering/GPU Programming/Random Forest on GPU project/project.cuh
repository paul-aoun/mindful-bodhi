//////////////////////////////////////////////////////
// Paul Aoun										//
//													//
// Johns Hopkins University							//				
//													//
// 12/13/2020										//
//													//
//CUDA header file for all the kernel functions.    //
//Has the helper functions needed for Random Forest.//
//////////////////////////////////////////////////////

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <bits/stdc++.h>

#include "Node.h"
#include "RandomForest.h"

#ifndef PROJECT_HELPER_DEVICE
#define PROJECT_HELPER_DEVICE

/* Error handling for CUDA functions */
static void checkError( cudaError_t error, const char *file, int line ) {
    if (error != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( error ),
        file, line );
        exit( EXIT_FAILURE );
    }
}
#define CHECK_ERROR( error ) (checkError( error, __FILE__, __LINE__ ))

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

	/* we have to initialize the state */
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
				blockIdx.x, /* the sequence number should be different for each core (unless you want all
							   cores to get the same sequence of numbers for some reason - use thread id! */
				0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
				&states[blockIdx.x]);
  }
  
  /* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void randoms(curandState_t* states, float* numbers, int max) {
  /* curand works like rand - except that it takes a state as a parameter */
  numbers[blockIdx.x] = (curand(&states[blockIdx.x]) % max);
}

//Initialize an array with random values generated using cuRAND
void randomInitializeArray(unsigned int arrayLength, float **arrayHost, int max) {
	
	//States to hold the seed for the CUDA random numbers generator
	curandState_t* states;

	//Array of unsigned float on the GPU
	float* gpu_nums;

	CHECK_ERROR( cudaMalloc((void**) &states, arrayLength*sizeof(curandState_t)));
	init<<<arrayLength, 1>>>(time(NULL), states);
	CHECK_ERROR( cudaMalloc((void**) &gpu_nums, arrayLength * sizeof(float)));
	randoms<<<arrayLength, 1>>>(states, gpu_nums, max);
	CHECK_ERROR( cudaMemcpy(*arrayHost, gpu_nums, arrayLength * sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree (gpu_nums);
	cudaFree (states);

}

/* Allocate and initialize host matrices for the calculations */
/* flag values are: -1 for pageable, 0 for page-locked, and 1 for Portable*/
// void allocateArraysHost ( unsigned int array1HostLength, unsigned int array2HostLength, unsigned int resultHostLength,
// 	float** array1Host, float** array2Host, float** arrayResultHost, unsigned int flag ) {
void allocateArraysHost ( unsigned int array1HostLength, float** array1Host, unsigned int flag ) {
	
	/* Allocate memory based on the flag passed  */
	if (flag == 0) {
		if (DEBUG)
			printf("Flag = %d. Using Pageable host memory\n", flag);
		*array1Host = (float*)malloc( array1HostLength*sizeof(float) );
	} else if (flag == 1) {
		if (DEBUG)
			printf("Flag = %d. Using Page-locked host memory\n", flag);
		CHECK_ERROR (cudaHostAlloc((void**)array1Host, array1HostLength*sizeof(float), cudaHostAllocDefault));
	} else if (flag == 2) {
		if (DEBUG)
			printf("Flag = %d. Using Portable host memory\n", flag);
		CHECK_ERROR (cudaHostAlloc((void**)array1Host, array1HostLength*sizeof(float), cudaHostAllocPortable));
	}

}

/* Free host memory */
void freeArraysHost (float** array1Host, unsigned int  flag ) {
	
	if (flag == 0) {
		free(*array1Host);
	} else {
		CHECK_ERROR( cudaFreeHost(*array1Host));
	}
}

/* Parallel calculation of Gini for each column on GPU */
__global__ 
void calculateColumnGiniDevice(int arraySize, int *columnValuesArray,  
	int *decisionValuesArray, float *resultColumnGini ) {
	
	/* Getting the thread id */
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// Store the temporary addition for the threads in the block 
	int temp11 = 0; //positives counter
	int temp10 = 0; //fales positives counter
	int temp01 = 0; //false negatives counter
	int temp00 = 0; //negatives counter

	float giniValueLeft = 0.0f, giniValueRight = 0.0f, giniValue = 0.0f;

	//Calculate positives, negatives, false postives and false negatives
	for (int i=0; i<arraySize; i++) {
		switch (columnValuesArray[i]) {
			case 1:
				if (decisionValuesArray[i]==1){
					++temp11;
				} else if (decisionValuesArray[i]==0) {
					++temp10;
				}
				break;
			case 0:
				if (decisionValuesArray[i]==1){
					++temp01;
				} else if (decisionValuesArray[i]==0) {
					++temp00;
				}
				break;
		}
	}

	//Calculate Gini probability for correct and incorrect predition
	if ((temp11==0) &(temp10==0))
		giniValueLeft=0.0f;
	else
		giniValueLeft = 1.0f - pow(((float)temp11/((float)temp11 + (float)temp10)), 2) - pow(((float)temp10/((float)temp10 + (float)temp10)), 2);
	
	if ((temp01==0) &(temp01==0))
		giniValueRight=0.0f;
	else
		giniValueRight = 1.0f - pow(((float)temp01/((float)temp01 + (float)temp00)), 2) - pow(((float)temp00/((float)temp01 + (float)temp00)), 2);

	//Gini value for the column is the weighted average of Gini left and Gini right
	giniValue = giniValueLeft* (temp10 + temp11)/(temp10+temp11+temp01+temp00) 
		+giniValueRight* (temp00 + temp01)/(temp10+temp11+temp01+temp00);

	resultColumnGini[0] = giniValue;
}

//Call calculateColumnGiniDevice to calculate the columns Gini impurities on the device
void calculateColumnGini(int decisionColumn, std::vector<std::vector<std::string>> bootStrapRecords, 
	std::vector<Node*> &nodes) {
		
	//Get the columns from the bootStrapRecords vector
	std::vector<std::string> row = bootStrapRecords[0];

	//Each column will get its own stream to calculate its Gini value
	unsigned int numberOfStreams = nodes.size();
	cudaStream_t streams[numberOfStreams];
	//Initialize the streams to NULL
	for (int i=0; i<numberOfStreams; i++)
		streams[i] = NULL;
	
	int arraySize = bootStrapRecords.size();

	//Device arrays
	int *columnValuesArrayDevice, *decisionValuesArrayDevice;
	float *resultColumnGiniDevice;
	// Create CUDA timing events //
	cudaEvent_t start, stop; float elapsedTimeDevice;
	CHECK_ERROR( cudaEventCreate(&start) );
	CHECK_ERROR( cudaEventCreate(&stop) );
	
	//Host arrays
	int columnValuesArray[arraySize], decisionValuesArray[arraySize];
	float resultColumnGini[1];

	CHECK_ERROR( cudaMalloc((void**) &columnValuesArrayDevice, arraySize*sizeof(int)));
	CHECK_ERROR( cudaMalloc((void**) &decisionValuesArrayDevice, arraySize*sizeof(int)));
	CHECK_ERROR( cudaMalloc((void**) &resultColumnGiniDevice, sizeof(float)));

	int k{0};
	for (std::vector<std::string> row : bootStrapRecords) {
		decisionValuesArray[k++] = std::stoi(row [decisionColumn]);
	}

	//Calculate the Gini impurity all the nodes (columns) in the data file
	//Calculations are done on the device
	int i{0};
	for (Node *node : nodes) {
		if (node->getSelectedColumn()) {
			CHECK_ERROR( cudaStreamCreate( &streams[i] ) );
			int j{0};
			for (std::vector<std::string> row : bootStrapRecords) {
				columnValuesArray[j++] = std::stoi(row [node->getColumnNumber()]);
			}

			CHECK_ERROR( cudaEventRecord(start, 0) );

			CHECK_ERROR (cudaMemcpyAsync(columnValuesArrayDevice, columnValuesArray, arraySize * sizeof(int), cudaMemcpyHostToDevice, streams[i]) );
			CHECK_ERROR (cudaMemcpyAsync(decisionValuesArrayDevice, decisionValuesArray, arraySize * sizeof(int), cudaMemcpyHostToDevice, streams[i]) );
			calculateColumnGiniDevice<<<1, 1, 0, streams[i]>>>
				(arraySize, columnValuesArrayDevice, decisionValuesArrayDevice, resultColumnGiniDevice);
			CHECK_ERROR( cudaMemcpyAsync(resultColumnGini, resultColumnGiniDevice, sizeof(float), cudaMemcpyDeviceToHost, streams[i]) );

			// Retrieve timing events and time elapsed for CUDA arithmetic operation //
			CHECK_ERROR( cudaEventRecord(stop, 0) );
			CHECK_ERROR( cudaEventSynchronize(stop) );

			CHECK_ERROR( cudaEventElapsedTime(&elapsedTimeDevice, start, stop) );

			//Set the Gini impurity value for the node based on the result from the device
			node->setGiniImpurity(resultColumnGini[0]);
			
			if (DEBUG)
				std::cout << "calculateColumnGini: Calculated column " << node->getColumnNumber() << " gini impurity value=" 
					<< node->getGiniImpurity() << " in " << elapsedTimeDevice << " ms on device." << std::endl;
			
			numberOfStreams=i;
			i++;
		} else {
			//Node not selected, set impossibly high Gini value
			node->setGiniImpurity(2.0f);
		}
	}

	//Clear the streams
	for (int i = 0; i<numberOfStreams; i++ )
	{
		if (streams[i] != NULL)
			CHECK_ERROR( cudaStreamDestroy( streams[i] ));
	}
		

	//Free device memory
	CHECK_ERROR( cudaFree(columnValuesArrayDevice));
	CHECK_ERROR( cudaFree(decisionValuesArrayDevice));
	CHECK_ERROR( cudaFree(resultColumnGiniDevice));

	/* Destroy the events */
	CHECK_ERROR( cudaEventDestroy(start) );
	CHECK_ERROR( cudaEventDestroy(stop) ) ;

}

//Randomly select columns to be used for generating the tree
void selectColumns(int minColumnNumber, int maxColumnNumber, int decisionColumn,
	int maxRandomNumbers, int flag, std::vector<Node*> &nodes) {
	
	// Initialize random number to select the columns
	float *columns;
	std::vector<int> columnsNumbers;

	//Allocate and initialize the host arrays needed
	allocateArraysHost (maxRandomNumbers, &columns, flag);
	randomInitializeArray (maxRandomNumbers, &columns, maxColumnNumber);

	int i{0}, j{0};

	//Get the column numbers between min and max 
	while((i<maxRandomNumbers) && (j<maxColumnNumber - minColumnNumber +1)) {
		if (((int)columns[i] >= minColumnNumber) && (((int)columns[i] <= maxColumnNumber))) {
			if ((int)columns[i] != decisionColumn) {
				//Check if column already selected
				std::vector<int>::iterator it = 
					find (columnsNumbers.begin(), columnsNumbers.end(), (int)columns[i]);
				if (it == columnsNumbers.end()) { 
					columnsNumbers.push_back((int)columns[i]);
					for (Node *node : nodes) {
						if (node->getColumnNumber() == ((int)columns[i])) {
							node->setSelectedColumn(true);
						}
					}
					if (DEBUG) {
						std::cout << "selectColumns: column number= " << columnsNumbers.at(j) << " selected." << std::endl;
					}
					j++;
				}
			}
		}
		i++;
	}
}

#endif