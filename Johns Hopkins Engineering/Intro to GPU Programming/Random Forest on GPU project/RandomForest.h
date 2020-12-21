//////////////////////////////////////////////////////
// Paul Aoun										//
//													//
// Johns Hopkins University							//				
//													//
// 11/15/2020										//
//													//
// This is the file for the RandomForest class.   	//
// It defines the functions needed to				//
// manage the RandomForest building .               //
//////////////////////////////////////////////////////

#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>

#include "Node.h"

#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

//Class to provide all the needed RandomForest functions.
class RandomForest {

private:
	
	static int ROW_NUMBER; //Length of the data file	
	static int COLUMN_NUMBER; //Number of columns in the data file 
	std::vector<std::vector<std::string>> allRecords; //Vector of all rows in the data file
	std::vector<Node*> allNodes;

public:

	//Constructor for the Random Forest class. It will load the data file in memory
	// and set file length and number of columns variables
	RandomForest (char *fileName, char splitChar);
	
	//Return the number of rows in the data file
	int getNumberOfRows();

	//Return the number of columns in the data file
	int getNumberOfColumns();
	
	//Return all the records in a vector
	std::vector<std::vector<std::string>> getAllRecords();

	//Return all nodes for the columns in the data file
	std::vector<Node*> getAllNodes();

	//Get bootstrap records from the CSV file
	std::vector<std::vector<std::string>> getBootStrapRecords (char *fileName, char splitChar, 
		int numberOfRandomNumbers, float **randomNumbers);
	
};

#endif