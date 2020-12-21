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


#include "RandomForest.h"
#include "Node.h"

int RandomForest::ROW_NUMBER;
int RandomForest::COLUMN_NUMBER;

//Constructor for the Random Forest class. It will load the data file in memory
// and set file length and number of columns variables
RandomForest::RandomForest (char *fileName, char splitChar) {
	
	std::ifstream myfile;
	unsigned int fileLength { 0 }; 
	std::string line;
    
    COLUMN_NUMBER = 0;
    ROW_NUMBER = 0;

	myfile.open (fileName);
   
   //Open the data file and loads the records in the vector of strings
    if (myfile.is_open()) { 
        while ( getline (myfile,line) )
        {
			fileLength++;
            std::istringstream split(line);
            std::vector<std::string> row;

            //Split the line into a row of strings
            for (std::string each; std::getline(split, each, splitChar); row.push_back(each));
            
            COLUMN_NUMBER = row.size();
            allRecords.push_back(row);
            // if (DEBUG)
            //     std::cout << "RandomForest::RandomForest row[0]=" << row[0] << std::endl;
		}

        if (DEBUG)
            std::cout << "RandomForest::RandomForest COLUMN_NUMBER=" << COLUMN_NUMBER << std::endl;        
	} 
	
	ROW_NUMBER = fileLength;

    //Create the nodes for all the columns in the data file
    std::vector<std::string> row = allRecords.front();
    int counter{0};
    for (std::string column : row) {
        Node *n = new Node(counter);
        n->setName(column);
        n->setColumnNumber(counter);
        allNodes.push_back(n);
        if (DEBUG)
            std::cout << "Added column name=" << column << " and number=" << counter <<std::endl;
        counter++;
    }
	
	if (DEBUG)
		std::cout << "RandomForest::RandomForest: ROW_NUMBER= " << fileLength << std::endl;
    
    myfile.close();

};

//Return number of rows in the data file
int RandomForest::getNumberOfRows() {
	return RandomForest::ROW_NUMBER;
};

//Return number of columns in the data file
int RandomForest::getNumberOfColumns() {
	return RandomForest::COLUMN_NUMBER;
};

//Return all records loaded from the data file
std::vector<std::vector<std::string>> RandomForest::getAllRecords () {
    return allRecords;
}

//Return all nodes for the columns in the data file
std::vector<Node*> RandomForest::getAllNodes() {
    return allNodes;
}


//Get bootstrap dataset to build the RandomForest
std::vector<std::vector<std::string>> RandomForest::getBootStrapRecords (char *fileName, char splitChar, 
    int numberOfRandomNumbers, float **randomNumbers) {

    std::vector<std::vector<std::string>> bootStrapRecords;

    if (DEBUG)
        for (int i=0; i<numberOfRandomNumbers; i++)
            std::cout << "randomNumbers[" << i << "]=" << (*randomNumbers)[i] << std::endl;
    
    std::vector<float> bootStrapRecordsRandomId(*randomNumbers, *randomNumbers + numberOfRandomNumbers);
    
    int i = 0;
    for (float &element : bootStrapRecordsRandomId) {
        
        //Ignore headers
        if ((int)element != 0) {
            std::string recordId = std::to_string((int)element);
               
            for (std::vector<std::string> record : allRecords) {
                if (DEBUG) {
                    std::cout << "recordId=" << recordId << std::endl;
                    std::cout << "record[0]=" << record[0] << std::endl;
                }
                if (record[0] == recordId) {
                    bootStrapRecords.push_back(record);
                    if (DEBUG)
                        std::cout << "Matching record with record[0]=" << record[0] << std::endl;
                }
            }
        }

        if (DEBUG) {
            std::cout << "bootStrapRecordsRandomId[" << i++ << "]=" << element << std::endl;
        }
    }
    
    return bootStrapRecords;

}