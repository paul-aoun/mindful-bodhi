//////////////////////////////////////////////////////
// Paul Aoun										//
//													//
// Johns Hopkins University							//				
//													//
// 12/13/2020										//
//													//
// This is the main source code file.            	//
// It has the logic to setup, initialize, and       //
// generate the Random Forest using both the host   //
// and device for the implementation.               //
//////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>

#include "project.cuh"
#include "RandomForest.h"
#include "Node.h"

//Data file to generate the tree for the random forest
#define FILE_NAME "spy.us.short.csv"
//Comma delimited CSV file
//Split charater to read the data file. 44 is a "," comma.
#define SPLIT_CHAR 44

//Decision column in the data file based on which the Gini impurities will be calculated
#define DECISION_COLUMN 20 

#define START_COLUMN 10 //first column in the data file to be used in the Trees
#define MAX_RANDOM_NUMBERS 1000 //used to get the column numbers between MIN and MAX
#define MIN_COLUMN_NUMBER 10 // lowest column number to be included in the tree building
#define NUMBER_OF_TREES 5 //Number of trees per forest


// Flag are Pageable, Page-locked or Portable host memory [0, 1, or 2]
#define FLAG 0

//Generate a Tree for the Random Forest, using recursion and depth-first algorithm
Node* generateTree (int decisionColumn, std::vector<Node*>&nodes, 
    int numberOfNodes, int numberOfSelectedNodes, Node *rootNode) {

    if (DEBUG) 
    {
        std::cout << "generateTree: decisionColumn=" << decisionColumn << std::endl;
    }

    //Initialize the impurity with 2 (impossible) since the Gini values are between 0 and 1.
    rootNode->setGiniImpurity(2.0f);
    std::vector<std::vector<std::string>> records = rootNode->getRecords();

    //Find root node with the lowest Gini impurity
    int nodeCounter1{0};   
    for (Node *node : nodes) { 
        if (rootNode->getGiniImpurity() > node->getGiniImpurity()) {
            rootNode = node;
            if (DEBUG)
                std::cout << "generateTree: Temp root node for Tree is column " << rootNode->getColumnNumber() 
                    << " with Gini impurity " << rootNode->getGiniImpurity() << std::endl;
        }
    }

    //Prepare the record for the left side and right side of the binary tree
    std::vector<std::vector<std::string>> leftRecords; 
    std::vector<std::vector<std::string>> rightRecords; 

    //Remove column data from the rows
    for (std::vector<std::string> &row : records) {
        leftRecords.push_back(row);
        rightRecords.push_back(row);
    }

    std::cout << "generateTree: Final root node column number is " << rootNode->getColumnNumber() 
            << " with Gini impurity of " << rootNode->getGiniImpurity() << std::endl;
   
    
    //Recursive depth-first generation of the tree
    if (nodes.size() > numberOfNodes - numberOfSelectedNodes + 1) 
    {
        //Remove current node from the vector and reset Gini impurities for the nodes
        int nodeCounter{0};
        for (Node *tempNode : nodes) {
            if (tempNode->getColumnNumber() == rootNode->getColumnNumber()) 
            {
                nodes.erase(nodes.begin() + nodeCounter);
                if (DEBUG)
                    std::cout <<"generateTree: set node=" << rootNode->getColumnNumber() 
                        << " in Tree." <<std::endl; 
            }
            else
            {
                tempNode->setGiniImpurity(2.0f);
            }
            nodeCounter++;
        }
        
        if (DEBUG)
            for (Node *node: nodes) {
                std::cout << "generateTree: Node column number=" << node->getColumnNumber() << std::endl;
            }
        // Calculate Gini impurities for the selected columns and store in the nodes
        if (nodes.size() > numberOfNodes - numberOfSelectedNodes + 1) {

            //Calculate right-side Gini impurities for the selected columns and store in the nodes
            calculateColumnGini(decisionColumn, leftRecords, nodes);
            Node *leftNode = new Node(rootNode->getId() + 1);
            leftNode->setRecords(leftRecords);
            //Recursively call the generateTree to continue building it
            rootNode->setLeftNode(generateTree (decisionColumn, nodes, numberOfNodes, numberOfSelectedNodes, leftNode));

            //Calculate right-side Gini impurities for the selected columns and store in the nodes
            calculateColumnGini(decisionColumn, rightRecords, nodes);
            Node *rightNode = new Node (rootNode->getId() + 1);
            //Recursively call the generateTree to continue building it
            rightNode->setRecords(rightRecords);
            rootNode->setRightNode(generateTree (decisionColumn, nodes, numberOfNodes, numberOfSelectedNodes, rightNode));
        } 
    } else 
    {//Leaf node and end of the recursion
        std::cout << "generateTree: recursion completed and Tree generated." << std::endl;
    }
    
    return rootNode;
}

//Setup root node for the tree and call generateTree to recursively build the tree
void generateTreeControl (RandomForest randomForest, int numberOfTrees, int decisionColumn) {

    std::vector<std::vector<std::string>> bootStrapRecords;

    float *randomNumbers;
    int numberOfRows = randomForest.getNumberOfRows();
    int numberOfColumns = randomForest.getNumberOfColumns();

    //Allocate and initialize host arrays
    allocateArraysHost (numberOfRows, &randomNumbers, FLAG);
    randomInitializeArray(numberOfRows, &randomNumbers, numberOfColumns);

    //Get the bootstrapRecords from the Forest object
    bootStrapRecords = 
        randomForest.getBootStrapRecords (FILE_NAME, SPLIT_CHAR, numberOfRows, &randomNumbers);

 
    //Release the random numbers array
    freeArraysHost(&randomNumbers, FLAG);
    
    std::vector<Node*> nodes = randomForest.getAllNodes();

    //Get the columns to use in generating the tree
    selectColumns(MIN_COLUMN_NUMBER, numberOfColumns, DECISION_COLUMN, MAX_RANDOM_NUMBERS, 
        FLAG, nodes);

    int numberOfSelectedNodes{0};
    if (DEBUG)
        for (Node *node : nodes) 
        {
            std::cout << "generateRandomForest: Column " << node->getColumnNumber() 
                << " selected=" << node->getSelectedColumn() << std::endl;
                if (node->getSelectedColumn() == 1)
                    numberOfSelectedNodes++;
        }
    
    // Calculate Gini impurities for the selected columns and store in the nodes
    calculateColumnGini(DECISION_COLUMN, bootStrapRecords, nodes);
   
    if (DEBUG)
        for (Node *node : nodes) {
            std::cout << "Gini impurity for column " << node->getColumnNumber() 
                << " is " << node->getGiniImpurity() << std::endl;
        }

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    //Root node for the Tree
    Node *node = new Node(0);
    node->setRecords(bootStrapRecords);
    //Call generateTree with the root node for the Tree
    Node *rootNode = generateTree(DECISION_COLUMN, nodes, 
        numberOfColumns, numberOfSelectedNodes, node);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);	
    std::cout << "Tree generation completed in "<< time_span.count() << "ms." << std::endl;

}

//Generate the number of Trees for the Random Forest based on numberOfTrees
//Decision column is used to calculate the Gini impurities
void generateRandomForest (int numberOfTrees, int decisionColumn) 
{
    std::cout << std::fixed;

    //Create RandomForest object. The constructor will load the data in a vector of string
    RandomForest randomForest (FILE_NAME, SPLIT_CHAR);

    //Repeatedly generate a tree up to numberOfTrees
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    for (int i=0; i<numberOfTrees; i++) 
    {
        std::cout << "Tree number " << i << " generation started." << std::endl;
        generateTreeControl(randomForest, numberOfTrees, decisionColumn);
        std::cout << "Tree number " << i << " generation completed." << std::endl;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);	
    std::cout << "Random Forest of " << numberOfTrees << " completed in "<< time_span.count() << "ms." << std::endl;

}

int main () {

    std::cout << "Welcome to Intro to GPU - Project." << std::endl;

    //Generate the Random Forst with the required number of Trees,
    // and based on the decision column
    generateRandomForest(NUMBER_OF_TREES, DECISION_COLUMN);

    /* Reset device after being done with all the functions */
	cudaDeviceReset();    
    
    return 0;
}