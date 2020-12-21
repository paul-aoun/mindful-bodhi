//////////////////////////////////////////////////////
// Paul Aoun										//
//													//
// Johns Hopkins University							//				
//													//
// 11/15/2020										//
//													//
// This is the file for the Node class.            	//
// It defines the functions needed for the Nodes.   //
//////////////////////////////////////////////////////

#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>

#ifndef Node_H
#define Node_H

#define DEBUG false

//Class to provide all the needed Node functions.
class Node {

private:

    int id{0}, columnNumber{0};
    Node *left;
    Node *right;
    std::string name{""};
    float giniImpurity{0.0f};
    bool selectedColumn{false};
    std::vector<std::vector<std::string>> records;

public:

   //Contructor for Node class. Nodes are the building blocks for the Tree.
    Node (int nodeId);

    //Contructor for Node to be used for Binary Trees.
    Node (int nodeId, Node *leftNode, Node *rightNode);

    //Return the ID of the node
    int getId();

    //Set the ID of the node
    void setId(int nodeId);

    //Return the left branch of a binary tree used in Random Forest   
    Node getLeftNode ();

    //Set the left branch of a binary tree used in Random Forest  
    void setLeftNode (Node *leftNode);

    //Return the right branch of a binary tree used in Random Forest   
    Node getRightNode ();

    //Set the right branch of a binary tree used in Random Forest  
    void setRightNode (Node *rightNode);

    //Get the name (column name) of the node
    std::string getName();

    //Set the name (column name) of the node
    void setName(std::string aName);

    //Get the number (column number) of the node
    int getColumnNumber();

    //Get the number (column number) of the node
    void setColumnNumber(int aValue);

    //Get the Gini impurity of the node
    float getGiniImpurity();

    //Set the Gini impurity of the node
    void setGiniImpurity(float aValue);

    //Return the selectedColumn value
    //1=selected, 0=not selected
    bool getSelectedColumn();

    //Set the selectedColumn value
    //1=selected, 0=not selected
    void setSelectedColumn(bool aValue);

    //Return the records associated with the node
    std::vector<std::vector<std::string>> getRecords();

    //Set the records associated with the node
    void setRecords(std::vector<std::vector<std::string>> records);

};

#endif