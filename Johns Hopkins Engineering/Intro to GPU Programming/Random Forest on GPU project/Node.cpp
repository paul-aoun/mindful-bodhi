//////////////////////////////////////////////////////
// Paul Aoun										//
//													//
// Johns Hopkins University							//				
//													//
// 11/15/2020										//
//													//
// This is the file for the Node class.            	//
// It implements the functions needed for the Nodes.//
// Nodes are the building blocks for the Trees.     //
//////////////////////////////////////////////////////
 
 #include "Node.h"

//Contructor for Node class. Nodes are the building blocks for the Tree.
Node::Node (int nodeId) {
    id = nodeId;
}

//Contructor for Node to be used for Binary Trees.
Node::Node (int nodeId, Node *leftNode, Node *rightNode) {
    id = nodeId;
    left = leftNode;
    right = rightNode;
}

//Return the ID of the node
int Node::getId() {
    return id;
}

//Set the ID of the node
void Node::setId(int nodeId) {
    id = nodeId;
}

//Return the left branch of a binary tree used in Random Forest   
Node Node::getLeftNode () {
    return *left;
}

//Set the left branch of a binary tree used in Random Forest  
void Node::setLeftNode (Node *leftNode) {
    left = leftNode;
}

//Return the right branch of a binary tree used in Random Forest   
Node Node::getRightNode () {
    return *right;
}

//Set the right branch of a binary tree used in Random Forest  
void Node::setRightNode (Node *rightNode) {
    right = rightNode;
}

//Get the name (column name) of the node
std::string Node::getName() {
    return name;
}

//Set the name (column name) of the node
void Node::setName(std::string aName) {
    name=aName;
}

//Get the number (column number) of the node
int Node::getColumnNumber() {
    return columnNumber;
}

//Get the number (column number) of the node
void Node::setColumnNumber(int aValue) {
    columnNumber = aValue;
}

//Get the Gini impurity of the node
float Node::getGiniImpurity() {
    return giniImpurity;
}

//Set the Gini impurity of the node
void Node::setGiniImpurity(float aValue) {
    giniImpurity = aValue;
}

//Return the selectedColumn value
//1=selected, 0=not selected
bool Node::getSelectedColumn() {
    return selectedColumn;
}

//Set the selectedColumn value
//1=selected, 0=not selected
void Node::setSelectedColumn(bool aValue) {
    selectedColumn = aValue;
}

//Return the records associated with the node
std::vector<std::vector<std::string>> Node::getRecords(){
    return records;
}
//Set the records associated with the node
void Node::setRecords(std::vector<std::vector<std::string>> records) {
    Node::records = records;
}