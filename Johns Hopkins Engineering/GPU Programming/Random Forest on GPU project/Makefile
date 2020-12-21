all: project.cu
	g++ -c RandomForest.cpp Node.cpp
	nvcc -G -g project.cu -lcudart -lcuda -lcurand -o project.exe ./RandomForest.o ./Node.o

