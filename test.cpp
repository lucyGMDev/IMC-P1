#include "imc/MultilayerPerceptron.h"
#include <iostream>
int main(int argc, char **argv)
{
  imc::MultilayerPerceptron mlp;
  int numCapaOculta=1;
  int* estructura = new int[numCapaOculta+2];
  estructura[0] = 2;
  estructura[1] = 3;
  estructura[2] = 1;
  mlp.initialize(numCapaOculta+2,estructura);
  
  imc::Dataset* trainData = mlp.readData("trainDatasets/train_xor.dat");
  imc::Dataset* testData = mlp.readData("testDatasets/test_xor.dat");
  double* errorTrain = new double;
  double* errorTest = new double;
  mlp.runOnlineBackPropagation(trainData,testData,1000,errorTrain,errorTest);

  delete errorTrain;
  delete errorTest;  
}