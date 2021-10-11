#include "imc/MultilayerPerceptron.h"
#include <iostream>
int main(int argc, char **argv)
{
  imc::MultilayerPerceptron *mlp = new imc::MultilayerPerceptron();
  int *structure = new int[3];
  double *input = new double[2];
  input[0] = 2.1;
  input[1] = 1.8;
  structure[0] = 2;
  structure[1] = 1;
  structure[2] = 2;
  mlp->initialize(3, structure);
  double *target = new double[2];
  target[0] = 2;
  target[1] = 3;
  mlp->addInput(input);
  mlp->computeOutput();
  double error = mlp->getError(target);
  std::cout<<"Error: "<<error<<std::endl;
  delete mlp;
  delete[] structure;
  delete[] input;
}