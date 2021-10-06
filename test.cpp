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
  structure[2] = 1;
  mlp->initialize(3, structure);

  mlp->CopyWeights();
  mlp->TestCopyWeights();
  

  delete mlp;
  delete[] structure;
  delete[] input;
}