#include "imc/MultilayerPerceptron.h"
#include <iostream>
int main(int argc, char **argv)
{
  imc::MultilayerPerceptron* mlp = new imc::MultilayerPerceptron();
  int *structure = new int[4];
  double* input = new double[2];
  input[0]=2.1;
  input[1]=1.8;
  structure[0] = 2;
  structure[1] = 3;
  structure[2] = 5;
  structure[3] = 2;
  mlp->initialize(4, structure);
  
  mlp->show();
  mlp->addInput(input);

  delete mlp;
  delete[] structure; 
  delete[] input;

}