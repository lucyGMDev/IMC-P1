#include "imc/MultilayerPerceptron.h"
#include <iostream>
int main(int argc, char **argv)
{
  imc::MultilayerPerceptron mlp;
  int *structure = new int[4];
  structure[0] = 2;
  structure[1] = 3;
  structure[2] = 4;
  structure[3] = 2;
  std::cerr << "Structura está bien creada" << std::endl;
  mlp.initialize(4, structure);
  mlp.show();
}