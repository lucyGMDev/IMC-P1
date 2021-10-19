/*********************************************************************
 * File  : MultilayerPerceptron.cpp
 * Date  : 2020
 *********************************************************************/

#include "MultilayerPerceptron.h"

#include "util.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()
#include <limits>
#include <math.h>

using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
}

// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[])
{
	srand(time(NULL));
	this->nOfLayers = nl;
	this->layers = new Layer[this->nOfLayers];
	this->eta = 0.1;

	for (int i = 0; i < nl; i++)
	{
		this->layers[i].nOfNeurons = npl[i];
		this->layers[i].neurons = new Neuron[npl[i]];
		for (int j = 0; j < npl[i]; j++)
		{
			if (i > 0) // Inicializo los los pesos de las neuronas (la primera capa no tiene neuronas)
			{
				int numWeightInputs = npl[i - 1] + 1; //+1 por el sesgo
				this->layers[i].neurons[j].w = new double[numWeightInputs];
				this->layers[i].neurons[j].wCopy = new double[numWeightInputs];
				this->layers[i].neurons[j].deltaW = new double[numWeightInputs];
				this->layers[i].neurons[j].lastDeltaW = new double[numWeightInputs];
				for (int k = 0; k < numWeightInputs; k++)
				{
					this->layers[i].neurons[j].w[k] = (double)((rand()) % 201 / (double)100) - 1;
					this->layers[i].neurons[j].wCopy[k] = 0;
					this->layers[i].neurons[j].deltaW[k] = 0;
					this->layers[i].neurons[j].lastDeltaW[k] = 0;
				}
			}
			else
			{
				this->layers[i].neurons[j].w = nullptr;
				this->layers[i].neurons[j].wCopy = nullptr;
				this->layers[i].neurons[j].deltaW = nullptr;
				this->layers[i].neurons[j].lastDeltaW = nullptr;
			}
			this->layers[i].neurons[j].out = 0;
			this->layers[i].neurons[j].delta = 0;
		}
	}

	return 1;
}

// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron()
{
	freeMemory();
}

// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory()
{
	for (int i = 0; i < nOfLayers; i++)
	{
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			layers[i].neurons[j].out = 0;
			layers[i].neurons[j].delta = 0;
			delete[] layers[i].neurons[j].w;
			layers[i].neurons[j].w = nullptr;
			delete layers[i].neurons[j].deltaW;
			layers[i].neurons[j].deltaW = nullptr;
			delete layers[i].neurons[j].lastDeltaW;
			layers[i].neurons[j].lastDeltaW = nullptr;
			delete[] layers[i].neurons[j].wCopy;
			layers[i].neurons[j].wCopy = nullptr;
		}
		delete[] layers[i].neurons;
		layers[i].neurons = nullptr;
		layers[i].nOfNeurons = 0;
	}
	delete[] layers;
	layers = nullptr;
	nOfLayers = 0;
}

// ------------------------------
// Feel all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights()
{
	srand(time(NULL));
	for (int i = 1; i < nOfLayers - 1; i++)
	{
		for (int j = 0; j < this->layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < this->layers[i - 1].nOfNeurons; k++)
			{
				this->layers[i].neurons[j].w[k] = (double)((rand()) % 201 / (double)100) - 1;
			}
		}
	}
}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double *input)
{
	int inputLength = sizeof(input) / sizeof(double);
	for (int i = 0; i < this->layers[0].nOfNeurons; i++)
	{
		this->layers[0].neurons[i].out = input[i];
	}
}

// ------------------------------
// Get the outputs predicted by the network (out vector the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double *output)
{
	for (int i = 0; i < this->layers[nOfLayers - 1].nOfNeurons; i++)
	{
		output[i] = this->layers[nOfLayers - 1].neurons[i].out;
	}
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights()
{
	// TODO Check Function
	for (int i = 1; i < this->nOfLayers; i++)
	{
		for (int j = 0; j < this->layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < this->layers[i - 1].nOfNeurons; k++)
			{
				this->layers[i].neurons[j].wCopy[k] = this->layers[i].neurons[j].w[k];
			}
		}
	}
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights()
{
	for (int i = 1; i < this->nOfLayers; i++)
	{
		for (int j = 0; j < this->layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < this->layers[i - 1].nOfNeurons; k++)
			{
				this->layers[i].neurons[j].w[k] = this->layers[i].neurons[j].wCopy[k];
			}
		}
	}
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate()
{
	for (int i = 1; i < this->nOfLayers; i++)
	{
		for (int j = 0; j < this->layers[i].nOfNeurons; j++)
		{
			double sum = 0;
			int k;
			for (k = 0; k < this->layers[i - 1].nOfNeurons; k++)
			{

				sum += this->layers[i - 1].neurons[k].out * this->layers[i].neurons[j].w[k];
			}
			sum += this->layers[i].neurons[j].w[k];
			this->layers[i].neurons[j].out = 1 / (1 + exp(-sum));
		}
	}
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
double MultilayerPerceptron::obtainError(double *target)
{
	int outputSize = this->layers[nOfLayers - 1].nOfNeurons;
	double *outputs = new double[outputSize];
	this->getOutputs(outputs);
	double error = 0;
	for (int i = 0; i < outputSize; i++)
	{

		error += pow(target[i] - outputs[i], 2);
	}
	error /= outputSize;
	return error;
}

// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
void MultilayerPerceptron::backpropagateError(double *target)
{
	for (int i = nOfLayers - 1; i > 0; i--)
	{
		if (i == nOfLayers - 1)
		{
			for (int j = 0; j < this->layers[i].nOfNeurons; j++)
			{
				double output = this->layers[i].neurons[j].out;
				this->layers[i].neurons[j].delta = (target[j] - output) * output * (1 - output);
			}
		}
		else
		{
			for (int j = 0; j < this->layers[i].nOfNeurons; j++)
			{
				double sumDeltaWeight = 0;
				double output = this->layers[i].neurons[j].out;
				for (int k = 0; k < this->layers[i + 1].nOfNeurons; k++)
				{
					sumDeltaWeight += this->layers[i + 1].neurons[k].delta * this->layers[i + 1].neurons[k].w[j];
				}
				this->layers[i].neurons[j].delta = sumDeltaWeight * output * (1 - output);
			}
		}
	}
}

// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange()
{
	for (int i = nOfLayers - 1; i > 0; i--)
	{
		for (int j = 0; j < this->layers[i].nOfNeurons; j++)
		{
			int k;
			for (k = 0; k < this->layers[i - 1].nOfNeurons; k++)
			{
				this->layers[i].neurons[j].deltaW[k] = this->layers[i].neurons[j].delta * this->layers[i - 1].neurons[k].out;
			}
			this->layers[i].neurons[j].deltaW[k] = this->layers[i].neurons[j].delta;
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment()
{
	for (int i = 1; i < nOfLayers; i++)
	{
		for (int j = 0; j < this->layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < this->layers[i - 1].nOfNeurons + 1; k++)
			{
				double learningRate = this->eta * pow(this->decrementFactor, -(this->nOfLayers - i));
				this->layers[i].neurons[j].w[k] += this->layers[i].neurons[j].deltaW[k] * learningRate + this->mu * this->layers[i].neurons[j].lastDeltaW[k] * learningRate;

				this->layers[i].neurons[j].lastDeltaW[k] = this->layers[i].neurons[j].deltaW[k];
			}
		}
	}
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork()
{
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
void MultilayerPerceptron::performEpochOnline(double *input, double *target)
{
	feedInputs(input);
	forwardPropagate();
	backpropagateError(target);
	accumulateChange();
	weightAdjustment();
}

// ------------------------------
// Read a dataset from a file name and return it
Dataset *MultilayerPerceptron::readData(const char *fileName)
{

	ifstream myFile(fileName); // Create an input stream

	if (!myFile.is_open())
	{
		cout << "ERROR: I cannot open the file " << fileName << endl;
		return NULL;
	}

	Dataset *dataset = new Dataset;
	if (dataset == NULL)
		return NULL;

	string line;
	int i, j;

	if (myFile.good())
	{
		getline(myFile, line); // Read a line
		istringstream iss(line);
		iss >> dataset->nOfInputs;
		iss >> dataset->nOfOutputs;
		iss >> dataset->nOfPatterns;
	}
	dataset->inputs = new double *[dataset->nOfPatterns];
	dataset->outputs = new double *[dataset->nOfPatterns];

	for (i = 0; i < dataset->nOfPatterns; i++)
	{
		dataset->inputs[i] = new double[dataset->nOfInputs];
		dataset->outputs[i] = new double[dataset->nOfOutputs];
	}

	i = 0;
	while (myFile.good())
	{
		getline(myFile, line); // Read a line
		if (!line.empty())
		{
			istringstream iss(line);
			for (j = 0; j < dataset->nOfInputs; j++)
			{
				double value;
				iss >> value;
				if (!iss)
					return NULL;
				dataset->inputs[i][j] = value;
			}
			for (j = 0; j < dataset->nOfOutputs; j++)
			{
				double value;
				iss >> value;
				if (!iss)
					return NULL;
				dataset->outputs[i][j] = value;
			}
			i++;
		}
	}

	myFile.close();

	return dataset;
}

// ------------------------------
// Perform an online training for a specific trainDataset
void MultilayerPerceptron::trainOnline(Dataset *trainDataset)
{
	int i;
	for (i = 0; i < trainDataset->nOfPatterns; i++)
	{
		performEpochOnline(trainDataset->inputs[i], trainDataset->outputs[i]);
	}
}

// ------------------------------
// Test the network with a dataset and return the MSE
double MultilayerPerceptron::test(Dataset *testDataset)
{
	double sumMSE = 0;
	for (int i = 0; i < testDataset->nOfPatterns; i++)
	{
		double *outputs = new double[this->layers[this->nOfLayers - 1].nOfNeurons];
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		sumMSE += getError(testDataset->outputs[i]);
	}
	return sumMSE / testDataset->nOfPatterns;
}

// Optional - KAGGLE
// Test the networreturn -1.0;k with a dataset and return the MSE
// Your have to use the format from Kaggle: two columns (Id y predictied)
void MultilayerPerceptron::predict(Dataset *pDatosTest)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers - 1].nOfNeurons;
	double *obtained = new double[numSalidas];

	cout << "Id,Predicted" << endl;

	for (i = 0; i < pDatosTest->nOfPatterns; i++)
	{

		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(obtained);

		cout << i;

		for (j = 0; j < numSalidas; j++)
			cout << "," << obtained[j];
		cout << endl;
	}
}

// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
void MultilayerPerceptron::runOnlineBackPropagation(Dataset *trainDataset, Dataset *pDatosTest, int maxiter, double *errorTrain, double *errorTest)
{
	cerr << this->validationRatio << std::endl;
	int countTrain = 0;
	int iterWithoutImprovingValidation = 0;
	// Random assignment of weights (starting point)
	randomWeights();
	double minTrainError = 0;
	int iterWithoutImproving = 0;
	double testError = 0;

	double validationError = 1;
	double lastValidationError = 1;

	bool useValidation = false;
	Dataset *validationDataset;
	// Generate validation data
	if (validationRatio > 0 && validationRatio < 1)
	{
		useValidation = true;
		validationDataset = new Dataset();
		int validationSize = validationRatio * trainDataset->nOfPatterns;
		validationDataset->nOfPatterns = validationSize;
		validationDataset->nOfInputs = trainDataset->nOfInputs;
		validationDataset->nOfOutputs = trainDataset->nOfOutputs;
		validationDataset->inputs = new double *[validationDataset->nOfPatterns];
		validationDataset->outputs = new double *[validationDataset->nOfPatterns];

		cerr << validationSize << endl;
		cerr << trainDataset->nOfPatterns - validationSize << endl;

		for (int i = 0; i < validationSize; i++)
		{
			validationDataset->inputs[i] = new double[validationDataset->nOfInputs];
			validationDataset->outputs[i] = new double[validationDataset->nOfOutputs];
			for (int j = 0; j < validationDataset->nOfInputs; j++)
			{
				validationDataset->inputs[i][j] = trainDataset->inputs[trainDataset->nOfPatterns - validationSize + i][j];
			}
			for (int j = 0; j < validationDataset->nOfOutputs; j++)
			{
				validationDataset->outputs[i][j] = trainDataset->inputs[trainDataset->nOfPatterns - validationSize + i][j];
			}
		}
		trainDataset->nOfPatterns = trainDataset->nOfPatterns - validationSize;
	}

	// Learning
	do
	{

		trainOnline(trainDataset);
		double trainError = test(trainDataset);
		
		if (countTrain == 0 || trainError < minTrainError)
		{
			minTrainError = trainError;
			copyWeights();
			iterWithoutImproving = 0;
		}
		else if ((trainError - minTrainError) < 0.00001)
			iterWithoutImproving = 0;
		else
			iterWithoutImproving++;
		if (iterWithoutImproving == 50)
		{
			cout << "We exit because the training is not improving!!" << endl;
			restoreWeights();
			countTrain = maxiter;
		}

		if (useValidation)
		{
			validationError = test(validationDataset);
			if (validationError < lastValidationError)
			{
				iterWithoutImprovingValidation = 0;
			}
			else if (validationError - lastValidationError < 0.00001)
			{
				iterWithoutImprovingValidation = 0;
			}
			else
			{
				iterWithoutImprovingValidation++;
			}

			if (iterWithoutImprovingValidation == 50)
			{
				cout << "We exit because the validation is not improving!!" << endl;
				restoreWeights();
				countTrain = maxiter;
			}

			lastValidationError = validationError;
		}
		// cerr<<countTrain<<endl;
		countTrain++;

		// Check validation stopping condition and force it
		// BE CAREFUL: in this case, we have to save the last validation error, not the minimum one
		// Apart from this, the way the stopping condition is checked is the same than that
		// applied for the training set

		// cout << "Iteration " << countTrain << "\t Training error: " << trainError << "\t Validation error: " << validationError << endl;

	} while (countTrain < maxiter);

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for (int i = 0; i < pDatosTest->nOfPatterns; i++)
	{
		double *prediction = new double[pDatosTest->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		// for (int j = 0; j < pDatosTest->nOfOutputs; j++)
		// 	cout << pDatosTest->outputs[i][j] << " -- " << prediction[j] << " ";
		// cout << endl;
		delete[] prediction;
	}

	testError = test(pDatosTest);
	*errorTest = testError;
	*errorTrain = minTrainError;
}

// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char *archivo)
{
	// Object for writing the file
	ofstream f(archivo);

	if (!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for (int i = 0; i < nOfLayers; i++)
		f << " " << layers[i].nOfNeurons;
	f << endl;

	// Write the weight matrix of every layer
	for (int i = 1; i < nOfLayers; i++)
		for (int j = 0; j < layers[i].nOfNeurons; j++)
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
				f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;
}

// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char *archivo)
{
	// Object for reading a file
	ifstream f(archivo);

	if (!f.is_open())
		return false;

	// Number of layers and number of neurons in every layer
	int nl;
	int *npl;

	// Read number of layers
	f >> nl;

	npl = new int[nl];

	// Read number of neurons in every layer
	for (int i = 0; i < nl; i++)
		f >> npl[i];

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for (int i = 1; i < nOfLayers; i++)
		for (int j = 0; j < layers[i].nOfNeurons; j++)
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
				f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
