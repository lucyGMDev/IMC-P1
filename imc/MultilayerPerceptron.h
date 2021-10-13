/*********************************************************************
* File  : MultilayerPerceptron.cpp
* Date  : 2020
*********************************************************************/

#ifndef _MULTILAYERPERCEPTRON_H_
#define _MULTILAYERPERCEPTRON_H_
#include <iostream>
namespace imc
{

	// Suggested structures
	// ---------------------
	struct Neuron
	{
		double out;					/* Output produced by the neuron (out_j^h)*/
		double delta;				/* Derivative of the output produced by the neuron (delta_j^h)*/
		double *w;					/* Input weight vector (w_{ji}^h)*/
		double *deltaW;			/* Change to be applied to every weight (\Delta_{ji}^h (t))*/
		double *lastDeltaW; /* Last change applied to the every weight (\Delta_{ji}^h (t-1))*/
		double *wCopy;			/* Copy of the input weights */
	};

	struct Layer
	{
		int nOfNeurons;	 /* Number of neurons of the layer*/
		Neuron *neurons; /* Vector with the neurons of the layer*/
	};

	struct Dataset
	{
		int nOfInputs;		/* Number of inputs */
		int nOfOutputs;		/* Number of outputs */
		int nOfPatterns;	/* Number of patterns */
		double **inputs;	/* Matrix with the inputs of the problem */
		double **outputs; /* Matrix with the outputs of the problem */
	};

	class MultilayerPerceptron
	{
	private:
		int nOfLayers; /* Total number of layers in the network */
		Layer *layers; /* Vector containing every layer */

		// Free memory for the data structures
		void freeMemory();

		// Feel all the weights (w) with random numbers between -1 and +1
		void randomWeights();

		// Feed the input neurons of the network with a vector passed as an argument
		void feedInputs(double *input);

		// Get the outputs predicted by the network (out vector the output layer) and save them in the vector passed as an argument
		void getOutputs(double *output);

		// Make a copy of all the weights (copy w in wCopy)
		void copyWeights();

		// Restore a copy of all the weights (copy wCopy in w)
		void restoreWeights();

		// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
		void forwardPropagate();

		// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
		double obtainError(double *target);

		// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
		void backpropagateError(double *target);

		// Accumulate the changes produced by one pattern and save them in deltaW
		void accumulateChange();

		// Update the network weights, from the first layer to the last one
		void weightAdjustment();

		// Print the network, i.e. all the weight matrices
		void printNetwork();

		// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
		// input is the input vector of the pattern and target is the desired output vector of the pattern
		void performEpochOnline(double *input, double *target);

	public:
		// Values of the parameters (they are public and can be updated from outside)
		double eta;							// Learning rate
		double mu;							// Momentum factor
		double validationRatio; // Ratio of training patterns used as validation (e.g.
														// if validationRatio=0.2, a 20% of the training patterns
														// are used for validation; if validationRatio=0, there is no validation)
		double decrementFactor; // Decrement factor used for eta in the different layers

		// Constructor: Default values for all the parameters
		MultilayerPerceptron();

		// DESTRUCTOR: free memory
		~MultilayerPerceptron();

		// Allocate memory for the data structures
		// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
		// Give values to Layer* layers
		int initialize(int nl, int npl[]);

		// Read a dataset from a file name and return it
		Dataset *readData(const char *fileName);

		// Test the network with a dataset and return the MSE
		double test(Dataset *dataset);

		// Obtain the predicted outputs for a dataset
		void predict(Dataset *testDataset);

		// Perform an online training for a specific dataset
		void trainOnline(Dataset *trainDataset);

		// Run the traning algorithm for a given number of epochs, using trainDataset
		// Once finished, check the performance of the network in testDataset
		// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
		void runOnlineBackPropagation(Dataset *trainDataset, Dataset *testDataset, int maxiter, double *errorTrain, double *errorTest);

		// Optional Kaggle: Save the model weights in a textfile
		bool saveWeights(const char *archivo);

		// Optional Kaggle: Load the model weights from a textfile
		bool readWeights(const char *archivo);

		//Function to testing
		void show()
		{
			std::cout << "Mostrando neurona" << std::endl;
			std::cout << "Tiene " << this->nOfLayers << " capas" << std::endl;
			for (int i = 0; i < this->nOfLayers; i++)
			{
				std::cout << "Mostrando capa: " << i << std::endl;
				std::cout << "Esta capa tiene: " << this->layers[i].nOfNeurons << " neuronas" << std::endl;
				if (i > 0)
				{
					for (int j = 0; j < this->layers[i].nOfNeurons; j++)
					{
						std::cout << "Mostrando pesos de la neurona " << j << " de la capa " << i << std::endl;
						for (int k = 0; k < this->layers[i - 1].nOfNeurons + 1; k++)
						{
							std::cout << this->layers[i].neurons[j].w[k] << std::endl;
						}
					}
				}
			}
		}
		void showOutput()
		{
			double *output = new double[this->layers[this->nOfLayers - 1].nOfNeurons];
			this->getOutputs(output);
			std::cout << "Output: " << std::endl;
			for (int i = 0; i < this->layers[this->nOfLayers - 1].nOfNeurons; i++)
			{
				std::cout << output[i] << std::endl;
			}
		}
		void showDelta()
		{
			int cont = 1;
			for (int i = 1; i < nOfLayers; i++)
			{
				for (int j = 0; j < this->layers[i].nOfNeurons; j++)
				{
					std::cout << "Delta " << cont << ": " << this->layers[i].neurons[j].delta << std::endl;
					cont++;
				}
			}
		}
		void addInput(double *input) { this->feedInputs(input); }
		void showInputs()
		{
			for (int i = 0; i < this->layers[0].nOfNeurons; i++)
			{
				std::cout << this->layers[0].neurons[i].out << std::endl;
			}
		}
		void computeOutput()
		{
			this->forwardPropagate();
		}
		void computeDelta(double *target)
		{
			this->backpropagateError(target);
		}
		double getError(double *target)
		{
			return this->obtainError(target);
		}
		void AccumulateChange()
		{
			this->accumulateChange();
		}
		void seeDeltaW()
		{
			for (int i = 1; i < nOfLayers; i++)
			{
				for (int j = 0; j < this->layers[i].nOfNeurons; j++)
				{
					for (int k = 0; k < this->layers[i - 1].nOfNeurons+1; k++)
					{
						std::cout << "Capa " << i << " conexión neurona " << j << " con neurona " << k << " de la capa " << i - 1 << ": " << this->layers[i].neurons[j].deltaW[k] << std::endl;
					}
				}
			}
		}
		//End function to testing
	};
};

#endif
