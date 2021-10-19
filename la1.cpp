//============================================================================
// Introduction to computational models
// Name        : la1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>   // To obtain current time time()
#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()
#include <string.h>
#include <math.h>

#include "imc/MultilayerPerceptron.h"

using namespace imc;
using namespace std;

int main(int argc, char **argv)
{
    // Process arguments of the command line
    bool lflag = 0, Tflag = 0, wflag = 0, pflag = 0, tflag = 0, iflag = 0, hflag = 0, eflag = 0, mflag = 0, vflag = 0;
    char *Tvalue = NULL, *wvalue = NULL, *tvalue = NULL, dflag = 0;

    int c, ivalue = 0, lvalue = 0, hvalue = 0;
    double evalue = 0, mvalue = 0, vvalue = 0, dvalue = 0;

    opterr = 0;

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "t:r:T::w:pi:l:h:e:m:v:d:")) != -1)
    {
        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
        switch (c)
        {
        case 'T':
            Tflag = true;
            Tvalue = optarg;
            break;
        case 'w':
            wflag = true;
            wvalue = optarg;
            break;
        case 'p':
            pflag = true;
            break;
        case 't':
            tflag = true;
            tvalue = optarg;
            break;
        case 'i':
            iflag = true;
            ivalue = atoi(optarg);
            break;
        case 'l':
            lflag = true;
            lvalue = atoi(optarg);
            break;
        case 'h':
            hflag = true;
            hvalue = atoi(optarg);
            break;
        case 'e':
            eflag = true;
            evalue = atof(optarg);
            break;
        case 'm':
            mflag = true;
            mvalue = atof(optarg);
            break;
        case 'v':
            vflag = true;
            vvalue = atof(optarg);
            break;
        case 'd':
            dflag = true;
            dvalue = atof(optarg);
            break;
        case '?':
            if (optopt == 'T' || optopt == 'w' || optopt == 'p')
                fprintf(stderr, "The option -%c requires an argument.\n", optopt);
            else if (isprint(optopt))
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf(stderr,
                        "Unknown character `\\x%x'.\n",
                        optopt);
            return EXIT_FAILURE;
        default:
            return EXIT_FAILURE;
        }
    }
    if (Tvalue == NULL)
        Tvalue = tvalue;
    if (!pflag)
    {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Parameters of the mlp. For example, mlp.eta = value;
        int iterations = iflag ? ivalue : 1000;
        mlp.eta = eflag ? evalue : 0.1;
        mlp.mu = mflag ? mvalue : 0.9;
        mlp.validationRatio = vflag ? vvalue : 0;
        mlp.decrementFactor = dflag ? dvalue : 1;
        // Read training and test data: call to mlp.readData(...)
        Dataset *trainDataset;
        trainDataset = mlp.readData(tvalue);
        Dataset *testDataset;
        testDataset = mlp.readData(Tvalue);
        if (trainDataset == NULL || testDataset == NULL)
        {
            std::cerr << "Train and Test datasets are required" << std::endl;
            exit(-1);
        }
        // Initialize topology vector
        int layers = lflag ? lvalue : 1;
        int *topology = new int[layers + 2]; // This should be corrected
        topology[0] = trainDataset->nOfInputs;
        for (int i = 0; i <= layers; i++)
            topology[i] = hflag ? hvalue : 5;
        topology[layers + 1] = trainDataset->nOfOutputs;

        // Initialize the network using the topology vector
        mlp.initialize(layers + 2, topology);

        // Seed for random numbers
        int seeds[] = {1, 2, 3, 4, 5};
        double *testErrors = new double[5];
        double *trainErrors = new double[5];
        double bestTestError = 1;
        for (int i = 0; i < 5; i++)
        {
            cout << "**********" << endl;
            cout << "SEED " << seeds[i] << endl;
            cout << "**********" << endl;
            srand(seeds[i]);
            mlp.runOnlineBackPropagation(trainDataset, testDataset, iterations, &(trainErrors[i]), &(testErrors[i]));
            cout << "We end!! => Final test error: " << testErrors[i] << endl;

            // We save the weights every time we find a better model
            if (wflag && testErrors[i] <= bestTestError)
            {
                mlp.saveWeights(wvalue);
                bestTestError = testErrors[i];
            }
        }

        cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

        double averageTestError = 0, stdTestError = 0;
        double averageTrainError = 0, stdTrainError = 0;

        // Obtain training and test averages and standard deviations
        for (int i = 0; i < 5; i++)
            averageTestError += testErrors[i];

        for (int i = 0; i < 5; i++)
            averageTrainError += trainErrors[i];

        averageTestError = averageTestError / (double)5;
        averageTrainError = averageTrainError / (double)5;

        for (int i = 0; i < 5; i++)
            stdTestError += pow((testErrors[i] - averageTestError), 2);
        stdTestError /= 5;
        stdTestError = sqrt(stdTestError);

        for (int i = 0; i < 5; i++)
            stdTrainError += pow((trainErrors[i] - averageTrainError), 2);
        stdTrainError /= 5;
        stdTrainError = sqrt(stdTrainError);

        cout << "FINAL REPORT" << endl;
        cout << "************" << endl;
        cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << endl;
        cout << "Test error (Mean +- SD):          " << averageTestError << " +- " << stdTestError << endl;
        return EXIT_SUCCESS;
    }
    else
    {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////

        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Initializing the network with the topology vector
        if (!wflag || !mlp.readWeights(wvalue))
        {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading training and test data: call to mlp.readData(...)
        Dataset *testDataset;
        testDataset = mlp.readData(Tvalue);
        if (testDataset == NULL)
        {
            cerr << "The test file is not valid, we can not continue" << endl;
            exit(-1);
        }

        mlp.predict(testDataset);

        return EXIT_SUCCESS;
    }
}
