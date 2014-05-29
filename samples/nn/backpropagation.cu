#include <stdio.h>
#include "neural_network.hpp"

using namespace nn;

backPropagationNetwork::backPropagationNetwork()
{
	printf("Back Propagation Network!!!\n");
}

backPropagationNetwork::~backPropagationNetwork()
{
}

float transferFunctions::evaluate(const int transferFunction, float input)
{
	switch(transferFunction)
	{
	case Sigmoid:
		return sigmoid(input);
	case None :
	default:
		return 0.0f;
	}
}
float transferFunctions::evaluateDerivative(const int transferFunction, float input)
{
	switch(transferFunction)
	{
	case Sigmoid:
		return sigmoidDerivative(input);
	case None :
	default:
		return 0.0f;
	}
}
float transferFunctions::sigmoid(float x)
{
	return 1.0f/(1.0f * expf(-x));
}
float transferFunctions::sigmoidDerivative(float x)
{
	return sigmoid(x)* (1-sigmoid(x));
}