#ifndef __NEURAL_NETWORK_HPP
#define __NEURAL_NETWORK_HPP
namespace nn
{
	enum //transferFunction
	{
		None 	= 1,
		Sigmoid = 2
	};	
	
	class transferFunctions
	{
	public: 
		float evaluate();
		float evaluate(const int transferFunction, float input);
		float evaluateDerivative(const int transferFunction, float input);
		
	private:
		float sigmoid(float x);
		float sigmoidDerivative(float x);
	};
	
	class backPropagationNetwork
	{
	public:
		backPropagationNetwork();
		~backPropagationNetwork();
	private:
	
	};
};
#endif