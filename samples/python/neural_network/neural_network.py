import numpy as np
#---------------------------------------------------------------------------------------------------
class BackPropagationNetwork:
	"""A Back-Propagation Network"""
	# Class members
	layerCount 	= 0		# number of layers
	shape 		= None 	# tuple
	weights 	= [] 	# weights of empty list 
	
	# Class methods
	def __init__(self, layerSize):
		"""Initialize the network"""
		# Layer info
		self.layerCount 	= len(layerSize)-1 # Why -1? not count the input layer
		self.shape      	= layerSize
		
		# Input/Output from the last Run
		self._layerInput  	= []
		self._layerOutput 	= []
		
		# Create the weight array
		for (l1, l2) in zip(layerSize[:-1], layerSize[1:]):
			self.weights.append(np.random.normal(scale=0.1, size=(l2, l1+1)))
		
	# Run methods
	def Run(self, input):
		"""Run the network based on the input data"""
		lnCases = input.shape[0]
		# Clear the previous intermediate values lists
		self._layerInput  	= []
		self._layerOutput  	= []
		
		# Run it:
		for index in range(self.layerCount):
			if index == 0:
				layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, lnCases])]))
			else:	
				layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, lnCases])]))
			
			self._layerInput.append(layerInput)
			self._layerOutput.append(self.sgm(layerInput))
			
		return self._layerOutput[-1].T
		
	# Train epoch methods
	def TrainEpoch(self, input, target, trainingRate = 0.2):
		"""This method trains the network for one epoch"""
		delta = []
		lnCases = input.shape[0]
		
		# First run the network
		self.Run(input)
		
		#Calculate deltas
		for index in reversed(range(self.layerCount)):
			if index == self.layerCount-1:
				#compare the target values
				output_delta = self._layerOutput[index] - target.T
				error = np.sum(output_delta**2)
				delta.append(output_delta*self.sgm(self._layerInput[index], True))
			else:
				#compare the following layer's delta
				delta_pullback = self.weights[index+1].T.dot(delta[-1])
				delta.append(delta_pullback[:-1,:] * self.sgm(self._layerInput[index], True))
				
		# Compute weight delta
		for index in range(self.layerCount):
			delta_index = self.layerCount - 1 - index
			
			if index==0:
				layerOutput = np.vstack([input.T, np.ones([1, lnCases])])
			else:
				layerOutput = np.vstack([self._layerOutput[index-1], np.ones([1, self._layerOutput[index-1].shape[1]])])
				
			weightDelta = np.sum(\
								 layerOutput[None,:,:].transpose(2, 0, 1) * delta[delta_index][None,:,:].transpose(2, 1, 0)\
								 , axis = 0)
		
			self.weights[index]  -= trainingRate * weightDelta
			
		return error
	# Transfer function
	def sgm(self, x, Derivative=False):
		if not Derivative:
			return 1 / (1 + np.exp(-x))
		else:
			out = self.sgm(x)
			return out*(1-out)
#---------------------------------------------------------------------------------------------------
# Create a test object
if __name__ == "__main__":
	bpn = BackPropagationNetwork((2,2,2))
	print(bpn.shape)
	# print(bpn.weights)
	
	# lvInput  = np.array([[0, 0], [1, 1], [-1, 0.5]])
	# lvOutput = bpn.Run(lvInput)
	
	# print("Input: {0}\nOutput: {1}". format(lvInput, lvOutput))
	
	lvInput  = np.array([[0, 0], [1, 1], [0, 1], [0, 1]])
	lvTarget = np.array([[0.05], [0.05], [0.95], [0.95]])
	lnMax = 100000
	lnErr = 1e-5
	
	for i in range(lnMax-1):
		err = bpn.TrainEpoch(lvInput, lvTarget)
		if i % 1000== 0:
			print("Iteration {0}\tError: {1:0.6f}".format(i, err))
		if err <= lnErr:
			print("Minimum error reached at iteration {0}".format(i))
			break
	# Display output
	lvOutput = bpn.Run(lvInput)
	print("Input: {0}\nOutput: {1}". format(lvInput, lvOutput))
			