# import modules
import numpy as np
import random as rand


# define the class for each network
class nodeNet():
    def __init__(self, Ninputs=10, Nnodes=10, Noutputs=10):
        # use matrix multiplication and other matrix functions
        
        #set array sizes
        self.Ninputs = Ninputs
        self.Nnodes = Nnodes
        self.Noutputs = Noutputs
        
        #default fitness to 0
        self.fitness = 0;
        
        # inputs weights will be given by inputWeights[input][node]
        self.inputWeights = [[1 for x in xrange(Nnodes)] for x in xrange(Ninputs+1)]  

        # output weights will be given by outputWeights[output][node]
        self.outputWeights = [[1 for x in xrange(Nnodes)] for x in xrange(Noutputs)]  
        
        
    # process inputs will take in a list of length Ninputs and return a list of Noutputs     
    def processInputs(self, inputs):
        tempInputs = inputs
        inputs.append(-1) # this last index is for the bias
        node_arg = np.dot(inputs, self.inputWeights)
        node_values = 1/(1+np.exp(np.negative(node_arg))) # the activation function (a sigmoid)
        return np.dot(self.outputWeights, node_values)
        
    
    # the input values will be MxNinputs array and the output values will be an MxNoutputs array
    # this functions will process each input and return some metric of fitness, with lower being more fit
    # this metric could be sum((sum((err of outputs for a give input row)^2) for all rows)^2)
    def setFitness(self, inputs, trueOutputs):
        Ncases = inputs.__len__()
        self.fitness = 0;
        for i in xrange(Ncases):
            diffs = np.subtract(trueOutputs[i], self.processInputs(inputs[i]))
            contribution = np.sum(np.square(diffs))
            self.fitness = self.fitness + contribution
        
        
    # mutate will adjust all the weights by a normal distributed random amount
    # the standard deviation of the adjustment will be variation*(the weight) 
    def mutate(self, variation):
        for i in xrange(self.Nnodes):   
            for j in xrange(self.Ninputs+1):
                stdDev = variation * self.inputWeights[j][i]
                self.inputWeights[j][i] = rand.normalvariate(self.inputWeights[j][i], stdDev)
            for j in xrange(self.Noutputs):
                stdDev = variation * self.outputWeights[j][i]
                self.outputWeights[j][i] = rand.normalvariate(self.outputWeights[j][i], stdDev)



# define the class for to hold all networks and evolve them
class geneticEvolution():
    def __init__(self, nodeNetObj = nodeNet()):
        self.nodesNets = nodeNetObj
        self.curPop = len(self.nodesNets)
        
        
    # this will replace the list of nodeNet objects with a new list of length newPop  
    # the new list will be built from random sampling of weights from the last generation & then mutated by variation
    # random sampling could pull individual weights, entire columns in the weight matrix or entire rows. which is best?
    def newGeneration(self, newPop = 20, variation = 0.1):
         print "WE STILL NEED TO BUILD newGeneration!"
   
   
    # findFitness will loop through each nodeNets object and update the fitness given inputs and outputs
    # inputs is a list of lists containing the inputs of the nodesNets objects w/ each outer lists being a new case
    # outputs is a list of lists containing the expected outputs of nodesNets object w/ outer lists corresponding to cases 
    # i.e. ..puts[case][inputN]
    def findFitness(self, inputs = [], trueOutputs = []):
        for i in xrange(self.curPop):
            self.nodesNets[i].setFitness(inputs, trueOutputs)
        
    
    # this function will sort the nodeNet object list (self.nodeNets) by fitness and remove all but the top Nsurvivors
    def selectBest(self, Nsurvivors = 5):
         print "WE STILL NEED TO BUILD selectBest!"
        
        
        
        
        
        
    
    
    