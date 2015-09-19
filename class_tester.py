# the call to setFitness below breaks because of python's unusual scope rules
# see scope_test in Reed's sandbox for an explanation
# we need to go through and adjust things accordingly but it doesn't sounds like there is an easy way to make a copy ever  

import numpy as np
from nueralnet_classes import nodeNet

#create simple nodeNet with 3 inputs 5 nodes and 2 outputs
simpleNet = nodeNet(3, 5, 2)

# print the results of some simple inputs
print "---should give 5 very activated nodes time weight 1 so both outputs should be almost five:"
test_list = np.array([3, 3, 3])
print simpleNet.processInputs(test_list)

print "---set fitness by above metric, fitness should be very low (low fitness is good):"
answer = np.array([[5, 5]])
simpleNet.setFitness(test_list,answer)
print simpleNet.fitness
 
print "---should give 5 very un-activated nodes time weight 1 so both outputs should be almost 0:"
test_list = np.array([-3, -3, -3])
print simpleNet.processInputs(test_list)
 
print "---see the input weights:"
print simpleNet.inputWeights
print "---see the output weights:"
print simpleNet.outputWeights 

print "---vary them by 50% and then print again:"
simpleNet.mutate(0.5)
print simpleNet.inputWeights
print simpleNet.outputWeights

print "---process the inputs that gave five before again:"
test_list = np.array([3, 3, 3])
print simpleNet.processInputs(test_list)
 
print "---check that fitness has increased (i.e. the net is less fit):"
simpleNet.setFitness(test_list,answer)
print simpleNet.fitness
 
# ****************************
# the results to all of the above tests seem reasonable: Apr. 27, 2014
# ****************************