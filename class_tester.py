# the call to setFitness below breaks because of python's unusual scope rules
# see scope_test in Reed's sandbox for an explanation
# we need to go through and adjust things accordingly but it doesn't sounds like there is an easy way to make a copy ever  


from nueralnet_classes import nodeNet

# create simple nodeNet with 3 inputs 5 nodes and 2 outputs
simpleNet = nodeNet(3, 5, 2)


# print the results of some simple inputs

# should give 5 very activated nodes time weight 1 so both outputs should be almost five
test_list = [3, 3, 3]
print simpleNet.processInputs(test_list)

# set fitness by above metric, fitness should be very low
answer = [5, 5]
simpleNet.setFitness([test_list],[answer])
print simpleNet.fitness

# should give 5 very un-activated nodes time weight 1 so both outputs should be almost 0
test_list = [-3, -3, -3]
print simpleNet.processInputs(test_list)

# see the inputs weights
print simpleNet.inputWeights

# see the outputs weights
print simpleNet.outputWeights

# vary them by 50% and then print again
simpleNet.mutate(0.5)
print simpleNet.inputWeights
print simpleNet.outputWeights

# process the inputs that gave five before again
test_list = [3, 3, 3]
print simpleNet.processInputs(test_list)

# check that fitness has increased (ie the net is less fit)
simpleNet.setFitness(test_list,answer)
print simpleNet.fitness

# ****************************
# the results to all of the above tests seem reasonable: Apr. 27, 2014
# ****************************
 
 