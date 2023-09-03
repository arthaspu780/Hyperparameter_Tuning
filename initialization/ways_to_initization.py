import numpy as np
def initialize_parameters_random(layers_dims):
    L=len(layers_dims)
    parameter={}
    for i in range(1,L):
        parameter["w"+str(i)]=np.random.randn(layers_dims[i],layers_dims[i-1])
        parameter["b"+str(i)]=np.zeros((layers_dims[i],1))
    return parameter
layers_dims=[2,3,4]
param=initialize_parameters_random(layers_dims)
print(param)
def initialize_parameters_he(layers_dims):
    L = len(layers_dims)
    parameter = {}
    for i in range(1, L):
        parameter["w" + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1])*np.sqrt(2/layers_dims[i-1])
        parameter["b" + str(i)] = np.zeros((layers_dims[i], 1))
    return parameter
layers_dims=[2,3,4]
param=initialize_parameters_random(layers_dims)
print(param)


