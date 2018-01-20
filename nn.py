# =====STEPS=====

# =FORWARD PROPAGATION=
# we want to run the model on each set of inputs and calculate the 'error':
# Take each row split into input a, input b and target.
# Run a & b through the nodes to calculate the output.
# Each node is calculated by taking the sum of inputs * weights from previous layer
# (and possibly applying a function eg. relu)


# =CALCULATE ERROR=
# Calculate the error for each row (pair of inputs) by calculating predicted - target
# Store a list of errors in a list
# Calculate mean squared error for all individual errors

# =BACK PROPAGATION=
# Calculate the slope of loss function at the output and adjust weights
    # by subtracting slope * learning rate. -> weight = weight - (slope * learning_rate)
# Repeat this process going back towards the input adjusting the weights at each layer.

# After all weights have been adjusted repeat process of forward propagation, calculate error
# and back propagation until slope = 0

#=CALCULATING THE SLOPE=
'''To calculate the slope for a weight we need to multiply:
        1) The slope of the loss function with respect to the value at the node we feed into (see below).
        2) The value of the node that feeds into our weight
        3) The slope of the activation function with respect to the value we feed into
            (in the case of relu() this last element will always be 0 or 1) - we will ignore this step for now

    To work out point 1 above calculate: 2 * Error

'''

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


prods = pd.read_csv('product.csv')

input_data_arrays = []
target_list = []

#Splits the pandas dataframe into a list of arrays for input and a list of target values
for row in prods.itertuples():
    input_data_arrays.append(np.asarray([row.a, row.b]))
    target_list.append(row.product)



print('List of input arrays: ',input_data_arrays)
print('Target list: ', target_list)

input_data = np.array([4, 8]) #A simple single input to test the predict_with_network function

#The weights are the only things we change to make the network generate more accurate predictions.
#In the current model we only have 1 hidden layer with 2 nodes.
weights_0 = {'node_0':[1,1],
             'node_1':[1,1],
             'output':[1,1]
             }

def relu(n):
    return max(0,n)

def predict_with_network(input_data_point, weights):
    '''Predicts the output of a network given 2 input data points
    and a set of weights for 2 nodes + 1 output.
    This function was copied direct from datacamp.
    Note the use of the 'relu' function which returns 0 if n<0 else returns n
    The data points must be in the form of numpy arrays in order to multiply with the weights
    '''
    node_0_input = (input_data_point * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    node_1_input = (input_data_point * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    hidden_layer_values = np.array([node_0_output, node_1_output])
    input_to_final_layer = (hidden_layer_values * weights['output']).sum()
    model_output = relu(input_to_final_layer)

    return (model_output, hidden_layer_values)



model_output_1 = predict_with_network(input_data, weights_0)[0]

print('model output 1 = ', model_output_1)

model_output_list = []

# Loop over input_data
for row in input_data_arrays:
    # Append prediction to model_output_list
    model_output_list.append(predict_with_network(row, weights_0)[0])

print('model output list: ', model_output_list) #List of predictions
print('Length of output list: ', len(model_output_list))
print('Mean of output list: ',np.mean(model_output_list))
print('List of errors: ', [np.asarray(model_output_list) - np.asarray(target_list)])

# Calculate the mean squared error for model_output_0: mse_0
mse = mean_squared_error(target_list, model_output_list)

print('mse= ', mse)


# lets try and write a weight adjusting process that works with just one input:
learning_rate = 0.1
target1 = sum(input_data)
prediction1, hidden_layer = predict_with_network(input_data, weights_0)
error1 = prediction1 - target1
print('Input data: ', input_data)
print('Target1: ', target1)
print('Prediction1: ', prediction1)
print('Error:', error1)
print('Weights: ', weights_0)
print('Hidden Layer: ', hidden_layer)




#These steps below calculate the gradient at output and adjust the last set of weights accordingly:
gradient_at_output = 2 * hidden_layer * error1
print('Gradient at output:', gradient_at_output)
print('Type gradient at output:', type(gradient_at_output))
print('Gradient at output[0]:', gradient_at_output[0])
weights_0['output'] = weights_0['output'] - (gradient_at_output * 0.01)
print('Adjusted weights:', weights_0)

#This while loop repeats the steps outlined above until the gradient is close to zero
while abs(gradient_at_output[0]) > 0.1:
    print('-----NEW ROUND-----')
    prediction1, hidden_layer = predict_with_network(input_data, weights_0)
    error1 = prediction1 - target1
    gradient_at_output = 2 * hidden_layer * error1
    print('Gradient at output:', gradient_at_output)
    weights_0['output'] = weights_0['output'] - (gradient_at_output * 0.001)
    print('Adjusted weights:', weights_0)
    print('Prediction1: ', prediction1)
    print('Error:', error1)






