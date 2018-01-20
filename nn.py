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
# Calculate the slope of loss function at the output and adjust weights by weight - (slope * learning rate)
# Repeat this process going back towards the input adjusting the weights at each layer.

# After all weights have been adjusted repeat process of forward propagation, calculate error
# and back propagation until slope = 0


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



print(input_data_arrays)
print(target_list)

input_data = np.array([2, 3]) #A simple single input to test the predict_with_network function

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
    I think the data points must be in the form of numpy array in order to multiply with the weights
    '''
    node_0_input = (input_data_point * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    node_1_input = (input_data_point * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    hidden_layer_values = np.array([node_0_output, node_1_output])
    input_to_final_layer = (hidden_layer_values * weights['output']).sum()
    model_output = relu(input_to_final_layer)

    return (model_output)



model_output_1 = predict_with_network(input_data, weights_0)

print('model output 1 = ', model_output_1)

model_output_error_list = []

# Loop over input_data
for row in input_data_arrays:
    # Append prediction to model_output_0
    model_output_error_list.append(predict_with_network(row, weights_0))

# Calculate the mean squared error for model_output_0: mse_0
mse = mean_squared_error(target_list, model_output_error_list)

print('mse= ', mse)