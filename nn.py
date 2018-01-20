import pandas as pd
prods = pd.read_csv('product.csv')
print(prods.head())

#=====STEPS=====
#=FORWARD PROPAGATION=
#we want to run the model on each set of inputs and calculate the 'error':
#Take each row split into input a, input b and target.
#Run a & b through the nodes to calculate the output.
#Each node is calculated by taking the sum of inputs * weights from previous layer
#Calculate the error for each row (pair of inputs) by calculating output and subtracting from target
#Store a list of errors in a list
#Calculate mean squared error for all individual errors
#=BACK PROPAGATION=
#Calculate the slope of loss function at the output and adjust weights by weight - (slope * learning rate)