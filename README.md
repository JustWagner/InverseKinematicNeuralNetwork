# InverseKinematicNeuralNetwork

This script consists of a function of a forward kinematic. A simple neural network and a training and evaluation loop.

## Forward Kinematic
The forward kinematic is for a 2 segment continuum robot. The input consists of the curvature of the segments, the relative rotational angle of the curvature direction and the lengths of the segments.

## Training Data

The main idea of the script is to generate an array of random input data for the forward kinematic and calculate the corresponding positions. 
Then a Neural network is constructed so it can use the output of the forward kinematic as input and predict the random data.
