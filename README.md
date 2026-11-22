# Perceptron.cpp
The Perceptron.cpp file contains the implementation of the network: weight initialization, a forward pass through the layers, training using the backpropagation method, and saving and loading weights.
# Perceptron.h
The Perceptron.h file is a header file that defines the NeuralClass class—that is, the structure and interface of a neural network. It does not contain any computations; instead, it specifies what the network consists of and what operations can be performed on it. As a result, Main.cpp can include the file using the command #include “Perceptron.h” and work with the network without knowing the details of its implementation.
# Main.cpp
The Main.cpp file is the program's main file. It does not contain the neural network itself, but rather manages the entire process: it prepares the data, runs the training, checks the quality, and recognizes new images. It uses the network through the NeuralClass class from Perceptron.h.
# Generate Shape 
This is a separate program wrote in C++ it generates the shapes of the forms in the 64x64 size for the network. 
