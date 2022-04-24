# project-M
Project-M is a neural network library made in C.


# Documentation
This documentation will be explaining the code written in example.c and more.
## Making your first network
Start by making your network struct.
```C
struct Network base;
/*
the struct looks like this.
struct Network{
  double learning Rate; //stores the learning rate of your network
  int numLayers; //stores the numbers of layers including the input and output
  int* sizes; //stores the size of each layer(only the amount of nodes needed, no need for the weight sizes)
  int* activations; //stores which activation function will be used for each layer.
  double** matrixN; //stores the value of each node in the network
  double** matrixNRELU; //stores the value of each node in the network with the activation function used
  double** matrixW; //stores the value of each weight;
  double** matrixB; //stores the value of each bias;
  double** matrixCW; //stores the cost of the weights;
  double** matrixCB; //stores the cost of the biases;
  double** layerLoses; //stores the cost of the layers;
};
*/
```
then set the amount of layers you need, and the activation functions that will be used, and the sizes of each layer
```C
base.numLayers = 4;
int* layerSizes = calloc(base.numLayers,sizeof(int));
layerSizes[0] = 28*28;
layerSizes[1] = 125;
layerSizes[2] = 125;
layerSizes[3] = 10;
int* activationFunction = calloc(base.numLayers-1,sizeof(int));
activationFunction[0] = 3;
activationFunction[1] = 3;
activationFunction[2] = 7;
base.activations = activationFunction;
base.sizes = layerSizes;
```
then we can initalize our network with beginNet
```C
beginNet(&base);
```

