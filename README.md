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
then set the amount of layers you need, and the activation functions that will be used, and the sizes of each layer and the learning rate
```C
base.learningRate = 0.09999999999;
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
Finally you have made your first network.
The weights and biases and set to random numbers, and the nodes are set to 0.
```C
matrixH2WB(&base); //this initalizes the weights and biases to the H2 tequnique
```

## Training the Network
Training the network consits of setting the input matrix, normalizing your data(if needed), doing the forward pass, set your target matrix, calculating the loss,
preform backpropegation, reseting your nodes, and finally chaning the weights and biases.
Setting your input matrix can be done by changing the matrixN and matrixNRELU as mentioned above.
```C
base.matrixN[0][0] = 1; //the first 0 is the layer number and the second one is the element number. So we are setting the first element of the first layer which is the input layer to 1.
matrixNormalize(base.sizes[0],1,base.matrixN[0]); //this normalizes the input layer
matrixCopy(base.sizes[0],1,base.matrixN[0],base.matrixN[0]); //we need to copy the values of the first layer(without the activation function) to the first layer(with the activation function)
```
now that we have set out input we can run the Forward pass. This is easily done by using the Forward(net) method
```C
Forward(&base);
```
Now that the forward pass is done the output of the network is stored in the matrixNRELU[net.numLayer-1] matrix.
Moving on setting up the target matrix. This can be done by declaring a normal array/matrix with the same size as your output in this case 10.
Next comes calculating the loss of the network. There are many types of losses that are already implemented in the library. Go to lossFunctions.c to look throught them all in our case we are using catagorical cross entropy.
```C
matrixCatagoricalCrossEntropy(base.sizes[base.numLayers-1],1,base.matrixNRELU[base.numLayers-1],targetV,lossFunction); //targetV is your target array/matrix the loss Function should already be inisalized with the size of your output in our case 10.
```
Now comes Backpropegation note that backpropegation does not change the weights and biases and only calculates the loss.
```C
Backprop(&base,lossFunction);
```
Note that backpropegation adds the cost of each iteration so it allows you to use mini batch gradient descent if needed.
After that reseting the nodes is an extreamly important job as it can ruin the forward pass calculations.
```C
resetNet(&base);
```
Finally to changing the weights and biases there are many optimizers that have been implemented please go to optimizers.c to check all of them out.
I will be using momentum for this case. So I will need to inialize some additional matrices for the implementation.
```C
//this should be at the begining of the file not inside a loop
double** SDW = calloc(base.numLayers-1,sizeof(double*));
double** SDB = calloc(base.numLayers-1,sizeof(double*));
for(int i=0;i<base.numLayers-1;i++){
  SDW[i] = calloc(base.sizes[i+1]*base.sizes[i],sizeof(double));
  SDB[i] = calloc(base.sizes[i+1],sizeof(double));
}
//calling the optimizer
Momentum(&base,SDW,SDB,0.9);
```
Note that some optimizers need extra parameters other than the learning rate.
Congrats you have finally trained your AI.

## Extra Functions

  

