#include<math.h>
#include"MatrixOperations.h"
#include"activation.h"
#ifndef NETWORKOPERATIONS_H 
#define NETWORKOPERATIONS_H 


//all network functions
struct Network{
	double learningRate; 
	int numLayers;
	int* sizes;
	int* activations;
	double** matrixN;
	double** matrixNRELU;
	double** matrixW;
	double** matrixB;
	double** matrixCW;
	double** matrixCB;
	double** layerLoses;
};

//preformes the activation function specified by the type
void matrixActivationDev(int row,int col,double* matrix,int type,double param,double* matrixRes);
double activationDev(double num,int type,double param);

//preformes the activation Dev function specified by the type
void matrixActivation(int row,int col,double* matrix,int type,double param,double* matrixRes);
double activation(double num,int type,double param);

//imports the networks weights and biases from a .txt file
void netImport(struct Network* net);

//exports the networks weights and biases onto a .txt file
void netExport(struct Network* net);

//sets the weights to the H2 weight convnetion
void matrixH2W(struct Network* net);

//declares the weights and biases of the network
void beginNet(struct Network* net);

//does the forward loop of the neural net
void Forward(struct Network* net);

//frees all data collected from the Forward pass
void resetNet(struct Network* net);

//frees all weight and biases data from network 
void freeNet(struct Network* net);

//changes the weight and biases
void Backprop(struct Network* net,double* target);


void matrixH2WB(struct Network* net){
	for(int i=0;i<net->numLayers-1;i++){
		matrixMultiplyScalar(net->sizes[i+1],net->sizes[i],net->matrixW[i],sqrt((double)2/net->sizes[i]));
	}
		
}

void beginNet(struct Network* net){
	net->matrixN = calloc(net->numLayers,sizeof(double*));
	net->matrixNRELU = calloc(net->numLayers,sizeof(double*));
	net->matrixB = calloc(net->numLayers-1,sizeof(double*));
	net->matrixW = calloc(net->numLayers-1,sizeof(double*));
	net->matrixCW = calloc(net->numLayers-1,sizeof(double*));
	net->matrixCB = calloc(net->numLayers-1,sizeof(double*));
	net->layerLoses = calloc(net->numLayers-1,sizeof(double*));
	for(int i=0;i<net->numLayers;i++){
		net->matrixN[i] = calloc(net->sizes[i],sizeof(double));
		net->matrixNRELU[i] = calloc(net->sizes[i],sizeof(double));
	}
	for(int i=0;i<net->numLayers-1;i++){
		net->layerLoses[i] = calloc(net->sizes[i+1],sizeof(double));
		net->matrixB[i] = calloc(net->sizes[i+1],sizeof(double));
		matrixMake(net->sizes[i+1],1,net->matrixB[i]);
		net->matrixW[i] = calloc(net->sizes[i+1]*net->sizes[i],sizeof(double));
		matrixMake(net->sizes[i+1],net->sizes[i],net->matrixW[i]);
		net->matrixCW[i] = calloc(net->sizes[i+1]*net->sizes[i],sizeof(double));
		net->matrixCB[i] = calloc(net->sizes[i+1],sizeof(double));
	}
}

void resetNet(struct Network* net){
	for(int i=0;i<net->numLayers;i++){
		matrixSet(net->sizes[i],1,net->matrixN[i],0);
		matrixSet(net->sizes[i],1,net->matrixNRELU[i],0);
	}
}

void freeNet(struct Network* net){
	for(int i=0;i<net->numLayers-1;i++){
		free(net->matrixB[i]);
		free(net->matrixCB[i]);
		free(net->matrixW[i]);
		free(net->matrixCW[i]);
		free(net->layerLoses[i]);
	}
	for(int i=0;i<net->numLayers;i++){
		free(net->matrixN[i]);
		free(net->matrixNRELU[i]);
	}
	free(net->layerLoses);
	free(net->matrixCW);
	free(net->matrixCB);
	free(net->matrixB);
	free(net->matrixW);
	free(net->matrixN);
	free(net->matrixNRELU);
	free(net->activations);
	free(net->sizes);
}

void Forward(struct Network* net){
	double sum = 0;
	for(int i=0;i<net->numLayers-1;i++){
		for(int j=0;j<net->sizes[i+1];j++){
			sum = 0;
			for(int k=0;k<net->sizes[i];k++){
				sum+=net->matrixW[i][j*net->sizes[i]+k]*net->matrixNRELU[i][k];
			}
			sum+=net->matrixB[i][j];
			net->matrixN[i+1][j] = sum;
		}
		matrixActivation(net->sizes[i+1],1,net->matrixN[i+1],net->activations[i],0,net->matrixNRELU[i+1]);
	}
}

void Backprop(struct Network* net,double* lossFunction){
	//calculating error in the last layer( C hamard zL) 
	double* lastLayerActivationDev = calloc(net->sizes[net->numLayers-1],sizeof(double));
	matrixActivationDev(net->sizes[net->numLayers-1],1,net->matrixN[net->numLayers-1],net->activations[net->numLayers-2],0,lastLayerActivationDev);
	matrixHamard1(net->sizes[net->numLayers-1],1,lossFunction,lastLayerActivationDev,net->layerLoses[net->numLayers-2],0,1);

	//calculating the error for the other layers
	double* weightTranspose;
	double* weightXloss;
	double* layerActivationDev;
	for(int i=net->numLayers-3;i>=0;i--){
		weightTranspose = calloc(net->sizes[i+2]*net->sizes[i+1],sizeof(double));
		matrixTranspose(net->sizes[i+2],net->sizes[i+1],net->matrixW[i+1],weightTranspose);
		weightXloss = calloc(net->sizes[i+1],sizeof(double));
		matrixMultiply1(net->sizes[i+1],net->sizes[i+2],weightTranspose,net->sizes[i+2],1,net->layerLoses[i+1],weightXloss,1,0);
		layerActivationDev = calloc(net->sizes[i+1],sizeof(double));
		matrixActivationDev(net->sizes[i+1],1,net->matrixN[i+1],net->activations[i],0,layerActivationDev);
		matrixHamard1(net->sizes[i+1],1,weightXloss,layerActivationDev,net->layerLoses[i],1,1);
	}

	//calculating the slope for schastic gradient decent
	double* weightLoss;
	for(int i=0;i<net->numLayers-1;i++){
		weightLoss = calloc(net->sizes[i+1]*net->sizes[i],sizeof(double));
		matrixMultiply1(net->sizes[i+1],1,net->layerLoses[i],1,net->sizes[i],net->matrixNRELU[i],weightLoss,0,0);
		matrixAdditionEqual1(net->sizes[i+1],net->sizes[i],weightLoss,net->matrixCW[i],1);
		matrixAdditionEqual(net->sizes[i+1],1,net->layerLoses[i],net->matrixCB[i]);
	}
}

void netExport(struct Network* net){
	FILE* input;
	input = fopen("export.txt","w");
	for(int i=0;i<net->numLayers-1;i++){
		for(int j=0;j<net->sizes[i+1]*net->sizes[i];j++)
			fprintf(input,"%f ",net->matrixW[i][j]);
		fprintf(input,"\n");
		for(int j=0;j<net->sizes[i+1];j++)
			fprintf(input,"%f ",net->matrixB[i][j]);
	}
	fclose(input);
}

void netImport(struct Network* net){
	FILE* input;
	input = fopen("export.txt","r");
	for(int i=0;i<net->numLayers-1;i++){
		for(int j=0;j<net->sizes[i+1]*net->sizes[i];j++)
			fscanf(input,"%le ",&net->matrixW[i][j]);
		fscanf(input,"\n");
		for(int j=0;j<net->sizes[i+1];j++)
			fscanf(input,"%le ",&net->matrixB[i][j]);
	}
	fclose(input);
}

//double activation(double num,int type,double param){
//	switch(type){
//		case 1:
//			return RELU(num);
//		case 2:
//			return Sigmoid(num);
//		case 3:
//			return Tanh(num);
//		case 4:
//			return LeakyRELU(num);
//		case 5:
//			return ParamRELU(num,param);
//		case 6:
//			return ELU(num,param);
//		//case 7:
//		//	return matrixSoftMax(row,col,matrix);
//		case 8:
//			return Swish(num,param);
//		case 9:
//			return GELU(num);
//		case 10:
//			return SELU(num);
//	}
//}

//double activationDev(double num,int type,double param){
//	switch(type){
//		case 1:
//			return RELUDev(num);
//		case 2:
//			return SigmoidDev(num);
//		case 3:
//			return TanhDev(num);
//		case 4:
//			return LeakyRELUDev(num);
//		case 5:
//			return ParamRELUDev(num,param);
//		case 6:
//			return ELUDev(num,param);
//		//case 7:
//		//	return matrixSoftMaxDev(row,col,matrix);
//		case 8:
//			return SwishDev(num,param);
//		case 9:
//			return GELUDev(num);
//		case 10:
//			return SELUDev(num);
//	}
//}

void matrixActivation(int row,int col,double* matrix,int type,double param,double* matrixRes){
	switch(type){
		case 1:
			matrixRELU(row,col,matrix,matrixRes);
			break;
		case 2:
			matrixSigmoid(row,col,matrix,matrixRes);
			break;
		case 3:
			matrixTanh(row,col,matrix,matrixRes);
			break;
		case 4:
			matrixLeakyRELU(row,col,matrix,matrixRes);
			break;
		case 5:
			matrixParamRELU(row,col,matrix,matrixRes,param);
			break;
		case 6:
			matrixELU(row,col,matrix,matrixRes,param);
			break;
		case 7:
			matrixSoftMax(row,col,matrix,matrixRes);
			break;
		case 8:
			matrixSwish(row,col,matrix,matrixRes,param);
			break;
		case 9:
			matrixGELU(row,col,matrix,matrixRes);
			break;
		case 10:
			matrixSELU(row,col,matrix,matrixRes);
			break;
		default:
			matrixCopy(row,col,matrix,matrixRes);
	}
}

void matrixActivationDev(int row,int col,double* matrix,int type,double param,double* matrixRes){
	switch(type){
		case 1:
			matrixRELUDev(row,col,matrix,matrixRes);
			break;
		case 2:
			matrixSigmoidDev(row,col,matrix,matrixRes);
			break;
		case 3:
			matrixTanhDev(row,col,matrix,matrixRes);
			break;
		case 4:
			matrixLeakyRELUDev(row,col,matrix,matrixRes);
			break;
		case 5:
			matrixParamRELUDev(row,col,matrix,matrixRes,param);
			break;
		case 6:
			matrixELUDev(row,col,matrix,matrixRes,param);
			break;
		case 7:
			matrixSoftMaxDev(row,col,matrix,matrixRes);
			break;
		case 8:
			matrixSwishDev(row,col,matrix,matrixRes,param);
			break;
		case 9:
			matrixGELUDev(row,col,matrix,matrixRes);
			break;
		case 10:
			matrixSELUDev(row,col,matrix,matrixRes);
			break;
		default:
			matrixCopy(row,col,matrix,matrixRes);
	}
}
#endif
