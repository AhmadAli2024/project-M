#include<stdio.h>
#include<stdlib.h>
#include<math.h>

struct Network{
	int numLayers;
	int* sizes;
	double*** matrixN;
	double*** matrixNRELU;
	double*** matrixW;
	double*** matrixB;
	double*** matrixCW;
	double*** matrixCB;
};
//imports the networks weights and biases from a .txt file
void netImport(struct Network* net);

//exports the networks weights and biases onto a .txt file
void netExport(struct Network* net);

//sets the weights to the H2 weight convnetion
void matrixH2WB(struct Network* net);

//preformes matrix2-=matrix1
void matrixSubtractionEqual(int row,int col,double** matrix1,double** matrix2,int f1,int f2);

//preformes matrix2+=matrix1
void matrixAdditionEqual(int row,int col,double** matrix1,double** matrix2,int f1,int f2);

//returns the transpose of the matrix
double** matrixTranspose(int row,int col,double** matrix);

//frees a matrix
void matrixFree(int row,int col,double** matrix);

//display a 2d matrix
void matrixDisplay(int row,int col,double** table);

//allocate memory for a 2D matrix 
double** matrix2D(int row,int col);

//adds 2 2D matricies
double** matrixAddition(int row,int col,double** matrix1,double** matrix2,int f1,int f2);

//subtracts matrix 1 to matrix 2
double** matrixSubtraction(int row,int col,double** matrix1,double** matrix2,int f1,int f2);

//returns a random double number
double randomDouble(double min, double max);

//returns a matrix with random values between -1 and 1
double** matrixMake(int row,int col);

//multiplys 2 matrices return the resault
double** matrixMultiply(int rows,int colums,double** Matrix,int rows1,int colums1,double** Matrix1,int f1,int f2);

//returns the matrix relu of matrix
double** matrixRELU(int row,int col,double** matrix);

//returns the matrix relu dev of matrix
double** matrixRELUDev(int row,int col,double** matrix);

//declares the weights and biases of the network
void beginNet(struct Network* net);

//does the forward loop of the neural net
void Forward(struct Network* net);

//frees all data collected from the Forward pass
void resetNet(struct Network* net);

//frees all weight and biases data from network 
void freeNet(struct Network* net);

//changes the weight and biases
void Backprop(struct Network* net,double** target);

//normalizes the matrix
void matrixNormalize(int row,int col,double** matrix);

//returns the sum of the matrix
double matrixSum(int row,int col,double** matrix);

//multiplys all matrix elements by a num
void matrixMultiplyScalar(int row,int col,double** matrix,double num);

//preformes hamard matrix multiplication
double** matrixHamard(int row,int col,double** matrix,double** matrix1,int f1,int f2);

//returns index of max element in a vector
int matrixMaxIndex(int row,int col,double** matrix);

void matrixDisplay(int row,int col,double** table){
	for(int i=0;i<row;i++){
		for(int j =0;j<col;j++)
			printf("%f ",table[i][j]);
		printf("\n");
	}
}

double** matrix2D(int row,int col){
	double** table = calloc(row,sizeof(double*));
	for(int i=0;i<row;i++)
		table[i] = calloc(col,sizeof(double));
	return table;
}

double** matrixAddition(int row,int col,double** matrix1,double** matrix2,int f1,int f2){
	double** matrixRes = matrix2D(row,col);
	for(int i=0;i<row;i++)
		for(int j=0;j<col;j++)
			matrixRes[i][j] = matrix1[i][j]+matrix2[i][j];
	if(f1)
		matrixFree(row,col,matrix1);
	if(f2)
		matrixFree(row,col,matrix2);
	return matrixRes;
}

void matrixAdditionEqual(int row,int col,double** matrix1,double** matrix2,int f1,int f2){
	for(int i=0;i<row;i++)
		for(int j=0;j<col;j++)
			matrix2[i][j]+=matrix1[i][j];
	if(f1)
		matrixFree(row,col,matrix1);
	if(f2)
		matrixFree(row,col,matrix2);
}

double** matrixSubtraction(int row,int col,double** matrix1,double** matrix2,int f1,int f2){
	double** matrixRes = matrix2D(row,col);
	for(int i=0;i<row;i++)
		for(int j=0;j<col;j++)
			matrixRes[i][j] = matrix1[i][j]-matrix2[i][j];
	if(f1)
		matrixFree(row,col,matrix1);
	if(f2)
		matrixFree(row,col,matrix2);
	return matrixRes;
}

void matrixSubtractionEqual(int row,int col,double** matrix1,double** matrix2,int f1,int f2){
	for(int i=0;i<row;i++)
		for(int j=0;j<col;j++)
			matrix2[i][j]-=matrix1[i][j];
	if(f1)
		matrixFree(row,col,matrix1);
	if(f2)
		matrixFree(row,col,matrix2);
}

double** matrixMake(int row,int col){
	double** matrixRes = matrix2D(row,col);
	for(int i=0;i<row;i++)
		for(int j=0;j<col;j++)
			matrixRes[i][j] = randomDouble(-1,1);
	return matrixRes;
}

double** matrixMultiply(int rows,int colums,double** Matrix,int rows1,int colums1,double** Matrix1,int f1,int f2){
	double** MatrixRes = matrix2D(rows,colums1);
	for(int i=0;i<rows;i++)
		for(int j=0;j<colums1;j++)
			for(int k=0;k<colums;k++)
				MatrixRes[i][j] += Matrix[i][k]*Matrix1[k][j];
	if(f1)
		matrixFree(rows,colums,Matrix);
	if(f2)
		matrixFree(rows1,colums1,Matrix1);
	return MatrixRes;
}

double randomDouble(double min, double max){
	return min + (rand() / (RAND_MAX/(max-min)));
}

void matrixFree(int row,int col,double** matrix){
	for(int i=0;i<row;i++)
		free(matrix[i]);
	free(matrix);
}

double** matrixRELU(int row,int col,double** matrix){
	double** matrixRes = matrix2D(row,col);
	for(int i=0;i<row;i++)
		for(int j=0;j<col;j++)
			if(matrix[i][j] > 0)
				matrixRes[i][j] = matrix[i][j];
	return matrixRes;
}

double** matrixRELUDev(int row,int col,double** matrix){
	double** matrixRes = matrix2D(row,col);
	for(int i=0;i<row;i++)
		for(int j=0;j<col;j++)
			matrixRes[i][j] = matrix[i][j] > 0 ? 1 : 0;
	return matrixRes;
}


void matrixH2WB(struct Network* net){
	for(int i=0;i<net->numLayers-1;i++)
		matrixMultiplyScalar(net->sizes[i+1],net->sizes[i],net->matrixW[i],sqrt((double)2/net->sizes[i]));
}

void beginNet(struct Network* net){
	net->matrixN = calloc(net->numLayers,sizeof(double**));
	net->matrixNRELU = calloc(net->numLayers,sizeof(double**));
	net->matrixB = calloc(net->numLayers-1,sizeof(double**));
	net->matrixW = calloc(net->numLayers-1,sizeof(double**));
	net->matrixCW = calloc(net->numLayers-1,sizeof(double**));
	net->matrixCB = calloc(net->numLayers-1,sizeof(double**));
	for(int i=0;i<net->numLayers-1;i++){
		net->matrixB[i] = matrixMake(net->sizes[i+1],1);
		net->matrixW[i] = matrixMake(net->sizes[i+1],net->sizes[i]);
		net->matrixCW[i] = matrix2D(net->sizes[i+1],net->sizes[i]);
		net->matrixCB[i] = matrix2D(net->sizes[i+1],1);
	}
}

void resetNet(struct Network* net){
	for(int i=0;i<net->numLayers;i++){
		matrixFree(net->sizes[i],1,net->matrixN[i]);
		matrixFree(net->sizes[i],1,net->matrixNRELU[i]);
	}
}
		
void freeNet(struct Network* net){
	for(int i=0;i<net->numLayers-1;i++){
		matrixFree(net->sizes[i+1],1,net->matrixB[i]);
		matrixFree(net->sizes[i+1],net->sizes[i],net->matrixW[i]);
		matrixFree(net->sizes[i+1],net->sizes[i],net->matrixCW[i]);
		matrixFree(net->sizes[i+1],1,net->matrixCB[i]);
	}
	free(net->matrixCW);
	free(net->matrixCB);
	free(net->matrixB);
	free(net->matrixW);
	free(net->matrixN);
	free(net->matrixNRELU);
	free(net->sizes);
}

double matrixSum(int row,int col,double** matrix){
	double sum = 0;
	for(int i =0;i<row;i++)
		for(int j=0;j<col;j++)
			sum+=matrix[i][j];
	return sum;
}

void matrixNormalize(int row,int col,double** matrix){
	double sum = matrixSum(row,col,matrix);
	sum/=(row*col);
	for(int i =0;i<row;i++)
		for(int j=0;j<col;j++)
			matrix[i][j]-=sum;
}

void Forward(struct Network* net){
	for(int i=0;i<net->numLayers-1;i++){
		net->matrixN[i+1] = matrixAddition(net->sizes[i+1],1,matrixMultiply(net->sizes[i+1],net->sizes[i],net->matrixW[i],net->sizes[i],1,net->matrixNRELU[i],0,0),net->matrixB[i],1,0);
		net->matrixNRELU[i+1] = matrixRELU(net->sizes[i+1],1,net->matrixN[i+1]);
		matrixNormalize(net->sizes[i+1],1,net->matrixNRELU[i+1]);
	}
}

void matrixMultiplyScalar(int row,int col,double** matrix,double num){
	for(int i=0;i<row;i++)
		for(int j=0;j<col;j++)
			matrix[i][j]*=num;
}

double** matrixHamard(int row,int col,double** matrix,double** matrix1,int f1,int f2){
	double** matrixRes = matrix2D(row,col);
	for(int i=0;i<row;i++)
		for(int j=0;j<col;j++)
			matrixRes[i][j] = matrix[i][j]*matrix1[i][j];
	if(f1)
		matrixFree(row,col,matrix);
	if(f2)
		matrixFree(row,col,matrix1);
	return matrixRes;
}

double** matrixTranspose(int row,int col,double** matrix){
	double** MatrixRes = matrix2D(col,row);
    for(int i=0;i<col;i++){
        for(int j=0;j<row;j++){
            MatrixRes[i][j] = matrix[j][i];
        }
    }
	return MatrixRes;
}

int matrixMaxIndex(int row,int col,double** matrix){
	int max =0;
	for(int i=0;i<row;i++)
		if(matrix[max][0] < matrix[i][0])
			max = i;
	return max;
}

void Backprop(struct Network* net,double** targetV){
	double*** layerLoses = calloc(net->numLayers-1,sizeof(double**));
	//calculating error in the last layer( C hamard zL) 
	double** lossFunction = matrixSubtraction(net->sizes[net->numLayers-1],1,net->matrixNRELU[net->numLayers-1],targetV,0,1);
	matrixMultiplyScalar(net->sizes[net->numLayers-1],1,lossFunction,2);
	matrixNormalize(net->sizes[net->numLayers-1],1,lossFunction);
	layerLoses[net->numLayers-2] = matrixHamard(net->sizes[net->numLayers-1],1,lossFunction,matrixRELUDev(net->sizes[net->numLayers-1],1,net->matrixN[net->numLayers-1]),1,1);

	//calculating the error for the other layers
	for(int i=net->numLayers-3;i>=0;i--)
		layerLoses[i] = matrixHamard(net->sizes[i+1],1,matrixMultiply(net->sizes[i+1],net->sizes[i+2],matrixTranspose(net->sizes[i+2],net->sizes[i+1],net->matrixW[i+1]),net->sizes[i+2],1,layerLoses[i+1],1,0),matrixRELUDev(net->sizes[i+1],1,net->matrixN[i+1]),1,1);

	//calculating the slope for schastic gradient decent
	for(int i=0;i<net->numLayers-1;i++){
		matrixAdditionEqual(net->sizes[i+1],net->sizes[i],matrixMultiply(net->sizes[i+1],1,layerLoses[i],1,net->sizes[i],matrixTranspose(net->sizes[i],1,net->matrixNRELU[i]),0,1),net->matrixCW[i],1,0);
		matrixAdditionEqual(net->sizes[i+1],1,layerLoses[i],net->matrixCB[i],0,0);
	}

	//freeing the losses errors in the layerLoss array
	for(int i=0;i<net->numLayers-1;i++)
		matrixFree(net->sizes[i+1],1,layerLoses[i]);
	free(layerLoses);
}

void netExport(struct Network* net){
	FILE* input;
	input = fopen("export.txt","w");
	for(int i=0;i<net->numLayers-1;i++){
		for(int j=0;j<net->sizes[i+1];j++)
			for(int k=0;k<net->sizes[i];k++)
				fprintf(input,"%f ",net->matrixW[i][j][k]);
		fprintf(input,"\n");
		for(int j=0;j<net->sizes[i+1];j++)
			fprintf(input,"%f ",net->matrixB[i][j][0]);
	}
	fclose(input);
}

void netImport(struct Network* net){
	FILE* input;
	input = fopen("export.txt","r");
	for(int i=0;i<net->numLayers-1;i++){
		for(int j=0;j<net->sizes[i+1];j++)
			for(int k=0;k<net->sizes[i];k++)
				fscanf(input,"%le ",&net->matrixW[i][j][k]);
		fscanf(input,"\n");
		for(int j=0;j<net->sizes[i+1];j++)
			fscanf(input,"%le ",&net->matrixB[i][j][0]);
	}
	fclose(input);
}
