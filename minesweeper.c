#include<stdlib.h>
#include "matrix.h"
#include "translate.h"



int main(){
	struct Network base;
	base.numLayers = 4;
	int* layerSizes = calloc(base.numLayers,sizeof(int));
	//setting the sizes of the layers
	layerSizes[0] = 28*28;
	layerSizes[1] = 256;
	layerSizes[2] = 256; 
	layerSizes[3] = 10; 

	base.sizes = layerSizes;

	beginNet(&base);
	netImport(&base);

	//matrixH2WB(&base);

	int epoch = 100;
	int batchSize = 500;
	double LR = 0.02;
	int right = 0;
	int action = 0;
	int rightAction = 0;

	//learning loop	
	//for(int i=0;i<epoch;i++){
	//	for(int j=0;j<batchSize;j++){
	//		char* filename = calloc(50,sizeof(char));
	//		double* input = getinput(getimage(),filename);
	//		for(int k=0;k<28*28;k++)
	//			input[k] = input[k] > 0 ? 1 : 0;
	//		base.matrixN[0] = matrix2D(28*28,1);
	//		for(int k=0;k<28*28;k++)
	//			base.matrixN[0][k][0] = input[k];
	//		matrixNormalize(base.sizes[0],1,base.matrixN[0]);
	//		base.matrixNRELU[0] = matrix2D(28*28,1);
	//		for(int k=0;k<28*28;k++)
	//			base.matrixNRELU[0][k][0] = base.matrixN[0][k][0];
	//		Forward(&base);
	//		action = matrixMaxIndex(base.sizes[base.numLayers-1],1,base.matrixNRELU[base.numLayers-1]);
	//		rightAction = filename[18]-48;
	//		double** targetV = matrix2D(base.sizes[base.numLayers-1],1);
	//		targetV[rightAction][0] = 1;
	//		matrixNormalize(base.sizes[base.numLayers-1],1,targetV);
	//		Backprop(&base,targetV);
	//		resetNet(&base);
	//		free(filename);
	//		free(input);
	//	}
	//	for(int j=0;j<base.numLayers-1;j++){
	//		matrixMultiplyScalar(base.sizes[j+1],1,base.matrixCB[j],LR/batchSize);
	//		matrixNormalize(base.sizes[j+1],1,base.matrixCB[j]);
	//		matrixMultiplyScalar(base.sizes[j+1],base.sizes[j],base.matrixCW[j],LR/batchSize);
	//		matrixNormalize(base.sizes[j+1],base.sizes[j],base.matrixCW[j]);
	//	}
	//	for(int j=0;j<base.numLayers-1;j++){
	//		matrixSubtractionEqual(base.sizes[j+1],1,base.matrixCB[j],base.matrixB[j],0,0);		
	//		matrixSubtractionEqual(base.sizes[j+1],base.sizes[j],base.matrixCW[j],base.matrixW[j],0,0);		
	//	}
	//	for(int j=0;j<base.numLayers-1;j++){
	//		matrixMultiplyScalar(base.sizes[j+1],1,base.matrixCB[j],0);
	//		matrixMultiplyScalar(base.sizes[j+1],base.sizes[j],base.matrixCW[j],0);
	//	}
	//	printf("epoch = %i\n",i);
	//}

	//testing loop
	for(int i=0;i<10;i++){
		char* filename = calloc(50,sizeof(char));
		double* input = getinput(getimage(),filename);
		for(int k=0;k<28*28;k++)
			input[k] = input[k] > 0 ? 1 : 0;

		for(int j=0;j<28*28;j++){
			if(j%29 != 0){
				if((int)input[j] > 0){
					printf("%i",(int)input[j]);
				}
				else{
					printf("%i",(int)input[j]);		
				}
			}
			else{
				j++;
				printf("\n");
			}
		}

		base.matrixN[0] = matrix2D(28*28,1);
		for(int k=0;k<28*28;k++)
			base.matrixN[0][k][0] = input[k];
		matrixNormalize(base.sizes[0],1,base.matrixN[0]);
		base.matrixNRELU[0] = matrix2D(28*28,1);
		for(int k=0;k<28*28;k++)
			base.matrixNRELU[0][k][0] = base.matrixN[0][k][0];
		Forward(&base);
		action = matrixMaxIndex(base.sizes[base.numLayers-1],1,base.matrixNRELU[base.numLayers-1]);
		rightAction = filename[18]-48;
		printf("action = %i, right action = %i",action,rightAction);
		if(action == rightAction)
			right++;
		resetNet(&base);
		free(filename);
		free(input);
	}
		
	netExport(&base);	
	

	printf("\n\nResult = %f\n\n",(double)right/10);

	printf("\n\nFreeing...\n\n");
	freeNet(&base);
}
