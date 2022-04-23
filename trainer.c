#include<stdlib.h>
#include "translate.h"
#include "MatrixOperations.h"
#include "NetworkOperations.h"
#include "activation.h"
#include "lossFunctions.h"
#include "optimizers.h"

int main(){
	struct Network base;
	base.numLayers = 4;

	int* layerSizes = calloc(base.numLayers,sizeof(int));
	layerSizes[0] = 28*28;
	layerSizes[1] = 125;
	layerSizes[2] = 125;
	layerSizes[3] = 10;

	int* activation = calloc(base.numLayers-1,sizeof(int));
	activation[0] = 3; 
	activation[1] = 3; 
	activation[2] = 7;
	//best: 3,8,9,10
	//setting the sizes of the layers
	base.activations = activation;
	base.sizes = layerSizes;

	beginNet(&base);
	netImport(&base);

	//matrixH2WB(&base);

	int epoch = 1000;
	int batchSize = 1;
	//double LR = 0.0211;
	double LR = 0.0999999999999; //LR for Momentum
	//double LR = 0.0001; //LR for RMS prop
	int right = 0;
	int action = 0;
	int rightAction = 0;

	base.learningRate = LR;

	double* targetV = calloc(base.sizes[base.numLayers-1],sizeof(double));
	double* lossFunction = calloc(base.sizes[base.numLayers-1],sizeof(double));
	char* filename = calloc(50,sizeof(char));
	

	double** SDW = calloc(3,sizeof(double*));
	double** SDB = calloc(3,sizeof(double*));
	for(int i=0;i<base.numLayers-1;i++){
		SDW[i] = calloc(base.sizes[i+1]*base.sizes[i],sizeof(double));
		SDB[i] = calloc(base.sizes[i+1],sizeof(double));
	}

	double** VDW = calloc(3,sizeof(double*));
	double** VDB = calloc(3,sizeof(double*));
	for(int i=0;i<base.numLayers-1;i++){
		VDW[i] = calloc(base.sizes[i+1]*base.sizes[i],sizeof(double));
		VDB[i] = calloc(base.sizes[i+1],sizeof(double));
	}

	clock_t begin = clock();
	//learning loop	
	for(int i=0;i<epoch;i++){
		for(int j=0;j<batchSize;j++){
			while(1){
				if(getimage(filename) == 1){
					break;
				}
			}
			getinput(filename,base.matrixN[0]);
			for(int k=0;k<28*28;k++){
				base.matrixN[0][k] = base.matrixN[0][k] > 0 ? 1 : 0;
			}

			//if(j == 1){
			//	for(int k=0;k<28*28;k++){
			//		if(k%29 != 0){
			//			printf("%i",(int)base.matrixN[0][k]);
			//		}
			//		else{
			//			k++;
			//			printf("\n");
			//		}
			//	}
			//}

			matrixNormalize(base.sizes[0],1,base.matrixN[0]);
			matrixCopy(base.sizes[0],1,base.matrixN[0],base.matrixNRELU[0]);
			Forward(&base);

			//matrixDisplay(10,1,base.matrixNRELU[3]);

			action = matrixMaxIndex(base.sizes[base.numLayers-1],1,base.matrixNRELU[base.numLayers-1]);
			rightAction = filename[18]-48;
			targetV[rightAction] = 1;
			matrixNormalize(base.sizes[base.numLayers-1],1,targetV);
			matrixCategoricalCrossEntropy(base.sizes[base.numLayers-1],1,base.matrixNRELU[base.numLayers-1],targetV,lossFunction);
			//matrixMSE(base.sizes[base.numLayers-1],1,base.matrixNRELU[base.numLayers-1],targetV,lossFunction);
			//matrixSH(base.sizes[base.numLayers-1],1,base.matrixNRELU[base.numLayers-1],targetV,lossFunction);
			Backprop(&base,lossFunction);
			resetNet(&base);
			matrixSet(base.sizes[base.numLayers-1],1,targetV,0);
			for(int k=0;k<50;k++)
				filename[k] = ' ';
		}
		//MBGD(&base,100);
		Momentum(&base,SDW,SDB,0.9);
		//RMS(&base,SDW,SDB,0.9);
		//Adam(&base,SDW,SDB,VDW,VDB,0.999,0.005);
		//AGD(&base);
		printf("epoch = %i\n",i);
	}
	clock_t end = clock();
	long double time_spent = (long double)(end - begin) / CLOCKS_PER_SEC;
	printf("\ntime spent = %Le\n",time_spent);

	//testing loop
	for(int i=0;i<1000;i++){
		while(1){
			if(getimage(filename) == 1){
				break;
			}
		}
		getinput(filename,base.matrixN[0]);
		for(int k=0;k<28*28;k++)
			base.matrixN[0][k] = base.matrixN[0][k] > 0 ? 1 : 0;

		//for(int k=0;k<28*28;k++){
		//	if(k%29 != 0){
		//		printf("%i",(int)base.matrixN[0][k]);
		//	}
		//	else{
		//		k++;
		//		printf("\n");
		//	}
		//}

		matrixNormalize(base.sizes[0],1,base.matrixN[0]);

		matrixCopy(base.sizes[0],1,base.matrixN[0],base.matrixNRELU[0]);
		Forward(&base);
		action = matrixMaxIndex(base.sizes[base.numLayers-1],1,base.matrixNRELU[base.numLayers-1]);
		rightAction = filename[18]-48;
		if(action == rightAction)
			right++;
		//printf("action = %i\nright action = %i\n\n",action,rightAction);
		resetNet(&base);
		for(int k=0;k<50;k++)
			filename[k] = ' ';
	}

	netExport(&base);	

	printf("\n\nResult = %f\n\n",(double)right/10);

	printf("\n\nFreeing...\n\n");
	freeNet(&base);
	free(targetV);
	free(lossFunction);
	free(filename);
	return 0;
}
