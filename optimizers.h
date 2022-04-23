#include<stdio.h>
#include<math.h>
#include "NetworkOperations.h"
#include "MatrixOperations.h"
#ifndef OPTIMIZERS_H 
#define OPTIMIZERS_H 


void GD(struct Network* net){
	for(int i=0;i<net->numLayers-1;i++){
		for(int j=0;j<net->sizes[i+1]*net->sizes[i];j++)
			net->matrixW[i][j]-= net->matrixCW[i][j]*net->learningRate;
		for(int j=0;j<net->sizes[i+1];j++)
			net->matrixB[i][j]-= net->matrixCB[i][j]*net->learningRate;
		matrixSet(net->sizes[i+1],1,net->matrixCB[i],0);
		matrixSet(net->sizes[i+1],net->sizes[i],net->matrixCW[i],0);
	}
}

void MBGD(struct Network* net,int batchSize){
	for(int i=0;i<net->numLayers-1;i++){
		for(int j=0;j<net->sizes[i+1]*net->sizes[i];j++){
			net->matrixW[i][j]-=net->matrixCW[i][j]*(net->learningRate/batchSize);
		}
		for(int j=0;j<net->sizes[i+1];j++){
			net->matrixB[i][j]-=net->matrixCB[i][j]*(net->learningRate/batchSize);
		}
		matrixSet(net->sizes[i+1],1,net->matrixCB[i],0);
		matrixSet(net->sizes[i+1],net->sizes[i],net->matrixCW[i],0);
	}
}

void Momentum(struct Network* net,double** weightMomentum,double** biasMomentum,double beta){
	for(int i=0;i<net->numLayers-1;i++){
		for(int j=0;j<net->sizes[i+1]*net->sizes[i];j++){
			weightMomentum[i][j] = ((weightMomentum[i][j]*beta)+((1-beta)*net->matrixCW[i][j]))*net->learningRate;
		}
		for(int j=0;j<net->sizes[i+1];j++){
			biasMomentum[i][j] = ((biasMomentum[i][j]*beta)+((1-beta)*net->matrixCB[i][j]))*net->learningRate;
		}
		matrixSubtractionEqual(net->sizes[i+1],1,biasMomentum[i],net->matrixB[i]);		
		matrixSubtractionEqual(net->sizes[i+1],net->sizes[i],weightMomentum[i],net->matrixW[i]);		
		matrixSet(net->sizes[i+1],1,net->matrixCB[i],0);
		matrixSet(net->sizes[i+1],net->sizes[i],net->matrixCW[i],0);
	}
}

void AGD(struct Network* net){
	double alphaW;
	double alphaB;
	double nW;
	double nB;
	for(int i=0;i<net->numLayers-1;i++){
		for(int j=0;j<net->sizes[i+1]*net->sizes[i];j++)
			alphaW+= net->matrixCW[i][j]*net->matrixCW[i][j];
		for(int j=0;j<net->sizes[i+1];j++)
			alphaB+= net->matrixCB[i][j]*net->matrixCB[i][j];
		nW = net->learningRate/(sqrt(alphaW+0.000001));
		nB = net->learningRate/(sqrt(alphaB+0.000001));
		for(int j=0;j<net->sizes[i+1]*net->sizes[i];j++)
			net->matrixW[i][j]-= net->matrixCW[i][j]*nW;
		for(int j=0;j<net->sizes[i+1];j++)
			net->matrixB[i][j]-= net->matrixCB[i][j]*nB;
		matrixSet(net->sizes[i+1],1,net->matrixCB[i],0);
		matrixSet(net->sizes[i+1],net->sizes[i],net->matrixCW[i],0);
		alphaW = 0;
		alphaB = 0;
	}
}

void RMS(struct Network* net,double** weightMomentum,double** biasMomentum,double beta){
	for(int i=0;i<net->numLayers-1;i++){
		for(int j=0;j<net->sizes[i+1]*net->sizes[i];j++){
			weightMomentum[i][j] = (weightMomentum[i][j]*beta)+((1-beta)*net->matrixCW[i][j]*net->matrixCW[i][j]);
			net->matrixW[i][j]-= (net->learningRate/(sqrt(weightMomentum[i][j]+0.0000001)))*net->matrixCW[i][j];
		}
		for(int j=0;j<net->sizes[i+1];j++){
			biasMomentum[i][j] = (biasMomentum[i][j]*beta)+((1-beta)*net->matrixCB[i][j]*net->matrixCB[i][j]);
			net->matrixB[i][j]-= (net->learningRate/(sqrt(biasMomentum[i][j]+0.0000001)))*net->matrixCB[i][j];
		}
		matrixSet(net->sizes[i+1],net->sizes[i],net->matrixCW[i],0);
		matrixSet(net->sizes[i+1],1,net->matrixCB[i],0);
	}
}	

void Adam(struct Network* net,double** SDW,double** SDB,double** VDW,double** VDB,double beta,double beta2){
	for(int i=0;i<net->numLayers-1;i++){
		for(int j=0;j<net->sizes[i+1]*net->sizes[i];j++){
			SDW[i][j] = (SDW[i][j]*beta)+((1-beta)*net->matrixCW[i][j]*net->matrixCW[i][j]);
			VDW[i][j] = (VDW[i][j]*beta2)+((1-beta2)*net->matrixCW[i][j]);
			net->matrixW[i][j]-= (net->learningRate/(sqrt(SDW[i][j]+0.0000001)))*VDW[i][j];
		}
		for(int j=0;j<net->sizes[i+1];j++){
			SDB[i][j] = (SDB[i][j]*beta)+((1-beta)*net->matrixCB[i][j]*net->matrixCB[i][j]);
			VDB[i][j] = (VDB[i][j]*beta2)+((1-beta2)*net->matrixCB[i][j]);
			net->matrixB[i][j]-= (net->learningRate/(sqrt(SDB[i][j]+0.0000001)))*VDB[i][j];
		}
		matrixSet(net->sizes[i+1],net->sizes[i],net->matrixCW[i],0);
		matrixSet(net->sizes[i+1],1,net->matrixCB[i],0);
	}
}	
#endif
