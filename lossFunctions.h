#include<stdio.h>
#include "MatrixOperations.h"
#ifndef LOSSFUNCTIONS_H 
#define LOSSFUNCTIONS_H 

double MSE(double a,double y){ return (a-y)*2; }

void matrixMSE(int row,int col,double* matrix,double* matrix1,double* matrixRes){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = MSE(matrix[i],matrix1[i]);
}

double MAE(double a,double y){ return ((a-y)/absD(a-y)); }

void matrixMAE(int row,int col,double* matrix,double* matrix1,double* matrixRes){
	for(int i=0;i<row*col;i++){
		matrixRes[i] = MAE(matrix[i],matrix1[i]);
	}
}

double Hubor(double a,double y,double param){ return abs(a-y) < param ? a-y : param; }

void matrixHubor(int row,int col,double* matrix,double* matrix1,double* matrixRes,double param){
	for(int i=0;i<row*col;i++){
		matrixRes[i] = Hubor(matrix[i],matrix1[i],param);
	}
}

double SH(double a,double y){ return a*y >= 1 ? 0 : 2*y*(a*y-1); }

void matrixSH(int row,int col,double* matrix,double* matrix1,double* matrixRes){
	for(int i=0;i<row*col;i++){
		matrixRes[i] = SH(matrix[i],matrix1[i]);
	}
}

double BinaryCrossEntropy(double a,double y){ return ((-2*a*y)+a+y)/((y-1)*y); }

void matrixBinaryCrossEntropy(int row,int col,double* matrix,double* matrix1,double* matrixRes){
	for(int i=0;i<row*col;i++){
		matrixRes[i] = BinaryCrossEntropy(matrix[i],matrix1[i]);
	}
}

double CrossEntropy(double a,double y){ return -y/a; }

void matrixCrossEntropy(int row,int col,double* matrix,double* matrix1,double* matrixRes){
	for(int i=0;i<row*col;i++){
		matrixRes[i] = CrossEntropy(matrix[i],matrix1[i]);
	}
}

double CategoricalCrossEntropy(double a,double y){ return y > 0 ? a-1 : a; }

void matrixCategoricalCrossEntropy(int row,int col,double* matrix,double* matrix1,double* matrixRes){
	for(int i=0;i<row*col;i++){
		matrixRes[i] = CategoricalCrossEntropy(matrix[i],matrix1[i]);
	}
}
#endif
