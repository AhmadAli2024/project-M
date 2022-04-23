#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "GeneralFunctions.h"
#ifndef MATRIXOPERATIONS_H 
#define MATRIXOPERATIONS_H 


//returns the copy of a matrix
void matrixCopy(int row,int col,double* matrix,double* matrixRes);

//sets all elements of a matrix to num
void matrixSet(int row,int col,double* matrix,double num);

//preformes matrix2-=matrix1
void matrixSubtractionEqual(int row,int col,double* matrix1,double* matrix2);

//preformes matrix2+=matrix1
void matrixAdditionEqual(int row,int col,double* matrix1,double* matrix2);

//returns the transpose of the matrix
void matrixTranspose(int row,int col,double* matrix,double* matrixRes);

//frees a matrix
void matrixFree(int row,int col,double** matrix);

//display a 2d matrix
void matrixDisplay(int row,int col,double* table);

//allocate memory for a 2D matrix 
double* matrix2D(int row,int col);

//adds 2 2D matricies
void matrixAddition(int row,int col,double* matrix1,double* matrix2,double* matrixRes);

//subtracts matrix 1 to matrix 2
void matrixSubtraction(int row,int col,double* matrix1,double* matrix2,double* matrixRes);

//returns a matrix with random values between -1 and 1
void matrixMake(int row,int col,double* matrixRes);

//multiplys 2 matrices return the resault
void matrixMultiply(int rows,int colums,double* Matrix,int rows1,int colums1,double* Matrix1,double* matrixRes);

//normalizes the matrix
void matrixNormalize(int row,int col,double* matrix);

//returns the sum of the matrix
double matrixSum(int row,int col,double* matrix);

//multiplys all matrix elements by a num
void matrixMultiplyScalar(int row,int col,double* matrix,double num);

//preformes hamard matrix multiplication
void matrixHamard(int row,int col,double* matrix,double* matrix1,double* matrixRes);

//returns index of max element in a vector
int matrixMaxIndex(int row,int col,double* matrix);

//made if you want to free matrix1 or matrix2 after calling this function
void matrixAddition1(int row,int col,double* matrix1,double* matrix2,double* matrixRes,int f1,int f2);

//made if you want to free matrix1 after calling this function
void matrixAdditionEqual1(int row,int col,double* matrix1,double* matrix2,int f1);

//made if you want to free matrix1 or matrix2 after calling this function
void matrixSubtraction1(int row,int col,double* matrix1,double* matrix2,double* matrixRes,int f1,int f2);

//made if you want to free matrix1 after calling this function
void matrixSubtractionEqual1(int row,int col,double* matrix1,double* matrix2,int f1);

//made if you want to free matrix1 or matrix2 after calling this function
void matrixMultiply1(int rows,int colums,double* Matrix,int rows1,int colums1,double* Matrix1,double* matrixRes,int f1,int f2);

//made if you want to free matrix1 or matrix2 after calling this function
void matrixHamard1(int row,int col,double* matrix,double* matrix1,double* matrixRes,int f1,int f2);

void matrixCopy(int row,int col,double* matrix,double* matrix1){
	for(int i=0;i<row*col;i++)
		matrix1[i] = matrix[i];
}

void matrixSet(int row,int col,double* matrix,double num){
	for(int i=0;i<row*col;i++)
		matrix[i] = num;
}

void matrixTranspose(int row,int col,double* matrix,double* matrixRes){
	for(int i=0;i<row;i++)
		for(int j=0;j<col;j++)
			matrixRes[j*row+i] = matrix[i*col+j];
}

int matrixMaxIndex(int row,int col,double* matrix){
	int max =0;
	for(int i=0;i<row*col;i++)
		if(matrix[max] < matrix[i])
			max = i;
	return max;
}

void matrixFree(int row,int col,double** matrix){
	for(int i=0;i<row;i++)
		free(matrix[i]);
	free(matrix);
}

double matrixSum(int row,int col,double* matrix){
	double sum = 0;
	for(int i =0;i<row*col;i++)
		sum+=matrix[i];
	return sum;
}

void matrixNormalize(int row,int col,double* matrix){
	double sum = matrixSum(row,col,matrix);
	sum/=(row*col);
	for(int i =0;i<row*col;i++)
		matrix[i]-=sum;
}

void matrixMultiplyScalar(int row,int col,double* matrix,double num){
	for(int i=0;i<row*col;i++)
		matrix[i]*=num;
}

void matrixDisplay(int row,int col,double* table){
	for(int i=0;i<row;i++){
		for(int j =0;j<col;j++)
			printf("%f ",table[i*col+j]);
		printf("\n");
	}
}

void matrixMake(int row,int col,double* matrixRes){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = randomDouble(-1,1);
}

void matrixAddition(int row,int col,double* matrix1,double* matrix2,double* matrixRes){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = matrix1[i]+matrix2[i];
}

void matrixAddition1(int row,int col,double* matrix1,double* matrix2,double* matrixRes,int f1,int f2){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = matrix1[i]+matrix2[i];
	if(f1)
		free(matrix1);
	if(f2)
		free(matrix2);
}

void matrixAdditionEqual(int row,int col,double* matrix1,double* matrix2){
	for(int i=0;i<row*col;i++)
		matrix2[i]+=matrix1[i];
}

void matrixAdditionEqual1(int row,int col,double* matrix1,double* matrix2,int f1){
	for(int i=0;i<row*col;i++)
		matrix2[i]+=matrix1[i];
	if(f1)
		free(matrix1);
}

void matrixSubtraction(int row,int col,double* matrix1,double* matrix2,double* matrixRes){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = matrix1[i]-matrix2[i];
}

void matrixSubtraction1(int row,int col,double* matrix1,double* matrix2,double* matrixRes,int f1,int f2){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = matrix1[i]-matrix2[i];
	if(f1)
		free(matrix1);
	if(f2)
		free(matrix2);
}

void matrixSubtractionEqual(int row,int col,double* matrix1,double* matrix2){
	for(int i=0;i<row*col;i++)
		matrix2[i]-=matrix1[i];
}

void matrixSubtractionEqual1(int row,int col,double* matrix1,double* matrix2,int f1){
	for(int i=0;i<row*col;i++)
		matrix2[i]-=matrix1[i];
	if(f1)
		free(matrix1);
}

void matrixMultiply(int rows,int colums,double* Matrix,int rows1,int colums1,double* Matrix1,double* matrixRes){
	for(int i=0;i<rows;i++)
		for(int j=0;j<colums1;j++)
			for(int k=0;k<colums;k++)
				matrixRes[i*colums1+j] += Matrix[i*colums+k]*Matrix1[k*colums1+j];
}

void matrixMultiply1(int rows,int colums,double* Matrix,int rows1,int colums1,double* Matrix1,double* matrixRes,int f1,int f2){
	for(int i=0;i<rows;i++)
		for(int j=0;j<colums1;j++)
			for(int k=0;k<colums;k++)
				matrixRes[i*colums1+j] += Matrix[i*colums+k]*Matrix1[k*colums1+j];
	if(f1)
		free(Matrix);
	if(f2)
		free(Matrix1);
}

void matrixHamard(int row,int col,double* matrix,double* matrix1,double* matrixRes){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = matrix[i]*matrix1[i];
}

void matrixHamard1(int row,int col,double* matrix,double* matrix1,double* matrixRes,int f1,int f2){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = matrix[i]*matrix1[i];
	if(f1)
		free(matrix);
	if(f2)
		free(matrix1);
}
#endif
