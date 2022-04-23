#include<math.h>
#include "MatrixOperations.h"
#ifndef ACTIVATION_H 
#define ACTIVATION_H 


//all activation function and their derivitive
//resources https://blog.knoldus.com/activation-function-in-neural-network

//1
double RELU(double num);
void matrixRELU(int row,int col,double* matrix,double* matrixRes);
double RELUDev(double num);
void matrixRELUDev(int row,int col,double* matrix,double* matrixRes);

//2
double Sigmoid(double num);
void matrixSigmoid(int row,int col,double* matrix,double* matrixRes);
double SigmoidDev(double num);
void matrixSigmoidDev(int row,int col,double* matrix,double* matrixRes);

//3
double Tanh(double num);
void matrixTanh(int row,int col,double* matrix,double* matrixRes);
double TanhDev(double num);
void matrixTanhDev(int row,int col,double* matrix,double* matrixRes);

//4
double LeakyRELU(double num);
void matrixLeakyRELU(int row,int col,double* matrix,double* matrixRes);
double LeakyRELUDev(double num);
void matrixLeakyRELUDev(int row,int col,double* matrix,double* matrixRes);

//5
double ParamRELU(double num,double param);
void matrixParamRELU(int row,int col,double* matrix,double* matrixRes,double param);
double ParamRELUDev(double num,double param);
void matrixParamRELUDev(int row,int col,double* matrix,double* matrixRes,double param);

//6
double ELU(double num,double param);
void matrixELU(int row,int col,double* matrix,double* matrixRes,double param);
double ELUDev(double num,double param);
void matrixELUDev(int row,int col,double* matrix,double* matrixRes,double param);

//7
void matrixSoftMax(int row,int col,double* matrix,double* matrixRes);
void matrixSoftMaxDev(int row,int col,double* matrix,double* matrixRes);

//8
double Swish(double num,double param);
void matrixSwish(int row,int col,double* matrix,double* matrixRes,double param);
double SwishDev(double num,double param);
void matrixSwishDev(int row,int col,double* matrix,double* matrixRes,double param);

//9
double GELU(double num);
void matrixGELU(int row,int col,double* matrix,double* matrixRes);
double GELUDev(double num);
void matrixGELUDev(int row,int col,double* matrix,double* matrixRes);

//10
double SELU(double num);
void matrixSELU(int row,int col,double* matrix,double* matrixRes); 
double SELUDev(double num);
void matrixSELUDev(int row,int col,double* matrix,double* matrixRes); 

//SELU Function
double SELU(double num){ double long h = 1.0507009873554804934193349852946;double long a = 1.6732632423543772848170429916717; return num > 0 ? num*h : (a*h)*(pow(M_E,num)-1); }

void matrixSELU(int row,int col,double* matrix,double* matrixRes){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = SELU(matrix[i]);
}

double SELUDev(double num){ double long h = 1.0507009873554804934193349852946;double long a = 1.6732632423543772848170429916717; return num > 0 ? h : a*h*pow(M_E,num); }

void matrixSELUDev(int row,int col,double* matrix,double* matrixRes){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = SELUDev(matrix[i]);
}


//GELU Function
double GELU(double num){ return (0.5*num)*(1+Tanh(sqrt(2/M_PI)*(num+(0.044715*pow(num,3))))); }

void matrixGELU(int row,int col,double* matrix,double* matrixRes){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = GELU(matrix[i]);
}

double GELUDev(double num){ return 0.5*Tanh(0.0356774*pow(num,3)+0.797885*num)+(0.0535161*pow(num,3)+0.398942*num)*TanhDev(0.0356774*pow(num,3)+0.797885*num)+0.5; }

void matrixGELUDev(int row,int col,double* matrix,double* matrixRes){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = GELUDev(matrix[i]);
}

//Swish Function 
double Swish(double num,double param){ return num*Sigmoid(num*param); }

void matrixSwish(int row,int col,double* matrix,double* matrixRes,double param){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = Swish(matrix[i],param);
}

double SwishDev(double num,double param){ double sw = Swish(num,param); return (sw*param)+(Sigmoid(param*num)*(1-(sw*param))); }

void matrixSwishDev(int row,int col,double* matrix,double* matrixRes,double param){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = SwishDev(matrix[i],param);
}


//softMax function
void matrixSoftMax(int row,int col,double* matrix,double* matrixRes){
	long double sum = 0;
	for(int i=0;i<row*col;i++){
		sum+=pow(M_E,matrix[i]);
	}
	for(int i=0;i<row*col;i++){
		matrixRes[i] = pow(M_E,matrix[i])/sum;
	}
}

void matrixSoftMaxDev(int row,int col,double* matrix,double* matrixRes){
	double* buffer = calloc(row*col,sizeof(double));
	for(int i=0;i<row*col;i++)
		buffer[i] = pow(M_E,matrix[i]);
	double sum = matrixSum(row,col,buffer);
	sum*=sum;
	double test = 0;
	for(int i=0;i<row*col;i++){
		for(int j=0;j<row*col;j++){
			if(i!=j){
				test+=buffer[i];
			}
		}
		matrixRes[i] = pow(M_E,matrix[i])*test/sum;
		test = 0;
	}
	free(buffer);
}

//ELU function
double ELU(double num,double param){ return num >= 0 ? num : param*(pow(M_E,num)-1); }

void matrixELU(int row,int col,double* matrix,double* matrixRes,double param){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = ELU(matrix[i],param);
}

double ELUDev(double num,double param){ return num >= 0 ? num : ELU(num,param)+param; }

void matrixELUDev(int row,int col,double* matrix,double* matrixRes,double param){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = ELUDev(matrix[i],param);
}

//ParamRELU function
double ParamRELU(double num,double param){ return num >=0 ? num : num*param; }

void matrixParamRELU(int row,int col,double* matrix,double* matrixRes,double param){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = ParamRELU(matrix[i],param);
}

double ParamRELUDev(double num,double param){ return num >=0 ? 1 : param; }

void matrixParamRELUDev(int row,int col,double* matrix,double* matrixRes,double param){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = ParamRELUDev(matrix[i],param);
}

//LeakyRELU function
double LeakyRELU(double num){ return num >= 0 ? num : num*0.1; }

void matrixLeakyRELU(int row,int col,double* matrix,double* matrixRes){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = LeakyRELU(matrix[i]);
}

double LeakyRELUDev(double num){ return num >= 0 ? 1 : 0.01; }

void matrixLeakyRELUDev(int row,int col,double* matrix,double* matrixRes){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = LeakyRELUDev(matrix[i]);
}

//Tanh Function
double Tanh(double num){ return (pow(M_E,num)-pow(M_E,-num))/(pow(M_E,num)+pow(M_E,-num)); }

void matrixTanh(int row,int col,double* matrix,double* matrixRes){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = Tanh(matrix[i]);
}

double TanhDev(double num){ return 4/(pow(pow(M_E,-num)+pow(M_E,num),2));}

void matrixTanhDev(int row,int col,double* matrix,double* matrixRes){
	for(int i=0;i<row;i++)
		matrixRes[i] = TanhDev(matrix[i]);
}

//Sigmoid Function
double Sigmoid(double num){ return 1/(1+pow(M_E,-num)); }

void matrixSigmoid(int row,int col,double* matrix,double* matrixRes){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = Sigmoid(matrix[i]);
}

double SigmoidDev(double num){ double buff = Sigmoid(num); return num*(1-num); }

void matrixSigmoidDev(int row,int col,double* matrix,double* matrixRes){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = SigmoidDev(matrix[i]);
}

//RELU function
double RELU(double num){ return num >= 0 ? num : 0;}

void matrixRELU(int row,int col,double* matrix,double* matrixRes){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = RELU(matrix[i]);
}

double RELUDev(double num){ return num >= 0 ? 1 : 0;}

void matrixRELUDev(int row,int col,double* matrix,double* matrixRes){
	for(int i=0;i<row*col;i++)
		matrixRes[i] = RELUDev(matrix[i]);
}
#endif
