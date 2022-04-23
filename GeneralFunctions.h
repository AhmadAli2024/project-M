#include<stdio.h>
#include<math.h>
#ifndef GENERALFUNCTIONS_H 
#define GENERALFUNCTIONS_H 



//abs of a number
double absD(double x);

//returns a random double number
double randomDouble(double min, double max);


double randomDouble(double min, double max){
	return min + (rand() / (RAND_MAX/(max-min)));
}

double absD(double x){ return x < 0 ? x*-1 : x; } 

#endif
