#include<stdio.h>
#include<stdlib.h>
#include<dirent.h>
#include<time.h>
#ifndef TRANSLATE_H 
#define TRANSLATE_H 


void printarr(int length,int* arr);

void setarr(int length,int num,int* arr);

void getChunk(FILE* input,int length,int* arr);

void getSeq(int length,int* arr,int* seq);

int power(int x,int pow);

int bigEd(int* arr,int length);

int smallEd(int* arr,int length);

int decode(int* seq,int length,int* decoded);

void lengthtable(int num,int* bit,int* len);

void offsettable(int x,int* bit,int* dis);

int getrandom(int min,int max);

int getimage(char* final);
//int getimage(char* name,int num);

int check(int* seq);

void getinput(char* filename,double* decoded);

void getinput(char* filename,double* decodeDouble){
	FILE* input;
	input = fopen(filename,"r");
	int* buffer = calloc(36,sizeof(int));
	getChunk(input,36,buffer);
	int* buffer1 = calloc(4,sizeof(int));
	int IDATlength = getc(input);
	getChunk(input,4,buffer1);
	int* IDAT = calloc(IDATlength,sizeof(int));
	int* IDATseq = calloc(IDATlength*8,sizeof(int));
	getChunk(input,IDATlength,IDAT);
	getSeq(IDATlength,IDAT,IDATseq);
	int* decoded = calloc(28*28,sizeof(int));
	decode(IDATseq,IDATlength,decoded);
	for(int i=0;i<28*28;i++)
		decodeDouble[i] = (int)decoded[i];
	int zerocount =0;
	int onecount = 0;
	for(int i=0;i<28*28;i++)
		if((int)decodeDouble[i] == 0)
			zerocount++;
		else
			onecount++;
	free(IDAT);
	free(buffer);
	free(buffer1);
	free(IDATseq);
	free(decoded);
	fclose(input);
	//if(zerocount >= 28*27 || onecount>= 28*28/3){
	//	while(1)
	//		if(getimage(filename) == 1)
	//			break;
	//	getinput(filename,decodeDouble);
	//}
}



int check(int* seq){
	int arr[] = {0,0,0,1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,0};
	for(int i=0;i<27;i++)
		if(arr[i] != seq[i])
			return 0;
	return 1;
}

//int getimage(char* name,int num){
//    char image[] = "./mnist_png/trait/";
//	for(int i=0;i<18;i++)
//		name[i] = image[i];
//	name[18] = 0+48;
//	name[19] = '/';
//	name[20] = '1';
//	name[21] = ' ';
//	name[22] = '(';
//	int counter = 1;
//	while(num > 0){
//		name[22+counter] = (num%10)+48;
//		num/=10;
//		counter++;
//	}
//	//name[22+counter] = num+48;
//	//counter++;
//	name[22+counter] = ')';
//	counter++;
//	name[22+counter] = '.';
//	name[22+counter+1] = 'p';
//	name[22+counter+2] = 'n';
//	name[22+counter+3] = 'g';
//	name[22+counter+4] = '\0';
//	FILE* test;
//	if(test = fopen(name,"r")){
//		fclose(test);
//		return 1;
//	}
//	else{
//		free(test);
//		for(int i=0;i<50;i++)
//			name[i] = ' ';
//		return 0;
//	}
//}


int getimage(char* name){
    char image[] = "./mnist_png/trait/";
	for(int i=0;i<18;i++)
		name[i] = image[i];
	name[18] = getrandom(0,9)+48;
	name[19] = '/';
	name[20] = '1';
	name[21] = ' ';
	name[22] = '(';
	int num = getrandom(1,5400);
	int rev = 0;
	int rem = 0;
	while(num !=0){
		rem = num%10;
		rev = rev*10+rem;
		num/=10;
	}
	num = rev;
	int counter = 1;
	if(num == 0){
		name[22+counter] = '0';
		counter++;
	}
	else{
		while(num != 0){
			name[22+counter] = num%10+48;
			num/=10;
			counter++;
		}
	}
	name[22+counter] = ')';
	counter++;
	name[22+counter] = '.';
	name[22+counter+1] = 'p';
	name[22+counter+2] = 'n';
	name[22+counter+3] = 'g';
	name[22+counter+4] = '\0';
	FILE* test;
	if(test = fopen(name,"r")){
		fclose(test);
		return 1;
	}
	else{
		free(test);
		for(int i=0;i<50;i++)
			name[i] = ' ';
		return 0;
	}
}

int getrandom(int min,int max){
	return (rand()%(max-min+1))+min;
}

int decode(int* seq,int length,int* decoded){
    unsigned int arrlen = 27;
    unsigned int decodelen = 0;
    int x =0;
    while(1){
		if(arrlen+40 >= length*8)
			return 1;
        //get 7 bits to identify the right interval in the fixxed huffman table
        int* buffer = calloc(7,sizeof(int));
        for(int i=0;i<7;i++){
            buffer[i] = seq[arrlen];
            arrlen++;
        }
        x = bigEd(buffer,7);
        //interval 1
        if(x>=24 && x<=95){
           buffer = (int*) realloc(buffer,sizeof(int[8]));
           buffer[7] = seq[arrlen];
           arrlen++;
           decoded[decodelen] = bigEd(buffer,8)-48;
           decodelen++;
		   free(buffer);
        } 
        //interval 2
        else if(x>=100 && x<=127){
           buffer = (int*) realloc(buffer,sizeof(int[9]));
           buffer[7] = seq[arrlen];
           arrlen++;
           buffer[8] = seq[arrlen];
           arrlen++;
           decoded[decodelen] = bigEd(buffer,9)-400+144;
           decodelen++;
		   free(buffer);
        }        
        //interval 3 
        else if(x>=0 && x<=23){
           int* extrabit = calloc(1,sizeof(int));
           int* length = calloc(1,sizeof(int));
           x+=256;
           lengthtable(x,extrabit,length);
		   if(*extrabit < 0){
			   free(extrabit);
			   free(length);
			   free(buffer);
			   return 0;
		   }
           if(*extrabit !=0){
               int* bits = calloc(*extrabit,sizeof(int));
               for(int i=0;i<*extrabit;i++){
                   bits[i] = seq[arrlen];
                   arrlen++;
               }
               *length+= smallEd(bits,*extrabit);
			   free(bits);
           }
           int* distance = calloc(1,sizeof(int));
           *extrabit = 0;
           int* offsetbit = calloc(5,sizeof(int));
           for(int i=0;i<5;i++){
               offsetbit[i] = seq[arrlen];
               arrlen++;
           }
           offsettable(bigEd(offsetbit,5),extrabit,distance);
		   free(offsetbit);
           if(*extrabit !=0){
               int* bits = calloc(*extrabit,sizeof(int));
               for(int i=0;i<*extrabit;i++){
                   bits[i] = seq[arrlen];
                   arrlen++;
               }
               *distance += smallEd(bits,*extrabit);
			   free(bits);
           }
		   if(*distance > decodelen || decodelen+*length > 28*28){
			   free(distance);
			   free(length);
			   free(extrabit);
			   free(buffer);
			   return 0;
		   }
		   int finallen = decodelen-*distance;
		   int* final = calloc((finallen+*length)-(finallen),sizeof(int));
		   for(int i=finallen;i<finallen+*length;i++)
			   final[i-(finallen)] = decoded[i];
		   for(int i=0;i<(finallen+*length)-(finallen);i++){
			   decoded[decodelen] = final[i];
			   decodelen++;
		   }
		   free(buffer);
		   free(extrabit);
		   free(final);
		   free(distance);
		   free(length);
        }
        //interval 4 
        else if(x>=96 && x<=99){
           buffer = (int*) realloc(buffer,sizeof(int[8]));
           buffer[7] = seq[arrlen];
           arrlen++;
           int* extrabit = calloc(1,sizeof(int));
           int* length = calloc(1,sizeof(int));
           x = bigEd(buffer,8)-192+280;
           lengthtable(x,extrabit,length);
           if(*extrabit > 0){
               int* bits = calloc(*extrabit,sizeof(int));
               for(int i=0;i<*extrabit;i++){
                   bits[i] = seq[arrlen];
                   arrlen++;
               }
               *length+=smallEd(bits,*extrabit);
			   free(bits);
           }
           int* distance = calloc(1,sizeof(int));
           *extrabit = 0;
           int* offsetbits = calloc(5,sizeof(int));
           for(int i=0;i<5;i++){
               offsetbits[i] = seq[arrlen];
               arrlen++;
           }
           offsettable(bigEd(offsetbits,5),extrabit,distance);
		   free(offsetbits);
           if(*extrabit !=0){
               int* bits = calloc(*extrabit,sizeof(int));
               for(int i=0;i<*extrabit;i++){
                   bits[i] = seq[arrlen];
                   arrlen++;
               }
               *distance += smallEd(bits,*extrabit);
			   free(bits);
           }
		   if(*distance > decodelen || decodelen+*length > 28*28){
			   free(extrabit);
			   free(buffer);
			   free(length);
			   free(distance);
			   return 0;
		   }
		   if(*distance == 0){
			   free(buffer);
			   free(distance);
			   free(length);
			   free(extrabit);
			   return 0;
		   }
		   int finallen = decodelen-*distance;
           int* final = calloc(*distance,sizeof(int));
           for(int i=finallen;i<decodelen;i++)
               final[i-(finallen)] = decoded[i];
           for(int i=0;i<(*length)-1;i++){
			   decoded[decodelen] = final[i%*distance];
			   decodelen++;
           }
		   free(buffer);
		   free(final);
		   free(distance);
		   free(length);
		   free(extrabit);
        }
		if(arrlen>= length*8 || decodelen >= 28*28)
			return 1;
    }
}


        
int bigEd(int* arr,int length){
    int final = 0;
    int value = 1;
    for(int i=0;i<length;i++){
        final+=arr[length-i-1]*value;
        value*=2;
    }
    return final;
}

int smallEd(int* arr,int length){
    int final = 0;
    int value = 1;
    for(int i=0;i<length;i++){
        final+=arr[i]*value;
        value*=2;
    }
    return final;
}

int power(int x,int pow){
    for(int i=0;i<pow;i++)
       x*=x; 
    return x;
}

void getSeq(int length,int* arr,int* seq){
    int* binary = calloc(8,sizeof(int));
    int x =0;
    for(int i=0;i<length;i++){
        x = arr[i];
        setarr(8,0,binary);
        for(int q=0;x>0;q++){
            binary[q] = x%2;
            x/=2;
        }
        for(int q=0;q<8;q++){
            seq[i*8+q]=binary[q];
        }
    }
	free(binary);
}


void printarr(int length,int* arr){
    for(int i=0;i<length;i++){
        printf("%i ",arr[i]);
    }
} 

void getChunk(FILE* input,int length,int* arr){
    for(int i=0;i<length;i++){
        arr[i] = getc(input);
    }
}

void setarr(int length,int num,int* arr){
    for(int i=0;i<length;i++){
        arr[i] = num;
    }
}

void lengthtable(int num,int* bit,int* len){
    switch(num){
        case(257):
            *bit = 0;
            *len = 3;
            break;
        case(258):
            *bit = 0;
            *len = 4;
            break;
        case(259):
            *bit = 0;
            *len = 5;
            break;
        case(260):
            *bit = 0;
            *len = 6;
            break;
        case(261):
            *bit = 0;
            *len = 7;
            break;
        case(262):
            *bit = 0;
            *len = 8;
            break;
        case(263):
            *bit = 0;
            *len = 9;
            break;
        case(264):
            *bit = 0;
            *len = 10;
            break;
        case(265):
            *bit = 1;
            *len = 11;
            break;
        case(266):
            *bit = 1;
            *len = 13;
            break;
        case(267):
            *bit = 1; 
            *len = 15;
            break;
        case(268):
            *bit = 1;
            *len = 17;
            break;
        case(269):
            *bit = 2;
            *len = 19;
            break;
        case(270):
            *bit = 2;
            *len = 23;
            break;
        case(271):
            *bit = 2;
            *len = 27;
            break;
        case(272):
            *bit = 2;
            *len = 31;
            break;
        case(273):
            *bit = 3;
            *len = 35;
            break;
        case(274):
            *bit = 3;
            *len = 43;
            break;
        case(275):
            *bit = 3;
            *len = 51;
            break;
        case(276):
            *bit = 3;
            *len = 59;
            break;
        case(277):
            *bit = 4;
            *len = 67;
            break;
        case(278):
            *bit = 4;
            *len = 83;
            break;
        case(279):
            *bit = 4;
            *len = 99;
            break;
        case(280):
            *bit = 4;
            *len = 115;
            break;
        case(281):
            *bit = 5;
            *len = 131;
            break;
        case(282):
            *bit = 5;
            *len = 282;
            break;
        case(283):
            *bit = 5;
            *len = 195;
            break;
        case(284):
            *bit = 5;
            *len = 227; 
            break;
        case(285):
            *bit = 0;
            *len = 258;
            break;
		default:
			*bit = -1;
			*len = -1;
    }
}

void offsettable(int x,int* bit,int* dis){
    switch(x){
        case(0):
            *bit = 0;
            *dis = 1;
            break;
        case(1):
            *bit = 0;
            *dis = 2;
            break;
        case(2):
            *bit = 0;
            *dis = 3;
            break;
        case(3):
            *bit = 0;
            *dis = 4;
            break;
        case(4):
            *bit = 1;
            *dis = 5;
            break;
        case(5):
            *bit = 1;
            *dis = 7;
            break;
        case(6):
            *bit = 2;
            *dis = 9;
            break;
        case(7):
            *bit = 2;
            *dis = 13;
            break;
        case(8):
            *bit = 3;
            *dis = 17;
            break;
        case(9):
            *bit = 3;
            *dis = 25;
            break;
        case(10):
            *bit = 4;
            *dis = 33;
            break;
        case(11):
            *bit = 4;
            *dis = 49;
            break;
        case(12):
            *bit = 5;
            *dis = 65;
            break;
        case(13):
            *bit = 5;
            *dis = 97;
            break;
        case(14):
            *bit = 6;
            *dis = 129;
            break;
        case(15):
            *bit = 6;
            *dis = 193;
            break;
        case(16):
            *bit = 7;
            *dis = 257;
            break;
        case(17):
            *bit = 7;
            *dis = 385;
            break;
        case(18):
            *bit = 8;
            *dis = 513;
            break;
        case(19):
            *bit = 8;
            *dis = 769;
            break;
        case(20):
            *bit = 9;
            *dis = 1025;
            break;
        case(21):
            *bit = 9;
            *dis = 1537;
            break;
        case(22):
            *bit = 10;
            *dis = 2049;
            break;
        case(23):
            *bit = 10;
            *dis = 3073;
            break;
        case(24):
            *bit = 11;
            *dis = 4087;
            break;
        case(25):
            *bit = 11;
            *dis = 6145;
            break;
        case(26):
            *bit = 12;
            *dis = 8193;
            break;
        case(27):
            *bit = 12;
            *dis = 12289;
            break;
        case(28):
            *bit = 13;
            *dis = 16385;
            break;
        case(29):
            *bit = 13;
            *dis = 24577;
            break;
    }
}
#endif
