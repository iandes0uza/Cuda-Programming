//-----------------------------------------------
//			ELEC374: Machine Problem 2
//					Ian DeSouza
//					 20232372
//				 20iagd@queensu.ca
//-----------------------------------------------

#include "cuda_runtime.h"
#include <stdio.h> 
#include <string.h> 

//This library will help us fetch necessary information
#include "device_launch_parameters.h"

//Creating Function Instance
void hostRunner(cudaDeviceProp dev, int num);
void matrixElement(float* c, float* a, float* b, int num);
void matrixRow(float* c, float* a, float* b, int num);
void matrixColumn(float* c, float* a, float* b, int num);
void matrixCPU(float* c, float* a, float* b, int num);

//CPU Multiplication
void matrixCPU(float* c, float* a, float* b, int num)
{
	//Perform CPU operations
	for (int i = 0; i < (num*num); i++)
	{
		c[i] = a[i] + b[i];
	}
}

//GPU Multiplication by Element
_global_ void matrixElement(float* c, float* a, float* b, int num)
{
	//Initiate the columns and rows
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	//Find the # of iterations necessary
	int i = row * N + col;
	if (col < N && row < N )
	{
		c[i] = a[i] + b[i];
	}
}

//GPU Multiplication by Row
_global_ void matrixRow(float* c, float* a, float* b, int num)
{
	//Initiate rows
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	//Find the # of iterations necessary
	if (row < num) return;
	for (int i = 0; i < num; i++)
	{
		int j = row * num + 1;
		c[j] = a[j] + b[j];
	}
}

//GPU Multiplication by Column
_global_ void matrixColumn(float* c, float* a, float* b, int num)
{
	//Initiate columns
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	//Find the # of iterations necessary
	if (col < num) return;
	for (int i = 0; i < num; i++)
	{
		int j = i * num + col;
		c[j] = a[j] + b[j];
	}
}


