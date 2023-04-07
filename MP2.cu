//-----------------------------------------------
//			ELEC374: Machine Problem 2
//					Ian DeSouza
//					 20232372
//				 20iagd@queensu.ca
//-----------------------------------------------

#include "cuda_runtime.h"
#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <math.h>

//This library will help us fetch necessary information
#include "device_launch_parameters.h"

//Creating Global Variables
#define BLOCK_DIM 16
#define TOLERANCE 0.0001

//Creating Function Instance
__global__ void matrixElement(float *a, float *b, float *result, int num);
__global__ void matrixRow(float *a, float *b, float *result, int num);
__global__ void matrixColumn(float *a, float *b, float *result, int num);
void matrixCPU(float *a, float *b, float *result, int num);
float absol(float a);
void checkTest(float *cpu, float *gpu, int size);

int main()
{
	const int arraySize = 5;
	int testSize[arraySize] = { 125, 250, 500, 1000, 2000 };
	for (int i = 0; i < arraySize; i++)
	{
		printf("*****************************************************************************\n");
		printf("----------------------------- %d X %d Matrix -----------------------------\n", testSize[i], testSize[i]);
		printf("*****************************************************************************\n");

		//Create Event Objects
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		float time = 0.0;

		//Allocate Pointer Memory
		size_t size = testSize[i] * testSize[i] * sizeof(float);
		float* matrixA = (float*)malloc(size);
		float* matrixB = (float*)malloc(size);
		float* cpuResult = (float*)malloc(size);
		float* gpuElement = (float*)malloc(size);
		float* gpuRow = (float*)malloc(size);
		float* gpuCol = (float*)malloc(size);

		//Allocate Device Memory
		float* devA;
		float* devB;
		float* devResult;
		cudaMalloc((void**)&devA, size);
		cudaMalloc((void**)&devB, size);
		cudaMalloc((void**)&devResult, size);

		//Fill Matrix
		for (int x = 0; x < testSize[i]; x++)
		{
			for (int y = 0; y < testSize[i]; y++)
			{
				*(matrixA + x * testSize[i] + y) = (rand() % 100) / 10.0;
				*(matrixB + x * testSize[i] + y) = (rand() % 100) / 10.0;
				*(cpuResult + x * testSize[i] + y) = 0.0;
				*(gpuElement + x * testSize[i] + y) = 0.0;
				*(gpuRow + x * testSize[i] + y) = 0.0;
				*(gpuCol + x * testSize[i] + y) = 0.0;
			}
		}

		//Defining Block & Grid Size
		int NumBlocks = testSize[i] / BLOCK_DIM;
		if (testSize[i] % BLOCK_DIM) NumBlocks++;
		dim3 dimGrid(NumBlocks, NumBlocks);
		dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);

		//Elapsed CPU Addition
		cudaEventRecord(start);
		matrixCPU(matrixA, matrixB, cpuResult, testSize[i]);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		printf("Elapsed CPU:\t\t\t\t\t%0.2f \n", time);
		printf("------------------------------------------------------------------------------\n");

		//Copy Memory (Host -> Device)
		cudaMemcpy(devA, matrixA, size, cudaMemcpyHostToDevice);
		cudaMemcpy(devB, matrixB, size, cudaMemcpyHostToDevice);

		//Elapsed GPU Addition by Element
		cudaEventRecord(start);
		matrixElement << < dimGrid, dimBlock >> > (devA, devB, devResult, testSize[i]);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		printf("Elapsed GPU-Element:\t\t\t\t%0.2f \n", time);

		//Copy Memory (Device -> Host)
		cudaMemcpy(gpuElement, devResult, size, cudaMemcpyDeviceToHost);

		//Check if Test Passed
		checkTest(cpuResult, gpuElement, testSize[i]);
		printf("------------------------------------------------------------------------------\n");

		//Elapsed GPU Addition by Row
		cudaEventRecord(start);
		matrixRow << < ceil(testSize[i] / BLOCK_DIM), BLOCK_DIM >> > (devA, devB, devResult, testSize[i]);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		printf("Elapsed GPU-Row:\t\t\t\t%0.2f \n", time);

		//Copy Memory (Device -> Host)
		cudaMemcpy(gpuRow, devResult, size, cudaMemcpyDeviceToHost);

		//Check if Test Passed
		checkTest(cpuResult, gpuRow, testSize[i]);
		printf("------------------------------------------------------------------------------\n");

		//Elapsed GPU Addition by Column
		cudaEventRecord(start);
		matrixColumn << < ceil(testSize[i] / BLOCK_DIM), BLOCK_DIM >> > (devA, devB, devResult, testSize[i]);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		printf("Elapsed GPU-Column:\t\t\t\t%0.2f \n", time);

		//Copy Memory (Device -> Host)
		cudaMemcpy(gpuCol, devResult, size, cudaMemcpyDeviceToHost);

		//Check if Test Passed
		checkTest(cpuResult, gpuCol, testSize[i]);

		//Free Memory
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		cudaFree(devA);
		cudaFree(devB);
		free(matrixA);
		free(matrixB);
	}
}


//CPU (Host) Addition
void matrixCPU(float *a, float *b, float *result, int num)
{
	for (int x = 0; x < num; x++)
	{
		for (int y = 0; y < num; y++)
		{
			result[x * num + y] = a[x * num + y] + b[x * num + y];
		}
	}

}

//GPU Addition by Element
__global__ void matrixElement(float *a, float *b, float *result, int num)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < num && row < num)
	{
		result[row * num + col] = a[row * num + col] + b[row * num + col];
	}
}

//GPU Addition by Row
__global__ void matrixRow(float *a, float *b, float *result, int num)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < num)
	{
		for (int i = 0; i < num; i++)
		{
			result[row * num + i] = a[row * num + i] + b[row * num + i];
		}
	}
}

//GPU Addition by Column
__global__ void matrixColumn(float *a, float *b, float *result, int num)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < num)
	{
		for (int i = 0; i < num; i++)
		{
			result[i * num + col] = a[i * num + col] + b[i * num + col];
		}
	}
}

//Overloading Absolute Function
float absol(float a)
{
	if (a < 0) return -a;
	else return a;
}

//Tolerance Checking
void checkTest(float *cpu, float *gpu, int size)
{
	for (int x = 0; x < size * size; x++)
	{

		if (absol(cpu[x] - gpu[x]) > TOLERANCE)
		{
			printf("\t\t\t\t\t**TEST FAILED**\n");
			return;
		}

	}
	printf("\t\t\t\t\t--TEST PASSED--\n");
	return;
}