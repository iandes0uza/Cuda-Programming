//-----------------------------------------------
//			ELEC374: Machine Problem 3.3
//					Ian DeSouza
//					 20232372
//				 20iagd@queensu.ca
//-----------------------------------------------

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <ctime>

__global__ void gpuMultiplication(float* inputA, float* inputB, float* result, int num);
void cpuMultiplication(float* inputA, float* inputB, float* result, int num);

int main()
{
	const int blockNum = 5;
	const int arraySize = 5;
	int blockWidth[blockNum] = { 2, 4, 10, 20, 25 };
	int testSize[arraySize] = { 125, 250, 500, 1000, 2000 };
	for (int i = 0; i < blockNum; i++)
	{
		printf("*****************************************************************************\n");
		printf("---------------------------- %d X %d Block Width ----------------------------\n", blockWidth[i], blockWidth[i]);
		printf("*****************************************************************************\n");
		for (int j = 0; j < arraySize; j++)
		{
			printf("*****************************************************************************\n");
			printf("---------------------------- %d X %d Matrix ----------------------------\n", testSize[j], testSize[j]);
			printf("*****************************************************************************\n");

			//Create Event Objects
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			float time = 0.0;

			//Allocate Pointer Memory
			size_t size = testSize[j] * testSize[j] * sizeof(float);
			float* matrixA = (float*)malloc(size);
			float* matrixB = (float*)malloc(size);
			float* cpuResult = (float*)malloc(size);
			float* gpuResult = (float*)malloc(size);

			//Allocate Device Memory
			float* devA;
			float* devB;
			float* devResult;
			cudaMalloc((void**)&devA, size);
			cudaMalloc((void**)&devB, size);
			cudaMalloc((void**)&devResult, size);

			//Fill Matrix
			for (int x = 0; x < testSize[j]; x++)
			{
				for (int y = 0; y < testSize[j]; y++)
				{
					*(matrixA + x * testSize[j] + y) = (rand() % 100) / 10.0;
					*(matrixB + x * testSize[j] + y) = (rand() % 100) / 10.0;
					*(cpuResult + x * testSize[j] + y) = 0.0;
					*(gpuResult + x * testSize[j] + y) = 0.0;
				}
			}

			//Defining Block & Grid Size
			int NumBlocks = testSize[j] / blockWidth[i];
			if (testSize[j] % blockWidth[i]) NumBlocks++;
			dim3 dimGrid(NumBlocks, NumBlocks);
			dim3 dimBlock(blockWidth[i], blockWidth[i]);

			//Copy Memory (Host -> Device)
			cudaMemcpy(devA, matrixA, size, cudaMemcpyHostToDevice);
			cudaMemcpy(devB, matrixB, size, cudaMemcpyHostToDevice);

			//Print GPU Completion Time
			cudaEventRecord(start);
			gpuMultiplication << < dimGrid, dimBlock >> > (devA, devB, devResult, testSize[j]);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			printf("Elapsed GPU Computation:\t\t\t\t%0.2f \n", time);
			printf("------------------------------------------------------------------------------\n");

			//Copy Memory (Device -> Host)
			cudaMemcpy(gpuResult, devResult, size, cudaMemcpyDeviceToHost);

			//Print CPU Completion Time
			cudaEventRecord(start);
			cpuMultiplication(matrixA, matrixB, cpuResult, testSize[j]);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			printf("Elapsed CPU Computation:\t\t\t\t%0.2f \n", time);

			//Free Memory
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			cudaFree(devA);
			cudaFree(devB);
			cudaFree(devResult);
			free(matrixA);
			free(matrixB);
			free(cpuResult);
			free(gpuResult);
		}
	}
}

__global__ void gpuMultiplication(float *a, float *b, float *result, int num)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	float ans = 0.0;

	if (row < num && col < num)
	{
		for (int i = 0; i < num; i++)
		{
			ans = ans + a[row * num + i] * b[i * num + col];
			result[row * num + col] = ans;
		}
	}
}

void cpuMultiplication(float *a, float *b, float *result, int num)
{
	float ans = 0.0;
	for (int x = 0; x < num; x++)
	{
		for (int y = 0; y < num; y++)
		{
			ans = 0.0;
			for (int i = 0; i < num; i++)
			{
				ans += a[x * num + i] * b[i * num + y];
			}
			result[x * num + y] = ans;
		}
	}
}