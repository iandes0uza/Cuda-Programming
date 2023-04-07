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
#include <stdlib.h>
#include <ctime>

#define BLOCK_WIDTH 16

__global__ void gpuTiledMultiplication(int *a, int *b, int *c, int size);
void cpuTiledMultiplication(int *A, int *B, int *C, int size);

int main()
{
	const int tileNum = 5;
	const int arraySize = 5;
	int tileWidth[tileNum] = { 2, 4, 10, 20, 25 };
	int testSize[arraySize] = { 125, 250, 500, 1000, 2000 };
	for (int i = 0; i < tileNum; i++)
	{
		printf("*****************************************************************************\n");
		printf("--------------------------- %d X %d Tile Width ---------------------------\n", blockWidth[i], blockWidth[i]);
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
			int NumBlocks = testSize[j] / tileWidth[i];
			if (testSize[j] % tileWidth[i]) NumBlocks++;
			dim3 dimGrid(NumBlocks, NumBlocks);
			dim3 dimBlock(tileWidth[i], tileWidth[i]);

			//Copy Memory (Host -> Device)
			cudaMemcpy(devA, matrixA, size, cudaMemcpyHostToDevice);
			cudaMemcpy(devB, matrixB, size, cudaMemcpyHostToDevice);

			//Print GPU Completion Time
			cudaEventRecord(start);
			tiledMultiplication << < dimGrid, dimBlock >> > (devA, devB, devResult, testSize[j], tileWidth[i]);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			printf("Elapsed GPU Computation:\t\t\t\t%0.2f \n", time);
			printf("------------------------------------------------------------------------------\n");

			//Copy Memory (Device -> Host)
			cudaMemcpy(gpuResult, devResult, size, cudaMemcpyDeviceToHost);

			//Print CPU Completion Time
			cudaEventRecord(start);
			cpuTiledMultiplication(matrixA, matrixB, cpuResult, testSize[j]);
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

//GPU Multiplication
__global__ void gpuTiledMultiplication(int *a, int *b, int *result, int num, int tileWidth) 
{
	__shared__ int tile1[tileWidth][tileWidth];
	__shared__ int tile2[tileWidth][tileWidth];

	int row = blockIdx.y * tileWidth + threadIdx.y;
	int col = blockIdx.x * tileWidth + threadIdx.x;
	
	int tX = threadIdx.x;
	int tY = threadIdx.y;

	int tempA = 0;
	int tempB = 0;
	int location = 0;

	for (int x = 0; x < gridDim.x; x++) 
	{
		location = (x * tileWidth + tY)* num + col;
		
		if (location >= num * num)
			tile2[tY][tX] = 0;
		else
			tile2[tY][tX] = b[location];
			
		location = row * num + x * tileWidth + tX;
		
		if (location >= num * num)
			tile1[tY][tX] = 0;
		else
			tile1[tY][tX] = a[location];

		for (int y = 0; y < tileWidth; y++)
		{
			tempA = tempA + tile1[tY][y] * tile2[y][tX];
		}

		__syncthreads();
	}

	if (row < num && col < num) 
	{
		tempB = row * num + col;
		result[tempB] = tempA;
	}

}

//CPU Multiplication (can't figure out how to do it tiled???)
void cpuTiledMultiplication(int *a, int *b, int *result, int num) 
{
	float ans = 0.0;
	for (int x = 0; x < num; x++)
	{
		for (int y = 0; y < num; y++)
		{
			for (int i = 0; i < num; i++) {
				offset1 = x * num + i;
				offset2 = i * num + y;
				ans = ans + a[x * num + i] * b[i * num + y];
			}
			result[x * num + y] = ans;
		}
	}
}