//-----------------------------------------------
//			ELEC374: Machine Problem 3.1
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



int main() 
{
	const int arraySize = 5;
	int testSize [arraySize] = {125, 250, 500, 1000, 2000};
	for (int i = 0; i < arraySize; i++)
	{
		//Create Event Objects
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		float time = 0.0;

		//Allocate Pointer Memory
		size_t size = testSize[i] * testSize[i] * sizeof(float);
		float* matrixA = (float*)malloc(size);
		float* matrixB = (float*)malloc(size);

		//Allocate Device Memory
		int* devA;
		int* devB;
		cudaMalloc((void**)&devA, size);
		cudaMalloc((void**)&devB, size);

		//Fill Matrix
		for (int x = 0; x < testSize[i]; x++)
		{
			for (int y = 0; y < testSize[i]; y++)
			{
				*(matrixA + x * testSize[i] + y) = (rand() % 100) / 10.0;
				*(matrixB + x * testSize[i] + y) = (rand() % 100) / 10.0;
			}
		}

		printf("Moving %d x %d Matrix To Device", testSize[i], testSize[i]);
		cudaEventRecord(start);
		cudaMemcpy(devA, matrixA, size, cudaMemcpyHostToDevice);
		cudaMemcpy(devB, matrixB, size, cudaMemcpyHostToDevice);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		printf("\tMatrices transfer time: %0.2f \n", time);

		printf("Moving %d x %d Matrix To Host", testSize[i], testSize[i]);
		cudaEventRecord(start);
		cudaMemcpy(devA, matrixA, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(devB, matrixB, size, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		printf("\tMatrices transfer time: %0.2f \n", time);

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		cudaFree(devA);
		cudaFree(devB);
		free(matrixA);
		free(matrixB);
	}
	/*
	cudaEvent_t startTransfer, stopTransfer, startGPU, stopGPU;
	cudaEventCreate(&startTransfer);
	cudaEventCreate(&stopTransfer);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);

	float transTime = 0.0f;
	float gpuTime = 0.0f;

	//synchronize
	cudaDeviceSynchronize(start);


	//get the size of the matrix
	size_t size = S*S*sizeof(int);

	//Allocate Pointer Memory
	int* a = (int*)malloc(size);
	int* b = (int*)malloc(size);
	int* cpuResult = (int*)malloc(size);
	int* gpuResult = (int*)malloc(size);

	//Initialize Multiplication Matrices
	for (int i = 0; i < S; i++)
	{
	for (int j = 0; j < S; j++)
	{
	int rand1 = rand() % 10;
	int rand2 = rand() % 10;
	*(a + i * S + j) = rand1;
	*(b + i * S + j) = rand2;
	}
	}

	//Allocate Device Memory
	int* devA, devB, devResult;
	cudaMalloc((void**)&devA, size);
	cudaMalloc((void**)&devB, size);
	cudaMalloc((void**)&devResult, size);

	//Transfer Time Recording
	cudaEventRecord(startTransfer);
	cudaMemcpy(devA, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(devB, b, size, cudaMemcpyHostToDevice);
	cudaEventRecord(stopTransfer);
	cudaEventSynchronize(stopTransfer);
	cudaEventElapsedTime(&transTime, startTransfer, stopTransfer);
	printf("Matrices transfer time: %0.2f \n", transTime);

	dim3 threadsPerBlock(16, 16);
	dim3 numberOfBlocks(ceil(S / threadsPerBlock.x), ceil(S / threadsPerBlock.y), 1);


	cudaEventRecord(startGPU);
	DeviceMatrixMultiplication << <numberOfBlocks, threadsPerBlock >> >(devA, devB, devResult, S);
	cudaEventRecord(stopGPU);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&gpuTime, startGPU, stopGPU);

	printf("for 16x16: \n");
	printf("number of blocks in x and y, respectively: %d, %d\n", (int)S / (int)16, (int)S / (int)16);
	printf("time taken : %0.2f ", gpuTime);
	cudaMemcpy(gpuResult, devResult, size, cudaMemcpyDeviceToHost);

	HostMatrixMultiplication(a, b, cpuResult, S);


	for (int x = 0; x < S; x++) {
	for (int y = 0; y < S; y++) {
	if (*(cpuResult + x * S + y) != *(gpuResult + x * S + y))
	flag = 1;
	}
	}*/
}
