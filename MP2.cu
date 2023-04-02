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

//Creating Global Variables
//#define int TOLERANCE = 0.001; ???
#define int TEST_SIZE = 5;
#define experiment[TEST_SIZE] = [125, 250, 500, 1000, 2000];


//Creating Function Instance
void hostRunner(float* a, float* b, float* c, int num, void (*matrixFunc)(float*, float*, float*, int), size_t size)
void matrixCreate(float* a, float* b, float* c);
void matrixElement(float* c, float* a, float* b, int num);
void elementRunner(float* c, float* a, float* b, int num);
void matrixRow(float* c, float* a, float* b, int num);
void rowRunner(float* c, float* a, float* b, int num);
void matrixColumn(float* c, float* a, float* b, int num);
void columnRunner(float* c, float* a, float* b, int num);
void matrixCPU(float* c, float* a, float* b, int num);

void main()
{
	float *a[], *b[], *c[];
	
	//Perform Test on all Matrix Sizes
	for (int i = 0; i < TEST_SIZE; i++)
	{
		int num = experiment[i];
		size_t arraySize = num * num * sizeof(float);
		a = (float*)malloc(arraySize);
		b = (float*)malloc(arraySize);
		c = (float*)malloc(arraySize);
		hostRunner(a, b, c, num, matrixElement, arraySize);
		hostRunner(a, b, c, num, matrixRow, arraySize);
		hostRunner(a, b, c, num, matrixColumn, arraySize);
	}
}

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
__global__ void matrixElement(float* c, float* a, float* b, int num)
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

//GPU Element Multiplication Allocator
void elementRunner(float* c, float* a, float* b, int num)
{
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((int)ceil(N / (float)threadsPerBlock.x), (int)ceil(N / (float)threadsPerBlock.y));
	matrixRow<<<numBlocks, threadsPerBlock>>>(matrixA, matrixB, matrixC, num);
}

//GPU Multiplication by Row
__global__ void matrixRow(float* c, float* a, float* b, int num)
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

//GPU Row Multiplication Allocator
void rowRunner(float* c, float* a, float* b, int num)
{
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((int)ceil(num / (float)threadsPerBlock.x), 1);
	matrixRow<<<numBlocks, threadsPerBlock>>>(matrixA, matrixB, matrixC, num);
}

//GPU Multiplication by Column
__global__ void matrixColumn(float* c, float* a, float* b, int num)
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

//GPU Column Multiplication Allocator
void columnRunner(float* c, float* a, float* b, int num)
{
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((int)ceil(num / (float)threadsPerBlock.y));
	matrixColumn<<<numBlocks, threadsPerBlock>>>(matrixA, matrixB, matrixC, num);
}

//Create Random Float Filled Vectors
void matrixCreate(float* a, float* b, float* c, int num)
{
	for (int x = 0; x < num; x++)
	{
		for (int y = 0;  y < num; y++)
		{
			int z = x + y * num;
			(*a)[z] = rand() % 100 / 10.0
			(*b)[z] = rand() % 100 / 10.0
			(*c)[z] = 0.0f;
		}
	}
}
			

void hostRunner(float* a, float* b, float* c, int num, void (*matrixFunc)(float*, float*, float*, int), size_t size)
{
	//Initialize Matricies & Pointers
	matrixCreate(a, b, c));
	float *matrixA, *matrixB, *matrixC;

	//Allocate Device Memory
	cudaMalloc((void**)&matrixA, size);
	cudaMalloc((void**)&matrixB, size);
	cudaMalloc((void**)&matrixC, size);

	//Copy Host Memory to Device Memory
	cudaMemcpy(matrixA, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(matrixB, B, size, cudaMemcpyHostToDevice);
	cudaMemcpy(matrixC, C, size, cudaMemcpyHostToDevice);

	//KERNEL CALL
	float time = 0;
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);
	addHandler(pA, pB, pC, N);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	printf("Kernal function time: %f\n", time);

	//Copy result from device memory to host memory
	cudaMemcpy(C, pC, (N*N)*sizeof(float), cudaMemcpyDeviceToHost);

	//Use the CPU to compute addition
	time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);
	matrixCPU(matrixA, matrixB, matrixC, num);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	printf("CPU time: %f\n", time);


	//Check GPU computed against CPU computed
	int good = 1;
	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			int index = i + j * N;
			float diff = (*CTemp)[index] - (*C)[index]; //Compute difference
			if (absf(diff) > TOLERANCE) {
				good = 0;
			}
		}
	}

	if (good == 1) {
		printf("TEST PASSED\n");
	} else {
		printf("TEST FAILED\n");
	}

	// free device memory
	cudaFree(pA);
	cudaFree(pB);
	cudaFree(pC);
}