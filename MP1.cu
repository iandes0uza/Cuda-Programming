//-----------------------------------------------
//			ELEC374: Machine Problem 1
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
void devGetInfo(cudaDeviceProp dev, int num);
int devGetCores(cudaDeviceProp dev);

int main(void)
{
	//Declare the number of devices on the GPU Servers
	int numOfDev;
	cudaGetDeviceCount(&numOfDev);

	//Iterate through each device & their properties
	for (int devNum = 0; devNum < numOfDev; devNum++)
	{
		//Call certain device
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, devNum);

		//Call per device
		devGetInfo(devProp, devNum);

	}

	return 0;
}

void devGetInfo(cudaDeviceProp dev, int num)
{
	printf("*****************************************************************************\n");
	printf("------------------------------ Device #%d ------------------------------------\n", num);
	printf("*****************************************************************************\n");

	//Printing Device Properties
	printf("Device Name:\t\t\t\t\t%s\n", dev.name);
	printf("Clock Rate:\t\t\t\t\t%d\n", dev.clockRate);
	printf("# of SM's:\t\t\t\t\t%d\n", dev.multiProcessorCount);

	//Fetch and Print Core Count
	int cores = devGetCores(dev);
	if (cores < 0) printf("# of Cores:\t\t\t\t\t%d\n", cores);
	else printf("# of Cores:\t\t\t\t\t%d\n", cores);

	printf("Warp Size:\t\t\t\t\t%d\n", dev.warpSize);
	printf("Amount of Global Memory:\t\t\t%d\n", dev.totalGlobalMem);
	printf("Amount of Constant Memory:\t\t\t%d\n", dev.totalConstMem);
	printf("Amount of Shared Memory (per block):\t\t%d\n", dev.sharedMemPerBlock);
	printf("# of Registers Available (per block):\t\t%d\n", dev.regsPerBlock);
	printf("Max # of Threads (per block):\t\t\t%d\n", dev.maxThreadsPerBlock);
	printf("Max Dimension Sizes for Block:\t\t\tx: %d, y: %d, z: %d\n", dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
	printf("Max Dimension Sizes for Grid:\t\t\tx: %d, y: %d, z: %d\n", dev.maxGridSize[0], dev.maxGridSize[1], dev.maxGridSize[2]);
}

int devGetCores(cudaDeviceProp dev)
{
	int minor = dev.minor;
	switch (dev.major)
	{
		case 2: 
			if (dev.minor) return dev.multiProcessorCount * 48;
			else return dev.multiProcessorCount * 32;
		case 3:
			return dev.multiProcessorCount * 192;
		case 5:
			return dev.multiProcessorCount * 128;
		case 6:
			if (dev.minor == 1 || dev.minor == 2) return dev.multiProcessorCount * 128;
			else if (dev.minor == 0) return dev.multiProcessorCount * 64;
			else return -1;
		case 7:
			if (dev.minor == 0 || dev.minor == 5) return dev.multiProcessorCount * 64;
			else return -1;
		case 8:
			return dev.multiProcessorCount * 64;
			break;
		default:
			return -1;
	}
}
