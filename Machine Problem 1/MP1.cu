//-----------------------------------------------
//			ELEC374: Machine Problem 1
//					Ian DeSouza
//					 20232372
//				 20iagd@queensu.ca
//-----------------------------------------------

#include "cuda_runtime.h"

//This library will help us fetch necessary information
#include "device_launch_parameters.h"


int main(void)
{
	//Declare the number of devices on the GPU Servers
	int numOfDev;
	cudaGetDeviceCount(&numOfDev);

	//Iterate through each device & their properties
	for (int devNum = 0; devNum < numOfDev; devNum++)
	{
		//Call per device
	}
}

