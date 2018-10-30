
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include<math.h>
#include <stdio.h>
#include<time.h>
#include <iostream>
#include <stdlib.h>
#include "GpuTimer.h"
using namespace std;

#define BLOCK_SIZE 1024
#define NUM_BINS 512


//Compute Histogram
// Serial implementation for running on CPU using a single thread.
void HistogramCpu(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins)
{
	//@@ Insert Your Code Here for the CPU Function to Compute the Histogram with the output bins saturated at 127.
	for (int i = 0; i < num_elements; i++)
	{
		if (bins[input[i]] < 127)
		{
			bins[input[i]]++;
		}
	}
}


//GPU Kernel for Basic Histogram Computation without sauration
__global__ void histogram_kernel(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins)
{
	//@@ Insert Your Code Here for the CUDA Kernel for Basic Histogram Computation without saturation
	int i = threadIdx.x + blockIdx.x*blockDim.x;


	if (i < num_elements)
	{
		atomicAdd(&bins[input[i]], 1);
	}

}


//GPU kernel for converting the output bins into saturated bins at a maximum value of 127
__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins)
{
	//@@ Insert Your Code Here for the CUDA Kernel for ensuring that the output bins are saturated at 127.
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	if (bins[i] > 127)
	{
		bins[i] = 127;
	}

}

int main(void)
{

	int inputLength;
	unsigned int *hostInput;
	unsigned int *hostBins;
	unsigned int *hostBins_CPU;
	unsigned int *deviceInput;
	unsigned int *deviceBins;

	//ask the user to enter the length of the input vector
	printf("Please enter the length of the input array\n");
	scanf("%u", &inputLength);

	
	//Allocate the host memory for the input array and output histogram array
	hostInput = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
	hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
	hostBins_CPU = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));


	//Random Initialize input array. 
	//There are several ways to do this, such as making functions for manual input or using random numbers. 

	// Set the Seed for the random number generator rand() 
	srand(clock());
	for (int i = 0; i < inputLength; i++)
	{
		unsigned int j = rand() % NUM_BINS;
		hostInput[i] = j;
	}
		//hostInput[i] = static_cast <unsigned int> (rand()) / (static_cast <unsigned int> (RAND_MAX / (NUM_BINS))); //the values will range from 0 to (NUM_BINS - 1)


	for (int i = 0; i<NUM_BINS; i++)
		hostBins_CPU[i] = 0;//initialize CPU histogram array to 0

	//Allocate memory on the device for input array and output histogram array and record the needed time


	GpuTimer timer;
	timer.Start();

	//@@Insert Your Code Here to allocate memory for deviceInput and deviceBins
	cudaError_t err;
	deviceInput = NULL;
	err = cudaMalloc((void **)&deviceInput, inputLength * sizeof(unsigned int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector deviceInput (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	deviceBins = NULL;
	err = cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector deviceBins (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	//initialize deviceBins to zero
	cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

	timer.Stop();
	printf("Time to allocate memory on the device is: %f msecs.\n", timer.Elapsed());



	//Copy the input array from the host to the device and record the needed time
	GpuTimer timer1;
	timer1.Start();

	//@@ Insert Your Code Here to copy input array from Host to Device
	cudaMemcpy(deviceInput,hostInput , inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);

	timer1.Stop();
	printf("Time to copy the input array from the host to the device is: %f msecs.\n", timer1.Elapsed());


	//Do the Processing on the GPU for Basic Histogram computation without saturation
	//@@ Insert Kernel Execution Configuration Parameters for the histogram_kernel
	dim3 dimBlock(BLOCK_SIZE, 1, 1);

	dim3 dimGrid((inputLength - 1) / BLOCK_SIZE + 1, 1, 1);

	//Invoke the histogram_kernel kernel and record the needed time for its execution
	GpuTimer timer2;
	timer2.Start();
	//@@ Insert Your Code Here for Kernel Invocation
	histogram_kernel <<<dimGrid, dimBlock >> >(deviceInput, deviceBins, inputLength, NUM_BINS);
	timer2.Stop();
	printf("Implemented CUDA code for basic histogram calculation ran in: %f msecs.\n", timer2.Elapsed());


	//Do the Processing on the GPU for convert_kernel
	//@@ Insert Kernel Execution Configuration Parameters for the convert_kernel
	dim3 dimBlock2(BLOCK_SIZE, 1, 1);
	dim3 dimGrid2((NUM_BINS - 1) / BLOCK_SIZE + 1, 1, 1);

	//Invoke the convert_kernel kernel and record the needed time for its execution
	GpuTimer timer3;
	timer3.Start();
	//@@ Insert Your Code Here for Kernel Invocation
	convert_kernel<<<dimGrid2,dimBlock2>>>(deviceBins, NUM_BINS);
	timer3.Stop();
	printf("Implemented CUDA code for output saturation ran in: %f msecs.\n", timer3.Elapsed());


	//Copy resulting histogram array from device to host and record the needed time
	GpuTimer timer4;
	timer4.Start();
	//@@ Insert Your Code Here to Copy the resulting histogram deviceBins from device to the Host hostBins
	cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	timer4.Stop();
	printf("Time to copy the resulting Histogram from the device to the host is: %f msecs.\n", timer4.Elapsed());


	//Do the Processing on the CPU
	clock_t begin = clock();
	//@@ Insert Your Code Here to call the CPU function HistogramCpu where the resulting vector is hostBins_CPU
	HistogramCpu(hostInput, hostBins_CPU, inputLength, NUM_BINS);
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
	printf("Implemented CPU serial code ran in: %f msecs.\n", time_spent);

	//Verify Results Computed by GPU and CPU
	for (int i = 0; i < NUM_BINS; i++)
	{
		//printf("CPU: %d, GPU: %d \n", hostBins_CPU[i], hostBins[i]);
		if (abs(int(hostBins_CPU[i] - hostBins[i])) > 0)
		{
			fprintf(stderr, "Result verification failed at element (%d)!\n", i);
			exit(EXIT_FAILURE);
		}
	}
	printf("Test PASSED\n");


	//Free host memory
	free(hostBins);
	free(hostBins_CPU);
	free(hostInput);


	//Free device memory
	//@@ Insert Your Code Here to Free Device Memory
	cudaFree(deviceBins);
	cudaFree(deviceInput);

	return 0;

}