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

// Given a list (lst) of length n
// Output its  Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...+ lst[n-1]}


//GPU Kernel that implements inclusive scan on input vector of length len, 
//the results of scan done by the thread blocks is stored in a vector output and 
//an aux array is generated from the last elements of the scanned sections in each 
//thread block
__global__ void scan(float *input, float *output, float *aux, int len) {

	//@@ Load a segment of the input vector into shared memory
	//@@ write the scan iterations
	//@@ copy the resulting vector from shared memory into output vector
	//@@ in the global memory
	//@@ generate the vector aux
	__shared__ float ScanBlock[2*BLOCK_SIZE];

	int tx = threadIdx.x;
	int index = blockIdx.x*blockDim.x*2 + threadIdx.x;

	if (index < len)
	{
		ScanBlock[tx] = input[index];
	}
	else
	{
		ScanBlock[tx] = 0.0f;
	}

	if (index + blockDim.x < len)
	{
		ScanBlock[tx + blockDim.x] = input[index + blockDim.x];
	}
	else
	{
		ScanBlock[tx + blockDim.x] = 0.0f;
	}
	__syncthreads();

	for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
		int index2 = (tx + 1)*stride * 2 - 1;
		if (index2 < 2 * BLOCK_SIZE)
			ScanBlock[index2] += ScanBlock[index2 - stride];
		__syncthreads();
	}

	for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		__syncthreads();
		int index3 = (tx + 1)*stride * 2 - 1;
		if (index3 + stride< 2 * BLOCK_SIZE) {
			ScanBlock[index3 + stride] += ScanBlock[index3];
		}
	}
	__syncthreads();
	if (index< len) output[index] = ScanBlock[threadIdx.x];
	if (index + BLOCK_SIZE<len) output[index + BLOCK_SIZE] = ScanBlock[threadIdx.x + BLOCK_SIZE];




	if (index < len)
	{
		if (tx == (BLOCK_SIZE - 1))
		{
			aux[blockIdx.x] = ScanBlock[tx+blockDim.x];
		}
	}

}


//GPU kernel that fixes the intermdiate results from scan using the aux array
__global__ void fixup(float *input, float *aux, int len) {

	//@@ Insert your code here to implement the fixup operation
	int tx = threadIdx.x;
	int index = blockIdx.x*blockDim.x + tx;

	if (blockIdx.x > 0 && index < len)
		input[index] += aux[blockIdx.x - 1];
}

// Serial implementation for running on CPU using a single thread.
void ScanCpu(float *input, float *output, int len)
{
	//@@ Insert Your Code Here for the CPU Function that implements inclusive scan on an input vector of length len 
	//@@ to generate output vector
	output[0] = input[0];
	for (int i = 1; i < len; i++)
	{
		output[i] = input[i] + output[i - 1];
	}

}


int main(void)
{

	int inputLength;  // number of elements in the input list
	float *hostInput; // The input 1D list
	float *hostOutput; // The output list
	float *CPUOutput; //the output of sequential implementation
	float *deviceInput;
	float *deviceOutput;
	float *deviceAuxArray, *deviceAuxScannedArray;


	//ask the user to enter the length of the input vector
	printf("Please enter the length of the input array\n");
	scanf("%d", &inputLength);


	int Grid_dim = ceil((float)inputLength / (float)(BLOCK_SIZE));

	//Allocate the host memory for the input list and output list
	hostInput = (float *)malloc(inputLength * sizeof(float));
	hostOutput = (float *)malloc(inputLength * sizeof(float));
	CPUOutput = (float *)malloc(inputLength * sizeof(float));



	//Random Initialize input array. 
	//There are several ways to do this, such as making functions for manual input or using random numbers. 

	// Set the Seed for the random number generator rand() 
	srand(clock());
	for (int i = 0; i < inputLength; i++)

		hostInput[i] = roundf((float)rand() / float(RAND_MAX) * 100) / 100;//To round to two digits after decimal point


																		   //Allocate memory on the device for input list, output list,deviceAuxArray,deviceAuxScannedArray  and record the needed time

	GpuTimer timer;
	timer.Start();

	//@@Insert Your Code Here to allocate memory for deviceInput and deviceOutput
	cudaError_t err;
	deviceInput = NULL;
	err = cudaMalloc((void **)&deviceInput, inputLength * sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector deviceInput (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	timer.Stop();
	printf("Time to allocate memory on the device is: %f msecs.\n", timer.Elapsed());
	deviceOutput = NULL;
	err = cudaMalloc((void **)&deviceOutput, inputLength * sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector deviceOutput (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	deviceAuxArray = NULL;
	err = cudaMalloc((void **)&deviceAuxArray, Grid_dim* sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector deviceAuxArray (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	deviceAuxScannedArray = NULL;
	err = cudaMalloc((void **)&deviceAuxScannedArray, Grid_dim * sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector deviceAuxScannedArray (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//Copy the input list from the host to the device and record the needed time
	GpuTimer timer1;
	timer1.Start();

	//@@ Insert Your Code Here to copy input array from Host to Device
	cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(float), cudaMemcpyHostToDevice);


	timer1.Stop();
	printf("Time to copy the input array from the host to the device is: %f msecs.\n", timer1.Elapsed());


	//Do the Processing on the GPU for scan
	//@@ Insert Kernel Execution Configuration Parameters for the total kernel
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid2(Grid_dim, 1, 1);


	//Invoke the scan kernel and record the needed time for its execution
	GpuTimer timer2;
	timer2.Start();
	//@@ Insert Your Code Here for Kernel Invocation
	scan << <dimGrid2, dimBlock >> >(deviceInput, deviceOutput, deviceAuxArray, inputLength);
	timer2.Stop();
	printf("Implemented CUDA code for first call of scan ran in: %f msecs.\n", timer2.Elapsed());

	cudaDeviceSynchronize();

	//@@ Insert Kernel Execution Configuration Parameters for a second call of the scan kernel
	//which performes a scan on the auxiliary array, note that the number of elements in this array is the number of blocks 
	dim3 dimGrid(1, 1, 1);


	//Invoke the scan kernel and record the needed time for its execution
	GpuTimer timer3;
	timer3.Start();
	//@@ Insert Your Code Here for Kernel Invocation
	scan << <dimGrid, dimBlock >> >(deviceAuxArray, deviceAuxScannedArray, deviceAuxArray, Grid_dim);
	timer3.Stop();
	printf("Implemented CUDA code for second call of scan ran in: %f msecs.\n", timer3.Elapsed());

	cudaDeviceSynchronize();

	//@@ Insert Kernel Execution Configuration Parameters for the FixUp kernel
	//which takes as input the output of the first call of scan and the scanned auxiliary array to 
	//generate the final output
	//	dim3 dimBlock(BLOCK_SIZE, 1, 1);



	//Invoke the FixUp kernel and record the needed time for its execution
	GpuTimer timer4;
	timer4.Start();
	//@@ Insert Your Code Here for Kernel Invocation
	fixup << <dimGrid2, dimBlock >> >(deviceOutput, deviceAuxScannedArray, inputLength);
	timer4.Stop();
	printf("Implemented CUDA code for FixUp kernel ran in: %f msecs.\n", timer4.Elapsed());

	cudaDeviceSynchronize();

	//Copy resulting output list from device to host and record the needed time
	GpuTimer timer5;
	timer5.Start();
	//@@ Insert Your Code Here to Copy the resulting output from deviceOutput to hostOutput 
	cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);
	timer5.Stop();
	printf("Time to copy the resulting output list from the device to the host is: %f msecs.\n", timer5.Elapsed());




	//Do the Processing on the CPU
	clock_t begin1 = clock();
	//@@ Insert Your Code Here to call the CPU function ScanCpu where the resulting output is CPUOutput
	ScanCpu(hostInput, CPUOutput, inputLength);
	clock_t end1 = clock();
	double time_spent1 = (double)(end1 - begin1) / CLOCKS_PER_SEC * 1000;
	printf("Implemented CPU serial code ran in: %f msecs.\n", time_spent1);

	//Verify Results Computed by GPU and CPU
	printf("Verify Results:\n");
	printf("The values of the last elements are: CPU=%.2f, GPU=%.2f\n", CPUOutput[inputLength - 1], hostOutput[inputLength - 1]);
	for (int i = 0; i<inputLength; i++)
	{

		if (fabs(hostOutput[i] - CPUOutput[i]) > 1)
		{
			fprintf(stderr, "Result verification failed at %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}
	printf("Test PASSED\n");


	//Free host memory

	free(hostInput);
	free(hostOutput);
	free(CPUOutput);


	//Free device memory
	//@@ Insert Your Code Here to Free Device Memory
	cudaFree(deviceInput);
	cudaFree(deviceOutput);
	cudaFree(deviceAuxArray);
	cudaFree(deviceAuxScannedArray);

	return 0;

}