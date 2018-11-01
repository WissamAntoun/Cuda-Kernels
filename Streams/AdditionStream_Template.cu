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

#define BLOCK_SIZE 256
#define SegLength 1024



//CUDA Kernel Device code
//Computes the element-wise vector addition of A and B into C: C[i] = A[i] + B[i].
//The 3 vectors have the same number of elements numElements.
__global__ void vecAdd(float *A, float *B, float *C, int numElements)

{
	//@@ Insert  your code here to implement vector addition
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < numElements)
		C[i] = A[i] + B[i];

}


int main2()
{
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	float EPS = 0.0001;
	int numElements = 4*1200 * 1024;
	size_t size = numElements * sizeof(float);
	//printf("[Vector addition of %d elements]\n", numElements);

	//Implement Vector Addition without using CUDA Streams
	// Allocate the host input vector A
	float *h_A = (float *)malloc(size);

	// Allocate the host input vector B
	float *h_B = (float *)malloc(size);

	// Allocate the host output vector C
	float *h_C = (float *)malloc(size);

	// Verify that allocations succeeded
	if (h_A == NULL || h_B == NULL || h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = float(i);
		h_B[i] = 1 / (i + EPS);
	}

	// Allocate the device input vector A
	float *d_A = NULL;
	err = cudaMalloc((void **)&d_A, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	float *d_B = NULL;
	err = cudaMalloc((void **)&d_B, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	float *d_C = NULL;
	err = cudaMalloc((void **)&d_C, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the host input vectors A and B in host memory to the device input vectors in
	// device memory
	GpuTimer timer;
	timer.Start();
	//printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	timer.Stop();
	//printf("Time to copy the input array A from the host to the device is: %f msecs.\n", timer.Elapsed());


	GpuTimer timer2;
	timer2.Start();

	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	timer2.Stop();
	//printf("Time to copy the input array B from the host to the device is: %f msecs.\n", timer2.Elapsed());


	GpuTimer timer3;
	timer3.Start();
	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	//printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vecAdd << <blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, numElements);
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	timer3.Stop();
	//printf("Implemented CUDA code for vector addition ran in: %f msecs.\n", timer3.Elapsed());


	GpuTimer timer4;
	timer4.Start();
	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	//printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	timer4.Stop();
	//printf("Time to copy the output array C from the Device to the Host is: %f msecs.\n", timer4.Elapsed());


	// Verify that the result vector is correct
	for (int i = 0; i < numElements; ++i)
	{
		if (fabs((h_A[i] + h_B[i]) - h_C[i]) > 1e-5)
		{
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}

	printf("Test PASSED\n");

	// Free device global memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);



	//Implement Vector Addition Using CUDA Streams

	//@@ Insert your code here to implement Vector Addition using streams and Time your implementation.

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);

	printf("Done main 2\n");

	return 0;
}
/**
* Host main routine
*/
int main(void)
{
	GpuTimer timer;
	timer.Start();
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	float EPS = 0.0001;
	int SegSize = 1024 ;
	int numElements = 4*1200*1024;
	size_t size = numElements * sizeof(float);
	printf("[Vector addition of %d elements]\n", numElements);

	//Implement Vector Addition without using CUDA Streams
	// Allocate the host input vector A
	float *h_A = (float *)malloc(size);

	// Allocate the host input vector B
	float *h_B = (float *)malloc(size);

	// Allocate the host output vector C
	float *h_C = (float *)malloc(size);


	// Verify that allocations succeeded
	if (h_A == NULL || h_B == NULL || h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = float(i);
		h_B[i] = 1 / (i + EPS);
	}

	
	
	// Allocate the device input vector A
	float *d_A[4] = { NULL, NULL, NULL, NULL };
	for (int i = 0; i < 4; i++)
	{
		err = cudaMalloc((void **)&d_A[i], SegSize* sizeof(float));
	}
	

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B

	float *d_B[4] = { NULL, NULL, NULL, NULL };
	for (int i = 0; i < 4; i++)
	{
		err = cudaMalloc((void **)&d_B[i], SegSize * sizeof(float));
	}

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C

	float *d_C[4] = { NULL, NULL, NULL, NULL };
	for (int i = 0; i < 4; i++)
	{
		err = cudaMalloc((void **)&d_C[i], SegSize * sizeof(float));
	}

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	

	cudaStream_t stream0, stream1, stream2, stream3;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	

	for (int i = 0; i<numElements; i += SegSize * 4 ) 
	{
		cudaMemcpyAsync(d_A[0], h_A + i, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_B[0], h_B + i, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream0);

		cudaMemcpyAsync(d_A[1], h_A + i + SegSize, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(d_B[1], h_B + i + SegSize, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream1);
		
		cudaMemcpyAsync(d_A[2], h_A + i + 2*SegSize, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream2);
		cudaMemcpyAsync(d_B[2], h_B + i + 2*SegSize, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream2);

		cudaMemcpyAsync(d_A[3], h_A + i + 3 * SegSize, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream3);
		cudaMemcpyAsync(d_B[3], h_B + i + 3 * SegSize, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream3);

		vecAdd << <SegSize / 256, 256, 0, stream0 >> > (d_A[0], d_B[0], d_C[0], SegSize);
		vecAdd << <SegSize / 256, 256, 0, stream1 >> > (d_A[1], d_B[1], d_C[1], SegSize);
		vecAdd << <SegSize / 256, 256, 0, stream2 >> > (d_A[2], d_B[2], d_C[2], SegSize);
		vecAdd << <SegSize / 256, 256, 0, stream3 >> > (d_A[3], d_B[3], d_C[3], SegSize);


		cudaMemcpyAsync(h_C + i, d_C[0], SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(h_C + i + SegSize, d_C[1], SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream1);
		cudaMemcpyAsync(h_C + i + 2*SegSize, d_C[2], SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream2);
		cudaMemcpyAsync(h_C + i + 3*SegSize, d_C[3], SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream3);

		cudaStreamSynchronize(stream0);
		cudaStreamSynchronize(stream1);
		cudaStreamSynchronize(stream2);
		cudaStreamSynchronize(stream3);
	}


	timer.Stop();
	printf("Implemented CUDA code for improved vector sum ran in: %f msecs.\n", timer.Elapsed());

	


	// Verify that the result vector is correct
	for (int i = 0; i < numElements; ++i)
	{
		if (fabs((h_A[i] + h_B[i]) - h_C[i]) > 1e-5)
		{
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}

	printf("Test PASSED\n");
	printf("Done\n");
	GpuTimer timer2;
	timer2.Start();
	main2();
	timer2.Stop();
	printf("Implemented CUDA code for vector sum ran in: %f msecs.\n", timer2.Elapsed());
	// Free device global memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);



	//Implement Vector Addition Using CUDA Streams

	//@@ Insert your code here to implement Vector Addition using streams and Time your implementation.

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);

	
	

	return 0;
}

