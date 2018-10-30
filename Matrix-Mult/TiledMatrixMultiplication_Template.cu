
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<math.h>
#include <stdio.h>
#include<time.h>
#include <iostream>
#include <stdlib.h>
#include "GpuTimer.h"
#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
using namespace std;

#define BLOCK_SIZE 32


//Compute C=A*B
// Serial implementation for running on CPU using a single thread.
void MatrixMultiplyCpu(float* A, float* B, float* C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns)
{
	//@@ Insert Your Code Here for the CPU Function to Compute Matrix Maltiply
	float temp = 0;
	for (int Row = 0; Row < numARows; ++Row)
	{
		for (int Col = 0; Col < numBColumns; ++Col)
		{
			temp = 0;
			for (int k = 0; k<numAColumns; ++k)
			{

				temp += A[Row*numAColumns + k] * B[k * numBColumns + Col];
			}
			C[Row*numCColumns + Col] = temp;
		}
	}
}


//GPU Kernel for Tiled Matrix Multiplication
__global__ void TiledMatrixMultiply(float* A, float* B, float* C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns)
{

	__shared__ float ds_A[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float ds_B[BLOCK_SIZE][BLOCK_SIZE];
	//@@ Insert Your Code Here for the CUDA Kernel for Tiled Matrix Multiply
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((Row < numARows) && (Col < numBColumns))
	{
		float pvalue = 0;
		int pmax = (numAColumns - 1) / BLOCK_SIZE + 1;
		for (int p = 0; p <pmax; p++)
		{
			ds_A[threadIdx.y][threadIdx.x] = A[Row*numAColumns + p*BLOCK_SIZE + threadIdx.x];
			ds_B[threadIdx.y][threadIdx.x] = B[(p*BLOCK_SIZE + threadIdx.y)*numBColumns + Col];
			__syncthreads();


			for (int i = 0; i < BLOCK_SIZE; i++)
			{
				pvalue += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
			}
			__syncthreads();
		}
		C[Row*numCColumns + Col] = pvalue;
	}
}

//GPU Kernel for Basic Matrix Multiplication
__global__ void BasixMatrixMultiply(float* A, float* B, float* C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns)
{
	//@@ Insert Your Code Here for the CUDA Kernel for Basic Matrix Multiply
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((Row < numARows) && (Col < numBColumns))
	{
		float temp = 0;
		for (int i = 0; i < numAColumns; ++i)
		{
			temp += A[Row*numAColumns + i] * B[i * numBColumns + Col];
		}
		C[Row*numCColumns + Col] = temp;
	}

}
int main(void)
{

	int numARows = 960; // number of rows in the matrix A
	int numAColumns = 640; // number of columns in the matrix A
	int numBRows = 640; // number of rows in the matrix B
	int numBColumns = 800; // number of columns in the matrix B

	int numCRows; // number of rows in the matrix C (you have to set this)
	int numCColumns; // number of columns in the matrix C (you have to set this)

					 //@@ Insert Your Code Here to Set numCRows and numCColumns
	numCRows = 960;
	numCColumns = 800;

	//Allocate the host memory for the input and output matrices
	float *h_A = (float *)malloc(sizeof(float)*numARows*numAColumns);
	float *h_B = (float *)malloc(sizeof(float)*numBRows*numBColumns);
	float *h_C = (float *)malloc(sizeof(float)*numCRows*numCColumns);
	float *h_C_CPU = (float *)malloc(sizeof(float)*numCRows*numCColumns);


	//Random Initialize Matrix A. 
	//There are several ways to do this, such as making functions for manual input or using random numbers. 
	//In this case, we simply use a for loop to fill the cells with trigonometric values of the indices:
	// Set the Seed for the random number generator rand() 
	//srand(clock());
	float r3;
	for (int i = 0; i<numARows; i++)
	{
		for (int j = 0; j<numAColumns; j++)
		{
			r3 = -125.0f + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (125.0f)));
			h_A[i*numAColumns + j] = 0;
			//h_A[i*numAColumns + j] = sin(i);
		}
	}

	//Random Initialize Matrix B
	for (int i = 0; i<numBRows; i++)
	{
		for (int j = 0; j<numBColumns; j++)
		{
			r3 = -125.0f + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (125.0f)));
			h_B[i*numBColumns + j] = 0;
			//h_B[i*numBColumns + j] = cos(j);

		}
	}






	//Allocate memory on the device for input and output matrices and record the needed time
	float *d_A, *d_B, *d_C;
	GpuTimer timer;
	timer.Start();

	//@@Insert Your Code Here to allocate memory for d_A, d_B, d_C
	cudaError_t err;
	d_A = NULL;
	err = cudaMalloc((void **)&d_A, sizeof(float)*numARows*numAColumns);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	d_B = NULL;
	err = cudaMalloc((void **)&d_B, sizeof(float)*numBRows*numBColumns);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	d_C = NULL;
	err = cudaMalloc((void **)&d_C, sizeof(float)*numCRows*numCColumns);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	timer.Stop();
	printf("Time to allocate memory on the device is: %f msecs.\n", timer.Elapsed());



	//Copy the input matrices A and B from the host to the device and record the needed time
	GpuTimer timer1;
	timer1.Start();

	//@@ Insert Your Code Here to copy matrices A and B from Host to Device
	cudaMemcpy(d_A, h_A, sizeof(float)*numARows*numAColumns, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sizeof(float)*numBRows*numBColumns, cudaMemcpyHostToDevice);

	timer1.Stop();
	printf("Time to copy the Matrix from the host to the device is: %f msecs.\n", timer1.Elapsed());


	//Do the Processing on the GPU
	//@@ Insert Kernel Execution Configuration Parameters
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 dimGrid((numCColumns - 1) / BLOCK_SIZE + 1, (numCRows - 1) / BLOCK_SIZE + 1, 1);


	//Invoke the BasicMatrixMultiply kernel and record the needed time for its execution
	GpuTimer timer2;
	timer2.Start();
	//@@ Insert Your Code Here for Kernel Invocation
	TiledMatrixMultiply<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
	timer2.Stop();
	printf("Implemented tiled CUDA code ran in: %f msecs.\n", timer2.Elapsed());

	//Copy resulting matrix from device to host and record the needed time
	GpuTimer timer3;
	timer3.Start();
	//@@ Insert Your Code Here to Copy the resulting Matrix d_C from device to the Host h_C
	cudaMemcpy(h_C, d_C, sizeof(float)*numCRows*numCColumns, cudaMemcpyDeviceToHost);
	timer3.Stop();
	printf("Time to copy the resulting Matrix from the device to the host is: %f msecs.\n", timer3.Elapsed());


	//Do the Processing on the CPU
	clock_t begin = clock();
	//@@ Insert Your Code Here to call the CPU function MatrixMultiplyCpu where the resulting matrix is h_C_CPU
	MatrixMultiplyCpu(h_A, h_B, h_C_CPU, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
	printf("Implemented CPU serial code ran in: %f msecs.\n", time_spent);

	//Verify Results Computed by GPU and CPU
	for (int i = 0; i<numCRows; i++)
		for (int j = 0; j<numCColumns; j++)
		{
			//fprintf(stderr, "c CPU:%d  c:%d at (%d,%d)!\n", h_C_CPU[i*numCColumns + j], h_C[i*numCColumns + j], i, j);
			if (fabs(h_C_CPU[i*numCColumns + j] - h_C[i*numCColumns + j]) > 1e-5)
			{

				fprintf(stderr, "Result verification failed at element (%d,%d)!\n", i, j);
				exit(EXIT_FAILURE);
			}
		}
	printf("Test PASSED\n");


	

	//Free device memory
	//@@ Insert Your Code Here to Free Device Memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	GpuTimer timer4;
	timer4.Start();

	//@@Insert Your Code Here to allocate memory for d_A, d_B, d_C
	d_A = NULL;
	err = cudaMalloc((void **)&d_A, sizeof(float)*numARows*numAColumns);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	d_B = NULL;
	err = cudaMalloc((void **)&d_B, sizeof(float)*numBRows*numBColumns);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	d_C = NULL;
	err = cudaMalloc((void **)&d_C, sizeof(float)*numCRows*numCColumns);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	timer4.Stop();
	printf("Time to allocate memory on the device is: %f msecs.\n", timer4.Elapsed());



	//Copy the input matrices A and B from the host to the device and record the needed time
	GpuTimer timer5;
	timer5.Start();

	//@@ Insert Your Code Here to copy matrices A and B from Host to Device
	cudaMemcpy(d_A, h_A, sizeof(float)*numARows*numAColumns, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sizeof(float)*numBRows*numBColumns, cudaMemcpyHostToDevice);

	timer5.Stop();
	printf("Time to copy the Matrix from the host to the device is: %f msecs.\n", timer5.Elapsed());



	//Invoke the BasicMatrixMultiply kernel and record the needed time for its execution
	GpuTimer timer6;
	timer6.Start();
	//@@ Insert Your Code Here for Kernel Invocation
	BasixMatrixMultiply<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
	timer6.Stop();
	printf("Implemented basic CUDA code ran in: %f msecs.\n", timer6.Elapsed());

	//Copy resulting matrix from device to host and record the needed time
	GpuTimer timer7;
	timer7.Start();
	//@@ Insert Your Code Here to Copy the resulting Matrix d_C from device to the Host h_C
	cudaMemcpy(h_C, d_C, sizeof(float)*numCRows*numCColumns, cudaMemcpyDeviceToHost);
	timer7.Stop();
	printf("Time to copy the resulting Matrix from the device to the host is: %f msecs.\n", timer7.Elapsed());

	//Free device memory
	//@@ Insert Your Code Here to Free Device Memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	//Free host memory
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_C_CPU);


	return 0;

}