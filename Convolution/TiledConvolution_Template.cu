#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "GpuTimer.h"
#include<time.h>

using namespace cv;
using namespace std;


#define Mask_width 5

#define Mask_radius Mask_width / 2

#define O_TILE_WIDTH 12

#define BLOCK_WIDTH (O_TILE_WIDTH + Mask_width - 1)

//__constant__ float M[Mask_width][Mask_width] ={ {1,4,7,4,1 }, { 4,16,26,16,4 }, {7,26,41,26,7 }, {4,16,26,16,4 }, {1,4,7,4,1 }};

//In OpenCV the image is read in BGR format, that is for each pixel, the Blue, Green, then Red components are read from the image file.

// Serial implementation for running on CPU using a single thread.
void ConvolutionCpu_2D(unsigned char* InputImage, unsigned char* OutputImage, float* M, int numRows, int numCols, int Channels)
{
	//@@ Insert your code here
	for (int row = 0; row < numRows; row++)
	{
		for (int col = 0; col < numCols; col++)
		{
			
			for (int c = 0; c < Channels; c++)
			{
				
				float sum = 0.0f;
				

				for (int i = -Mask_radius; i < Mask_radius; i++)
				{
					for (int j = -Mask_radius; j < Mask_radius; j++)
					{
						int xoffset = j + col;
						int yoffset = i + row;
						if (xoffset >= 0 && xoffset < numCols&&yoffset >= 0 && yoffset < numRows)
						{
							int index = (yoffset*numCols + xoffset)*Channels + c;
							sum += InputImage[index] * M[(i + Mask_radius)*Mask_width + j + Mask_radius];
						}
					}
				}

				OutputImage[(row*numCols + col)*Channels + c] = (unsigned char)sum;
			}
		}
	}
	
}


// we have 3 channels corresponding to B, G, and R components of each pixel
// The input image is encoded as unsigned characters [0, 255]

__global__ void TiledConvolution_2D(unsigned char * InputImage, unsigned char * OutputImage, const float *__restrict__ M,int numRows, int numCols, int Channels)
{
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row_o = blockIdx.y*O_TILE_WIDTH + ty;
	int col_o = blockIdx.x*O_TILE_WIDTH + tx;

	int row_i = row_o - Mask_radius;
	int col_i = col_o - Mask_radius;

	//old code with the bug
	/*int in_index;
	__shared__ float ds_In[BLOCK_WIDTH][BLOCK_WIDTH];
	for (int c = 0; c < Channels; c++)
	{	

		if ((row_i >= 0) && (row_i < numRows) && (col_i >= 0) && (col_i < numCols))
		{
			in_index = (row_i*numCols + col_i)*Channels + c;
			ds_In[ty][tx] = InputImage[in_index];
		}
		else
			ds_In[ty][tx] = 0.0f;

		__syncthreads();

		float sum = 0.0f;

		if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH)
		{
			for (int i = 0; i < Mask_width; i++)
			{
				for (int j = 0; j < Mask_width; j++)
				{
					sum += M[(i)*Mask_width + j] * ds_In[i + ty][j + tx];
				}
			}
		
			__syncthreads();
			if (row_o < numRows && col_o < numCols)
				OutputImage[(row_o*numCols + col_o)*Channels + c ] = (unsigned char)sum;
			__syncthreads();
		}
	}*/

	__shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH][3];
	if ((row_i >= 0) && (row_i < numRows) && (col_i >= 0) && (col_i < numCols))
	{
		Ns[ty][tx][0] = InputImage[(row_i*numCols + col_i) * 3 + 0];
		Ns[ty][tx][1] = InputImage[(row_i*numCols + col_i) * 3 + 1];
		Ns[ty][tx][2] = InputImage[(row_i*numCols + col_i) * 3 + 2];
	}
	else {
		Ns[ty][tx][0] = 0.0f;
		Ns[ty][tx][1] = 0.0f;
		Ns[ty][tx][2] = 0.0f;

	}
	__syncthreads();

	float blue = 0.0f;
	float green = 0.0f;
	float red = 0.0f;
	if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {
		for (int i = 0; i < Mask_width; i++) {
			for (int j = 0; j < Mask_width; j++) {
				blue += M[(i)*Mask_width + j] * Ns[i + ty][j + tx][0];
				green += M[(i)*Mask_width + j] * Ns[i + ty][j + tx][1];
				red += M[(i)*Mask_width + j] * Ns[i + ty][j + tx][2];
			}

		}
		__syncthreads();

		if (row_o < numRows && col_o < numCols) {
			OutputImage[(row_o*numCols + col_o) * 3 + 0] = blue;
			OutputImage[(row_o*numCols + col_o) * 3 + 1] = green;
			OutputImage[(row_o*numCols + col_o) * 3 + 2] = red;
		}

	}

	
}



int main(void)
{
	//Read the image using OpenCV
	Mat image; //Create matrix to read image
	image = imread("lena_color.bmp", CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		printf("Cannot read image file %s", "lena_color.bmp");
		exit(1);
	}

	
	int imageChannels = 3;
	int imageWidth = image.cols;
	int imageHeight = image.rows;

	//Allocate the host image vectors
	unsigned char *h_InputImage = (unsigned char *)malloc(sizeof(unsigned char)*imageWidth*imageHeight*imageChannels);
	unsigned char *h_OutputImage = (unsigned char *)malloc(sizeof(unsigned char)*imageWidth*imageHeight*imageChannels);
	unsigned char *h_OutputImage_CPU = (unsigned char *)malloc(sizeof(unsigned char)*imageWidth*imageHeight*imageChannels);
	float *hostMaskData=(float *)malloc(sizeof(float)*Mask_width*Mask_width);

	h_InputImage = image.data; //The data member of a Mat object returns the pointer to the first row, first column of the image.
							
	float mask[] ={1,4,7,4,1, 4,16,26,16,4, 7,26,41,26,7, 4,16,26,16,4, 1,4,7,4,1 };
	for (int i = 0; i < 25; i++)
	{
		mask[i] = mask[i] / 273.0f;
	}
	hostMaskData = mask;
	
	//Allocate memory on the device for the input image and the output image and record the needed time
	unsigned char *d_InputImage, *d_OutputImage;
	float *deviceMaskData;
	GpuTimer timer;
	cudaError_t err1 = cudaSuccess;
	cudaError_t err2 = cudaSuccess;
	timer.Start();

	//@@ Insert Your code Here to allocate memory on the device for input and output images
	err1 = cudaMalloc((void **)&d_InputImage, sizeof(unsigned char)*imageWidth*imageHeight*imageChannels);
	if (err1 != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_InputImage (error code %s)!\n", cudaGetErrorString(err1));
		exit(EXIT_FAILURE);
	}

	err1 = cudaMalloc((void **)&d_OutputImage, sizeof(unsigned char)*imageWidth*imageHeight*imageChannels);
	if (err1 != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_output (error code %s)!\n", cudaGetErrorString(err1));
		exit(EXIT_FAILURE);
	}
	
	//@@Insert your code Here to allocate memory on the device for the Mask data
	err2 = cudaMalloc((void **)&deviceMaskData, sizeof(float)*Mask_width*Mask_width);
	if (err2 != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector deviceMaskData (error code %s)!\n", cudaGetErrorString(err2));
		exit(EXIT_FAILURE);
	}

	timer.Stop();
	printf("Time to allocate memory on the device is: %f msecs.\n", timer.Elapsed());


	//Copy the input image and mask data from the host to the device and record the needed time
	GpuTimer timer1;
	timer1.Start();

	//@@ Insert your code here to Copy the input image from the host to the device
	//@@Insert your code here to copy the mask data from host to device
	cudaMemcpy(d_InputImage, h_InputImage, sizeof(unsigned char)*imageWidth*imageHeight*imageChannels, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskData, hostMaskData, sizeof(float)*Mask_width*Mask_width, cudaMemcpyHostToDevice);
	timer1.Stop();
	printf("Time to copy the input image from the host to the device is: %f msecs.\n", timer1.Elapsed());

	

	//Do the Processing on the GPU
	//Kernel Execution Configuration Parameters
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	dim3 dimGrid((imageWidth - 1) / O_TILE_WIDTH + 1, (imageHeight - 1) / O_TILE_WIDTH + 1, 1);

	//@@ Insert Your code Here for grid dimensions
	
	
	//Invoke the 2DTiledConvolution kernel and record the needed time for its execution
	//GpuTimer timer;
	GpuTimer timer2;
	timer2.Start();

	//@@ Insert your code here for kernel invocation
	TiledConvolution_2D << <dimGrid, dimBlock >> >(d_InputImage, d_OutputImage, deviceMaskData, imageHeight, imageWidth, imageChannels);

	timer2.Stop();
	printf("Implemented CUDA code ran in: %f msecs.\n", timer2.Elapsed());

	//Copy resulting output image from device to host and record the needed time
	GpuTimer timer3;
	timer3.Start();

	//@@ Insert your code here to Copy resulting output image from device to host 
	cudaMemcpy(h_OutputImage, d_OutputImage, sizeof(unsigned char)*imageWidth*imageHeight*imageChannels, cudaMemcpyDeviceToHost);

	timer3.Stop();
	printf("Time to copy the output image from the device to the host is: %f msecs.\n", timer3.Elapsed());

	//Do the Processing on the CPU
	clock_t begin = clock();

	//@@ Insert your code her to call the cpu function for 2DConvolution on the CPU	
	ConvolutionCpu_2D(h_InputImage, h_OutputImage_CPU, hostMaskData, imageHeight, imageWidth, imageChannels);
	

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
	printf("Implemented CPU code ran in: %f msecs.\n", time_spent);

	//Postprocess and Display the resulting images using OpenCV
	Mat Image1(imageHeight, imageWidth, CV_8UC3, h_OutputImage); //colored output image mat object
	Mat Image2(imageHeight, imageWidth, CV_8UC3, h_OutputImage_CPU); //colored output image mat object



	namedWindow("CPUImage", WINDOW_NORMAL); //Create window to display the image
	namedWindow("GPUImage", WINDOW_NORMAL);
	imshow("GPUImage", Image1);
	imshow("CPUImage", Image2); //Display the image in the window
	waitKey(0); //Wait till you press a key 



	//Free host memory
	//free(h_OutputImage);
	image.release();
	Image1.release();
	Image2.release();
	free(h_OutputImage);
	free(h_OutputImage_CPU);

	//Free device memory

	//@@ Insert your code here to free device memory
	cudaFree(d_InputImage);
	cudaFree(d_OutputImage);
	cudaFree(deviceMaskData);
	

	return 0;

}