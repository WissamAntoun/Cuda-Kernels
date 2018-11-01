#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "GpuTimer.h"
#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include<time.h>

using namespace cv;
using namespace std;

#define BLUR_SIZE 8
#define BlockSize 16



// Serial implementation for running on CPU using a single thread.
void ImageBlurCpu(unsigned char* out, unsigned char* in, int width, int height)
{
	
	//@@ Insert your code here
	for(int row = 0; row<height;row++)
		for (int col = 0; col < width; col++)
		{
			int temp = 0;
			int pixelcount = 0;
			for (int i = -BLUR_SIZE; i <= BLUR_SIZE; i++)
			{
				for (int j = -BLUR_SIZE; j <= BLUR_SIZE; j++)
				{
					int y = row + i;
					int x = col + j;

					if (y >= 0 && y < height&&x >= 0 && x < width)
					{
						temp += in[y*width + x];
						pixelcount++;
					}
				}
			}

			out[row*width + col] = (unsigned char)(temp / pixelcount);
		}
}


// The input image is grayscale and is encoded as unsigned characters [0, 255]

__global__ void ImageBlur(unsigned char *out, unsigned char *in, int width, int height)
{
	// Data cache: threadIdx.x , threadIdx.y
	__shared__ unsigned char ds_in[BlockSize + BLUR_SIZE * 2][BlockSize + BLUR_SIZE * 2];

	
	int x, y;	// image based coordinate

	
	int col = threadIdx.x + blockIdx.x* blockDim.x;
	int row = threadIdx.y + blockIdx.y* blockDim.y;

	// global mem address of this thread
	const int OGoffset = row * width + col;

	// First Quadrant
	x = col - BLUR_SIZE;
	y = row - BLUR_SIZE;

	if (x < 0 || y < 0)
		ds_in[threadIdx.x][threadIdx.y] = NULL;//accessing areas outside the image boundary will assign it to NULL
	else
		ds_in[threadIdx.x][threadIdx.y] = in[OGoffset - BLUR_SIZE - BLUR_SIZE*width];

	// Second Quadrant
	x = col + BLUR_SIZE;
	y = row - BLUR_SIZE;

	if (x > width - 1 || y < 0)
		ds_in[threadIdx.x + blockDim.x][threadIdx.y] = NULL;
	else
		ds_in[threadIdx.x + blockDim.x][threadIdx.y] = in[OGoffset + BLUR_SIZE - BLUR_SIZE*width];

	// Third Quadrant
		x = col - BLUR_SIZE;
	y = row + BLUR_SIZE;
	if (x < 0 || y > height - 1)
		ds_in[threadIdx.x][threadIdx.y + blockDim.y] = NULL;
	else
		ds_in[threadIdx.x][threadIdx.y + blockDim.y] = in[OGoffset - BLUR_SIZE + BLUR_SIZE*width];

	// Forth Quadrant
	x = col + BLUR_SIZE;
	y = row + BLUR_SIZE;
	if (x > width - 1 || y > height - 1)
		ds_in[threadIdx.x + blockDim.x][threadIdx.y + blockDim.y] = NULL;
	else
		ds_in[threadIdx.x + blockDim.x][threadIdx.y + blockDim.y] = in[OGoffset + BLUR_SIZE + BLUR_SIZE*width];

	__syncthreads();

	int temp = 0;
	int pixelcount = 0;

	//accounting for the substraction of BLUR_Size in the loop to prevent negative memory indices
	x = threadIdx.x + BLUR_SIZE;
	y = threadIdx.y + BLUR_SIZE;

	for (int i = -BLUR_SIZE; i <= BLUR_SIZE; i++)
	{
		for (int j = -BLUR_SIZE; j <= BLUR_SIZE; j++)
		{
			if (ds_in[x + i][y + j]!= NULL)//checking if it is inside the image area
			{
				temp += ds_in[x + i][y + j];
				pixelcount++;
			}
		}
	}

	out[OGoffset] = (unsigned char)(temp/pixelcount);
}



int main(void)
{
	//Read the image using OpenCV
	Mat image; //Create matrix to read iamge
	image= imread("Tiger2.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	if (image.empty()) {
		printf("Cannot read image file %s", "Tiger2.jpg");
		exit(1);
	}

	
	
	int imageWidth=image.cols;
	int imageHeight=image.rows;

	//Allocate the host image vectors
	unsigned char *h_OrigImage;
	unsigned char *h_BlurImage= (unsigned char *)malloc(sizeof(unsigned char)*imageWidth*imageHeight);
	unsigned char *h_BlurImage_CPU= (unsigned char *)malloc(sizeof(unsigned char)*imageWidth*imageHeight);

	h_OrigImage = image.data; //The data member of a Mat object returns the pointer to the first row, first column of the image.
							 //try image.ptr()


	//Allocate memory on the device for the original image and the blurred image and record the needed time
	unsigned char *d_OrigImage, *d_BlurImage;
	GpuTimer timer;
	timer.Start();
	
	//@@ Insert Your code Here to allocate memory on the device for original and blurred images
	cudaError_t err;
	d_OrigImage = NULL;
	err = cudaMalloc((void **)&d_OrigImage, sizeof(unsigned char)*imageWidth*imageHeight);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_origimage (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	d_BlurImage = NULL;
	err = cudaMalloc((void **)&d_BlurImage, sizeof(unsigned char)*imageWidth*imageHeight);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_BlurImage (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	timer.Stop();
	printf("Time to allocate memory on the device is: %f msecs.\n", timer.Elapsed());

	

	//Copy the original image from the host to the device and record the needed time
	GpuTimer timer1;
	timer1.Start();
	
	//@@ Insert your code here to Copy the original image from the host to the device
	cudaMemcpy(d_OrigImage, h_OrigImage, sizeof(unsigned char)*imageWidth*imageHeight, cudaMemcpyHostToDevice);

	timer1.Stop();
	printf("Time to copy the Original image from the host to the device is: %f msecs.\n", timer1.Elapsed());

	
	//Do the Processing on the GPU
	//Kernel Execution Configuration Parameters
	dim3 dimBlock(16, 16, 1);
	
	//@@ Insert Your code Here for grid dimensions
	dim3 dimGrid((imageWidth - 1) / 16 + 1, (imageHeight - 1) / 16 + 1, 1);
	
	
	
	//Invoke the ImageBlur kernel and record the needed time for its execution
	//GpuTimer timer;
	GpuTimer timer2;
	timer2.Start();

	//@@ Insert your code here for kernel invocation
	ImageBlur<<<dimGrid,dimBlock>>>(d_BlurImage, d_OrigImage, imageWidth, imageHeight);

	timer2.Stop();
	printf("Implemented ImageBlur Kernel ran in: %f msecs.\n", timer2.Elapsed());

	//Copy resulting blurred image from device to host and record the needed time
	GpuTimer timer3;
	timer3.Start();
	
	//@@ Insert your code here to Copy resulting blurred image from device to host
	cudaMemcpy(h_BlurImage, d_BlurImage, sizeof(unsigned char)*imageWidth*imageHeight, cudaMemcpyDeviceToHost);

	timer3.Stop();
	printf("Time to copy the blurred image from the device to the host is: %f msecs.\n", timer3.Elapsed());

	

	//Do the Processing on the CPU
	clock_t begin = clock();
	
	//@@ Insert your code her to call the cpu function for ImageBlur on the CPU	
	ImageBlurCpu(h_BlurImage_CPU, h_OrigImage, imageWidth, imageHeight);

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC*1000;
	printf("Implemented CPU code ran in: %f msecs.\n", time_spent);

	//Postprocess and Display the resulting images using OpenCV
	Mat Image1(imageHeight, imageWidth,CV_8UC1,h_BlurImage); //grayscale image mat object
	Mat Image2(imageHeight,imageWidth,CV_8UC1,h_BlurImage_CPU ); //grayscale image mat object

	

	namedWindow("CPUImage", WINDOW_NORMAL); //Create window to display the image
	namedWindow("GPUImage", WINDOW_NORMAL);
	namedWindow("OriginalImage", WINDOW_NORMAL);
	imshow("GPUImage",Image1);
	imshow("CPUImage",Image2); //Display the image in the window
	imshow("OriginalImage", image); //Display the original image in the window
	waitKey(0); //Wait till you press a key 

	
	
	//Free host memory
	image.release();
	Image1.release();
	Image2.release();
	free(h_BlurImage);
	free(h_BlurImage_CPU);

	//Free device memory
	
	//@@ Insert your code here to free device memory
	cudaFree(d_OrigImage);
	cudaFree(d_BlurImage);

	return 0;

}