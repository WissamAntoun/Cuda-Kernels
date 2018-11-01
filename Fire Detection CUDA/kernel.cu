
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <sstream>
#include <string>
#include <iostream>
#include <opencv\highgui.h>
#include <opencv\cv.h>
//#include <Windows.h>
#include <opencv2\video\background_segm.hpp>
#include <opencv2\photo\photo.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "GpuTimer.h"
#include <time.h>
using namespace std;
using namespace cv;

#define BLUR_Mask_width 5
#define BLUR_Mask_radius BLUR_Mask_width / 2

#define DILATE_Mask_width 9
#define DILATE_Mask_radius  DILATE_Mask_width / 2

#define ERODE_Mask_width 3
#define ERODE_Mask_radius  ERODE_Mask_width / 2

#define O_TILE_WIDTH 12
#define DILATE_O_TILE_WIDTH 8
#define ERODE_O_TILE_WIDTH 14

#define BLUR_BLOCK_WIDTH (O_TILE_WIDTH + BLUR_Mask_width - 1)
#define DILATE_BLOCK_WIDTH (DILATE_O_TILE_WIDTH + DILATE_Mask_width - 1)
#define ERODE_BLOCK_WIDTH (ERODE_O_TILE_WIDTH + ERODE_Mask_width - 1)

struct Coordinate
{
	int x, y;

	Coordinate(int paramx, int paramy) : x(paramx), y(paramy) {}
};

//default capture width and height
const int FRAME_WIDTH = 1280;
const int FRAME_HEIGHT = 720;

//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS = 50;

//minimum and maximum object area
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH / 1.5;

void drawObject(vector <Coordinate> fire, Mat &frame) {

	for (int i = 0; i < fire.size(); i++)
	{
		circle(frame, Point(fire.at(i).x, fire.at(i).y), 20, Scalar(0, 255, 0), 2);
		putText(frame, format("%d,%d", fire.at(i).x, fire.at(i).y), Point(fire.at(i).x, fire.at(i).y + 30), 1, 1, Scalar(0, 255, 0), 2);
	}

}

//create structuring element that will be used to "dilate" and "erode" image.
void morphOps(Mat &thresh) {

	//the element chosen here is a 3px by 3px rectangle
	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	
	//dilate with larger element so make sure object is nicely visible
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(9, 9));
	


	erode(thresh, thresh, erodeElement);
	erode(thresh, thresh, erodeElement);


	dilate(thresh, thresh, dilateElement);
	dilate(thresh, thresh, dilateElement);



}

//Tracking of object
void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed) {

	Mat temp;
	threshold.copyTo(temp);

	vector <Coordinate> fire;

	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//find contours of filtered image using openCV findContours function
	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	//use moments method to find our filtered object
	bool objectFound = false;
	if (hierarchy.size() > 0) {
		int numObjects = hierarchy.size();

		//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
		if (numObjects<MAX_NUM_OBJECTS) {
			for (int index = 0; index >= 0; index = hierarchy[index][0]) {

				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				//if the area is less than 20 px by 20px then it is probably just noise
				//if the area is the same as the 3/2 of the image size, probably just a bad filter
				//we only want the object with the largest area so we safe a reference area each
				//iteration and compare it to the area in the next iteration.
				if (area>MIN_OBJECT_AREA ) {
					x = moment.m10 / area;
					y = moment.m01 / area;
					fire.push_back(Coordinate(x, y));
					objectFound = true;
				}
				else
				{
					objectFound = false;
				}

			}
			//let user know you found an object
			if (objectFound == true) {
				putText(cameraFeed, "Tracking Object", Point(30, 60), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255),2);

				//draw object location on screen
				drawObject(fire, cameraFeed);
			}

		}
		else
		{
			putText(cameraFeed, "TOO MUCH NOISE! ADJUST FILTER", Point(30, 60), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255));
		}
	}
}


__global__ void TiledGpuBlur(unsigned char * InputImage, unsigned char * OutputImage, const float *__restrict__ M, int numRows, int numCols, int Channels)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row_o = blockIdx.y*O_TILE_WIDTH + ty;
	int col_o = blockIdx.x*O_TILE_WIDTH + tx;

	int row_i = row_o - BLUR_Mask_radius;
	int col_i = col_o - BLUR_Mask_radius;

	//loading the all the needed elements from the needed array will at max need 2 phases
	__shared__ float Ns[BLUR_BLOCK_WIDTH][BLUR_BLOCK_WIDTH][3];
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
		for (int i = 0; i < BLUR_Mask_width; i++) {
			for (int j = 0; j < BLUR_Mask_width; j++) {
				blue += M[(i)*BLUR_Mask_width + j] * Ns[i + ty][j + tx][0];
				green += M[(i)*BLUR_Mask_width + j] * Ns[i + ty][j + tx][1];
				red += M[(i)*BLUR_Mask_width + j] * Ns[i + ty][j + tx][2];
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

__global__ void TiledGpuDilate(unsigned char * InputImage, unsigned char * OutputImage, int numRows, int numCols  )
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row_o = blockIdx.y*DILATE_O_TILE_WIDTH + ty;
	int col_o = blockIdx.x*DILATE_O_TILE_WIDTH + tx;

	int row_i = row_o - DILATE_Mask_radius;
	int col_i = col_o - DILATE_Mask_radius;

	//loading the all the needed elements from the needed array will at max need 2 phases
	__shared__ float Ns[DILATE_BLOCK_WIDTH][DILATE_BLOCK_WIDTH];

	if ((row_i >= 0) && (row_i < numRows) && (col_i >= 0) && (col_i < numCols))
	{
		Ns[ty][tx] = InputImage[(row_i*numCols + col_i)];
		}
	else {
		Ns[ty][tx] = 0.0f;
	}

	__syncthreads();

	
	if (ty <DILATE_O_TILE_WIDTH && tx < DILATE_O_TILE_WIDTH) {
		for (int i = 0; i < DILATE_Mask_width; i++) {
			for (int j = 0; j < DILATE_Mask_width; j++) {
				if (Ns[i + ty][j + tx] == 255)
				{
					if (row_o < numRows && col_o < numCols) 
						OutputImage[(row_o*numCols + col_o)] = 255;
					return;
				}				
			}
		}
		OutputImage[(row_o*numCols + col_o)] = 0;
	}
}

__global__ void TiledGpuErode(unsigned char * InputImage, unsigned char * OutputImage, int numRows, int numCols)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row_o = blockIdx.y*ERODE_O_TILE_WIDTH + ty;
	int col_o = blockIdx.x*ERODE_O_TILE_WIDTH + tx;

	int row_i = row_o - ERODE_Mask_radius;
	int col_i = col_o - ERODE_Mask_radius;

	//loading the all the needed elements from the needed array will at max need 2 phases
	__shared__ float Ns[ERODE_BLOCK_WIDTH][ERODE_BLOCK_WIDTH];

	if ((row_i >= 0) && (row_i < numRows) && (col_i >= 0) && (col_i < numCols))
	{
		Ns[ty][tx] = InputImage[(row_i*numCols + col_i)];
	}
	else {
		Ns[ty][tx] = 0.0f;
	}

	__syncthreads();


	if (ty <ERODE_O_TILE_WIDTH && tx < ERODE_O_TILE_WIDTH) {
		for (int i = 0; i < ERODE_Mask_width; i++) {
			for (int j = 0; j < ERODE_Mask_width; j++) {
				if (Ns[i + ty][j + tx] == 0)
				{
					if (row_o < numRows && col_o < numCols)
						OutputImage[(row_o*numCols + col_o)] = 0;
					return;
				}
			}
		}
		OutputImage[(row_o*numCols + col_o)] = 255;
	}
}

__global__ void GPU_inrange(unsigned char * InputImage, int BLUE_MIN, int GREEN_MIN, int RED_MIN, int BLUE_MAX, int GREEN_MAX, int RED_MAX, unsigned char * Threshold, int numRows, int numCols)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = blockIdx.y*blockDim.y + ty;
	int col = blockIdx.x*blockDim.x + tx;

	

	if ((row < numRows) && (col < numCols))
	{
		unsigned char btemp = InputImage[(row*numCols + col) * 3];
		unsigned char gtemp = InputImage[(row*numCols + col) * 3 + 1];
		unsigned char rtemp = InputImage[(row*numCols + col) * 3 + 2];

		if (btemp > BLUE_MIN && btemp<BLUE_MAX && gtemp>GREEN_MIN && gtemp + 1< GREEN_MAX && rtemp > RED_MIN && rtemp< RED_MAX)
			Threshold[row*numCols + col] = 255;
		else
			Threshold[row*numCols + col] = 0;
	}

}

int main()
{

	VideoCapture capture;
	VideoCapture video("test_fire_5.avi");
	Mat originalFrame;
	Mat resultFrame;

	


	//for webcam
	/*capture.open(0);

	//set height and width of capture frame
	capture.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);*/

	//BackgroundSubtractorMOG2 mog;

	int x = 0, y = 0;

	int frameCounter = 0;
	int tick = 0;
	int fps;
	time_t timeBegin = time(0);

	int imageChannels = 3;
	int imageWidth = video.get(CV_CAP_PROP_FRAME_WIDTH);
	int imageHeight = video.get(CV_CAP_PROP_FRAME_HEIGHT);

	unsigned char *h_InputImage = (unsigned char *)malloc(sizeof(unsigned char)*imageWidth*imageHeight*imageChannels);
	unsigned char *h_OutputImage = (unsigned char *)malloc(sizeof(unsigned char)*imageWidth*imageHeight*imageChannels);
	unsigned char *h_threshold = (unsigned char*)malloc(sizeof(unsigned char)*imageWidth*imageHeight);
	unsigned char *h_threshold_dilated = (unsigned char*)malloc(sizeof(unsigned char)*imageWidth*imageHeight);
	unsigned char *h_threshold_eroded = (unsigned char*)malloc(sizeof(unsigned char)*imageWidth*imageHeight);


	float *hostMaskData = (float *)malloc(sizeof(float)*BLUR_Mask_width*BLUR_Mask_width);
	float mask[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	for (int i = 0; i < 25; i++)
	{
		mask[i] = mask[i] / 25.0f;
	}
	hostMaskData = mask;

	unsigned char *d_InputImage, *d_OutputImage;
	unsigned char *d_Threshold, *d_Threshold_dilated, *d_Threshold_eroded;
	float *BLUR_deviceMaskData;
	GpuTimer timer;
	cudaError_t err1 = cudaSuccess;
	cudaError_t err2 = cudaSuccess;
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

	err1 = cudaMalloc((void **)&d_Threshold, sizeof(unsigned char)*imageWidth*imageHeight);
	if (err1 != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_output (error code %s)!\n", cudaGetErrorString(err1));
		exit(EXIT_FAILURE);
	}
	err1 = cudaMalloc((void **)&d_Threshold_dilated, sizeof(unsigned char)*imageWidth*imageHeight);
	if (err1 != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_output (error code %s)!\n", cudaGetErrorString(err1));
		exit(EXIT_FAILURE);
	}
	err1 = cudaMalloc((void **)&d_Threshold_eroded, sizeof(unsigned char)*imageWidth*imageHeight);
	if (err1 != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_output (error code %s)!\n", cudaGetErrorString(err1));
		exit(EXIT_FAILURE);
	}

	//@@Insert your code Here to allocate memory on the device for the Mask data
	err2 = cudaMalloc((void **)&BLUR_deviceMaskData, sizeof(float)*BLUR_Mask_width*BLUR_Mask_width);
	if (err2 != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector deviceMaskData (error code %s)!\n", cudaGetErrorString(err2));
		exit(EXIT_FAILURE);
	}

	cudaMemcpy(BLUR_deviceMaskData, hostMaskData, sizeof(float)*BLUR_Mask_width*BLUR_Mask_width, cudaMemcpyHostToDevice);
	dim3 dimBlock(BLUR_BLOCK_WIDTH, BLUR_BLOCK_WIDTH, 1);
	dim3 BLUR_dimGrid((imageWidth - 1) / O_TILE_WIDTH + 1, (imageHeight - 1) / O_TILE_WIDTH + 1, 1);
	dim3 Thresh_dimGrid((imageWidth - 1) / BLUR_BLOCK_WIDTH + 1, (imageHeight - 1) / BLUR_BLOCK_WIDTH + 1, 1);

	dim3 Dilate_dimBlock(DILATE_BLOCK_WIDTH, DILATE_BLOCK_WIDTH, 1);
	dim3 Dilate_dimGrid((imageWidth - 1) / DILATE_O_TILE_WIDTH + 1, (imageHeight - 1) / DILATE_O_TILE_WIDTH + 1, 1);

	dim3 Erode_dimBlock(ERODE_BLOCK_WIDTH, ERODE_BLOCK_WIDTH, 1);
	dim3 Erode_dimGrid((imageWidth - 1) / ERODE_O_TILE_WIDTH + 1, (imageHeight - 1) / ERODE_O_TILE_WIDTH + 1, 1);



	int lower_bound []= { 40,0,90 };
	int higher_bound[] = { 256,95,256 };
	int BLUE_MIN = 40;
	int GREEN_MIN = 0;
	int RED_MIN = 95;
	int BLUE_MAX = 256;
	int GREEN_MAX = 95;
	int RED_MAX = 256;
	namedWindow("OriginalVideo", 0);
	namedWindow("ResultVideo", 0);

	

	while (1)
	{	
		
		//store image to matrix
		//capture.read(originalFrame);
		video >> originalFrame;
		if (originalFrame.empty())
		{
			break;
		}
		h_InputImage = originalFrame.data;
		cudaMemcpy(d_InputImage, h_InputImage, sizeof(unsigned char)*imageWidth*imageHeight*imageChannels, cudaMemcpyHostToDevice);

		//noise removal
		//medianBlur(originalFrame, resultFrame,3);
		TiledGpuBlur << <BLUR_dimGrid, dimBlock >> >(d_InputImage, d_OutputImage, BLUR_deviceMaskData, imageHeight, imageWidth, imageChannels);
		//cudaDeviceSynchronize();
		//cudaMemcpy(h_OutputImage, d_OutputImage, sizeof(unsigned char)*imageWidth*imageHeight*imageChannels, cudaMemcpyDeviceToHost);		
		//Mat(imageHeight, imageWidth, CV_8UC3, h_OutputImage).copyTo(resultFrame);

		//thresholding
		//inRange(resultFrame, Scalar(40,0,90), Scalar(256,95,256), resultFrame);
		GPU_inrange << <Thresh_dimGrid, dimBlock >> >(d_OutputImage, BLUE_MIN, GREEN_MIN, RED_MIN, BLUE_MAX, GREEN_MAX, RED_MAX, d_Threshold, imageHeight, imageWidth);
		//cudaMemcpy(h_threshold, d_Threshold, sizeof(unsigned char)*imageWidth*imageHeight, cudaMemcpyDeviceToHost);
		//Mat(imageHeight, imageWidth, CV_8U, h_threshold).copyTo(resultFrame);

		//dilate and erode
		//morphOps(resultFrame);
		TiledGpuErode << <Erode_dimGrid, Erode_dimBlock >> >(d_Threshold, d_Threshold_eroded, imageHeight, imageWidth);
		TiledGpuErode << <Erode_dimGrid, Erode_dimBlock >> >(d_Threshold_eroded, d_Threshold, imageHeight, imageWidth);
		//cudaMemcpy(h_threshold_eroded, d_Threshold, sizeof(unsigned char)*imageWidth*imageHeight, cudaMemcpyDeviceToHost);
		//Mat(imageHeight, imageWidth, CV_8U, h_threshold_eroded).copyTo(resultFrame);


		TiledGpuDilate<<<Dilate_dimGrid,Dilate_dimBlock>>>(d_Threshold, d_Threshold_dilated, imageHeight, imageWidth);
		TiledGpuDilate << <Dilate_dimGrid, Dilate_dimBlock >> >(d_Threshold_dilated, d_Threshold, imageHeight, imageWidth);
		cudaMemcpy(h_threshold_dilated, d_Threshold, sizeof(unsigned char)*imageWidth*imageHeight, cudaMemcpyDeviceToHost);
		Mat(imageHeight, imageWidth, CV_8U, h_threshold_dilated).copyTo(resultFrame);

		

		//trackFilteredObject(x, y, resultFrame, originalFrame);
		
		frameCounter++;

		time_t timeNow = time(0) - timeBegin;

		if (timeNow - tick >= 1)
		{
			tick++;
			fps = frameCounter;
			frameCounter = 0;
		}
		cout << fps << endl;
		/*//putText(originalFrame, format("Average FPS=%d", fps), Point(30, 200), FONT_HERSHEY_PLAIN, 10, Scalar(0, 0, 255), 10);
		
		imshow("OriginalVideo", originalFrame);
		imshow("ResultVideo", resultFrame);
		
		//every fram is displayed for 1 ms, ESC to quit
		if (waitKey(1) == 27) {
			break;
		}*/
		
	}
	originalFrame.release();
	resultFrame.release();
	
	cudaFree(d_Threshold);
	cudaFree(d_InputImage);
	cudaFree(d_OutputImage);
	cudaFree(BLUR_deviceMaskData);
	return 0;
}

