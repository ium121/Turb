// Heat_Sci.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>
#include <ctime>
#include <time.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\core\core.hpp"

#include<iostream>

using namespace cv;
using namespace std;

time_t t1, t2;
double diff = 0;
IplImage *image, *grey, *color;
IplImage *corrected = 0; IplImage *img2 = 0; IplImage *vel_x = 0, *vel_y = 0, *flow = 0;
IplImage *curr = 0, *current = 0;
float data_xf[256][256], data_yf[256][256]; 
float average2[256][256];

int main(int argc, char** argv)
{
	CvCapture* capture = 0;
	capture = cvCaptureFromAVI("video.avi");
	cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, 0);
	double x = cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
	printf("Frame Rate: %f ", x);
	if (!capture)
	{
		fprintf(stderr, "%d Could not initialize capturing...\n", argc);
		return -1;
	}
	cvNamedWindow("Input and Output", 1);
	
	time(&t1);
	for (int q = 0; q<100; q++)
	{
		cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, q + 1);
		IplImage* frame = 0;
		for (int p = 0; p<1; p++)
		{
			frame = cvQueryFrame(capture);
			if (!frame)
				break;


			if (!curr)
			{
				curr = cvCreateImage(cvGetSize(frame), 8, 3);
				curr->origin = frame->origin;
				current = cvCreateImage(cvGetSize(frame), 8, 1);
			}

			cvCopy(frame, curr, 0);
			cvConvertImage(curr, curr, CV_CVTIMG_FLIP);
			cvCvtColor(curr, current, CV_BGR2GRAY);


		}
		cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, q);
	
		for (int p = 0; p<6; p++)
		{
			IplImage* frame = 0;
			int i = 0, k = 0, c = 0;
			frame = cvQueryFrame(capture);
			if (!frame)
				break;
			if (!image)
			{
				image = cvCreateImage(cvGetSize(frame), 8, 3);
				image->origin = frame->origin;
				grey = cvCreateImage(cvGetSize(frame), 8, 1);
				vel_x = cvCreateImage(cvGetSize(grey), 32, 1);
				vel_y = cvCreateImage(cvGetSize(grey), 32, 1);
				flow = cvCreateImage(cvGetSize(grey), 32, 2);
				corrected = cvCreateImage(cvGetSize(grey), 8, 1);
			}
			cvCopy(frame, image, 0);
			cvConvertImage(image, image, CV_CVTIMG_FLIP);
			cvCvtColor(image, grey, CV_BGR2GRAY);
			
			int j = 0, r = 0;
			char s[20];
			int height, width, step;
			float *data_x1, *data_y1;
			
			cvCalcOpticalFlowFarneback(current, grey, flow, 0.5, 3, 31, 3, 5, 1.1, 0);
			cvSplit(flow, vel_x, vel_y, NULL, NULL);
			height = grey->height;
			width = grey->width;
			step = grey->widthStep;
			data_x1 = (float *)vel_x->imageData;
			data_y1 = (float *)vel_y->imageData;
			for (i = 0; i<height; i++)
			{
				for (j = 0; j<width; j++)
				{

					data_xf[i][j] = data_xf[i][j] + data_x1[i*step + j];

					data_yf[i][j] = data_yf[i][j] + data_y1[i*step + j];


					if (p == 5)
					{
						data_xf[i][j] = data_xf[i][j] / 5;
						data_yf[i][j] = data_yf[i][j] / 5;
						data_x1[i*step + j] = j - data_xf[i][j];
						data_y1[i*step + j] = i - data_yf[i][j];
						if (data_x1[i*step + j]<0)
							data_x1[i*step + j] = 0;
						if (data_y1[i*step + j]<0)
							data_y1[i*step + j] = 0;
						if (data_x1[i*step + j]>(width - 2))
							data_x1[i*step + j] = (float)(width - 2);
						if (data_y1[i*step + j]>(height - 2))
							data_y1[i*step + j] = (float)(height - 2);
					}

				}
			}

			if (p == 5)
			{
				cvRemap(current, corrected, vel_x, vel_y, CV_INTER_LINEAR, cvScalarAll(0));
				cvConvertImage(grey, grey, CV_CVTIMG_FLIP);
				cvConvertImage(corrected, corrected, CV_CVTIMG_FLIP);
				cvShowImage("Corrected", corrected);
				cv::Mat image = cv::cvarrToMat(corrected);
				Mat imgLap, imgResult;
				Mat kernel = (Mat_<float>(3, 3) <<
					0, 1, 0,
					1, -4, 1,
					0, 1, 0);
				filter2D(image, imgLap, CV_32F, kernel);
				image.convertTo(image, CV_32F);
				imgResult = image - imgLap;
				imgResult.convertTo(imgResult, CV_8U);
				imgLap.convertTo(imgLap, CV_8U);
		
				int dstWidth = grey->width + grey->width;
				int dstHeight = grey->height;
				IplImage* dst = cvCreateImage(cvSize(dstWidth, dstHeight), 8, 1);
	
				cvSetImageROI(dst, cvRect(0, 0, grey->width, grey->height));
				cvCopy(grey, dst, NULL);
				cvResetImageROI(dst);

				cvSetImageROI(dst, cvRect(corrected->width, 0, corrected->width, corrected->height));
				cvCopy(imgResult, dst, NULL);
				cvResetImageROI(dst);
				cvShowImage("Input and Output", dst);
				//sprintf_s(s, "frame%d.jpg", q);
				//cvSaveImage(s, corrected);
				c = cvWaitKey(1);
				if ((char)c == 27)
					break;
			}
		}

	}
	time(&t2);
	diff = difftime(t2, t1);
	printf("%f", diff);
	cvReleaseCapture(&capture);
	cvDestroyWindow("distorted");
	cvDestroyWindow("corrected");
	return 0;
}

