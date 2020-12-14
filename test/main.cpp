#include <opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/video.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/videoio.hpp>
#include "Extractor.h"
#include "Frame.h"

#include <iostream>
int main(int argc, char** argv)
{
	cv::Ptr<VideoCapture> cap;// = new cv::VideoCapture();
	if(argc<2)
	{
		std::cerr<<"Usage: ./run <Videofile Path>"<<std::endl;
		return -1;
	}
	else
	{
		cap = new cv::VideoCapture(argv[1]);
		if(!cap->isOpened())
		{
		std::cerr<<"There is no file exist"<<std::endl;
		return -1;
		}
	}
	PED_CAL::PostureExtractor pe(cap);

	// Main Loop

	for(size_t n = 0; n < cap->get(CAP_PROP_FRAME_COUNT);n++)
	{
		pe.ExtractBG();
		if (n<170) continue; // Before 170 frame, Learn Background
		pe.LabelPed();
	}

	std::cout<<"End Program"<<std::endl;
	return 0;
}
