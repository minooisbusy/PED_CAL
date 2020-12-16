#include <opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/video.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/videoio.hpp>
#include "Extractor.h"
#include "Detector.h"
#include "Frame.h"
#include "Estimator.h"

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
	PED_CAL::PeriodicDetector pd;
	PED_CAL::Estimator estimator;
	std::vector<PED_CAL::Posture> stack;// = pd.getData();
	std::vector<PED_CAL::Posture> stack_input;// = pd.getData();

	// Main Loop
	for(size_t n = 0; n < cap->get(CAP_PROP_FRAME_COUNT);n++)
	{
		pe.ExtractBG(); // Load Next frame, Learn Background
		if (n<170) continue; // Before 170 frame, Learn Background
		pd.push_back(pe.ExtractPed());
		if(pd.getSize() < 100) continue; // wait periodic data..

		Mat& pImg = pe.getImage();
		std::vector<PED_CAL::Posture> output = pd.ampd();
		std::vector<PED_CAL::Posture> input = pd.getData();

		stack.insert(stack.end(), output.begin(), output.end());
		stack_input.insert(stack_input.end(), input.begin(), input.end());
		for(int i=0; i<stack_input.size();i++)
		{
			circle(pImg,stack_input[i].foot,1,Scalar(0,255,0),-1);
			circle(pImg,stack_input[i].head,1,Scalar(255,0,0),-1);
			line(pImg,stack_input[i].foot,stack_input[i].head,Scalar(0,0,255),2,LINE_AA);
		}
		for(int i=0; i<stack.size();i++)
		{
			line(pImg,stack[i].foot,stack[i].head,Scalar(255,0,0),2,LINE_AA);
			//std::cout<<"output index="<<output[i].ratio<<std::endl;
		}
		estimator.grabPos(stack);
		Mat H = estimator.EstimateHomography();
		Mat F = estimator.EstimateFundamental();
    std::vector<Point2i> in(stack.size(),Point2i());
    std::vector<Point2i> out(stack.size(),Point2i());
    for(size_t i=0; i<stack.size(); i++)
    {
      out[i]=stack[i].foot;
      in[i]=stack[i].head;
    }
		Mat lines;
		cv::computeCorrespondEpilines(in,1, F, lines);
		for(int i=0; i<stack.size();i++)
		{
			Point2d p;
			const int x = stack[i].head.x; const int y = stack[i].head.y;
			const double den = H.at<double>(2,0)+ H.at<double>(2,1) + H.at<double>(2,2);
			p.x = (H.at<double>(0,0)*x + H.at<double>(0,1)*y + H.at<double>(0,2))/den;
			p.y = (H.at<double>(1,0)*x + H.at<double>(1,1)*y + H.at<double>(1,2))/den;
			cv::arrowedLine(pImg,stack[i].head, p,Scalar(0,255,0),2,LINE_AA);
		}
		pd.clear();
	imshow("image", pImg);
	waitKey(1);
	}
	std::cout<<"End Program"<<std::endl;
	return 0;
}
