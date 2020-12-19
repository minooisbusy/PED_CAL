#include <opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/video.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/videoio.hpp>
#include "Extractor.h"
#include "Detector.h"
//#include "Frame.h"
#include "Estimator.h"
#include <limits>

bool cmp(const PED_CAL::Posture& v1, const PED_CAL::Posture& v2)
{
	if(v1.ratio == v2.ratio)
		return v1.theta>v2.theta;
return v1.ratio>v2.ratio;

}
int main(int argc, char** argv)
{
	bool first =true;
	char key='n';
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
	std::vector<PED_CAL::Posture> stack_input;// = pd.getdata();

	// Main Loop
	//for(size_t n = 0; n < cap->get(CAP_PROP_FRAME_COUNT);n++)
	double reproj_err = std::numeric_limits<double>::max();
	Mat H;
	for(size_t n = 0; n < 3500;n++)
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
			//line(pImg,stack_input[i].foot,stack_input[i].head,Scalar(0,0,255),2,LINE_AA);
			circle(pImg,stack_input[i].foot,1,Scalar(0,255,0),-1);
			circle(pImg,stack_input[i].head,1,Scalar(255,0,0),-1);
		}

		std::vector<PED_CAL::Posture> stack_good;// = pd.getData();
		std::sort(stack.begin(), stack.end(), cmp);
		for(int i=0; i<stack.size()*0.5;i++)
		{
			stack_good.push_back(stack[i]);
		}
		for(int i=0; i<stack_good.size();i++)
		{
			line(pImg,stack_good[i].foot,stack_good[i].head,Scalar(255,0,0),2,LINE_AA);
			//std::cout<<"output index="<<output[i].ratio<<std::endl;
		}

		estimator.grabPos(stack_good);
		Mat H_cand=cv::Mat::eye(3,3,CV_32F);
		if(stack_good.size()>6)
		{
			H_cand = estimator.EstimateHomography();
			const double cand_reproj_err = PED_CAL::computeError(H_cand, stack_good);
			if(cand_reproj_err==0)
				continue;
			if(cand_reproj_err < reproj_err )
			{
				H = H_cand;
				reproj_err = cand_reproj_err;
				std::cout<<"Error update:"<<reproj_err<<std::endl;
			}
		}
		else
		{
			continue;
		}

    std::vector<Point2i> in(stack.size(),Point2i());
    std::vector<Point2i> out(stack.size(),Point2i());
    for(size_t i=0; i<stack.size(); i++)
    {
      out[i]=stack[i].foot;
      in[i]=stack[i].head;
    }
		Mat lines;
		//cv::computeCorrespondEpilines(in,1, F, lines);
		for(int i=0; i<stack_good.size();i++)
		{
			Point2d p;
			const int x = stack_good[i].head.x; const int y = stack_good[i].head.y;
			const double den = H.at<double>(2,0)*x+ H.at<double>(2,1)*y + H.at<double>(2,2);
			p.x = (H.at<double>(0,0)*x + H.at<double>(0,1)*y + H.at<double>(0,2))/den;
			p.y = (H.at<double>(1,0)*x + H.at<double>(1,1)*y + H.at<double>(1,2))/den;
			cv::line(pImg,stack_good[i].head, p,Scalar(0,255,0),2,LINE_AA);
		}
		pd.clear();
		imshow("image", pImg);
		key = waitKey(1);
		if(first)
		{
			first = false;
		}
	}
	waitKey(0);

	std::cout<<"End Program"<<std::endl;
	return 0;
}
