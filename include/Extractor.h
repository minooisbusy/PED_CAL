#ifndef EXTRACTOR_H
#define EXTRACTOR_H
#include<opencv2/opencv.hpp>
using namespace cv;
namespace PED_CAL
{
class PostureExtractor
{
  //float scale2;
  size_t mMaxFrame;
  size_t mCurrFrame;
  size_t numOfLabels;
  unsigned int width;
  unsigned int height;
  Ptr<VideoCapture> mCap;
  Mat img;
  Rect r;// rectangle
  Ptr<BackgroundSubtractor> mBg_model;
  Mat element;
  Mat foreground, labels, stats, centroids;
  std::vector<Point2f> blob;
  public:
    PostureExtractor(){};
    PostureExtractor(VideoCapture *capture);
    void ExtractBG();
    void LabelPed();
    void drawAxis(Mat&, Point, Point, Scalar, const float);
    Point2i bresenhamPointSerach(int, int, int, int,
                                   const Mat&,
                                   const std::vector<Point2i>&);
};
}

#endif