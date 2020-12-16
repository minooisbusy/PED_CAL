#ifndef EXTRACTOR_H
#define EXTRACTOR_H
#include<opencv2/opencv.hpp>
#include <memory>
#include <iostream>
using namespace cv;
namespace PED_CAL
{
class Posture
{
  public:
  Point2i head;
  Point2i foot;
  double theta;
  double ratio;
  Posture();
  Posture(int hx, int hy, int fx, int fy, double _theta, double _ratio);
  Posture(Point2i _head, Point2i _foot, double _theta, double _ratio);
  Posture(const Posture& posture);
};
class PostureExtractor
{
  // Output pedestrian
  std::shared_ptr<Posture> posture;
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
    Posture ExtractPed();
    void drawAxis(Mat&, Point, Point&, Scalar, const float);
    Point2i bresenhamPointSerach(int, int, int, int,
                                    Mat&,
                                   const std::vector<Point2i>&, const Point2i&);
    Point2i bresenhamPointSerach(Point2i, Point2i,
                                    Mat&,
                                   const std::vector<Point2i>&, const Point2i&);
    Mat& getImage();
};
}

#endif