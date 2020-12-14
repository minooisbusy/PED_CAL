#ifndef FRAME_H
#define FRAME_H
#include<opencv2/opencv.hpp>
#include<vector>
using namespace cv;
struct Pedestrian
{
  Point2f head;
  Point2f foot;
  Point2f center;
  float theta;
  float magnitude;
};

class Frame
{
  private:
    size_t mNumPed;
    std::vector<Pedestrian> mvPed;
    std::vector<bool> mbInlier;
    Mat foreground;
  public:
    Frame(const Mat& img);



};

class Data
{
  std::vector<Pedestrian> mvLambda;
  public:
};
#endif