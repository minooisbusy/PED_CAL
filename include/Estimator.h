#ifndef ESTIMATOR_H
#define ESTIMATOR_H
#include<opencv2/opencv.hpp>
#include"Detector.h"

using namespace cv;
namespace PED_CAL
{
class Estimator
{
  size_t N;
  Mat P;
  Mat R;
  Mat t;
  Mat K;
  Mat H;
  std::vector<Posture> data;
public:
  Estimator(){};
  Estimator(const std::vector<Posture>& );
  Mat EstimateHomography();
  Mat EstimateFundamental();
  Mat EstimateProjectionMatrix();
  void grabPos(const std::vector<Posture>& posture);
  Mat Estimate();
  void EstTwoVP(const Posture& A, const Posture& B, Point3d& EastVP, Point3d& VertVP);
};
}
#endif