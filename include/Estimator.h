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
Vec3d CrossProduct(Vec3d,Vec3d);
Point3d CrossProduct(Point3d,Point3d);
double EstFocalFromTwoVP(Vec3d Fu, Vec3d Fv, Size A);
Mat EstRotationFromTwoVP2(Vec3d Fu, Vec3d Fv, double Focal, Point pp);
Mat EstRotationFromTwoVP(Vec3d Fu, Vec3d Fv, double Focal, Point pp);
double DotProduct(Vec3d a, Vec3d b);
double computeError(const Mat& H, const std::vector<Posture>& pos);
Mat EstProj(const Mat &H);
Point3d operation(const Mat& H, Point3d v);
}
#endif