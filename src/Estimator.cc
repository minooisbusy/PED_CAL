#include"Estimator.h"

namespace PED_CAL
{
  Estimator::Estimator(const std::vector<Posture>& input)
  :data(input)
  {
    P = Mat::zeros(3,4,CV_32FC1);
    R = P(Range(0,3),Range(0,3));
    t = P.col(3);
  }
  void Estimator::grabPos(const std::vector<Posture>& posture)
  {
    data = posture;
    N = data.size();
  }
  Mat Estimator::EstimateHomography()
  {
    std::vector<Point2i> input(N,Point2i());
    std::vector<Point2i> output(N,Point2i());
    for(size_t i=0; i<N; i++)
    {
      output[i]=data[i].foot;
      input[i]=data[i].head;
    }
    return cv::findHomography(input,output);
  }
  Mat Estimator::EstimateFundamental()
  {
    std::vector<Point2i> input(N,Point2i());
    std::vector<Point2i> output(N,Point2i());
    for(size_t i=0; i<N; i++)
    {
      output[i]=data[i].foot;
      input[i]=data[i].head;
    }

    return cv::findFundamentalMat(input,output);
  }
  Mat Estimator::Estimate()
  {
    std::vector<Point2i> input(N,Point2i());
    std::vector<Point2i> output(N,Point2i());
    for(size_t i=0; i<N; i++)
    {
      output[i]=data[i].foot;
      input[i]=data[i].head;
    }
    //random sampling to data
    // modeling
    // reprojection error
    // optimize
    return Mat();
  }
  void Estimator::EstTwoVP(const Posture& A, const Posture& B, Point3d& EastVP, Point3d& VertVP)
{


	Vec3d footLine = CrossProduct(Vec3d(A.foot.x, A.foot.y, 1), Vec3d(B.foot.x, B.foot.y, 1));
	Vec3d headLine = CrossProduct(Vec3d(A.head.x, A.head.y, 1), Vec3d(B.head.x, B.head.y, 1));
	EastVP = CrossProduct(footLine, headLine);
	Vec3d ALine = CrossProduct(Vec3d(A.foot.x, A.foot.y, 1), Vec3d(A.head.x, A.head.y, 1));
	Vec3d BLine = CrossProduct(Vec3d(B.foot.x, B.foot.y, 1), Vec3d(B.head.x, B.head.y, 1));
	VertVP = CrossProduct(ALine, BLine);
}
Vec3d CrossProduct(Vec3d a, Vec3d b)
{
	Vec3d output;
	output[0] = a[1] * b[2] - a[2] * b[1];
	output[1] = a[2] * b[0] - a[0] * b[2];
	output[2] = a[0] * b[1] - a[1] * b[0];
	return output;
}
Point3d CrossProduct(Point3d a,Point3d b)
{
	Point3d output;
	output.x = a.y * b.z - a.z * b.y;
	output.y = a.z * b.x - a.x * b.z;
	output.z = a.x * b.y - a.y * b.x;
	return output;
}

double EstFocalFromTwoVP(Vec3d Fu, Vec3d Fv, Size A)
{
	Fu[0] = Fu[0] / Fu[2];
	Fu[1] = Fu[1] / Fu[2];
	Fu[2] = 1;
	Fv[0] = Fv[0] / Fv[2];
	Fv[1] = Fv[1] / Fv[2];
	Fv[2] = 1;
	Vec3d VL = CrossProduct(Fu, Fv);
	Vec3d pp = { A.width / 2.0,A.height / 2.0, 1 };
	Vec3d Puv;
	double denM = 1 / (VL[0] * VL[0] + VL[1] * VL[1]);
	double SQa = VL[0] * VL[0];
	double SQb = VL[1] * VL[1];
	double ab = VL[0] * VL[1];
	double ac = VL[0] * VL[2];
	double a = VL[0];
	double b = VL[1];
	Puv[0] = (1 - SQa * denM)*pp[0] - ab*denM*pp[1] - a*denM*VL[2];
	Puv[1] = (-ab*denM)*pp[0] + (1 - SQb*denM)*pp[1] - b*denM*VL[2];
	Puv[2] = 1;

	double Puv_O = sqrt(norm(Fu, Puv)*norm(Fv, Puv));
	double P_Puv = norm(Puv, pp);
	double focal = sqrt(pow(Puv_O, 2) - pow(P_Puv, 2));
	return focal;
}

Mat EstRotationFromTwoVP2(Vec3d Fu, Vec3d Fv, double Focal, Point pp)
{

	Fv[0] = Fv[0] / Fv[2];
	Fv[1] = Fv[1] / Fv[2];
	double roll = atan((double)(Fv[0] - pp.x) / (double)(Fv[1] - pp.y));
	roll = -180 - roll*(180 / CV_PI);

	// 3. Tilt angle
	double tilt = -atan2(sqrt(pow((double)(Fv[0] - pp.x), 2) + pow((double)(Fv[1] - pp.y), 2)), -Focal);
	tilt = tilt*(180 / CV_PI);
	printf("Roll=%lf\nTilt=%lf\n", roll, tilt);
	// extrinsic matrix
	double SR = sin(roll*(CV_PI / 180));
	double CR = cos(roll*(CV_PI / 180));
	double ST = sin(tilt*(CV_PI / 180));
	double CT = cos(tilt*(CV_PI / 180));
	double arr_roll[3][3] = { { CR, -SR, 0 },{ SR, CR, 0 },{ 0, 0, 1 } };
	double arr_tilt[3][3] = { { 1, 0, 0 },{ 0, CT, -ST },{ 0, ST, CT } };
	Mat mRoll = Mat(3, 3, CV_64FC1, arr_roll);
	Mat mTilt = Mat(3, 3, CV_64FC1, arr_tilt);
	Mat rotation = mRoll*mTilt;

	return rotation;
}

Mat EstRotationFromTwoVP(Vec3d Fu, Vec3d Fv, double Focal, Point pp)
{
	Vec3d mod_Fu, mod_Fv;

	Fu[0] = Fu[0] / Fu[2];
	Fu[1] = Fu[1] / Fu[2];

	Fv[0] = Fv[0] / Fv[2];
	Fv[1] = Fv[1] / Fv[2];

	mod_Fu[0] = Fu[0] - pp.x;
	mod_Fu[1] = Fu[1] - pp.y;
	mod_Fu[2] = 1;

	mod_Fv[0] = Fv[0] - pp.x;
	mod_Fv[1] = Fv[1] - pp.y;
	mod_Fv[2] = 1;

	//l_2 norm
	double s1 = sqrt(mod_Fu[0] * mod_Fu[0] + mod_Fu[1] * mod_Fu[1] + Focal*Focal);
	double s2 = sqrt(mod_Fv[0] * mod_Fv[0] + mod_Fv[1] * mod_Fv[1] + Focal*Focal);

	Vec3d u_PrimeSlashRc = { mod_Fu[0] / s1,mod_Fu[1] / s1,Focal / s1 };
	Vec3d v_PrimeSlashRc = { mod_Fv[0] / s2,mod_Fv[1] / s2,Focal / s2 };

	double inner = DotProduct(u_PrimeSlashRc, v_PrimeSlashRc);

	Vec3d w_PrimeSlashRc = CrossProduct(u_PrimeSlashRc, v_PrimeSlashRc);
	//w_PrimeSlashRc = w_PrimeSlashRc / norm(w_PrimeSlashRc);
	Mat rotation = Mat::zeros(3, 3, CV_64FC1);
	rotation.at<double>(0, 0) = u_PrimeSlashRc[0];
	rotation.at<double>(0, 1) = v_PrimeSlashRc[0];
	rotation.at<double>(0, 2) = w_PrimeSlashRc[0];

	rotation.at<double>(1, 0) = u_PrimeSlashRc[1];
	rotation.at<double>(1, 1) = v_PrimeSlashRc[1];
	rotation.at<double>(1, 2) = w_PrimeSlashRc[1];

	rotation.at<double>(2, 0) = u_PrimeSlashRc[2];
	rotation.at<double>(2, 1) = v_PrimeSlashRc[2];
	rotation.at<double>(2, 2) = w_PrimeSlashRc[2];
#define _ESTROTATION_
#ifdef _ESTROTATION_
	cout << "norm u_PRimSlashRc = " << norm(u_PrimeSlashRc) << endl;
	cout << "norm v_PRimSlashRc = " << norm(v_PrimeSlashRc) << endl;
	cout << "inner u.v = " << inner << endl;
	cout << "norm w_PRimSlashRc = " << norm(w_PrimeSlashRc) << endl;
	cout << "norm Rotation  = " << determinant(rotation) << endl;
#endif
	return rotation;
}
double DotProduct(Vec3d a, Vec3d b)
{
	return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}


double computeError(const Mat& H, const std::vector<Posture>& pos)
{
  double res = 0;
  int inliers =0;
  for(int i=0; i<pos.size();i++)
  {
			cv::Point2d p;
			const int x = pos[i].head.x; const int y = pos[i].head.y;
			const double den = H.at<double>(2,0)+ H.at<double>(2,1) + H.at<double>(2,2);
			p.x = (H.at<double>(0,0)*x + H.at<double>(0,1)*y + H.at<double>(0,2))/den;
			p.y = (H.at<double>(1,0)*x + H.at<double>(1,1)*y + H.at<double>(1,2))/den;
      const Point2d q = pos[i].foot;
      double value =  cv::norm(q-p);
      if(value<50)
      {
        res += value;
        inliers++;
      }

  }
  if(inliers==0)inliers=1;
  return res/(double)inliers;
}

Mat EstProj(const Mat &H, int width, int height)
{
  Point3d fleft(width/3, height/3,1);
  Point3d fright(width*2/3, height/3,1);
  Point3d fbottom(width/2, height*2/3, 1);

  Point3d hleft = operation(H,fleft);
  Point3d hright = operation(H,fright);
  Point3d hbottom = operation(H,fbottom);

  Point3d line1 = CrossProduct(hleft, hright);
  Point3d line2 = CrossProduct(hleft, hbottom);
  Point3d line3 = CrossProduct(hright, hbottom);

  Point3d vpleft = CrossProduct(line1,line2);
  Point3d vpright = CrossProduct(line2,line3);
  Point3d vpbottom = CrossProduct(line1,line3);

  Mat P=Mat::zeros(3,4,CV_64FC1);
  P.at<double>(0,0) = vpleft.x;
  P.at<double>(1,0) = vpleft.x;
  P.at<double>(2,0) = vpleft.x;

  P.at<double>(0,1) = vpright.x;
  P.at<double>(1,1) = vpright.x;
  P.at<double>(2,1) = vpright.x;

  P.at<double>(0,2) = vpbottom.x;
  P.at<double>(1,2) = vpbottom.x;
  P.at<double>(2,2) = vpbottom.x;
  Mat KR = P(Range(0,3),Range(0,3));

  Mat K;
  Mat KKT = KR*KR.t();
  //cv::Cholesky(KKT.ptr(),KKT.step1(),3,NULL)


 // Point3d hleft = H*fleft;
  return Mat();


}

Point3d operation(const Mat& H, Point3d v)
{
			Point3d p;
			const int x = v.x; const int y = v.y;
			const double den = H.at<double>(2,0)*x+ H.at<double>(2,1)*y + H.at<double>(2,2);
			p.x = (H.at<double>(0,0)*x + H.at<double>(0,1)*y + H.at<double>(0,2))/den;
			p.y = (H.at<double>(1,0)*x + H.at<double>(1,1)*y + H.at<double>(1,2))/den;
      p.z = 1;
    return p;
}


}