#include "Detector.h"


namespace PED_CAL
{
  PeriodicDetector::PeriodicDetector(){}
  void PeriodicDetector::push_back(Posture posture)
  {
    if(posture.ratio != -1.0)
    data.push_back(posture);
  }
  void PeriodicDetector::clear()
  {
    data.clear();
  }
  const size_t PeriodicDetector::getSize()
  {
    return data.size();
  }
  std::vector<Posture> PeriodicDetector::ampd()
  {
    vector<Posture> output;
    output.reserve(500);
    const uint N = data.size();
    const uint L = ceil(N/2.0) - 1;
    vector<Point2d> detrend(N, Point2d(0, 0));
    Mat A = Mat::zeros(Size(2, N), CV_64FC1);
    Mat y = Mat::zeros(Size(1, N), CV_64FC1);
    Mat x_Est = Mat::zeros(Size(2,1), CV_64FC1);

    for(size_t i = 0; i<N; i++)
    {
        A.at<double>(Point(0, i)) = static_cast<double>(i); // index of data
        A.at<double>(Point(1, i)) = 1.0;
        Point2d magVec = (data[i].head-data[i].foot);

        y.at<double>(Point(0, i)) = cv::norm(magVec);//static_cast<double>(data[i].ratio);
    }

    //cv::solve(A, y, x_Est,DECOMP_SVD);
    Mat pinvA = A.inv(DECOMP_SVD);
    x_Est = pinvA*y;
    double a  = x_Est.at<double>(0,0);
    double b  = x_Est.at<double>(1,0);
    //for(int i = 0; i < N; i++)
    //{
    //  detrend[i] = Point2d(0, a*i + b);
    //}
    //Mat data_dtrd(Size(2,N), CV_64FC1);// = Mat(data) - Mat(detrend);
    std::vector<Point2d> data_dtrd(N, Point2d(0,0));

    for(int i=0; i<N; i++)
    {
      data_dtrd[i].x = static_cast<double>(i);
      data_dtrd[i].y = y.at<double>(Point(0,i)) - a*i-b;//detrend[i].y;
    }


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    double alpha = 1.0;
    Mat lsm = Mat::zeros(L,N, CV_64FC1); // rows, columns
    // LSM = 1+ rand(RNG,L,N);
   // for(int k=0; k<L; k++)
   // {
   //   for(int i=0; i<N; i++)
   //   {
   //     lsm.at<double>(k,i) = dis(gen) + alpha; // row, column
   //   }
   // }
    for(int k=0; k<L+1; k++)
    {
      for(int i= k+1; i<N-k-1;i++)
      {
        if(data_dtrd[i].y <= data_dtrd[i-k-1].y||
           data_dtrd[i].y <= data_dtrd[i+k+1].y)
            lsm.at<double>(k,i) = 1;//dis(gen) + alpha; // row, column
      }
      /*
      for(int i=k; i<N-k; i++)
      {
        if(data_dtrd[i-1].y>data_dtrd[i-k-1].y)
          lsm.at<double>(k-1, i) = 0;

      }
      for(int i=k; i<N; i++)
        {
          if(data_dtrd[i].y>data_dtrd[i-k].y)
          lsm.at<double>(k-1, i) = 0;
        }
        */
    }


    std::vector<double> G(L, 0);
    for(int k = 0; k < L; k++)
    {
      for( int i = 0; i < N; i++)
      {
        G[k] +=lsm.at<double>(k, i);
      }
      /*
      for( int i = ceil(N); i > 0; i--)
      {
        G[k] *=(double)i;
      }
      */
    }

    // \lambda = argmin_{\gamma}{\gamma_k}
    double lambda = std::distance(G.begin(), std::min_element(G.begin(), G.end()));

    //for(int i=0; i<lambda;i++)
    //{
    //  Mat cols = lsm.rowRange(0,lambda).colRange(0,N);
    //  double minVal=0;
    //  cv::min(cols,minVal);
    //  std::cout<<cols<<std::endl;
    //  std::cout<<minVal<<std::endl;
    //  waitKey(0);
    //}
    //G.begin()+static_cast<int>(lambda);

    lsm = lsm(Range(0,static_cast<int>(lambda)),Range(0,N));
    std::vector<Point2d> sigma(N, Point2d(0,0));
    double tmp = 0;
    double mean = 0;
    double sigma_v = 0;
    for( int i=0; i < N; i++)
    {
      for(int k=0; k< static_cast<int>(lambda); k++)
      {
        mean += lsm.at<double>(k, i);
      }
      mean /= lambda;
      for( int k = 0; k < static_cast<int>(lambda); k++ )
      {
        tmp = lsm.at<double>(k, i) - mean;
        tmp *= tmp;
        sigma_v += tmp; // std
      }
      sigma_v = sqrt(sigma_v)/(lambda-1);

      sigma[i].x=i;
      sigma[i].y= sigma_v;

      mean = 0;
      tmp = 0;
      sigma_v = 0;
    }
    for(int i=0; i<sigma.size(); i++)
    {
      if(sigma[i].y == 0)
      {
        output.push_back(data[sigma[i].x]);//
      }
    }
    sigma.clear();
    return output;
  }
  std::vector<Posture>& PeriodicDetector::getData()
  {
    return data;
  }
}