#include "Extractor.h"

namespace PED_CAL
{
  Posture::Posture():
  head(Point2i(-1,-1)), foot(Point2i(-1,-1)), theta(-4.0), ratio(-1.0){}
  Posture::Posture(int hx, int hy, int fx, int fy, double _theta, double _ratio):
  head(Point2i(hx,hy)), foot(Point2i(fx,fy)), theta(_theta), ratio(_ratio){}
  Posture::Posture(Point2i _head, Point2i _foot, double _theta, double _ratio):
  head(_head), foot(_foot), theta(_theta), ratio(_ratio){}
  Posture::Posture(const Posture& posture):
  head(posture.head), foot(posture.foot), theta(posture.theta), ratio(posture.ratio){};

  PostureExtractor::PostureExtractor(VideoCapture *capture)
      : mMaxFrame(capture->get(CAP_PROP_FRAME_COUNT)),
        mCurrFrame(-1), width(capture->get(CAP_PROP_FRAME_WIDTH)),
        height(capture->get(CAP_PROP_FRAME_HEIGHT))
  {
    mCap = capture;
    mBg_model = createBackgroundSubtractorMOG2(500, 16.0, true);
    element = getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE,
                                    Size(5, 5), Point(-1, -1));

    r = Rect(160, 0, width - 320, height);
    width = r.width;

    std::cout << "# of Frame   : " << mMaxFrame << std::endl
              << "Image Width  : " << width << std::endl
              << "Image Height : " << height << std::endl;
  }

  void PostureExtractor::ExtractBG()
  {
    mCap->operator>>(img);
    Rect bound(0, 0, img.cols, img.rows);
    img = img(r);
    //cv::rectangle(img,r, Scalar(0,0,255),1);
    ++mCurrFrame;

    mBg_model->apply(img, foreground);
    cv::threshold(foreground, foreground, 200, 255, cv::THRESH_BINARY);
    //adaptiveThreshold(foreground,foreground,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY_INV,21, 20);
    morphologyEx(foreground, foreground, MORPH_OPEN, element);  // Remove Noise
    morphologyEx(foreground, foreground, MORPH_CLOSE, element); // Connect Inliers
    morphologyEx(foreground, foreground, MORPH_DILATE, element); // Connect Inliers
    morphologyEx(foreground, foreground, MORPH_DILATE, element); // Connect Inliers
  }

  Posture PostureExtractor::ExtractPed()
  {
    // remove background
    //for(int i=0; i< width; i++)
    //  for(int j=0; j< height; j++)
    Mat color;
    cvtColor(foreground, color, COLOR_GRAY2BGR);
    int numOfLabels =
        connectedComponentsWithStats(foreground, labels, stats, centroids, 8, CV_32S);
    const int j = 1;
    int left = stats.at<int>(j, CC_STAT_LEFT);
    int top = stats.at<int>(j, CC_STAT_TOP);
    int width = stats.at<int>(j, CC_STAT_WIDTH);
    int height = stats.at<int>(j, CC_STAT_HEIGHT);
    if (left <= 0 || (left + width) >= this->width // left and right
        || top <= 0 || (top + height) >= this->height || numOfLabels >= 3)
    {
      return Posture();//Posture(Point2i(0,0),Point2i(0,0),0,0); // Default constructor with degenerate case
    }
    for (int i = left; i < left + width; i++)
      for (int j = top; j < top + height; j++)
        if (foreground.at<uchar>(j, i) == 255)
        {
          blob.emplace_back(i, j);
        }

    Mat vBlob = Mat::zeros(blob.size(), 2, CV_64FC1);
    for (int i = 0; i < blob.size(); i++)
    {
      vBlob.at<double>(i, 0) = static_cast<double>(blob[i].x);
      vBlob.at<double>(i, 1) = static_cast<double>(blob[i].y);
    }
    //std::cout << "pca ON # of data = " << vBlob.rows << std::endl;
    PCA pca(vBlob, Mat(), cv::PCA::DATA_AS_ROW);
    Point cntr = Point(static_cast<int>(pca.mean.at<double>(0, 0)),
                       static_cast<int>(pca.mean.at<double>(0, 1)));
    std::vector<Point2d> eigen_vecs(2);
    std::vector<double> eigen_val(2);
    for (int i = 0; i < 2; i++)
    {
      eigen_vecs[i] = Point2d(pca.eigenvectors.at<double>(i, 0),
                              pca.eigenvectors.at<double>(i, 1));
      eigen_val[i] = pca.eigenvalues.at<double>(i);
    } // Draw the principal components
    //std::cout << "center = " << cntr << std::endl;
    blob.clear();
    circle(color, cntr, 3, Scalar(255, 0, 255), 1);
    Point p1 = cntr + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
    Point p2 = cntr - 0.02 * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
    drawAxis(color, cntr, p1, Scalar(0, 255, 0), 1);
    drawAxis(color, cntr, p2, Scalar(255, 255, 0), 5);


    //Find contour
    std::vector<std::vector<Point>> contours;
    Mat matContour=Mat::zeros(foreground.rows, foreground.cols,foreground.type());
    findContours(foreground, contours, RETR_LIST, CHAIN_APPROX_NONE);
    //std::cout<<"# of contours = " << contours.size() << std::endl;
    std::vector<Point2i> *pvPts= nullptr;
    double max = -1;
    for(size_t i=0; i<contours.size(); i++)
    {
      //std::cout <<i<< "-th contour length = " << contours[i].size()<<std::endl;
      double area = contourArea(contours[i]);
      if(area < 1e2 || 1e5 < area) continue;
      if(area>max)
      {
        max = area;
        pvPts = &contours[i];
      }
      drawContours(matContour, contours, static_cast<int>(i), Scalar(255, 255, 255), 2);
    }

    // p1 is major axis...
    p1 = p1 - cntr; // Make axis
    const int dx = p1.x;
    const int dy = p1.y;
    float tilt = dy/(dx + 1e-10);
    //Point2i head = bresenhamPointSerach(cntr,p1,*pvPts);

    /// Find Rectagle boundary
    // Legacy code do this in the box.
    // cntr point have to translated box coordinates
    Point2i center = cntr - Point2i(left, top);
    int y2 = height;
    int y1 = 0;
    int x1 = (int)((1 / tilt)*(double)(y1 - center.y) + center.x);
    int x2 = (int)((1 / tilt)*(double)(y2 - center.y) + center.x);
    if (!(x1 > 0))
    {
      x1 = 0;
      y1 = (int)((tilt)*(double)(x1 - center.x) + center.y);
    }
    else if (x1 >= width)
    {
      x1 = width - 1;
      y1 = (int)((tilt)*(double)(x1 - center.x) + center.y);
    }
    if (!(x2 > 0))
    {
      x2 = 0;
      y2 = (int)((tilt)*(double)(x2 - center.x) + center.y);
    }
    if (x2 >= width)
    {
      x2 = width - 1;
      y2 = (int)((tilt)*(double)(x2 - center.x) + center.y);
    }
    matContour = matContour.rowRange(top,top+height).colRange(left, left+width);
    std::vector<Point2i> conts;
    conts.reserve(height*width/10);
    for(int i=0; i<width; i++)
      for(int j=0; j<height; j++)
      {
        if(matContour.at<uchar>(Point2i(i,j)) == 255)
            conts.emplace_back(j, i);
      }
      //test.at<uchar>(conts[i].x, conts[i].y)=255;
    cvtColor(matContour,matContour,COLOR_GRAY2BGR);
    imshow("contour", matContour);
    Point2i offset(0,0);
    circle(matContour,center,2,Scalar(255,255,255),1);

    //HEAD Position
    //Point2i head = bresenhamPointSerach(center.y, center.x, y1, x1, matContour, *pvPts, offset);
    Point2i head = bresenhamPointSerach(center.y, center.x, y1, x1, matContour, conts, offset);
    head.x +=left;
    head.y +=top;
    //circle(matContour,head,3,Scalar(255,255,0),1);
    circle(color,head,3,Scalar(255,255,0),1);
    //FOOT Position
    Point2i foot = bresenhamPointSerach(center.y, center.x, y2, x2, matContour, conts, offset);
    foot.x +=left;
    foot.y +=top;
    //circle(matContour,foot,2,Scalar(0,255,0),1);
    circle(color,foot,3,Scalar(255,255,0),1);

    // Extra
    int cnt[1000] = {
        0,
    };
    int sum[1000] = {
        0,
    };
    int mean[1000] = {
        0,
    };
    for (int y = 0; y < labels.rows; ++y)
    {
      int *label = labels.ptr<int>(y);
      uchar *pixel = img.ptr<uchar>(y);
      for (int x = 0; x < labels.cols; ++x)
      {
        int intensity = pixel[x]; // label[x] == 0 은 배경영역에 대한 Blob 계산
        if (label[x] == 0)
        {
          cnt[label[x]] = 0;
          sum[label[x]] = 0;
        }
        else
        {
          cnt[label[x]] = cnt[label[x]] + 1;
          sum[label[x]] = sum[label[x]] + intensity;
        }
      }
    }
    for (int i = 0; i < numOfLabels; i++)
    {
      if (i == 0)
      {
        mean[i] = 0;
      }
      else
      {
        mean[i] = sum[i] / cnt[i];
      }
    }
    //cvtColor(image, image, CV_GRAY2BGR);
    for (int j = 1; j < numOfLabels; j++) // Start index with 1 means NOT include Background
    {
      int area = stats.at<int>(j, CC_STAT_AREA);
      int left = stats.at<int>(j, CC_STAT_LEFT);
      int top = stats.at<int>(j, CC_STAT_TOP);
      int width = stats.at<int>(j, CC_STAT_WIDTH);
      int height = stats.at<int>(j, CC_STAT_HEIGHT);
      int x = centroids.at<double>(j, 0);
      //중심좌표
      int y = centroids.at<double>(j, 1);
      /*
      std::cout << "Centroid1 = ("<<x <<", "<<y<<")"<<std::endl;
      std::cout << "Centroid2 = ("<<left + width/2 <<", "<<top + height/2<<")"<<std::endl;
      */
      //rectangle(color, Point(left, top), Point(left + width, top + height),
      // Scalar(0, 0, 255), 1);
      putText(color, std::to_string(j), Point(left + 20, top + 20), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 0, 255), 1);
      putText(color, std::to_string(mean[j]), Point(left, top + 20), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);
    }
    cv::imshow("result", color);
    waitKey(1);
    return Posture(head,foot,0.0, tilt);
  }

  void PostureExtractor::drawAxis(Mat &img, Point p, Point& q, Scalar colour, const float scale = 0.2)
  {
    //! [visualization1]
    double angle = atan2((double)p.y - q.y, (double)p.x - q.x); // angle in radians
    double hypotenuse = sqrt((double)(p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));

    // Here we lengthen the arrow by a factor of scale
    q.x = (int)(p.x - scale * hypotenuse * cos(angle));
    q.y = (int)(p.y - scale * hypotenuse * sin(angle));
    line(img, p, q, colour, 1, LINE_AA);

    // create the arrow hooks
    p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
    p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
    line(img, p, q, colour, 1, LINE_AA);

    p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
    p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
    line(img, p, q, colour, 1, LINE_AA);
    //! [visualization1]
  }

  //@x1, y1 origin
  //@x2, y2 end point
  //@points compare
  Point2i PostureExtractor::bresenhamPointSerach(int x1, int y1, int x2, int y2,
                                                  Mat& img,
                                                 const std::vector<Point2i> &points,
                                                 const Point2i& offset)
  {
    int dx, dy;
    int p_value;
    int inc_2dy;
    int inc_2dydx;
    int inc_value;
    int ndx;
    bool flag = false;
    dx = abs(x2 - x1);
    dy = abs(y2 - y1);
    Point2i result(-1, -1);

    if (dy <= dx)
    {
      inc_2dy = 2 * dy;
      inc_2dydx = 2 * (dy - dx);

      if (x2 < x1)
      {
        ndx = x1;
        x1 = x2;
        x2 = ndx; //swap?

        ndx = y1;
        y1 = y2;
        y2 = ndx; //swap?
      }
      if (y1 < y2)
        inc_value = 1;
      else
        inc_value = -1;

      p_value = 2 * dy - dx;
      for (ndx = x1; ndx < x2; ndx++)
      {
        if (0 > p_value)
          p_value += inc_2dy;
        else
        {
          p_value += inc_2dydx;
          y1 += inc_value;
        }
        //img.at<Vec3b>(Point(ndx, y1))[1] = 255;
        img.at<Vec3b>(Point(y1, ndx))[1] = 255;
        for (unsigned int i = 0; i < points.size(); i++)
        {
          if (ndx == points[i].x - offset.y &&
              y1 == points[i].y - offset.x
              //&&points[i].y > y1
              )
          {
            //std::cout<<"Hit!!"<<std::endl;
            result = Point(y1, ndx);
            //result = Point(ndx, y1);
            flag = true;
            return result;
            break;
          }
          #ifdef DRAW
          else
          {
            img.at<Vec3b>(ndx, y1)[0] =0;
            img.at<Vec3b>(ndx, y1)[1] =255;
            img.at<Vec3b>(ndx, y1)[2] =0;
          }
          #endif

        }
        if (flag == true)
        {
          flag = false;
          break;
        }
      }
    }
    else
    {
      inc_2dy = 2 * dx;
      inc_2dydx = 2 * (dx - dy);

      if (y2 < y1)
      {
        ndx = y1;
        y1 = y2;
        y2 = ndx;

        ndx = x1;
        x1 = x2;
        x2 = ndx;
      }
      if (x1 < x2)
        inc_value = 1;
      else
        inc_value = -1;

      p_value = 2 * dx - dy;
      for (ndx = y1; ndx < y2; ndx++)
      {
        if (0 > p_value)
          p_value += inc_2dy;
        else
        {
          p_value += inc_2dydx;
          x1 += inc_value;
        }
        //img.at<Vec3b>(Point(x1, ndx))[1] = 255;
        img.at<Vec3b>(Point(ndx, x1))[1] = 255;
        //vector<Point2i>::iterator iter1 = points.begin();
        //printf("points size: %d\n", points.size());
        //for (iter1 = points.begin(); iter1 != points.end(); ++iter1)
        for (unsigned int i = 0; i < points.size(); i++)
        {
          if (x1 == points[i].x - offset.y &&
              ndx == points[i].y - offset.x
          )
              //&&points[i].y > y1)
          {
            //std::cout<<"Hit!!"<<std::endl;
            //result = Point(x1, ndx);
            result = Point(ndx, x1);
            flag = true;
            return result;
            break;
          }
          #ifdef DRAW
          else
          {
            img.at<Vec3b>(x1, ndx)[0] =0;
            img.at<Vec3b>(x1, ndx)[1] =255;
            img.at<Vec3b>(x1, ndx)[2] =0;
          }
          #endif
        }

        //for (auto const & iter1 : points)
        //{
        //	printf("%d,", iter1.x);
        //	printf("%d,", iter1.y);

        //}

        if (flag == true)
        {
          flag = false;
          break;
        }
      }
    }
    return result;
  }

//wrapper
Point2i PostureExtractor::bresenhamPointSerach(Point2i p, Point2i q,
                                    Mat& img,
                                   const std::vector<Point2i>& pts, const Point2i& offset)
{
  return bresenhamPointSerach(p.x, p.y, q.x, q.y, img, pts, offset);
}
Mat& PostureExtractor::getImage()
{
  return img;
}
} // namespace PED_CAL