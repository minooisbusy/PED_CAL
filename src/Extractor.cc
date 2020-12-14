#include "Extractor.h"

namespace PED_CAL
{
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
    morphologyEx(foreground, foreground, MORPH_CLOSE, element); // Remove Noise
    morphologyEx(foreground, foreground, MORPH_OPEN, element);  // Connect Inliers
  }

  void PostureExtractor::LabelPed()
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
      return;
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
    std::cout << "pca ON # of data = " << vBlob.rows << std::endl;
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
    std::cout << "center = " << cntr << std::endl;
    blob.clear();
    circle(color, cntr, 3, Scalar(255, 0, 255), 2);
    Point p1 = cntr + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
    Point p2 = cntr - 0.02 * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
    drawAxis(color, cntr, p1, Scalar(0, 255, 0), 1);
    drawAxis(color, cntr, p2, Scalar(255, 255, 0), 5);

    //Find contour
    std::vector<std::vector<Point>> contours;
    Mat matContour=Mat::zeros(width,height,CV_8UC3);
    findContours(foreground, contours, RETR_LIST, CHAIN_APPROX_NONE);
    for(size_t i=0; i<contours.size(); i++)
    {
      double area = contourArea(contours[i]);
      if(area < 1e2 || 1e5 < area) continue;
      drawContours(matContour, contours, static_cast<int>(i), Scalar(0, 0, 255), 2);
    }
    imshow("contour", matContour);

    //Morphological processing
    // Do bresenham

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
    waitKey(0);
  }

  void PostureExtractor::drawAxis(Mat &img, Point p, Point q, Scalar colour, const float scale = 0.2)
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

  Point2i PostureExtractor::bresenhamPointSerach(int x1, int y1, int x2, int y2,
                                                 const Mat &image,
                                                 const std::vector<Point2i> &points)
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
    Point2i result;

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

      //waitKey(0);
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
        //image.at<uchar>(Point(ndx, y1)) = 100;
        for (unsigned int i = 0; i < points.size(); i++)
        {
          if (ndx == points[i].x &&
              y1 == points[i].y &&
              points[i].y > y1)
          {
            result = Point(ndx, y1);
            flag = true;
            break;
          }
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
        //image.at<uchar>(Point(x1, ndx)) = 100;
        //vector<Point2i>::iterator iter1 = points.begin();
        //printf("points size: %d\n", points.size());
        //for (iter1 = points.begin(); iter1 != points.end(); ++iter1)
        for (unsigned int i = 0; i < points.size(); i++)
        {
          if (x1 == points[i].x &&
              ndx == points[i].y &&
              points[i].y > y1)
          {

            result = Point(x1, ndx);
            flag = true;
            break;
          }
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

} // namespace PED_CAL