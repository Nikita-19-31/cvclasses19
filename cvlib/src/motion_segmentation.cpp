/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <iostream>

namespace cvlib
{
motion_segmentation::motion_segmentation()
{
    sizeKernel = cv::Size(3,3);
    N = 0;
}

void motion_segmentation::apply(cv::InputArray _image, cv::OutputArray _fgmask, double)
{
    if (_image.empty())
        return;
   cv::Mat image = _image.getMat();
   cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
   _fgmask.assign(image);
   updateModel(60);

   if (N == 0)
   {
       diff_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
       prev_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
       min_image = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(255));
       max_image = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
   }

   cv::GaussianBlur(image, image, sizeKernel, 0, 0);

   for (int i = 0; i < image.rows; i++)
   {
       for (int j = 0; j < image.cols; j++)
       {
            max_image.at<uchar>(i, j)  = std::max(image.at<uchar>(i, j), max_image.at<uchar>(i, j));
            min_image.at<uchar>(i, j)  = std::min(image.at<uchar>(i, j), min_image.at<uchar>(i, j));

            uchar diff_pixel = std::abs(image.at<uchar>(i, j) - prev_image.at<uchar>(i, j));
            diff_image.at<uchar>(i, j) = std::max(diff_image.at<uchar>(i, j), diff_pixel);

       }
   }
   uchar* image_line = diff_image.data;
   std::sort(image_line, image_line + image.cols * image.rows);
   double d_nu = image_line[image.cols * image.rows / 2] / 5;

   for (int i = 0; i < image.rows; i++)
   {
       for (int j = 0; j < image.cols; j++)
       {
            if (std::abs(max_image.at<uchar>(i, j) - image.at<uchar>(i, j)) > _threshold* d_nu)
                _fgmask.getMat().at<uchar>(i, j) = 255;
            else if (std::abs(min_image.at<uchar>(i, j) - image.at<uchar>(i, j)) > _threshold* d_nu)
                _fgmask.getMat().at<uchar>(i, j) = 255;
            else
                _fgmask.getMat().at<uchar>(i, j) = 0;
       }
   }

   N++;
}

void motion_segmentation::setVarThreshold(int threshold)
{
    _threshold = threshold;
}

void motion_segmentation::updateModel(int n)
{
    if (N > n)
        N = 0;

}
} // namespace cvlib
