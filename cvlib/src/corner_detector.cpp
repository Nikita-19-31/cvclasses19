/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

bool corner_detector_fast::checkPixel(cv::Mat &image, int i, int j, int N, int t)
{
    std::vector<int> _successVerify = std::vector<int>(16, -255);
    int count_black = 0;
    int count_white = 0;

    for (auto firstIndex : _initialVerify)
    {

        if (image.at<uchar>(cv::Point(j, i) + _pixelsAround[firstIndex - 1]) > image.at<uchar>(i, j) + t)
        {
            count_white++;
            _successVerify[firstIndex - 1] = 255;
        }
        else if (image.at<uchar>(cv::Point(j, i) + _pixelsAround[firstIndex - 1]) < image.at<uchar>(i, j) - t)
        {
            count_black++;
            _successVerify[firstIndex - 1] = 0;
        }
    }

    if ((count_black >= radius) || (count_white >= radius))
    {
        for (auto secondIndex : _secondaryVerify)
        {
            if (image.at<uchar>(cv::Point(j, i) + _pixelsAround[secondIndex - 1]) > image.at<uchar>(i, j) + t)
            {
                _successVerify[secondIndex - 1] = 255;
            }
            else if (image.at<uchar>(cv::Point(j, i) + _pixelsAround[secondIndex - 1]) < image.at<uchar>(i, j) - t)
            {
                _successVerify[secondIndex - 1] = 0;
            }
        }

        if (checkMaxLenSeqPix(_successVerify, N))
            return true;
    }

        return false;
}

bool corner_detector_fast::checkMaxLenSeqPix(std::vector<int> seq, int N)
{
    int countMax = 1;
    int count = 1;
    size_t size = seq.size();

    for (size_t k = 1; k < 2*size - 2; k++)
    {
        if (seq[k%size] == seq[(k - 1)%size])
            count++;
        else
            count = 1;

        if (count > countMax)
            countMax = count;
    }

    if (countMax >= N)
        return true;
    else
        return false;
}

void corner_detector_fast::detect(cv::InputArray _image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    keypoints.clear();
    int t = 15;
    int N = 11;
    cv::Mat image = _image.getMat();
    _image.getMat().copyTo(image);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(image, image, cv::Size(5, 5), 0, 0);
    cv::copyMakeBorder(image, image, radius, radius, radius, radius, cv::BORDER_REPLICATE);

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (checkPixel(image, i, j, N, t))
                keypoints.emplace_back(cv::Point(j, i), radius + 3);
        }
    }
}

void corner_detector_fast::compute(cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    std::srand(unsigned(std::time(0))); // \todo remove me
    // \todo implement any binary descriptor
    const int desc_length = 2;
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    int* ptr = reinterpret_cast<int*>(desc_mat.ptr());
    for (const auto& pt : keypoints)
    {
        for (int i = 0; i < desc_length; ++i)
        {
            *ptr = std::rand();
            ++ptr;
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray, cv::InputArray, std::vector<cv::KeyPoint>&, cv::OutputArray descriptors, bool /*= false*/)
{
    // \todo implement me
}

} // namespace cvlib
