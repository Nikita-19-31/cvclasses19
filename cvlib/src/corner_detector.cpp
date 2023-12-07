/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <random>
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

void corner_detector_fast::generateNormPoints(int s, int len_desc)
{
    int range_mask = s / 2;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, range_mask);
    int x1, y1, x2, y2;


    for (int i = 0; i < len_desc; i++)
    {
        x1 = (int)distribution(generator) % (range_mask + 1);
        y1 = (int)distribution(generator) % (range_mask + 1);
        x2 = (int)distribution(generator) % (range_mask + 1);
        y2 = (int)distribution(generator) % (range_mask + 1);

        _pairPixels.push_back(cv::Point(x1,y1));
        _pairPixels.push_back(cv::Point(x2, y2));
    }

}

void corner_detector_fast::detect(cv::InputArray _image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    keypoints.clear();
    int t = 15;
    int N = 11;
    cv::Mat image;
    _image.getMat().copyTo(image);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(image, image, cv::Size(5, 5), 0, 0);
    cv::copyMakeBorder(image, image, radius, radius, radius, radius, cv::BORDER_REPLICATE);

    for (int i = radius; i < image.rows - radius; i++)
    {
        for (int j = radius; j < image.cols - radius; j++)
        {
            if (checkPixel(image, i, j, N, t))
                keypoints.emplace_back(cv::Point(j - 3, i - 3), radius + 3);
        }
    }
}

void corner_detector_fast::compute(cv::InputArray _image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    cv::Mat image;
    _image.getMat().copyTo(image);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(image, image, cv::Size(5, 5), 0, 0);
    //Бинарный дескриптор BRIEF
    const int s = 25; //Размер окрестности особой точки SxS
    const int desc_length = 16;

    if (_pairPixels.empty())
    {
        generateNormPoints(s, desc_length * 16); //Генерация пар пикселей
    }

    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_16U);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);
    int half_s = s / 2 + 1;
    cv::copyMakeBorder(image, image, half_s, half_s, half_s, half_s, cv::BORDER_REPLICATE);
    uint16_t* ptr = reinterpret_cast<uint16_t*>(desc_mat.ptr());

    for (auto featPoint : keypoints)
    {
        featPoint.pt.x += half_s;
        featPoint.pt.y += half_s;

        int indx = 0;
        for (int i = 0; i < desc_length; i++)
        {
            uint16_t descrpt = 0;
            for (int j = 0; j < 2*8; j++)
            {
                uint8_t pix1 = image.at<uint8_t>(featPoint.pt + _pairPixels[indx]);
                uint8_t pix2 = image.at<uint8_t>(featPoint.pt + _pairPixels[indx+1]);
                int bit = (pix1 < pix2);
                descrpt |= bit << (15-j);
                indx += 2;
            }
            *ptr = descrpt;
            ++ptr;
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray image, cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors, bool /*= false*/)
{
    detect(image, keypoints);
    compute(image, keypoints, descriptors);
}

} // namespace cvlib
