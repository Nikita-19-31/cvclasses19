#include "cvlib.hpp"
#include <cmath>

namespace cvlib
{
Stitcher::Stitcher()
{
}

void Stitcher::init(cv::Mat init_image, int mather_ratio)
{
    init_image.copyTo(_stitcher_image);
    _corn_detector = cv::AKAZE::create();
    _decr_matcher = cvlib::descriptor_matcher(mather_ratio);
}

void Stitcher::stitch(cv::Mat input_image)
{
    std::vector<cv::KeyPoint> input_corners;
    cv::Mat input_decriptors;
    _corn_detector->detectAndCompute(input_image, cv::noArray(), input_corners, input_decriptors);
    _corn_detector->detectAndCompute(_stitcher_image, cv::noArray(), _stitcher_corners, _stitcher_descriptors);

    std::vector<std::vector<cv::DMatch>> matches;
    _decr_matcher.radiusMatch(input_decriptors, _stitcher_descriptors, matches, 100.0f);

        std::vector<cv::Point2f> srcPoints, dstPoints;
        for (size_t i = 0; i < matches.size(); ++i)
        {
            if (!matches[i].empty())
            {
                srcPoints.push_back(input_corners[matches[i][0].queryIdx].pt);
                dstPoints.push_back(_stitcher_corners[matches[i][0].trainIdx].pt);
            }
        }

        if ((srcPoints.size() < 4) || (dstPoints.size() < 4))
            return;

        cv::Mat H = findHomography(srcPoints, dstPoints, cv::RANSAC);

        //Создание панорамы
        cv::Size stitcher_size = cv::Size(input_image.cols + _stitcher_image.cols, _stitcher_image.rows);
        cv::Mat result;
        cv::warpPerspective(input_image, result, H, stitcher_size, cv::INTER_CUBIC);
        cv::Mat half(result, cv::Rect(0, 0, _stitcher_image.cols, _stitcher_image.rows));
        _stitcher_image.copyTo(half);
        result.copyTo(_stitcher_image);
        cv::Rect boundingRect = cv::Rect(cv::boundingRect(_stitcher_image != 0));
        cv::Mat croppedImage = _stitcher_image(boundingRect);
        croppedImage.copyTo(_stitcher_image);
}

cv::Mat Stitcher::get_panoram_image(void)
{
    return (_stitcher_image);
}
} // namespace cvlib
