/* Computer Vision Functions.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#ifndef __CVLIB_HPP__
#define __CVLIB_HPP__

#include <opencv2/opencv.hpp>

namespace cvlib
{
/// \brief Split and merge algorithm for image segmentation
/// \param image, in - input image
/// \param stddev, in - threshold to treat regions as homogeneous
/// \return segmented image
cv::Mat split_and_merge(const cv::Mat& image, double stddev);

/// \brief Segment texuture on passed image according to sample in ROI
/// \param image, in - input image
/// \param roi, in - region with sample texture on passed image
/// \param eps, in - threshold parameter for texture's descriptor distance
/// \return binary mask with selected texture
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps);

/// \brief Motion Segmentation algorithm
class motion_segmentation : public cv::BackgroundSubtractor
{
    public:
    /// \brief ctor
    motion_segmentation();

    /// \see cv::BackgroundSubtractor::apply
    void apply(cv::InputArray image, cv::OutputArray fgmask, double learningRate = -1) override;

    /// \see cv::BackgroundSubtractor::BackgroundSubtractor
    void getBackgroundImage(cv::OutputArray backgroundImage) const override
    {
        backgroundImage.assign(bg_model_);
    }
    void setVarThreshold(int threshold);
    void updateModel(int n);

    private:
    cv::Mat bg_model_;
    cv::Mat min_image;
    cv::Mat max_image;
    cv::Mat prev_image;
    cv::Mat diff_image;
    cv::Size sizeKernel;
    int N;
    int _threshold;
};

/// \brief FAST corner detection algorithm
class corner_detector_fast : public cv::Feature2D
{
    public:
    /// \brief Fabrique method for creating FAST detector
    static cv::Ptr<corner_detector_fast> create();

    /// \see Feature2d::detect
    virtual void detect(cv::InputArray _image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask = cv::noArray()) override;

    /// \see Feature2d::compute
    virtual void compute(cv::InputArray _image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) override;

    /// \see Feature2d::detectAndCompute
    virtual void detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors,
                                  bool useProvidedKeypoints = false) override;

    /// \see Feature2d::getDefaultName
    virtual cv::String getDefaultName() const override
    {
        return "FAST_Binary";
    }

    bool checkPixel(cv::Mat& image, int i, int j, int N, int t);
    bool checkMaxLenSeqPix(std::vector<int> seq, int N);
    void generateNormPoints(int s, int len_desc);

    std::vector<cv::Point> _pixelsAround = {
        cv::Point(0, -3), cv::Point(1, -3) , cv::Point(2, -2) , cv::Point(3, -1) ,
        cv::Point(3, 0) , cv::Point(3, 1)  , cv::Point(2, 2)  , cv::Point(1, 3)  ,
        cv::Point(0, 3) , cv::Point(-1, 3) , cv::Point(-2, 2) , cv::Point(-3, 1) ,
        cv::Point(-3, 0), cv::Point(-3, -1), cv::Point(-2, -2), cv::Point(-1, -3)
     };

    std::vector<int>  _initialVerify = {1, 5, 9, 13};
    std::vector<int>  _secondaryVerify = {2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16};
    int radius = 3;
    std::vector<cv::Point2f> _pairPixels;

};

/// \brief Descriptor matched based on ratio of SSD
class descriptor_matcher : public cv::DescriptorMatcher
{
    public:
    /// \brief ctor
    descriptor_matcher(float ratio = 1.5) : ratio_(ratio)
    {
    }

    /// \brief setup ratio threshold for SSD filtering
    void set_ratio(float r)
    {
        ratio_ = r;
    }

    protected:
    /// \see cv::DescriptorMatcher::knnMatchImpl
    virtual void knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k,
                              cv::InputArrayOfArrays masks = cv::noArray(), bool compactResult = false) override;

    /// \see cv::DescriptorMatcher::radiusMatchImpl
    virtual void radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float maxDistance,
                                 cv::InputArrayOfArrays masks = cv::noArray(), bool compactResult = false) override;

    /// \see cv::DescriptorMatcher::isMaskSupported
    virtual bool isMaskSupported() const override
    {
        return false;
    }

    /// \see cv::DescriptorMatcher::isMaskSupported
    virtual cv::Ptr<cv::DescriptorMatcher> clone(bool emptyTrainData = false) const override
    {
        cv::Ptr<cv::DescriptorMatcher> copy = new descriptor_matcher(*this);
        if (emptyTrainData)
        {
            copy->clear();
        }
        return copy;
    }

    private:
    float ratio_;
};

/// \brief Stitcher for merging images into big one
class Stitcher
{
    /// \todo design and implement
};
} // namespace cvlib

#endif // __CVLIB_HPP__
