/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include "cvlib.hpp"


class RegionsTree
{
public:

    RegionsTree(cv::Mat image): img(image), hasChilds(false) {}

    cv::Mat img;
    std::vector<RegionsTree> childs;
    bool hasChilds;
};
namespace
{

bool predicate(double stddev1, double stddev2, double stddev)
{
    return (stddev1 < stddev) && (stddev2 < stddev);
}
void merge_regions(RegionsTree* firstRegion, RegionsTree* secondRegion, double stddev)
{

    cv::Mat mean1, dev1;
    cv::Mat mean2, dev2;
    double d1;
    double d2;

    cv::meanStdDev(firstRegion->img, mean1, dev1);
    cv::meanStdDev(secondRegion->img, mean2, dev2);

    d1 = dev1.at<double>(0);
    d2 = dev2.at<double>(0);

    cv::Mat mean = (mean1 + mean2) / 2;

    if (predicate(d1, d2, stddev))
    {
        firstRegion->img.setTo(mean);
        secondRegion->img.setTo(mean);
    }
}
void merge_image(double stddev, RegionsTree* regions)
{
    while (regions->hasChilds)
    {
        bool hasAtLeastOneChild = false;

        for (int i = 0; i < 4; i++)
            hasAtLeastOneChild += regions->childs[i].hasChilds;

        if (hasAtLeastOneChild)
        {
            for (int i = 0; i < 4; i++)
                merge_image(stddev, &regions->childs[i]);
        }
        else
        {
            regions->hasChilds = false;
            merge_regions(&regions->childs[3], &regions->childs[0], stddev);
            merge_regions(&regions->childs[0], &regions->childs[1], stddev);
            merge_regions(&regions->childs[0], &regions->childs[2], stddev);

        }
    }
}

void split_image(cv::Mat image, double stddev, RegionsTree* regions)
{
    if ((image.cols < 3) || (image.rows < 3))
        return;

    int height = image.rows;
    int width = image.cols;
    cv::Mat mean;
    cv::Mat dev;

    cv::meanStdDev(image, mean, dev);

    if (dev.at<double>(0) <= stddev)
    {
        image.setTo(mean);
        return;
    }

    regions->img = image;
    regions->hasChilds = true;

    cv::Mat leftTop = image(cv::Range(0, height / 2), cv::Range(0, width / 2));
    cv::Mat rightTop = image(cv::Range(0, height / 2), cv::Range(width / 2, width));
    cv::Mat lefttBottom = image(cv::Range(height / 2, height), cv::Range(0, width / 2));
    cv::Mat rightBottom = image(cv::Range(height / 2, height), cv::Range(width / 2, width));

    regions->childs.push_back(RegionsTree(leftTop));
    regions->childs.push_back(RegionsTree(rightTop));
    regions->childs.push_back(RegionsTree(lefttBottom));
    regions->childs.push_back(RegionsTree(rightBottom));

    split_image(leftTop, stddev, &regions->childs[0]);
    split_image(rightTop, stddev, &regions->childs[1]);
    split_image(lefttBottom, stddev, &regions->childs[2]);
    split_image(rightBottom, stddev, &regions->childs[3]);
}
} // namespace


namespace cvlib
{
cv::Mat split_and_merge(const cv::Mat& image, double stddev)
{
    RegionsTree regions(image);
    cv::Mat res = image;

    split_image(res, stddev, &regions);
    merge_image(stddev, &regions);
    return res;
}
} // namespace cvlib
