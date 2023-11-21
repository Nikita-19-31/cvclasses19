/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

void hist(cv::Mat& input, cv::Mat& output)
{
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    cv::Mat hist;

    calcHist(&input, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    // Нормализация гистограммы
    normalize(hist, hist, 0, input.rows, cv::NORM_MINMAX, -1, cv::Mat());

    int histWidth = 800;
    int histHeight = 500;
    int numBars = 256; // Количество столбиков
    int spaceBetweenBars = 10; // Отступ между столбиками
    int totalSpace = spaceBetweenBars * (numBars - 1); // Общий отступ между столбиками
    int barWidth = (histWidth - totalSpace) / numBars; // Ширина столбиков
    cv::Mat histHamming(histHeight, histWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    // Отрисовка широких столбиков для гистограммы с отступами между ними
    for (int i = 0; i < numBars; i++)
    {
        rectangle(histHamming,
                  cv::Point((barWidth + spaceBetweenBars) * i, histHeight),
                  cv::Point((barWidth + spaceBetweenBars) * i + barWidth, histHeight - cvRound(hist.at<float>(i * (histSize / numBars)))),
                  cv::Scalar(255, 0, 0),
                  cv::FILLED);
    }

    histHamming.copyTo(output);
}

void convert8bitTo16bit(cv::Mat& src, cv::Mat& dst)
{
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j+=2)
        {
            dst.at<uint16_t>(i, j / 2) = (src.at<uint8_t>(i, j) << 8) | src.at<uint8_t>(i, j + 1);
        }
    }
}

int demo_feature_descriptor(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    cv::Mat frame;
    auto detector_a = cv::ORB::create();
    auto detector_b = cvlib::corner_detector_fast::create();
    std::vector<cv::KeyPoint> corners;
    cv::Mat descriptors;
    cv::Mat descriptorsORB;

    utils::fps_counter fps;
    int pressed_key = 0;
    while (pressed_key != 27) // ESC
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);

        detector_b->detect(frame, corners); // \todo use your detector (detector_b)
        cv::drawKeypoints(frame, corners, frame, cv::Scalar(0, 0, 255));

        utils::put_fps_text(frame, fps);
        const auto txtFont = cv::FONT_HERSHEY_SIMPLEX;
        const auto fontScale = 0.5;
        const auto thickness = 1;
        static const cv::Point textOrgPoint = {frame.rows / 8, frame.cols / 8};
        std::stringstream ss;
        ss << "detected: " << std::fixed << corners.size();

        cv::putText(frame, ss.str(), textOrgPoint, txtFont, fontScale, cv::Scalar(0, 255, 0), thickness, 8, false);
        cv::imshow(demo_wnd, frame);

        pressed_key = cv::waitKey(30);
        // \todo draw histogram of SSD distribution for all descriptors instead of dumping into the file
        if (pressed_key == ' ') // space
        {
            detector_a->compute(frame, corners, descriptorsORB);
            detector_b->compute(frame, corners, descriptors);

            cv::Mat d_hamming = cv::Mat(descriptors.rows, descriptors.cols, CV_16U);
            cv::Mat histHamming;
            cv::Mat descriptorsORB16bit = cv::Mat(descriptors.rows, descriptors.cols, CV_16U, cv::Scalar(0,0,0));
            cv::FileStorage file("descriptor.json", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
            //Конвертирование мат из 8 битного в 16 битный
            convert8bitTo16bit(descriptorsORB, descriptorsORB16bit);
            //Расстояние по Хэммингу
            cv::bitwise_xor(descriptors, descriptorsORB16bit, d_hamming);
            //Построение гистограммы
            hist(d_hamming, histHamming);
            // Отображение гистограммы
            namedWindow("Histogram", cv::WINDOW_AUTOSIZE);
            imshow("Histogram", histHamming);
            // json
            file << "ORB" << descriptorsORB;
            file << "detector_b" << descriptors;

            std::cout << "Dump descriptors complete! \n";
        }

        std::cout << "Feature points: " << corners.size() << "\r";
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
