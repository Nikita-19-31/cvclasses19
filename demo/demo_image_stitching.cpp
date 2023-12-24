/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

int demo_image_stitching(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
       if (!cap.isOpened())
           return -1;

       const auto main_wnd = "origina";
       const auto demo_wnd = "demo";

       cv::namedWindow(main_wnd);
       cv::namedWindow(demo_wnd);
       bool show_panoram_flag = false;
       std::vector<std::vector<cv::DMatch>> pairs;

       cv::Mat main_frame;
       cv::Mat demo_frame;
       cvlib::Stitcher stitcher;
       utils::fps_counter fps;
       int pressed_key = 0;
       while (pressed_key != 27) // ESC
       {
           cap >> main_frame;
           cv::cvtColor(main_frame, main_frame, cv::COLOR_BGR2GRAY);

           pressed_key = cv::waitKey(30);
           if (pressed_key == ' ') // space
           {
               demo_frame = main_frame.clone();
               stitcher.init(demo_frame);
               show_panoram_flag = true;
           }

           if (pressed_key == 's') // s
           {
               stitcher.stitch(main_frame);
           }
           utils::put_fps_text(main_frame, fps);
           cv::imshow(main_wnd, main_frame);

           if (show_panoram_flag)
           {
               cv::imshow(demo_wnd, stitcher.get_panoram_image());
           }
       }

       cv::destroyWindow(main_wnd);
       cv::destroyWindow(demo_wnd);

       return 0;
}
