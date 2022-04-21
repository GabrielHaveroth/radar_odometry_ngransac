#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "radar_utils.hpp"
#include "features.hpp"
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
namespace py = pybind11;


Eigen::MatrixXd getRadarDescriptor(std::string root_folder, std::string  file_name, int kp_extraction_type)
{
    int min_range = 58;              // min range of radar points (bin)
    float radar_resolution = 0.0432; // resolution of radar bins in meters per bin
    float cart_resolution = 0.2592;  // meters per pixel
    int cart_pixel_width = 964;      // height and width of cartesian image in pixels
    bool interp = true;
    int keypoint_extraction = kp_extraction_type; // 0: cen2018, 1: cen2019, 2: orb
    // cen2018 parameters
    float zq = 3.0;
    int sigma_gauss = 17;
    // cen2019 parameters
    int max_points = 10000;
    // ORB descriptor / matching parameters
    int patch_size = 21; // width of patch in pixels in cartesian radar image
    // Create ORB feature detector
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->setPatchSize(patch_size);
    detector->setEdgeThreshold(patch_size);
    // BRUTEFORCE_HAMMING for ORB descriptors FLANNBASED for cen2019 descriptors
    cv::Mat img, desc;
    std::vector<cv::KeyPoint> kp;
    Eigen::MatrixXd targets, cart_targets;
    std::vector<int64_t> t;
    std::vector<int64_t> times;
    std::vector<double> azimuths;
    std::vector<bool> valid;
    cv::Mat fft_data;
    load_radar(root_folder + "/" + file_name, times, azimuths, valid, fft_data);
    if (keypoint_extraction == 0)
        cen2018features(fft_data, zq, sigma_gauss, min_range, targets);
    if (keypoint_extraction == 1)
        cen2019features(fft_data, max_points, min_range, targets);
    if (keypoint_extraction == 0 || keypoint_extraction == 1)
    {
        radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width, interp, img, CV_8UC1); // NOLINT
        polar_to_cartesian_points(azimuths, times, targets, radar_resolution, cart_targets, t);
        convert_to_bev(cart_targets, cart_resolution, cart_pixel_width, patch_size, kp, t);
        // cen2019descriptors(azimuths, cv::Size(fft_data.cols, fft_data.rows), targets, cart_targets2,
        // radar_resolution, 0.3456, 722, desc);
        detector->compute(img, kp, desc);
        std::cout << img.size() << std::endl;

    }
    if (keypoint_extraction == 2)
    {
        detector->detect(img, kp);
        detector->compute(img, kp, desc);
        convert_from_bev(kp, cart_resolution, cart_pixel_width, cart_targets);
        getTimes(cart_targets, azimuths, times, t);
    }
    Eigen::MatrixXd desc_values = Eigen::MatrixXd::Zero(desc.rows, desc.cols);
    Eigen::MatrixXd kp_values = Eigen::MatrixXd::Zero(2, kp.size());
    
    for (uint i = 0; i < kp.size(); i++)
    {
        kp_values(0, i) = kp[i].pt.x;
        kp_values(1, i) = kp[i].pt.y;
    }
    std::cout << kp_values << std::endl;
    
    for (int i = 0; i < desc.rows; i++)
    {
        for (int j = 0; j < desc.cols; j++)
        {
             desc_values(i, j) = desc.at<uchar>(i, j);
        }
    }
    return desc_values;
}

std::vector<Eigen::MatrixXd> getRadardCorrespondeces(std::string root_folder, std::string file_name1, std::string file_name2, int kp_extraction_type)
{
    int min_range = 58;                 // min range of radar points (bin)
    float radar_resolution = 0.0432;    // resolution of radar bins in meters per bin
    float cart_resolution = 0.2592;     // meters per pixel
    int cart_pixel_width = 964;         // height and width of cartesian image in pixels
    bool interp = true;
    int keypoint_extraction = 0;        // 0: cen2018, 1: cen2019, 2: orb
    // cen2018 parameters
    float zq = 3.0;
    int sigma_gauss = 17;
    // cen2019 parameters
    int max_points = 10000;
    // ORB descriptor / matching parameters
    int patch_size = 21;                // width of patch in pixels in cartesian radar image
    float nndr = 0.80;                  // Nearest neighbor distance ratio
    // RANSAC
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->setPatchSize(patch_size);
    detector->setEdgeThreshold(patch_size);
    // BRUTEFORCE_HAMMING for ORB descriptors FLANNBASED for cen2019 descriptors
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    cv::Mat img1, img2, desc1, desc2;
    std::vector<cv::KeyPoint> kp1, kp2;
    Eigen::MatrixXd targets1, targets2, cart_targets1, cart_targets2;
    std::vector<int64_t> t1, t2;
    std::vector<int64_t> times1, times2;
    std::vector<double> azimuths1, azimuths2;
    std::vector<bool> valid1, valid2;
    cv::Mat fft_data1, fft_data2;
    load_radar(root_folder + "/" + file_name1, times1, azimuths1, valid1, fft_data1);
    load_radar(root_folder + "/" + file_name2, times2, azimuths2, valid2, fft_data2);
    if (keypoint_extraction == 0)
    {
        cen2018features(fft_data1, zq, sigma_gauss, min_range, targets1);
        cen2018features(fft_data2, zq, sigma_gauss, min_range, targets2);
    }

    if (keypoint_extraction == 1)
    {
        cen2019features(fft_data1, max_points, min_range, targets1);
        cen2019features(fft_data2, max_points, min_range, targets2);
    }        
    if (keypoint_extraction == 0 || keypoint_extraction == 1) {
        radar_polar_to_cartesian(azimuths1, fft_data1, radar_resolution, cart_resolution, cart_pixel_width, interp, img1, CV_8UC1);  // NOLINT
        polar_to_cartesian_points(azimuths1, times1, targets1, radar_resolution, cart_targets1, t1);
        convert_to_bev(cart_targets1, cart_resolution, cart_pixel_width, patch_size, kp1, t1);
        // cen2019descriptors(azimuths, cv::Size(fft_data.cols, fft_data.rows), targets, cart_targets2,
            // radar_resolution, 0.3456, 722, desc2);
        detector->compute(img1, kp1, desc1);

        radar_polar_to_cartesian(azimuths2, fft_data2, radar_resolution, cart_resolution, cart_pixel_width, interp, img2, CV_8UC1);  // NOLINT
        polar_to_cartesian_points(azimuths2, times2, targets2, radar_resolution, cart_targets2, t2);
        convert_to_bev(cart_targets2, cart_resolution, cart_pixel_width, patch_size, kp2, t2);
        // cen2019descriptors(azimuths, cv::Size(fft_data.cols, fft_data.rows), targets, cart_targets2,
            // radar_resolution, 0.3456, 722, desc2);
        detector->compute(img2, kp2, desc2);

    }
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(desc1, desc2, knn_matches, 2);
    // Filter matches using nearest neighbor distance ratio (Lowe, Szeliski)
    std::vector<cv::DMatch> good_matches;
    std::vector<float>good_ratio_values;
    float ratio;

    for (uint j = 0; j < knn_matches.size(); ++j) {
        if (!knn_matches[j].size())
            continue;
        if (knn_matches[j][0].distance < nndr * knn_matches[j][1].distance) 
        {
            good_matches.push_back(knn_matches[j][0]);
            ratio = knn_matches[j][0].distance / knn_matches[j][1].distance;
            good_ratio_values.push_back(ratio);  
        }
    }

    // Convert the good key point matches to Eigen matrices
    Eigen::MatrixXd p1 = Eigen::MatrixXd::Zero(2, good_matches.size());
    Eigen::MatrixXd p2 = p1;
    Eigen::MatrixXd ratios = Eigen::MatrixXd::Zero(good_matches.size(), 1);
    Eigen::MatrixXd kp1_values = Eigen::MatrixXd::Zero(good_matches.size(), 2);
    Eigen::MatrixXd kp2_values = kp1_values;

    for (uint j = 0; j < good_matches.size(); ++j) 
    {
        p1(0, j) = cart_targets1(0, good_matches[j].queryIdx);
        p1(1, j) = cart_targets1(1, good_matches[j].queryIdx);
        p2(0, j) = cart_targets2(0, good_matches[j].trainIdx);
        p2(1, j) = cart_targets2(1, good_matches[j].trainIdx);
        kp1_values(j, 0) = kp1[good_matches[j].queryIdx].pt.x;
        kp1_values(j, 1) = kp1[good_matches[j].queryIdx].pt.y;
        kp2_values(j, 0) = kp2[good_matches[j].trainIdx].pt.x;
        kp2_values(j, 1) = kp2[good_matches[j].trainIdx].pt.y;
        ratios(j, 0) = good_ratio_values[j];
    }

    std::vector<Eigen::MatrixXd> correspondences;
    correspondences.push_back(p1);
    correspondences.push_back(kp1_values);
    correspondences.push_back(p2);
    correspondences.push_back(kp2_values);
    correspondences.push_back(ratios);
    return correspondences;
}

PYBIND11_MODULE(radar_processor, m) {
    m.def("get_radar_descriptor", &getRadarDescriptor);
    m.def("get_radar_correspondeces", &getRadardCorrespondeces);
}
