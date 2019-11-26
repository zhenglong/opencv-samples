//
//  main.cpp
//  watermark
//
//  Created by HJ on 2019/10/29.
//  Copyright © 2019 HJ. All rights reserved.
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

using namespace cv;
using namespace std;

std::vector<Mat> allPlanes;
std::vector<Mat> planes;
Mat _complexImage;

Mat optimizeImageDim(Mat image) {
    Mat padded;
    int addPixelRows = getOptimalDFTSize(image.rows);
    int addPixelCols = getOptimalDFTSize(image.cols);
    
    copyMakeBorder(image, padded, 0, addPixelRows - image.rows, 0, addPixelCols - image.cols, BORDER_CONSTANT, Scalar::all(0));
    return padded;
}

Mat splitSrc(Mat image) {
    if (!allPlanes.empty()) {
        allPlanes.clear();
    }
    
    Mat optimizeImage = optimizeImageDim(image);
    Mat padded;
    
    split(optimizeImage, allPlanes);
    if (allPlanes.size() > 1) {
        padded = allPlanes[0];
    } else {
        padded = optimizeImageDim(image);
    }
    
    return padded;
}

void transformImageWithText(Mat image, const string &blindMarkText, Point point, double fontSize, Scalar scalar) {
    if (!planes.empty()) {
        planes.clear();
    }
    
    Mat padded = splitSrc(image);
    padded.convertTo(padded, CV_32F);
    planes.push_back(padded);
    planes.push_back(Mat::zeros(padded.size(), CV_32F));
    merge(planes, _complexImage);
    dft(_complexImage, _complexImage);
    putText(_complexImage, blindMarkText, point, FONT_HERSHEY_DUPLEX, fontSize, scalar, 3);
    flip(_complexImage, _complexImage, -1);
    putText(_complexImage, blindMarkText, point, FONT_HERSHEY_DUPLEX, fontSize, scalar, 3);
    flip(_complexImage, _complexImage, -1);
    planes.clear();
}

Mat antitransformImage() {
    Mat invDFT;
    idft(_complexImage, invDFT, DFT_SCALE | DFT_REAL_OUTPUT, 0);
    Mat restoredImage;
    invDFT.convertTo(restoredImage, CV_8U);
    
    allPlanes.erase(allPlanes.begin());
    allPlanes.insert(allPlanes.begin(), restoredImage);
    Mat lastImage;
    merge(allPlanes, lastImage);
    
    return lastImage;
}

void shiftDFT(Mat image) {
    image = image(Rect(0, 0, image.cols & (-2), image.rows & (-2)));
    int cx = image.cols / 2;
    int cy = image.rows / 2;
    
    Mat q0 = Mat(image, Rect(0, 0, cx, cy));
    Mat q1 = Mat(image, Rect(cx, 0, cx, cy));
    Mat q2 = Mat(image, Rect(0, cy, cx, cy));
    Mat q3 = Mat(image, Rect(cx, cy, cx, cy));
    
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

Mat createOptimizedMagnitude(Mat complexImage) {
    vector<Mat> newPlanes = {};
    Mat mag;
    
    split(complexImage, newPlanes);
    magnitude(newPlanes[0], newPlanes[1], mag);
    add(Mat::ones(mag.size(), CV_32F), mag, mag);
    log(mag, mag);
    
    shiftDFT(mag);
    mag.convertTo(mag, CV_8UC1);
    normalize(mag, mag, 0, 255, NORM_MINMAX, CV_8UC1);
    
    return mag;
}

Mat transformImage(Mat image) {
    if (!planes.empty()) {
        planes.clear();
    }
    Mat padded = splitSrc(image);
    padded.convertTo(padded, CV_32F);
    planes.push_back(padded);
    planes.push_back(Mat::zeros(padded.size(), CV_32F));
    merge(planes, _complexImage);
    dft(_complexImage, _complexImage);
    Mat magnitude = createOptimizedMagnitude(_complexImage);
    planes.clear();
    return magnitude;
}

int main(int argc, char ** argv)
{
    const char* filename = argc >=2 ? argv[1] : "/Users/hujiang/Downloads/2.jpg";
    // 以灰度形式读取图片
    Mat I = imread( samples::findFile( filename ), IMREAD_COLOR);
    if( I.empty()){
        cout << "Error opening image" << endl;
        return EXIT_FAILURE;
    }

    I.convertTo(I, CV_8UC4);
    Scalar color = Scalar(0, 255, 255);
    transformImageWithText(I, "Test", Point(65,65), 3, color);
    Mat cvMat = antitransformImage();
    Mat lastImage = transformImage(cvMat);
    imwrite("output.png",lastImage);
    
    waitKey();
    return EXIT_SUCCESS;
}
