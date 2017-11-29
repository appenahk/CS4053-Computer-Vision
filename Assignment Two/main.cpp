#include "opencv2/opencv.hpp"
#include "Utilities.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

Mat* ground_truth;

void groundTruth();
void detectText(Mat& img, Mat& dice_img, int img_num);
float diceCoefficient(Mat& img, Mat& dice_img);

int main(int argc, char** argv)
{

	int number_of_images = 8;
	Mat* image = new Mat[number_of_images];
	Mat* dice_image = new Mat[number_of_images];

	//Image file names to be loaded in
	char* file_location = "Media/";
	char* image_files[] = {
		"Notice1.jpg",
		"Notice2.jpg", //30
		"Notice3.jpg",
		"Notice4.jpg",
		"Notice5.jpg",
		"Notice6.jpg", //5
		"Notice7.jpg",
		"Notice8.jpg"
	};


	for (int file_no = 0; file_no < number_of_images; file_no++)
	{
		string filename(file_location);
		filename.append(image_files[file_no]);
		image[file_no] = imread(filename, -1);
		if (image[file_no].empty())
		{
			cout << "Could not open " << image[file_no] << endl;
			return -1;
		}
	}
	for (int i = 0; i < number_of_images; i++)
	{
		resize(image[i], image[i], Size(270, 286));
		dice_image[i] = image[i].clone();
	}
	
	groundTruth();

	float dice_avg = 0;
	for (int i = 0; i < number_of_images; i++) {
		
		detectText(image[i], dice_image[i], i);
		
		float dice = diceCoefficient(dice_image[i], ground_truth[i]);
		dice_avg = dice_avg + dice;
		cout << "Dice coefficient for Notice " << i + 1 << ": " << dice << endl;
	}

	dice_avg = dice_avg / 8;
	cout << "Average Dice coefficient is: " << dice_avg << endl;

	return 0;
}

void detectText(Mat& img, Mat& dice_img, int img_num)
{
	vector<Rect> boundRect;
	Mat kernel, morph, structe, element, img_grey, img_threshold, img_gradient, img_erode, img_dilate, img_dilate2, img_dilate3;
	Mat rect_image = img.clone();

	element = getStructuringElement(MORPH_RECT, Size(3, 1));
	kernel = getStructuringElement(MORPH_RECT, Size(2.5, 8));
	morph = getStructuringElement(MORPH_RECT, Size(8, 2));
	structe = getStructuringElement(MORPH_RECT, Size(6, 3));

	cvtColor(dice_img, img_grey, CV_BGR2GRAY);
	morphologyEx(img_grey, img_gradient, MORPH_GRADIENT, element);

	dilate(img_gradient, img_dilate, element);
	dilate(img_dilate, img_dilate2, kernel);

	threshold(img_dilate2, img_threshold, 150, 255, CV_THRESH_OTSU);

	erode(img_threshold, img_erode, structe);
	dilate(img_erode, img_dilate3, kernel);

	Mat img_open, img_close;

	morphologyEx(img_dilate3, img_close, CV_MOP_CLOSE, morph);
	morphologyEx(img_close, img_open, CV_MOP_OPEN, morph);
	
	vector< vector< Point> > contours;
	findContours(img_open, contours, 0, 1);
	vector<vector<Point> > contours_poly(contours.size());
	
	for (int i = 0; i < contours.size(); i++)
		if (contours[i].size() > 180 && contours[i].size() < 1600)
		{
			approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
			Rect appRect(boundingRect(Mat(contours_poly[i])));
			if (appRect.width >=  appRect.height && appRect.width < 250)
				boundRect.push_back(appRect);
			
		}
	dice_img.setTo(Scalar(0, 0, 0));
	
	for (int i = 0; i < boundRect.size(); i++) {
		rectangle(dice_img, boundRect[i], Scalar(255, 255, 255), CV_FILLED, 1, 0);
		rectangle(rect_image, boundRect[i], Scalar(0, 0, 255), 2, 8, 0);
				
	}

	cvtColor(ground_truth[img_num], ground_truth[img_num], CV_GRAY2BGR);
	imshow("Grayscale", img_grey);
	imshow("Morphological Gradient", img_gradient);
	imshow("First Dilation", img_dilate);
	imshow("Second Dilation", img_dilate2);

	imshow("Threshold", img_threshold);
	imshow("Erosion", img_erode);
	imshow("Third Dilation", img_dilate3);

	imshow("Closing", img_close);
	imshow("Opening", img_open);
	imshow("Final Result", rect_image);
	imshow("Dice", dice_img);
	//imshow("Ground", ground_truth[img_num]);
	
	char c = cvWaitKey();
	return;
}

float diceCoefficient(Mat& dice_img, Mat& ground_img) {

	float A = 0;
	float B = 0;
	float A_Intersection_B = 0;

	for (int i = 0; i < dice_img.rows; i++) {
		for (int j = 0; j < dice_img.cols; j++) {
			
			if (dice_img.at<Vec3b>(i, j)[0] == 255) {
				A++;
			}			
			if (ground_img.at<Vec3b>(i, j)[0] == 255) {
				B++;
			}			
			if (dice_img.at<Vec3b>(i, j)[0] == 255 && ground_img.at<Vec3b>(i, j)[0] == 255) {
				A_Intersection_B++;				
			}
		}
	}
	
	float dice = 100 * (2 * A_Intersection_B) / (A + B);
	return dice;
}

void groundTruth() {
	char* file_location = "Media/";
	char* image_files[] = {
		"Notice1.jpg",
		"Notice2.jpg", //30
		"Notice3.jpg",
		"Notice4.jpg",
		"Notice5.jpg",
		"Notice6.jpg", //5
		"Notice7.jpg",
		"Notice8.jpg"
	};

	// Load images
	int number_of_images = 8;
	ground_truth = new Mat[number_of_images];
	for (int file_no = 0; file_no < number_of_images; file_no++)
	{
		string filename(file_location);
		filename.append(image_files[file_no]);
		ground_truth[file_no] = imread(filename, -1);
		if (ground_truth[file_no].empty())
		{
			cout << "Could not open " << ground_truth[file_no] << endl;
			return;
		}
	}

	//set each image entirely black
	for (int i = 0; i < number_of_images; i++) {
		ground_truth[i].setTo(Scalar(0, 0, 0));
	}

	//ground truth 1
	rectangle(ground_truth[0], Point(34, 17), Point(286, 107), Scalar(255, 255, 255), CV_FILLED, 8, 0);
	rectangle(ground_truth[0], Point(32, 117), Point(297, 223), Scalar(255, 255, 255), CV_FILLED, 8, 0);
	rectangle(ground_truth[0], Point(76, 234), Point(105, 252), Scalar(255, 255, 255), CV_FILLED, 8, 0);

	//ground truth 2
	rectangle(ground_truth[1], Point(47, 191), Point(224, 253), Scalar(255, 255, 255), CV_FILLED, 8, 0);

	//ground truth 3
	rectangle(ground_truth[2], Point(142, 121), Point(566, 392), Scalar(255, 255, 255), CV_FILLED, 8, 0);

	//ground truth 4
	rectangle(ground_truth[3], Point(157, 72), Point(378, 134), Scalar(255, 255, 255), CV_FILLED, 8, 0);
	rectangle(ground_truth[3], Point(392, 89), Point(448, 132), Scalar(255, 255, 255), CV_FILLED, 8, 0);
	rectangle(ground_truth[3], Point(405, 138), Point(442, 152), Scalar(255, 255, 255), CV_FILLED, 8, 0);
	rectangle(ground_truth[3], Point(80, 157), Point(410, 245), Scalar(255, 255, 255), CV_FILLED, 8, 0);
	rectangle(ground_truth[3], Point(82, 258), Point(372, 322), Scalar(255, 255, 255), CV_FILLED, 8, 0);

	//ground truth 5
	rectangle(ground_truth[4], Point(112, 73), Point(598, 170), Scalar(255, 255, 255), CV_FILLED, 8, 0);
	rectangle(ground_truth[4], Point(108, 178), Point(549, 256), Scalar(255, 255, 255), CV_FILLED, 8, 0);
	rectangle(ground_truth[4], Point(107, 264), Point(522, 352), Scalar(255, 255, 255), CV_FILLED, 8, 0);

	//ground truth 6
	rectangle(ground_truth[5], Point(91, 54), Point(446, 227), Scalar(255, 255, 255), CV_FILLED, 8, 0);

	//ground truth 7
	rectangle(ground_truth[6], Point(64, 64), Point(476, 268), Scalar(255, 255, 255), CV_FILLED, 8, 0);
	rectangle(ground_truth[6], Point(529, 126), Point(611, 188), Scalar(255, 255, 255), CV_FILLED, 8, 0);
	rectangle(ground_truth[6], Point(545, 192), Point(603, 211), Scalar(255, 255, 255), CV_FILLED, 8, 0);
	rectangle(ground_truth[6], Point(210, 305), Point(595, 384), Scalar(255, 255, 255), CV_FILLED, 8, 0);

	//ground truth 8
	rectangle(ground_truth[7], Point(158, 90), Point(768, 161), Scalar(255, 255, 255), CV_FILLED, 8, 0);
	rectangle(ground_truth[7], Point(114, 174), Point(800, 279), Scalar(255, 255, 255), CV_FILLED, 8, 0);

	for (int i = 0; i < number_of_images; i++) {
		cvtColor(ground_truth[i], ground_truth[i], CV_BGR2GRAY);
		resize(ground_truth[i], ground_truth[i], Size(270, 286));
	}
	return;
}