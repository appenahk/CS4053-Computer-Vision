#include "opencv2/opencv.hpp"
#include "Utilities.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

Mat* ground_truth;


void detectPainting(Mat& img, Mat& dice_img, int img_num);

int main(int argc, char** argv)
{

	int number_of_images = 2;
	Mat* image = new Mat[number_of_images];
	Mat* dice_image = new Mat[number_of_images];

	//Image file names to be loaded in
	char* file_location = "Galleries/";
	char* image_files[] = {
		"Gallery1.jpg",
		"Gallery2.jpg", //30
		"Gallery3.jpg",
		"Gallery4.jpg"
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
		
		dice_image[i] = image[i].clone();
		detectPainting(image[i], dice_image[i], i);
	}

	//groundTruth();


	return 0;
}

void detectPainting(Mat& img, Mat& dice_img, int img_num)
{
	vector<Rect> boundRect;
	Mat kernel, morph, structe, element, img_grey, img_equalize, img_threshold, img_sobel, img_blur, img_laplace, img_gradient, img_canny;
	Mat rect_image = img.clone();
	element = getStructuringElement(MORPH_RECT, Size(5, 5));

	Canny(img, img_canny, 50, 200, 3);
	//GaussianBlur(dice_img, img_blur, Size(3, 3), 0, 0, BORDER_DEFAULT);
	cvtColor(img_canny, img_grey, CV_GRAY2BGR);
	//equalizeHist(img_grey, img_equalize);
	//morphologyEx(img_grey, img_gradient, MORPH_GRADIENT, element);

	dilate(img_grey, img_equalize, element);
	/*vector<Vec2f> lines;
	HoughLines(img_canny, lines, 1, CV_PI / 180, 80);
	
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(img_grey, pt1, pt2, Scalar(0, 0, 255), 3, 8);
	}*/
	vector<Vec4i> lines;
	HoughLinesP(img_canny, lines, 1, CV_PI/180, 80, 45, 5 );
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(img_equalize, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 6, 2);
	}
	imshow("Grayscale", img_grey);
	imshow("Morphological Gradient", img_canny);
	imshow("Morphological", img_equalize);
//	imshow("Morphological sobel", img_threshold);
	
	char c = cvWaitKey();
	return;
}

	
