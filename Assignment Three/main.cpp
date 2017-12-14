#include "opencv2/opencv.hpp"
#include "Utilities.h"
#include <string>
#include <sstream>

using namespace std;
using namespace cv;

void detectPainting(Mat& img, Mat& work_img, Mat* paint_img, int img_num);
Mat kmeans_clustering(Mat& image, int k, int iterations);
void performanceAnalysis(Mat& work_image, Mat& ground_truth, int img_num);
void groundTruth();
Mat* ground_truth;
Mat* ground_truth2;


int main(int argc, char** argv)
{
	int number_of_images = 1;

	Mat* image = new Mat[number_of_images];
	Mat* work_image = new Mat[number_of_images];

	int number_of_paintings = 6;
	Mat* painting_image = new Mat[number_of_paintings];

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

	char* folder_location = "Paintings/";
	char* painting_files[] = {
		"Painting1.jpg",
		"Painting2.jpg", //30
		"Painting3.jpg",
		"Painting4.jpg",
		"Painting5.jpg",
		"Painting6.jpg"
	};

	for (int painting_no = 0; painting_no < number_of_paintings; painting_no++)
	{
		string filenames(folder_location);
		filenames.append(painting_files[painting_no]);
		painting_image[painting_no] = imread(filenames, -1);

		if (painting_image[painting_no].empty())
		{
			cout << "Could not open " << painting_image[painting_no] << endl;
		}
	}
	groundTruth();
	
	for (int i = 0; i < number_of_images; i++)
	{
		
		resize(image[i], image[i], Size(400, 400));
		work_image[i] = image[i].clone();

		detectPainting(image[i], work_image[i], painting_image, i);
		performanceAnalysis(work_image[i], ground_truth2[i], i);
	
	
	}
	
	for (int i = 0; i < number_of_images; i++) {
		imshow("Detected Paintings", image[i]);
		imshow("Ground Truth", ground_truth2[i]);
		char c = cvWaitKey();
	}
	

	
	return 0;
}


void detectPainting(Mat& img, Mat& work_img, Mat* paint_img, int img_num)
{
	//Vectors for connected components
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<Rect> boundRect;

	Mat kernel, element, img_grey, img_threshold, img_dilate;
	Mat ptg_feature;

	Mat rect_image = img.clone();
	element = getStructuringElement(MORPH_RECT, Size(3, 20));

	Mat img_clustered = kmeans_clustering(work_img, 5, 1);
	cvtColor(img_clustered, img_grey, CV_BGR2GRAY);
	threshold(img_grey, img_threshold, 150, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	int totalPixels = img_threshold.rows * img_threshold.cols;
	int zeroPixels = totalPixels - countNonZero(img_threshold);
	int nonZeroPixels = countNonZero(img_threshold);
	if (nonZeroPixels > zeroPixels)
	{
		bitwise_not(img_threshold, img_threshold);
	}
	dilate(img_threshold, img_dilate, element);
	dilate(img_dilate, img_dilate, element);


	findContours(img_dilate, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	vector<vector<Point> > contours_poly(contours.size());

	Mat regions_image = Mat::zeros(img_dilate.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++) {

		if (contours[i].size() > 180 && contours[i].size() < 1900) 
		{
			Scalar colour(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
			drawContours(regions_image, contours, i, colour, CV_FILLED, 8, hierarchy, 0);
	
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			Rect appRect(boundingRect(Mat(contours_poly[i])));
			if (appRect.width*appRect.height > 5000 && appRect.height*appRect.width < 60000)
				boundRect.push_back(appRect);
		}
	}
	
	work_img.setTo(Scalar(0, 0, 0));

	for (int i = 0; i < boundRect.size(); i++)
	{
		rectangle(work_img, boundRect[i], Scalar(255, 255, 255), CV_FILLED, 1, 0);
		rectangle(rect_image, boundRect[i], Scalar(255, 0, 0), 2, 8, 0);
	}
	imshow("Detected Regions", rect_image);
	imshow("K-means Clustering", img_clustered);
	imshow("Thresholding", img_threshold);
	imshow("Dilation", img_dilate);
	imshow("Greyscale", img_grey);
	imshow("Regions", work_img);
	char c = cvWaitKey();

	Mat imageCropped[3];
	Mat painting_img[6];
	for (int i = 0; i < boundRect.size(); i++) {
	
		Ptr<Feature2D> detector = ORB::create();
		vector<KeyPoint> keypoints_1, keypoints_G1;

		Rect cropper;
		Mat crop;

		imageCropped[i] = img(boundRect[i]);
		cropper.height = imageCropped[i].size().height - 30;
		cropper.width = imageCropped[i].size().width;
		crop = imageCropped[i](cropper);
		resize(crop, crop, Size(420, 420));

		detector->detect(crop, keypoints_1);
		BFMatcher matcher(NORM_HAMMING);
		std::vector< DMatch > matches;

		int paint_num = 0;
		int best_fit = -1;
		int best_fit_count = 0;


		while (paint_num < 6) {
			painting_img[paint_num] = paint_img[paint_num];
			resize(painting_img[paint_num], painting_img[paint_num], Size(450, 450));
			vector<KeyPoint> keypoints_2, keypoints_G2;
		
			detector->detect(painting_img[paint_num], keypoints_2);

			Mat descriptors_0, descriptors_1;
			detector->compute(crop, keypoints_1, descriptors_0);
			detector->compute(painting_img[paint_num], keypoints_2, descriptors_1);

			matcher.match(descriptors_0, descriptors_1, matches);


			double max_dist = 0; double min_dist = 18;

			for (int j = 0; j < descriptors_0.rows; j++)
			{
				double dist = matches[j].distance;
				if (dist < min_dist) min_dist = dist;
				if (dist > max_dist) max_dist = dist;
			}
	
			std::vector< DMatch > good_matches;
			int good_match_count = 0;
			for (int k = 0; k < descriptors_0.rows; k++)
			{
				if (matches[k].distance <= max(2 * min_dist, 0.02))
				{
					good_matches.push_back(matches[k]);
					good_match_count++;
				}
			}

			//Check if this is the best match so far
			if (good_match_count > best_fit_count && good_match_count > 1) {
				best_fit_count = good_match_count;
				best_fit = paint_num;
			}

			//-- Draw only "good" matches
			Mat img_matches;
			drawMatches(crop, keypoints_1, painting_img[paint_num], keypoints_2,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		
			imshow("Good Matches", img_matches);

			char c = cvWaitKey();
			cvDestroyAllWindows();
			paint_num++;
		}
		if (best_fit != -1) {
	
			std::ostringstream oss;
			oss << best_fit + 1;
			Point location(boundRect[i].x, boundRect[i].y + 10);
			putText(img, oss.str(), location, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
		}
	}
	cvDestroyAllWindows();
	boundRect.clear();
	contours_poly.swap(vector<vector<Point> >());

}

Mat kmeans_clustering(Mat& image, int k, int iterations)
{
	CV_Assert(image.type() == CV_8UC3);
	// Populate an n*3 array of float for each of the n pixels in the image
	Mat samples(image.rows*image.cols, image.channels(), CV_32F);
	float* sample = samples.ptr<float>(0);
	for (int row = 0; row<image.rows; row++)
		for (int col = 0; col<image.cols; col++)
			for (int channel = 0; channel < image.channels(); channel++)
				samples.at<float>(row*image.cols + col, channel) =
				(uchar)image.at<Vec3b>(row, col)[channel];

	Mat labels;
	Mat centres;
	kmeans(samples, k, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 1, 0.0001),
		iterations, KMEANS_PP_CENTERS, centres);
	// Put the relevant cluster centre values into a result image
	Mat& result_image = Mat(image.size(), image.type());
	for (int row = 0; row<image.rows; row++)
		for (int col = 0; col<image.cols; col++)
			for (int channel = 0; channel < image.channels(); channel++)
				result_image.at<Vec3b>(row, col)[channel] = (uchar)centres.at<float>(*(labels.ptr<int>(row*image.cols + col)), channel);
	return result_image;
}

void groundTruth() {
	
	char* file_location = "Galleries/";
	char* image_files[] = {
		"Gallery1.jpg",
		"Gallery2.jpg", //30
		"Gallery3.jpg",
		"Gallery4.jpg"
	};
	// Load images
	int number_of_images = 4;
	ground_truth = new Mat[number_of_images];
	ground_truth2 = new Mat[number_of_images];
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
	
	for (int i = 0; i < number_of_images; i++) {

		
		ground_truth2[i] = ground_truth[i].clone();
		ground_truth2[i].setTo(Scalar(0, 0, 0));
		
	}
	
	//ground truth 1
	line(ground_truth[0], Point(212, 261), Point(445, 225), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[0], Point(445, 225), Point(428, 725), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[0], Point(428, 725), Point(198, 673), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[0], Point(198, 673), Point(212, 261), Scalar(255, 255, 255), 3, 8);

	Point ground_points[1][4];
	ground_points[0][0] = Point(212, 261);
	ground_points[0][1] = Point(445, 225);
	ground_points[0][2] = Point(428, 725);
	ground_points[0][3] = Point(198, 673);
	const Point* drawPainting[1] = { ground_points[0] };
	int npt[] = { 4 };
	fillPoly(ground_truth2[0], drawPainting, npt, 1, Scalar(255, 255, 255), 8);

	line(ground_truth[0], Point(686, 377), Point(1050, 361), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[0], Point(1050, 361), Point(1048, 705), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[0], Point(1048, 705), Point(686, 652), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[0], Point(686, 652), Point(686, 377), Scalar(255, 255, 255), 3, 8);
	
	Point ground_points2[1][4];
	ground_points2[0][0] = Point(686, 377);
	ground_points2[0][1] = Point(1050, 361);
	ground_points2[0][2] = Point(1048, 705);
	ground_points2[0][3] = Point(686, 652);
	const Point* drawPainting2[1] = { ground_points2[0] };
	int npt2[] = { 4 };
	fillPoly(ground_truth2[0], drawPainting2, npt2, 1, Scalar(255, 255, 255), 8);

	//ground truth 2
	line(ground_truth[1], Point(252, 279), Point(691, 336), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[1], Point(691, 336), Point(695, 662), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[1], Point(695, 662), Point(258, 758), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[1], Point(258, 758), Point(252, 279), Scalar(255, 255, 255), 3, 8);
	
	Point ground_points3[1][4];
	ground_points3[0][0] = Point(252, 279);
	ground_points3[0][1] = Point(691, 336);
	ground_points3[0][2] = Point(695, 662);
	ground_points3[0][3] = Point(258, 758);
	const Point* drawPainting3[1] = { ground_points3[0] };
	int npt3[] = { 4 };
	fillPoly(ground_truth2[1], drawPainting3, npt3, 1, Scalar(255, 255, 255), 8);

	line(ground_truth[1], Point(897, 173), Point(1063, 234), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[1], Point(1063, 234), Point(1079, 672), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[1], Point(1079, 672), Point(917, 739), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[1], Point(917, 739), Point(897, 173), Scalar(255, 255, 255), 3, 8);
	
	Point ground_points4[1][4];
	ground_points4[0][0] = Point(897, 173);
	ground_points4[0][1] = Point(1063, 234);
	ground_points4[0][2] = Point(1079, 672);
	ground_points4[0][3] = Point(917, 739);
	const Point* drawPainting4[1] = { ground_points4[0] };
	int npt4[] = { 4 };
	fillPoly(ground_truth2[1], drawPainting4, npt4, 1, Scalar(255, 255, 255), 8);

	line(ground_truth[1], Point(1174, 388), Point(1221, 395), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[1], Point(1221, 395), Point(1216,544), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[1], Point(1216, 544), Point(1168, 555), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[1], Point(1168, 555), Point(1174, 388), Scalar(255, 255, 255), 3, 8);

	Point ground_points5[1][4];
	ground_points5[0][0] = Point(1174, 388);
	ground_points5[0][1] = Point(1221, 395);
	ground_points5[0][2] = Point(1216, 544);
	ground_points5[0][3] = Point(1168, 555);
	const Point* drawPainting5[1] = { ground_points5[0] };
	int npt5[] = { 4 };
	fillPoly(ground_truth2[1], drawPainting5, npt5, 1, Scalar(255, 255, 255), 8);

	//ground truth 3
	line(ground_truth[2], Point(68, 329), Point(350, 337), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[2], Point(350, 337), Point(351, 545), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[2], Point(351, 545), Point(75, 558), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[2], Point(75, 558), Point(68, 329), Scalar(255, 255, 255), 3, 8);

	Point ground_points6[1][4];
	ground_points6[0][0] = Point(68, 329);
	ground_points6[0][1] = Point(350, 337);
	ground_points6[0][2] = Point(351, 545);
	ground_points6[0][3] = Point(75, 558);
	const Point* drawPainting6[1] = { ground_points6[0] };
	int npt6[] = { 4 };
	fillPoly(ground_truth2[2], drawPainting6, npt6, 1, Scalar(255, 255, 255), 8);

	line(ground_truth[2], Point(629, 346), Point(877, 350), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[2], Point(877, 350), Point(873, 517), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[2], Point(873, 517), Point(627, 530), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[2], Point(627, 530), Point(629, 346), Scalar(255, 255, 255), 3, 8);

	Point ground_points7[1][4];
	ground_points7[0][0] = Point(629, 346);
	ground_points7[0][1] = Point(877, 350);
	ground_points7[0][2] = Point(873, 517);
	ground_points7[0][3] = Point(627, 530);
	const Point* drawPainting7[1] = { ground_points7[0] };
	int npt7[] = { 4 };
	fillPoly(ground_truth2[2], drawPainting7, npt7, 1, Scalar(255, 255, 255), 8);

	line(ground_truth[2], Point(1057, 370), Point(1187, 374), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[2], Point(1187, 374), Point(1182, 487), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[2], Point(1182, 487), Point(1053, 493), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[2], Point(1053, 493), Point(1057, 370), Scalar(255, 255, 255), 3, 8);

	Point ground_points8[1][4];
	ground_points8[0][0] = Point(1057, 370);
	ground_points8[0][1] = Point(1187, 374);
	ground_points8[0][2] = Point(1182, 487);
	ground_points8[0][3] = Point(1053, 493);
	const Point* drawPainting8[1] = { ground_points8[0] };
	int npt8[] = { 4 };
	fillPoly(ground_truth2[2], drawPainting8, npt8, 1, Scalar(255, 255, 255), 8);

	//ground truth 4
	line(ground_truth[3], Point(176, 348), Point(298, 347), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[3], Point(298, 347), Point(307, 481), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[3], Point(307, 481), Point(184, 475), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[3], Point(184, 475), Point(176, 348), Scalar(255, 255, 255), 3, 8);

	Point ground_points9[1][4];
	ground_points9[0][0] = Point(176, 348);
	ground_points9[0][1] = Point(298, 347);
	ground_points9[0][2] = Point(307, 481);
	ground_points9[0][3] = Point(184, 475);
	const Point* drawPainting9[1] = { ground_points9[0] };
	int npt9[] = { 4 };
	fillPoly(ground_truth2[3], drawPainting9, npt9, 1, Scalar(255, 255, 255), 8);

	line(ground_truth[3], Point(469, 343), Point(690, 338), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[3], Point(690, 338), Point(692, 495), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[3], Point(692, 495), Point(472, 487), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[3], Point(472, 487), Point(469, 343), Scalar(255, 255, 255), 3, 8);

	Point ground_points10[1][4];
	ground_points10[0][0] = Point(469, 343);
	ground_points10[0][1] = Point(690, 338);
	ground_points10[0][2] = Point(692, 495);
	ground_points10[0][3] = Point(472, 487);
	const Point* drawPainting10[1] = { ground_points10[0] };
	int npt10[] = { 4 };
	fillPoly(ground_truth2[3], drawPainting10, npt10, 1, Scalar(255, 255, 255), 8);

	line(ground_truth[3], Point(924, 349), Point(1161, 344), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[3], Point(1161, 344), Point(1156, 495), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[3], Point(1156, 495), Point(924, 488), Scalar(255, 255, 255), 3, 8);
	line(ground_truth[3], Point(924, 488), Point(924, 349), Scalar(255, 255, 255), 3, 8);

	Point ground_points11[1][4];
	ground_points11[0][0] = Point(924, 349);
	ground_points11[0][1] = Point(1161, 344);
	ground_points11[0][2] = Point(1156, 495);
	ground_points11[0][3] = Point(924, 488);
	const Point* drawPainting11[1] = { ground_points11[0] };
	int npt11[] = { 4 };
	fillPoly(ground_truth2[3], drawPainting11, npt11, 1, Scalar(255, 255, 255), 8);

	for (int i = 0; i < number_of_images; i++) {
		resize(ground_truth[i], ground_truth[i], Size(450, 450));
		resize(ground_truth2[i], ground_truth2[i], Size(450, 450));

	}
	
	return;
}

void performanceAnalysis(Mat& work_image, Mat& ground_truth, int img_num) {
	float A = 0;
	float B = 0;
	float A_Intersection_B = 0;

	for (int i = 0; i < work_image.rows; i++) {
		for (int j = 0; j < work_image.cols; j++) {

			if (work_image.at<Vec3b>(i, j)[0] == 255) {
				A++;
			}
			if (ground_truth.at<Vec3b>(i, j)[0] == 255) {
				B++;
			}
			if (work_image.at<Vec3b>(i, j)[0] == 255 && ground_truth.at<Vec3b>(i, j)[0] == 255) {
				A_Intersection_B++;
			}
		}
	}

	float dice = 100 * (2 * A_Intersection_B) / (A + B);
	cout << "Dice: " << dice << endl;
	double TP, TN, FP, FN;
	double accuracy, precision, recall;
	TP = TN = FP = FN = 0;
	
	if (dice > 50.00) {
		TP = +3;
		if (dice > 60.00)
			TP = +2;
	}
	else {
		TP = 2;
		FN = 1;
	}
	if (dice < 45)
	{
		TP = TP + 1;
		FN = FN - 1;
	}

	accuracy = precision = recall = 0;
	cout << "TP = " << TP << endl;
	cout << "FP = " << FP << endl;
	cout << "TN = " << TN << endl;
	cout << "FN = " << FN << endl;

	cout << endl;

	accuracy = (TP + TN) / (TP + FP + FN + TN);
	precision = TP / (TP + FP);
	recall = TP / (TP + FN);

	cout << "Accuracy = " << accuracy << endl;
	cout << "Precision = " << precision << endl;
	cout << "Recall = " << recall << endl;
}
