/* reference code from https://github.com/davidshower/seam-carving/blob/master/seam-carving.cpp */

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <time.h>

using namespace cv;
using namespace std;

enum SeamDirection { VERTICAL, HORIZONTAL };

float energy_image_time = 0;
float cumulative_energy_map_time = 0;
float find_seam_time = 0;
float reduce_time = 0;
float modify_offset_map_time = 0;
float detect_line_time = 0;

bool demo;
bool debug;

bool straight_line_pre;
bool straight_line_manually;
bool faces_detect;
bool human_faces_detect;

Mat src, src_gray;
Mat dst, detected_edges;
Mat detected_straight_lines;
Mat offset_energy_map;
vector<Vec4i>lines;

int lowThreshold = 0;
int HoughThre = 0;
const int max_lowThreshold = 100;
const int Ratio = 3;
const int kernel_size = 3;

static void CannyThreshold(int, void*) {
	blur(src_gray, detected_edges, Size(3, 3));
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*Ratio, kernel_size);
	dst = Scalar::all(0);
	src.copyTo(dst, detected_edges);
	imshow("Edge Map", dst);
}

static void HoughThreshold(int, void*) {
	detected_straight_lines = src.clone();
	HoughLinesP(detected_edges, lines, 1, CV_PI / 180, HoughThre, 50, 10); 
	// draw the lines
	for (int i = 0; i < lines.size(); i++) {
		// in case it takes borders as straight lines
		Vec4i l = lines[i];
		if (l[0] == l[2] && l[0] < 10) {
			lines.erase(lines.begin() + i);
			i--;
			continue;
		}
		if (l[1] == l[3] && l[1] < 10) {
			lines.erase(lines.begin() + i);
			i--;
			continue;
		}
		line(detected_straight_lines, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
	}
	// show results
	imshow("Detected Lines(in red)", detected_straight_lines);
}

Mat createEnergyImage(Mat &image) {
	clock_t start = clock();
	Mat image_blur, image_gray;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat grad, energy_image;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S; 

	// apply a gaussian blur to reduce noise
	GaussianBlur(image, image_blur, Size(3, 3), 0, 0, BORDER_DEFAULT);

	// convert to grayscale
	cvtColor(image_blur, image_gray, CV_BGR2GRAY);

	// use Sobel to calculate the gradient of the image in the x and y direction
	//Sobel(image_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	//Sobel(image_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

	// use Scharr to calculate the gradient of the image in the x and y direction
	Scharr(image_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
	Scharr(image_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);

	// convert gradients to abosulte versions of themselves
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);

	// total gradient (approx) 
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	// convert the default values to double precision 
	grad.convertTo(energy_image, CV_64F, 1.0 / 255.0);

	if (straight_line_pre || faces_detect) {
		int rowsize = image.rows;
		int colsize = image.cols;
		for (int i = 0; i < rowsize; i++) {
			for (int j = 0; j < colsize; j++) {
				energy_image.at<double>(i, j) += offset_energy_map.at<Vec2d>(i, j)[0];
			}
		}
	}

	// create and show the newly created energy image
	if (demo) {
		namedWindow("Energy Image", CV_WINDOW_AUTOSIZE); imshow("Energy Image", energy_image);
	}

	// calculate time taken
	clock_t end = clock();
	energy_image_time += ((float)end - (float)start) / CLOCKS_PER_SEC;

	return energy_image;
}

Mat createCumulativeEnergyMap(Mat &energy_image, SeamDirection seam_direction) {
	clock_t start = clock();
	double a, b, c;

	// get the numbers of rows and columns in the image
	int rowsize = energy_image.rows;
	int colsize = energy_image.cols;

	// initialize the map with zeros
	Mat cumulative_energy_map = Mat(rowsize, colsize, CV_64F, double(0));

	// copy the first row
	if (seam_direction == VERTICAL) energy_image.row(0).copyTo(cumulative_energy_map.row(0));
	else if (seam_direction == HORIZONTAL) energy_image.col(0).copyTo(cumulative_energy_map.col(0));

	// take the minimum of the three neighbors and add to total, this creates a running sum which is used to determine the lowest energy path
	if (seam_direction == VERTICAL) {
		for (int row = 1; row < rowsize; row++) {
			for (int col = 0; col < colsize; col++) {
				a = cumulative_energy_map.at<double>(row - 1, max(col - 1, 0));
				b = cumulative_energy_map.at<double>(row - 1, col);
				c = cumulative_energy_map.at<double>(row - 1, min(col + 1, colsize - 1));

				cumulative_energy_map.at<double>(row, col) = energy_image.at<double>(row, col) + std::min(a, min(b, c));
			}
		}
	}
	else if (seam_direction == HORIZONTAL) {
		for (int col = 1; col < colsize; col++) {
			for (int row = 0; row < rowsize; row++) {
				a = cumulative_energy_map.at<double>(max(row - 1, 0), col - 1);
				b = cumulative_energy_map.at<double>(row, col - 1);
				c = cumulative_energy_map.at<double>(min(row + 1, rowsize - 1), col - 1);

				cumulative_energy_map.at<double>(row, col) = energy_image.at<double>(row, col) + std::min(a, min(b, c));
			}
		}
	}

	// create and show the newly created cumulative energy map converting map into color (similar to MATLAB's imagesc())
	if (demo) {
		Mat color_cumulative_energy_map;
		double Cmin;
		double Cmax;
		cv::minMaxLoc(cumulative_energy_map, &Cmin, &Cmax);
		float scale = 255.0 / (Cmax - Cmin);
		cumulative_energy_map.convertTo(color_cumulative_energy_map, CV_8UC1, scale);
		applyColorMap(color_cumulative_energy_map, color_cumulative_energy_map, cv::COLORMAP_JET); 

		namedWindow("Cumulative Energy Map", CV_WINDOW_AUTOSIZE); imshow("Cumulative Energy Map", color_cumulative_energy_map);
	}

	// calculate time taken
	clock_t end = clock();
	cumulative_energy_map_time += ((float)end - (float)start) / CLOCKS_PER_SEC;

	return cumulative_energy_map;
}

vector<int> findOptimalSeam(Mat &cumulative_energy_map, SeamDirection seam_direction) {
	clock_t start = clock();
	double a, b, c;
	int offset = 0;
	vector<int> path;
	double min_val, max_val;
	Point min_pt, max_pt;

	// get the number of rows and columns in the cumulative energy map
	int rowsize = cumulative_energy_map.rows;
	int colsize = cumulative_energy_map.cols;

	if (seam_direction == VERTICAL) { 
		// copy the data from the last row of the cumulative energy map
		Mat row = cumulative_energy_map.row(rowsize - 1); 

		// get min and max values and locations
		minMaxLoc(row, &min_val, &max_val, &min_pt, &max_pt); 

		// initialize the path vector
		path.resize(rowsize);
		int min_index = min_pt.x;
		path[rowsize - 1] = min_index;

		// starting from the bottom, look at the three adjacent pixels above current pixel, choose the minimum of those and add to the path
		for (int i = rowsize - 2; i >= 0; i--) { 
			a = cumulative_energy_map.at<double>(i, max(min_index - 1, 0));
			b = cumulative_energy_map.at<double>(i, min_index);
			c = cumulative_energy_map.at<double>(i, min(min_index + 1, colsize - 1));

			if (min(a, b) > c) { 
				offset = 1;
			}
			else if (min(a, c) > b) { 
				offset = 0;
			}
			else if (min(b, c) > a) { 
				offset = -1;
			}

			min_index += offset;
			min_index = min(max(min_index, 0), colsize - 1); // take care of edge cases 
			path[i] = min_index; // record the location of each point of seam
		}
	}

	else if (seam_direction == HORIZONTAL) {
		// copy the data from the last column of the cumulative energy map
		Mat col = cumulative_energy_map.col(colsize - 1);

		// get min and max values and locations
		minMaxLoc(col, &min_val, &max_val, &min_pt, &max_pt);

		// initialize the path vector
		path.resize(colsize);
		int min_index = min_pt.y;
		path[colsize - 1] = min_index;

		// starting from the right, look at the three adjacent pixels to the left of current pixel, choose the minimum of those and add to the path
		for (int i = colsize - 2; i >= 0; i--) {
			a = cumulative_energy_map.at<double>(max(min_index - 1, 0), i);
			b = cumulative_energy_map.at<double>(min_index, i);
			c = cumulative_energy_map.at<double>(min(min_index + 1, rowsize - 1), i);

			if (min(a, b) > c) {
				offset = 1;
			}
			else if (min(a, c) > b) {
				offset = 0;
			}
			else if (min(b, c) > a) {
				offset = -1;
			}

			min_index += offset;
			min_index = min(max(min_index, 0), rowsize - 1); // take care of edge cases
			path[i] = min_index;
		}
	}

	// calculate time taken
	clock_t end = clock();
	find_seam_time += ((float)end - (float)start) / CLOCKS_PER_SEC;

	return path;
}

void modifyOffsetMap(vector<int> path, SeamDirection seam_direction) {
	clock_t start = clock();
	int rowsize = offset_energy_map.rows;
	int colsize = offset_energy_map.cols;

	for (int i = 0; i < path.size(); i++) {
		int intersection_x, intersection_y;
		if (seam_direction == VERTICAL) { 
			intersection_x = i; 
			intersection_y = path[i];
		}
		else if (seam_direction == HORIZONTAL) {
			intersection_x = path[i];
			intersection_y = i;
		}
		// the intersection point with a straight line
		if (offset_energy_map.at<Vec2d>(intersection_x, intersection_y)[1] == 100) { 
			offset_energy_map.at<Vec2d>(intersection_x, intersection_y)[0] += 200;
			//split two channels
			vector<Mat>channels;
			split(offset_energy_map, channels);
			Mat offset_value = channels.at(0);
			// 7x7-pixel region
			int left_up_x = max(intersection_x - 3, 0); 
			int left_up_y = max(intersection_y - 3, 0);
			if (intersection_x + 3 >= rowsize) {
				left_up_x = rowsize - 7;
			}
			if (intersection_y + 3 >= colsize) {
				left_up_y = colsize - 7;
			}
			// note that the x and y in a Rect are contrary to that in a Mat
			Mat neighborhood = offset_value(Rect(left_up_y, left_up_x, 7, 7));
			GaussianBlur(neighborhood, neighborhood, Size(7, 7), 0, 0);
			neighborhood.copyTo(neighborhood);
			merge(channels, offset_energy_map);
		}
	}
	clock_t end = clock();
	modify_offset_map_time += ((float)end - (float)start) / CLOCKS_PER_SEC;
}

void reduce(Mat &image, vector<int> path, SeamDirection seam_direction) {
	clock_t start = clock();

	// get the number of rows and columns in the image
	int rowsize = image.rows;
	int colsize = image.cols;

	// create a 1x1x3 dummy matrix to add onto the tail of a new row to maintain image dimensions and mark for deletion
	Mat dummy(1, 1, CV_8UC3, Vec3b(0, 0, 0));
	Mat offset_dummy(1, 1, CV_64FC2, Scalar(0, 0));

	if (seam_direction == VERTICAL) { // reduce the width 
		for (int i = 0; i < rowsize; i++) {
			// take all pixels to the left and right of marked pixel and store them in appropriate subrow variables
			Mat new_row;
			Mat lower = image.rowRange(i, i + 1).colRange(0, path[i]); 
			Mat upper = image.rowRange(i, i + 1).colRange(path[i] + 1, colsize); 

			// merge the two subrows and dummy matrix/pixel into a full row
			if (!lower.empty() && !upper.empty()) {
				hconcat(lower, upper, new_row); 
				hconcat(new_row, dummy, new_row);
			}
			else {
				if (lower.empty()) { // the left-most pixel
					hconcat(upper, dummy, new_row);
				}
				else if (upper.empty()) { //the right-most pixel
					hconcat(lower, dummy, new_row);
				}
			}
			// take the newly formed row and place it into the original image
			new_row.copyTo(image.row(i));

			// remove the seam in the offset energy map
			if (straight_line_pre || faces_detect) {  
				Mat offset_new_row;
				Mat offset_lower = offset_energy_map.rowRange(i, i + 1).colRange(0, path[i]);
				Mat offset_upper = offset_energy_map.rowRange(i, i + 1).colRange(path[i] + 1, colsize);

				if (!lower.empty() && !upper.empty()) {
					hconcat(offset_lower, offset_upper, offset_new_row);
					hconcat(offset_new_row, offset_dummy, offset_new_row);
				}
				else {
					if (lower.empty()) { 
						hconcat(offset_upper, offset_dummy, offset_new_row);
					}
					else if (upper.empty()) { 
						hconcat(offset_lower, offset_dummy, offset_new_row);
					}
				}
				offset_new_row.copyTo(offset_energy_map.row(i));
			}
		}
		// clip the right-most side of the image 
		image = image.colRange(0, colsize - 1);

		if (straight_line_pre || faces_detect) {
			offset_energy_map = offset_energy_map.colRange(0, colsize - 1);
		}
	}
	else if (seam_direction == HORIZONTAL) { // reduce the height 
		for (int i = 0; i < colsize; i++) {
			// take all pixels to the top and bottom of marked pixel and store the in appropriate subcolumn variables
			Mat new_col;
			Mat lower = image.colRange(i, i + 1).rowRange(0, path[i]);
			Mat upper = image.colRange(i, i + 1).rowRange(path[i] + 1, rowsize);

			// merge the two subcolumns and dummy matrix/pixel into a full row
			if (!lower.empty() && !upper.empty()) {
				vconcat(lower, upper, new_col); 
				vconcat(new_col, dummy, new_col);
			}
			else {
				if (lower.empty()) {
					vconcat(upper, dummy, new_col);
				}
				else if (upper.empty()) {
					vconcat(lower, dummy, new_col);
				}
			}
			// take the newly formed column and place it into the original image
			new_col.copyTo(image.col(i));

			if (straight_line_pre || faces_detect) {
				Mat offset_new_col;
				Mat offset_lower = offset_energy_map.colRange(i, i + 1).rowRange(0, path[i]);
				Mat offset_upper = offset_energy_map.colRange(i, i + 1).rowRange(path[i] + 1, rowsize);

				// merge the two subcolumns and dummy matrix/pixel into a full row
				if (!lower.empty() && !upper.empty()) {
					vconcat(offset_lower, offset_upper, offset_new_col); 
					vconcat(offset_new_col, offset_dummy, offset_new_col);
				}
				else {
					if (lower.empty()) {
						vconcat(offset_upper, offset_dummy, offset_new_col);
					}
					else if (upper.empty()) {
						vconcat(offset_lower, offset_dummy, offset_new_col);
					}
				}
				// take the newly formed column and place it into the original image
				offset_new_col.copyTo(offset_energy_map.col(i));
			}
		}
		// clip the bottom-most side of the image
		image = image.rowRange(0, rowsize - 1);

		if (straight_line_pre || faces_detect) {
			offset_energy_map = offset_energy_map.rowRange(0, rowsize - 1);
		}
	}

	if (demo) {
		namedWindow("Reduced Image", CV_WINDOW_AUTOSIZE); imshow("Reduced Image", image);
	}

	// calculate time taken
	clock_t end = clock();
	reduce_time += ((float)end - (float)start) / CLOCKS_PER_SEC;
}

void showPath(Mat &energy_image, vector<int> path, SeamDirection seam_direction) {
	// loop through the image and change all pixels in the path to white
	if (seam_direction == VERTICAL) {
		for (int i = 0; i < energy_image.rows; i++) {
			energy_image.at<double>(i, path[i]) = 1;
		}
	}
	else if (seam_direction == HORIZONTAL) {
		for (int i = 0; i < energy_image.cols; i++) {
			energy_image.at<double>(path[i], i) = 1;
		}
	}

	// display the seam on top of the energy image
	namedWindow("Seam on Energy Image", CV_WINDOW_AUTOSIZE); imshow("Seam on Energy Image", energy_image);
}

// automatically choose a threshold for Hough Transform
int findthreshold(InputArray src, double rho, double theta) {
	Mat img = src.getMat();
	int i, j, threshold;
	double irho = 1 / rho;
	const uchar* image = img.ptr();
	int step = (int)img.step;
	int rowsize = img.rows;
	int colsize = img.cols;

	int max_rho = rowsize + colsize;
	int min_rho = -max_rho;
	int numangle = cvRound(CV_PI / theta);
	int numrho = cvRound(((max_rho - min_rho) + 1) / rho);

	Mat _accum = Mat::zeros((numangle + 2), (numrho + 2), CV_32SC1);
	AutoBuffer<double> _tabSin(numangle);
	AutoBuffer<double> _tabCos(numangle);
	int *accum = _accum.ptr<int>();
	double *tabSin = _tabSin.data(), *tabCos = _tabCos.data();

	// create sin and cos table
	double ang = static_cast<double>(0);
	for (int n = 0; n < numangle; ang += theta, n++) {
		tabSin[n] = sin(ang) * irho;
		tabCos[n] = cos(ang) * irho;
	}

	// find the maximum value in Hough space
	int maxcnt = 0;
	for (i = 0; i < rowsize; i++) {
		for (j = 0; j < colsize; j++) {
			if (image[i * step + j] != 0) {
				for (int n = 0; n < numangle; n++) {
					int r = cvRound(j * tabCos[n] + i * tabSin[n]);
					r += (numrho - 1) / 2;
					accum[(n + 1) * (numrho + 2) + r + 1]++;
					if (maxcnt < accum[(n + 1) * (numrho + 2) + r + 1]) {
						maxcnt = accum[(n + 1) * (numrho + 2) + r + 1];
					}
				}
			}
		}
	}
	threshold = 0.6 * maxcnt;
	return threshold;
}

void detectStraightLine() {
	dst.create(src.size(), src.type());
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	if (straight_line_manually) {
		namedWindow("Edge Map", WINDOW_AUTOSIZE);
		// manually choose the low threshold for Canny
		createTrackbar("Canny Min Threshold:", "Edge Map", &lowThreshold, max_lowThreshold, CannyThreshold);
		CannyThreshold(0, 0);
		waitKey(0);
		// manually choose the threshold for Hough
		createTrackbar("Hough Threshold:", "Edge Map", &HoughThre, 200, HoughThreshold);
		waitKey(0);
	}
	else { // automatically choose the threshold for Hough 
		blur(src_gray, detected_edges, Size(3, 3));
		Canny(detected_edges, detected_edges, 100, 200, 3);
		detected_straight_lines = src.clone();
		int threshold = findthreshold(detected_edges, 1, CV_PI / 180);
		HoughLinesP(detected_edges, lines, 1, CV_PI / 180, threshold, 50, 10);
		// draw the lines
		for (int i = 0; i < lines.size(); i++) {
			Vec4i l = lines[i];
			line(detected_straight_lines, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
		}
		imshow("Detected Lines(in red)", detected_straight_lines);
	}

	clock_t start = clock();
	int rowsize = src.rows;
	int colsize = src.cols;

	// find line pixels
	for (int i = 0; i < lines.size(); i++) {
		Vec4i l = lines[i];
		int x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
		int A = y2 - y1, B = x1 - x2, C = x2 * y1 - x1 * y2;
		double denominator = sqrt(A * A + B * B);

		int up = max(min(y1, y2) - 1, 0);
		int bottom = min(max(y1, y2) + 1, rowsize - 1);
		int left = max(min(x1, x2) - 1, 0);
		int right = min(max(x1, x2) + 1, colsize - 1);

		double dist_thre = 0.5, dist;
		for (int i = up; i <= bottom; i++) {
			for (int j = left; j <= right; j++) {
				// the distance between the point and the straight line
				dist = ((double)abs(A * j + B * i + C)) / denominator;
				if (dist <= dist_thre) {
					offset_energy_map.at<Vec2d>(i, j)[1] = 100; // considered as line pixel
				}
			}
		}
	}
	clock_t end = clock();
	detect_line_time += ((float)end - (float)start) / CLOCKS_PER_SEC;
}

void detectFaces(Mat &image) {
	int rowsize = offset_energy_map.rows;
	int colsize = offset_energy_map.cols;

	String cascadeFilePath;
	if (human_faces_detect) {  //detect human faces
		// the installation path of opencv
		cascadeFilePath = ".../opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml";
	}
	else {	// detect anime faces
		cascadeFilePath = ".../lbpcascade_animeface.xml"; // the installation path of this file
	}
	CascadeClassifier face_cascade;
	if (!face_cascade.load(cascadeFilePath)) {
		cout << "Could not load the haar data... Please check the file path of the cascade classifier..." << endl;
		return;
	}

	Mat gray_image;
	Mat src = image.clone();
	cvtColor(image, gray_image, COLOR_BGR2GRAY);
	equalizeHist(gray_image, gray_image);    
	imshow("Original Image", image);

	vector<Rect> faces;  // container for detected faces
	face_cascade.detectMultiScale(gray_image, faces, 1.1, 4, 0, Size(50, 50), Size(500, 500)); 
	for (int t = 0; t < faces.size(); t++) {
		// use a red rectangle to frame the face
		rectangle(src, faces[t], Scalar(0, 0, 255), 2, 8, 0);  
		
		for (int i = faces[t].x; i < faces[t].x + faces[t].width; i++) {
			for (int j = faces[t].y; j < faces[t].y + faces[t].height; j++) {
				offset_energy_map.at<Vec2d>(j, i)[0] += 50;
			}
		}
	}
	namedWindow("Detected Faces", CV_WINDOW_AUTOSIZE);
	imshow("Detected Faces", src);
	if (debug) {
		vector<Mat>channels;
		split(offset_energy_map, channels);
		Mat offset_value = channels.at(0);
		namedWindow("Offset Energy", CV_WINDOW_AUTOSIZE);
		imshow("Offset Energy", offset_value);
	}
}

void driver(Mat &image, SeamDirection seam_direction, int iterations) {
	clock_t start = clock();

	//namedWindow("Original Image", CV_WINDOW_AUTOSIZE); imshow("Original Image", image);
	if (faces_detect) {
		detectFaces(image);
	}
	if (straight_line_pre) {
		detectStraightLine();
	}

	// perform the specified number of reductions 
	for (int i = 0; i < iterations; i++) {
		Mat energy_image = createEnergyImage(image);
		Mat cumulative_energy_map = createCumulativeEnergyMap(energy_image, seam_direction);
		vector<int> path = findOptimalSeam(cumulative_energy_map, seam_direction);
		if (straight_line_pre) {
			modifyOffsetMap(path, seam_direction);
		}
		reduce(image, path, seam_direction);
		if (demo) {
			showPath(energy_image, path, seam_direction);
		}
	}

	// calculate and output time taken
	if (debug) {
		clock_t end = clock();
		float total_time = ((float)end - (float)start) / CLOCKS_PER_SEC;
		cout << "Final image size: " << image.rows << "x" << image.cols << endl;
		cout << "energy image time taken: "; cout << fixed; cout << setprecision(7); cout << energy_image_time << endl;
		cout << "cumulative energy map time taken: "; cout << fixed; cout << setprecision(7); cout << cumulative_energy_map_time << endl;
		cout << "find seam time taken: "; cout << fixed; cout << setprecision(7); cout << find_seam_time << endl;
		cout << "reduce time taken: "; cout << fixed; cout << setprecision(7); cout << reduce_time << endl;
		if (straight_line_pre) {
			cout << "detect straight line time taken: "; cout << fixed; cout << setprecision(7); cout << detect_line_time << endl;
			cout << "modify offset map time taken: "; cout << fixed; cout << setprecision(7); cout << modify_offset_map_time << endl;
		}
		cout << "total time taken: "; cout << fixed; cout << setprecision(7); cout << total_time << endl;
	}

	if (demo && straight_line_pre) {
		vector<Mat>channels;
		split(offset_energy_map, channels);
		Mat offset_value = channels.at(0);
		namedWindow("Offset Energy", CV_WINDOW_AUTOSIZE); imshow("Offset Energy", offset_value);
	}
	namedWindow("Reduced Image", CV_WINDOW_AUTOSIZE); 
	imshow("Reduced Image", image); 
	imwrite("result.jpg", image);
	return;
}

int main() {
	string filename, reduce_direction, width_height, s_iterations;
	SeamDirection seam_direction;
	int iterations;

	cout << "Please enter a filename: ";
	cin >> filename;

	Mat image = imread(filename);
	if (image.empty()) {
		cout << "Unable to load image, please try again." << endl;
		exit(EXIT_FAILURE);
	}

	cout << "Reduce height or reduce width? (Height:1 | Width:0): ";
	cin >> reduce_direction;

	if (reduce_direction == "0" || reduce_direction == "1") {
		if (reduce_direction == "0") {
			width_height = "width";
			seam_direction = VERTICAL;
		}
		else if (reduce_direction == "1") {
			width_height = "height";
			seam_direction = HORIZONTAL;
		}
	}
	else {
		cout << "Invalid choice, please re-run and try again" << endl;
		return 0;
	}

	cout << "Reduce " << width_height << " how many times? ";
	cin >> s_iterations;

	iterations = stoi(s_iterations);
	int rowsize = image.rows;
	int colsize = image.cols;

	// check that inputted number of iterations doesn't exceed the image size
	if (seam_direction == VERTICAL) {
		if (iterations > colsize) {
			cout << "Input is greater than image's width, please try again." << endl;
			return 0;
		}
	}
	else if (seam_direction == HORIZONTAL) {
		if (iterations > rowsize) {
			cout << "Input is greater than image's height, please try again." << endl;
			return 0;
		}
	}

	cout << "Detect faces? (Yes:1 | No:0): ";
	cin >> faces_detect;
	if (faces_detect) {
		cout << "Detect human faces or anime faces? (Human faces:1 | Anime faces:0): ";
		cin >> human_faces_detect;
	}
	cout << "Detect straight lines? (Yes:1 | No:0): ";
	cin >> straight_line_pre;
	if (straight_line_pre) {
		cout << "Detect straight lines manually or automatically? (Manually:1 | Automatically:0): ";
		cin >> straight_line_manually;
	}

	demo = false; // true;
	debug = false; // true;
 
	src = image.clone();	
	if (straight_line_pre || faces_detect) {
		offset_energy_map = Mat(rowsize, colsize, CV_64FC2, Scalar(0, 0));
	}
	driver(image, seam_direction, iterations);

	waitKey(0);
	return 0;
}