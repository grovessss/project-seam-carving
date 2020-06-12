// based on https://github.com/davidshower/seam-carving

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include <opencv2/saliency.hpp>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <time.h>
#include <stdio.h>
#include <cmath>

using namespace cv;
using namespace std;

enum SeamDirection { VERTICAL, HORIZONTAL };

float energy_image_time = 0;
float cumulative_energy_map_time = 0;
float find_seam_time = 0;
float reduce_time = 0;

bool demo;
bool debug;

enum energyMethod { SOBEL, SCHARR, LAPLACIAN, ENTROPY, Saliency_FT, SCHARR_FT, SCHARR_FT_Depth };

/* variants */
energyMethod method = Saliency_FT;
#define FORWARD_ENERGY
#define INPUT_DEPTHx  /* this is only defined if method is SCHARR_FT_Depth */
#define SAVE_MAP

/* energy functions */
Mat getSobel(Mat image)
{
    Mat image_gray, grad_x, grad_y, abs_grad_x, abs_grad_y, grad, energy_image;

    // convert to grayscale
    cvtColor(image, image_gray, CV_BGR2GRAY);

    // use Sobel to calculate the gradient of the image in the x and y direction
    Sobel(image_gray, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    Sobel(image_gray, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);

    // convert gradients to abosulte versions of themselves
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    
    // total gradient (approx)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    
    // convert the default values to double precision
    grad.convertTo(energy_image, CV_64F, 1.0/255.0);
    
    return energy_image;
}

Mat getScharr(Mat image)
{
    Mat image_gray, grad_x, grad_y, abs_grad_x, abs_grad_y, grad, energy_image;

    cvtColor(image, image_gray, CV_BGR2GRAY);
    Scharr(image_gray, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
    Scharr(image_gray, grad_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
            
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    grad.convertTo(energy_image, CV_64F, 1.0/255.0); // normalize
    
    return energy_image;
}

Mat getLaplacian(Mat image)
{
    Mat image_gray, grad, abs_grad, energy_image;
    cvtColor(image, image_gray, CV_BGR2GRAY);
    Laplacian(image_gray, grad, CV_16S, 3); 

    convertScaleAbs(grad, abs_grad);
    abs_grad.convertTo(energy_image, CV_64F, 1.0/255.0); // normalize

    return energy_image;
}

Mat getLocalEntropy(Mat image)
{
    int rowsize = image.rows;
    int colsize = image.cols;
    Mat image_gray, energy_image = Mat::zeros(rowsize, colsize, CV_64F);
    cvtColor(image, image_gray, CV_BGR2GRAY);

    int window_size = 9; // consider a window_size*window_size window around each pixel
    int offset = (window_size - 1)/2;
    //int bin_size = 10;
    for (int i = 0; i < rowsize; i++) // calculate entropy for each pixel
    {
        for (int j = 0; j < colsize; j++)
        {
            int start_x = max(i - offset, 0); int end_x = min(i + offset, rowsize - 1);
            int start_y = max(j - offset, 0); int end_y = min(j + offset, colsize - 1);
            int pixel_num = (end_y - start_y + 1)*(end_x - start_x + 1);
            int hist[256] = {0};
            for(int ii = start_x; ii <= end_x; ii++)
            {
                for(int jj = start_y; jj <= end_y; jj++)
                {
                    hist[(int)image_gray.at<uchar>(ii, jj)]++;
                }
            }
            for(int k = 0; k <= 255; k++)
            {
                if (hist[k])
                {
                    double pr = (double)hist[k]/pixel_num;
                    energy_image.at<double>(i, j) += -pr*log(pr)/log(2);
                }
            }
        }
    }
    return energy_image;
}

Mat getFT(Mat image)
{
    int rowsize = image.rows;
    int colsize = image.cols;

    Mat image_blur, image_Lab, image_Lab_blurred;
    Mat energy_image = Mat::zeros(rowsize, colsize, CV_64F);
    GaussianBlur(image, image_blur, Size(3,3), 0, 0, BORDER_DEFAULT); // get blurred image
    cvtColor(image, image_Lab, CV_BGR2Lab); // get image in Lab color space
    cvtColor(image_blur, image_Lab_blurred, CV_BGR2Lab);

    Scalar avg_Lab = mean(image_Lab);
    double avg_L = avg_Lab.val[0]*100/255; // average Lab pixel
    double avg_a = avg_Lab.val[1];
    double avg_b = avg_Lab.val[2];

    for (int i = 0; i < rowsize; i++) // calculate Euclidean distance in Lab color space
    { 
        for (int j = 0; j < colsize; j++)
        {
            Vec3b vec_3 = image_Lab_blurred.at<Vec3b>(i, j);
            double cur_L = vec_3[0]*100/255;
            double cur_a = vec_3[1];
            double cur_b = vec_3[2];
            double dist = (cur_L-avg_L)*(cur_L-avg_L)+(cur_a-avg_a)*(cur_a-avg_a)+(cur_b-avg_b)*(cur_b-avg_b);
            energy_image.at<double>(i, j) = dist;
        }
    }

    // FT image is not normalized

    /*  //  unblock if dilated
    Mat energy_image_ = Mat::zeros(rowsize, colsize, CV_64F), temp;
    double min_energy, max_energy;
    minMaxLoc(energy_image, &min_energy, &max_energy);
    double scale = 255/max_energy;
    energy_image.convertTo(energy_image_, CV_8U, scale);
    Mat structureElement = getStructuringElement(MORPH_CROSS, Size(3, 3), Point(-1, -1));
    dilate(energy_image_, energy_image_, structureElement, Point(-1, -1), 3);    
    energy_image_.convertTo(energy_image, CV_64F, 1/scale);
    //addWeighted(energy_image, 0.5, energy_image_, 0.5, 0, energy_image);
    */
    
    return energy_image;
}

void carveVertical(Mat &image, vector<int> path); // carve off a vertical path in an image

Mat createEnergyImage(Mat &image, Mat &depth_image) {
    clock_t start = clock();
    int rowsize = image.rows;
    int colsize = image.cols;

    Mat image_blur, image_gray;
    Mat energy_image = Mat::zeros(rowsize, colsize, CV_64F);
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    // apply a gaussian blur to reduce noise
    GaussianBlur(image, image_blur, Size(3,3), 0, 0, BORDER_DEFAULT);
    
    switch(method)
    {
        case SOBEL:
        {
            energy_image = getSobel(image_blur);
            break;
        }
        case SCHARR:
        {
            energy_image = getScharr(image_blur);
            break;
        }
        case LAPLACIAN:
        {
            energy_image = getLaplacian(image_blur);
            break;
        }
        case ENTROPY:
        {
            energy_image = getLocalEntropy(image);
            break;
        }
        case Saliency_FT:
        {
            energy_image = getFT(image);
            break;
        }
        case SCHARR_FT:
        {
            Mat energy_image1, energy_image2;
            energy_image1 = getScharr(image_blur);
            energy_image2 = getFT(image);

            double min_energy, max_energy; // normailize FT map
            minMaxLoc(energy_image2, &min_energy, &max_energy);
            energy_image2.convertTo(energy_image2, CV_64F, 1.0/max_energy);

            addWeighted(energy_image1, 0.5, energy_image2, 0.5, 0, energy_image);
            /*
            for (int i = 0; i < rowsize; i++){
                for (int j = 0; j < colsize; j++){
                    energy_image.at<double>(i, j) = energy_image2.at<double>(i, j)+energy_image1.at<double>(i, j);
                }
            }*/
            break;
        }
        case SCHARR_FT_Depth:
        {

#ifndef INPUT_DEPTH
    cout << "Depth image needed." << endl;
    exit(EXIT_FAILURE);
#endif
            Mat energy_image1, energy_image2, energy_image3;
            energy_image1 = getScharr(image_blur);
            energy_image2 = getFT(image);

            double min_energy, max_energy; // normalize FT map
            minMaxLoc(energy_image2, &min_energy, &max_energy);
            energy_image2.convertTo(energy_image2, CV_64F, 1.0/max_energy);

            // get standard deviation of the depth image
            depth_image.convertTo(energy_image3, CV_64F, 1.0);
            Mat mat_mean, mat_stddev;  
            meanStdDev(energy_image3, mat_mean, mat_stddev);
            double depth_stddev = mat_stddev.at<double>(0,0);  

            // calculate max_stddev with the brightest and darkest pixel in the depth image
            minMaxLoc(energy_image3, &min_energy, &max_energy);
            double max_stddev = 0.98*(max_energy - min_energy)/2; // absolute max std dev is not possible
            // alpha = depth_stddev/max_stddev
            double alpha = depth_stddev/max_stddev;
            
            energy_image3.convertTo(energy_image3, CV_64F, 1.0/255); // normalize depth image

            // combine gradient, FT & depth image, FT:depth = alpha:1-alpha
            addWeighted(energy_image2, alpha, energy_image3, 1-alpha, 0, energy_image2);
            addWeighted(energy_image1, 0.5, energy_image2, 0.5, 0, energy_image);
            
            //depth_image.convertTo(energy_image, CV_64F, 1.0);
        }
    } // end switch

    // create and show the newly created energy image
    if (demo) {
        namedWindow("Energy Image", CV_WINDOW_AUTOSIZE); imshow("Energy Image", energy_image);
    }
    
    // calculate time taken
    clock_t end = clock();
    energy_image_time += ((float)end - (float)start) / CLOCKS_PER_SEC;
    
    return energy_image;
}

Mat createCumulativeEnergyMap(Mat &energy_image, Mat &image, SeamDirection seam_direction) {
    clock_t start = clock();
    double a,b,c,C_U = 0,C_R = 0,C_L = 0; //forward energy
    Mat image_gray;
    cvtColor(image, image_gray, CV_BGR2GRAY);
    
    // get the numbers of rows and columns in the image
    int rowsize = energy_image.rows;
    int colsize = energy_image.cols;
    
    // initialize the map with zeros
    Mat cumulative_energy_map = Mat(rowsize, colsize, CV_64F, double(0));
    
    // copy the first row
    if (seam_direction == VERTICAL) energy_image.row(0).copyTo(cumulative_energy_map.row(0));
    else if (seam_direction == HORIZONTAL) energy_image.col(0).copyTo(cumulative_energy_map.col(0));

    double min_energy, max_energy; 
    minMaxLoc(energy_image, &min_energy, &max_energy);

    Mat image_blur, image_Lab_blurred;
    GaussianBlur(image, image_blur, Size(3,3), 0, 0, BORDER_DEFAULT); // get blurred image
    cvtColor(image_blur, image_Lab_blurred, CV_BGR2Lab); // get Lab image
    
    // take the minimum of the three neighbors and add to total, this creates a running sum which is used to determine the lowest energy path
    if (seam_direction == VERTICAL) {
        for (int row = 1; row < rowsize; row++) {
            for (int col = 0; col < colsize; col++) {
                C_U = 0; C_R = 0; C_L = 0;
                double l_L,l_a,l_b,u_L,u_a,u_b,r_L,r_a,r_b;

#ifdef FORWARD_ENERGY
                switch(method)
                {
                case SOBEL:case SCHARR:case LAPLACIAN:
                {
                    // forward energy for gradient energy map
                    if (col - 1 >= 0 && col + 1 < colsize) // three kinds of pixels
                    {
                        C_U = (abs((int)image_gray.at<uchar>(row, col - 1)-(int)image_gray.at<uchar>(row, col + 1)));
                        C_R = (abs((int)image_gray.at<uchar>(row - 1, col)-(int)image_gray.at<uchar>(row, col + 1))) + C_U;
                        C_L = (abs((int)image_gray.at<uchar>(row - 1, col)-(int)image_gray.at<uchar>(row, col - 1))) + C_U;
                    }
                    else if (col - 1 < 0)
                    {
                        C_U = (abs((int)image_gray.at<uchar>(row, col)-(int)image_gray.at<uchar>(row, col + 1)));
                        C_R = (abs((int)image_gray.at<uchar>(row - 1, col)-(int)image_gray.at<uchar>(row, col + 1))) + C_U;
                        C_L = (abs((int)image_gray.at<uchar>(row - 1, col)-(int)image_gray.at<uchar>(row, col))) + C_U;
                    }
                    else
                    {
                        C_U = (abs((int)image_gray.at<uchar>(row, col - 1)-(int)image_gray.at<uchar>(row, col)));
                        C_R = (abs((int)image_gray.at<uchar>(row - 1, col)-(int)image_gray.at<uchar>(row, col))) + C_U;
                        C_L = (abs((int)image_gray.at<uchar>(row - 1, col)-(int)image_gray.at<uchar>(row, col - 1))) + C_U;
                    }
                    break;
                }
                case Saliency_FT:
                {
                    // forward energy for FT saliency map (.ptr faster than .at)
                    uchar* row_=image_Lab_blurred.ptr<uchar>(row);
                    uchar* row__=image_Lab_blurred.ptr<uchar>(row-1);
                    u_L = row__[col*3]*100/255; u_a = row__[col*3+1]; u_b = row__[col*3+2];

                    // calculate Euclidean distance, three kinds of pixels
                    if (col - 1 >= 0 && col + 1 < colsize)
                    {
                        l_L = row_[(col-1)*3]*100/255; l_a = row_[(col-1)*3+1]; l_b = row_[(col-1)*3+2];
                        r_L = row_[(col+1)*3]*100/255; r_a = row_[(col+1)*3+1]; r_b = row_[(col+1)*3+2];
                    }
                    else if (col - 1 < 0)
                    {
                        l_L = row_[(col)*3]*100/255; l_a = row_[(col)*3+1]; l_b = row_[(col)*3+2];
                        r_L = row_[(col+1)*3]*100/255; r_a = row_[(col+1)*3+1]; r_b = row_[(col+1)*3+2];
                    }
                    else
                    {
                        l_L = row_[(col-1)*3]*100/255; l_a = row_[(col-1)*3+1]; l_b = row_[(col-1)*3+2];
                        r_L = row_[(col)*3]*100/255; r_a = row_[(col)*3+1]; r_b = row_[(col)*3+2];
                    }
                    C_U = (l_L-r_L)*(l_L-r_L)+(l_a-r_a)*(l_a-r_a)+(l_b-r_b)*(l_b-r_b);
                    C_L = (l_L-u_L)*(l_L-u_L)+(l_a-u_a)*(l_a-u_a)+(l_b-u_b)*(l_b-u_b) + C_U;
                    C_R = (r_L-u_L)*(r_L-u_L)+(r_a-u_a)*(r_a-u_a)+(r_b-u_b)*(r_b-u_b) + C_U;
                    break;
                }
                default: break;
                } // end switch
#endif          
                a = cumulative_energy_map.at<double>(row - 1, max(col - 1, 0)) + C_L;
                b = cumulative_energy_map.at<double>(row - 1, col) + C_U;
                c = cumulative_energy_map.at<double>(row - 1, min(col + 1, colsize - 1)) + C_R;
                
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
    double a,b,c;
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
            
            if (min(a,b) > c) {
                offset = 1;
            }
            else if (min(a,c) > b) {
                offset = 0;
            }
            else if (min(b, c) > a) {
                offset = -1;
            }
            
            min_index += offset;
            min_index = min(max(min_index, 0), colsize - 1); // take care of edge cases
            path[i] = min_index;
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
            
            if (min(a,b) > c) {
                offset = 1;
            }
            else if (min(a,c) > b) {
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

void reduce(Mat &image, vector<int> path, SeamDirection seam_direction, Mat &depth_image) {
    clock_t start = clock();
    
    // get the number of rows and columns in the image
    int rowsize = image.rows;
    int colsize = image.cols;
    
    // create a 1x1x3 dummy matrix to add onto the tail of a new row to maintain image dimensions and mark for deletion
    Mat dummy(1, 1, CV_8UC3, Vec3b(0, 0, 0));
#ifdef INPUT_DEPTH
    cvtColor(depth_image, depth_image, CV_GRAY2BGR);
#endif
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
                if (lower.empty()) {
                    hconcat(upper, dummy, new_row);
                }
                else if (upper.empty()) {
                    hconcat(lower, dummy, new_row);
                }
            }
            // take the newly formed row and place it into the original image
            new_row.copyTo(image.row(i));
        }
        // clip the right-most side of the image
        image = image.colRange(0, colsize - 1);
#ifdef INPUT_DEPTH
        carveVertical(depth_image, path)
#endif
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
        }
        // clip the bottom-most side of the image
        image = image.rowRange(0, rowsize - 1);
    }

#ifdef INPUT_DEPTH
    cvtColor(depth_image, depth_image, CV_BGR2GRAY);
#endif
   
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
            energy_image.at<double>(i,path[i]) = 1;
        }
    }
    else if (seam_direction == HORIZONTAL) {
        for (int i = 0; i < energy_image.cols; i++) {
            energy_image.at<double>(path[i],i) = 1;
        }
    }
    
    // display the seam on top of the energy image
    //namedWindow("Seam on Energy Image", CV_WINDOW_AUTOSIZE); imshow("Seam on Energy Image", energy_image);
}

void driver(Mat &image, SeamDirection seam_direction, int iterations, string filename, string savename, Mat &depth_image) {
    clock_t start = clock();
    
    Mat paths = image;
    Mat energy_image, cumulative_energy_map;
    vector<int> path;
    // perform the specified number of reductions
    for (int i = 0; i < iterations; i++) {
        if (method == Saliency_FT && i != 0) // FT method does not need to update energy map
        {
            Mat energy_image_3 = Mat::zeros(energy_image.rows,energy_image.cols,CV_64FC3);
            vector<Mat> channels;
            for (int j = 0; j < 3; j++)
                channels.push_back(energy_image);
            merge(channels,energy_image_3);

            carveVertical(energy_image_3, path);

            split(energy_image_3, channels);
            energy_image = channels[0];//

            cumulative_energy_map = createCumulativeEnergyMap(energy_image, image, seam_direction);
            path = findOptimalSeam(cumulative_energy_map, seam_direction);
            reduce(image, path, seam_direction, depth_image);
        }
        else
        {
            energy_image = createEnergyImage(image, depth_image);
            cumulative_energy_map = createCumulativeEnergyMap(energy_image, image, seam_direction);
            path = findOptimalSeam(cumulative_energy_map, seam_direction);
            reduce(image, path, seam_direction, depth_image);
        }
        if (demo) {
            showPath(energy_image, path, seam_direction);
        }
#ifdef SAVE_MAP
        if (i == 0)
        {
            string _filename = filename.substr(0, filename.rfind("."));

            // save energy map
            Mat _energy_image;
            double min_energy, max_energy;
            minMaxLoc(energy_image, &min_energy, &max_energy); //cout<<"minmax"<<min_energy<<' '<<max_energy<<endl;
            energy_image.convertTo(_energy_image, CV_8UC1, 255.0/max_energy);
            
            string Map_filename = _filename + "_energy_map" + ".jpg";
            imwrite(Map_filename, _energy_image);
            
            // save cumulative energy map
            Mat color_cumulative_energy_map;
            double Cmin, Cmax;
            minMaxLoc(cumulative_energy_map, &Cmin, &Cmax);
            float scale = 255.0 / (Cmax - Cmin);
            cumulative_energy_map.convertTo(color_cumulative_energy_map, CV_8UC1, scale);
            applyColorMap(color_cumulative_energy_map, color_cumulative_energy_map, cv::COLORMAP_JET);
            
            string Map_filename_ = _filename + "_cumulative_energy_map" + ".jpg";
            imwrite(Map_filename_, color_cumulative_energy_map);
        }
#endif
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
        cout << "total time taken: "; cout << fixed; cout << setprecision(7); cout << total_time << endl;
    }
    
    imwrite(savename, image);
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

    Mat depth_image = image; string filename_depth;
#ifdef INPUT_DEPTH
    cout << "Please enter the filename of the depth image: ";
    cin >> filename_depth;
    depth_image = imread(filename_depth);
    if (depth_image.empty()) {
        cout << "Unable to load image, please try again." << endl;
        exit(EXIT_FAILURE);
    }
    if(depth_image.rows != image.rows || depth_image.cols != image.cols){
        cout << "Depth image not matched." << endl;
        exit(EXIT_FAILURE);
    }
    cvtColor(depth_image, depth_image, CV_BGR2GRAY);
#endif

    /* block to always reduce width (added energy functions can not be used for height reduction)
    cout << "Reduce width or reduce height? (0 to reduce width | 1 to reduce height): ";
    cin >> reduce_direction;
    */
    reduce_direction = "0";

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
    
    int rowsize = image.rows;
    int colsize = image.cols;

    cout << "Image is " << image.rows << " * " << image.cols <<"; "; //added
    cout << "Reduce " << width_height << " how many times? ";
    cin >> s_iterations;
    
    iterations = stoi(s_iterations);
    
    string savename; //added
    cout << "Please enter a name for the result file: ";
    cin >> savename;
    
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

    demo = false;
    debug = false;
    
    if (demo) iterations = 1;
    
    driver(image, seam_direction, iterations, filename, savename, depth_image); //added

    return 0;
}                                          

void carveVertical(Mat &image, vector<int> path)
{
    int rowsize = image.rows;
    int colsize = image.cols;
    Mat dummy(1, 1, CV_MAKETYPE(image.type(),3), Vec3b(0, 0, 0));

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
            if (lower.empty()) {
                hconcat(upper, dummy, new_row);
            }
            else if (upper.empty()) {
                hconcat(lower, dummy, new_row);
            }
        }
        // take the newly formed row and place it into the original image
        new_row.copyTo(image.row(i));
    }
    // clip the right-most side of the image
    image = image.colRange(0, colsize - 1);
}