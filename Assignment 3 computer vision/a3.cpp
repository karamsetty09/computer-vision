#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/flann/miniflann.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <cctype>
#include <iterator>
#include <stdio.h>
#include <sstream>

#include <iostream>
#include <string>

#include <opencv2/nonfree/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>

#include<opencv2/video/background_segm.hpp>

using namespace cv;
using namespace std;

//Matrix Variables
Mat frame, est_bg_frame, finalframe;
Mat maskMOG2;

//Pointer Variables
Ptr<BackgroundSubtractor> pMOG2;

//Integer Variables
int keyboard;

//Function Declaration
void processVideo(char* videoFileName);

int main(int argc, char* argv[])
{
	//When No Arguments Specified, Display Help on How to execute the program.
	if (argc != 2)
	{
		cout << "Help:\ncountMovingObj" << endl;
		cout << "\n\Consider below example," << endl;
		cout << "countMovingObj.exe videofilename.avi" << endl;
	}

	//Create Background Subtractor. Used MOG2 Method with no Shadows.
	pMOG2 = new BackgroundSubtractorMOG2(0, 48, false); //No History. Threshold Value to 48. (Optimum value found based on test cases with 16, 32 and 64. 

														//When video file is given as argument.
	if (argc == 2)
	{
		//Call function.
		processVideo(argv[1]);
	}

	//Destroy all windows. 
	destroyAllWindows();
	return EXIT_SUCCESS;
}

void processVideo(char* videoFileName)
{
	//Capture frames from input video. 
	VideoCapture capture(videoFileName);

	//If file is not a video.
	if (!capture.isOpened())
	{
		//Display relavent message. 
		cout << "Unable to open video file: " << videoFileName << endl;
		exit(EXIT_FAILURE);
	}

	//While loop, until q or esc keys are pressed.
	while ((char)keyboard != 'q' && (char)keyboard != 27)
	{
		//Check for end of video frame.
		if (!capture.read(frame))
		{
			cout << "Unable to read next frame." << endl;
			cout << "End of Video." << endl;
			cout << "Exiting.." << endl;
			exit(EXIT_FAILURE);
		}

		//MOG2 method. 
		pMOG2 -> operator()(frame, maskMOG2, 0.1); //frame is compared with background frame. Result stored in maskMOG2. 
		pMOG2->getBackgroundImage(est_bg_frame); //capture background frame. 

												 //Steps to display frame count on the input video. For reference only. Not asked in the requirement. 
		stringstream ss;
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20), cv::Scalar(255, 255, 255), -1);
		ss << capture.get(CV_CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

		//Perform morph operations on maskMOG2 frame. 
		Mat thresh, dilated, open, eroded, close;

		//create two elements, to use different sizes. 
		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		Mat element2 = getStructuringElement(MORPH_RECT, Size(6, 6));

		//perform blur. Increases the point size.
		cv::blur(maskMOG2, thresh, Size(5, 5));

		//Perform Close operation. Helps in filling some part of the body. 
		morphologyEx(thresh, close, MORPH_CLOSE, element);
		threshold(close, thresh, 3, 255, THRESH_BINARY); //Threshold to convert grayscale to binary. 
		dilate(thresh, dilated, element2); //Dilate white pixels. 
		morphologyEx(dilated, open, MORPH_OPEN, element2); //Open operation. Extends white pixels further. 

														   //create dummy matrix to find number of objects. 
		cv::Mat obj;
		open.copyTo(obj); //Store final morphed frame to obj frame. 

						  //matrix needed to store 8 bit image for finding contour.
		cv::Mat obj_8u;
		obj.convertTo(obj_8u, CV_8U); //convert to 8bit. 

									  //find number of contours. 
		std::vector<std::vector<cv::Point>>contours;
		cv::findContours(obj_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		//size of contours gives number of seperate objects in frame. 
		int nobj = contours.size();

		//mask color frame to binary morphed frame.
		cv::cvtColor(maskMOG2, maskMOG2, CV_GRAY2RGB);
		cv::cvtColor(open, open, CV_GRAY2BGR);
		bitwise_and(frame, open, finalframe);

		//Matrix display of images in output.
		Mat hupper, hlower;
		hconcat(frame, est_bg_frame, hupper); //create upper layer. Contains original frame and estimated background frame. 
		hconcat(maskMOG2, finalframe, hlower); //create lower layer. Contains moving pixel and color detected object. 

		Mat vertical;
		vconcat(hupper, hlower, vertical); //create final display layer. Vertically adds the two horizontal layers. 

		namedWindow("countMovingObjs", CV_WINDOW_AUTOSIZE);
		imshow("countMovingObjs", vertical); //display all four frames to output. 

											 //display frame number and number of objects found in the frame. 
		std::cout << "Frame: " << frameNumberString << " - Objects: " << nobj << std::endl;

		keyboard = waitKey(10);
	}
	capture.release();
}
