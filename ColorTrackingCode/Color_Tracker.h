#pragma once
#ifndef COLOR_TRACKER_H_
#define COLOR_TRACKER_H_
#include "opencv2/core/core.hpp"
#include <vector>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include "sps.h"
#include "imgproc/imgproc.hpp"
#include <iostream>


using namespace std;
using namespace cv;

struct Feature_Color_HSV
{
	int bin1 = 16;//(0-2pai)*180/pai///0-360
	int bin2 = 16;//(0-1)*100///0-100;
	int bin3 = 5;//0-255
	float step1 = 256.0 / bin1;
	float step2 = 256.0 / bin2;
	float step3 = 256.0 / bin3;
	vector<float> color_hist;///uniquness -1 to 1;
	vector<float> color_number;
};

class Color_Tracker
{
public:
	Color_Tracker();
	~Color_Tracker();

public:
	bool voting=false;
	float surround_window_size_ = 0.5;// surrounding image size
	float search_window_size_ = 1.2;// surrounding image size

	float larger_size_noml = 120;//normalize the larger size of the target into 150 pixel. 

	float stader_ = 50;

	float ratio_;
	float representation_s_;
	Size normalized_enviroment_size_;
private:
	Mat shpae_mat_;
	bool CHECK_PINPUT = 1;
	Feature_Color_HSV feature_space_s_, feature_space_l_, feature_space_d_;// the feature space 
	Mat current_frame_, next_frame_, sub_img_ ;
	Rect current_rect_, next_rect_, sub_target_box_;
	int const BGR = 1;
	int const HSV = 2;
		int color_format_ = HSV; 
	//int color_format_ = BGR;

	int size_;
	float step1_, step2_, step3_, bin1_, bin2_, bin3_;
	Rect model_box_, enviroment_box_;

	vector<Scalar> characters_;//used to recode scale and motion changes.
	Scalar mean_characters_, stader_character_aviation_;
	int H = 13;
	float alpha_ = 1;
	float gama_ = 0.2;
	double const M_PI = 3.14159265358979323846;

public:
	
	void Color_Tracker::SetColorHistogram(int color_format, int bin1, int bin2, int bin3 = 1);
	double Color_Tracker::Track(Mat frame, Rect& result_box, Rect true_box, Mat& show_r=Mat());
	double Color_Tracker::Initial(Mat frame, Rect target_box);
	bool Color_Tracker::Track(Mat frame, Rect& result_box, Mat& show_r);
	double Color_Tracker::Track_s(Mat frame, Rect& result_box, Rect true_box);
	bool Color_Tracker::Update(Mat frame, Rect& target_box);
private:
	float Color_Tracker::CompareTwoHistograms(vector<float> h1, vector<float> h2);
	float Color_Tracker::CompareTwoHistograms_r(vector<float> h1, vector<float> h2);
	bool Color_Tracker::GeneratEnviromentBox(Mat& image, Rect& last_target_box, Rect& enviroment_box, float ENVIRO_WINDWO_SIZE);
	double Color_Tracker::InitColorHist(Mat& image, Rect target_box, Rect model_box, Rect enviroment_box, Feature_Color_HSV& s_feature_cells, Feature_Color_HSV& l_feature_cells);
	double Color_Tracker::InitColorHist_shape(Mat& image, Mat target_marsk, Rect target_box, Rect model_box, Feature_Color_HSV& s_feature_cells, Feature_Color_HSV& l_feature_cells,  bool do_need_check, Point move = Point(0, 0));
	double Color_Tracker::InitColorHist_rect(Mat& image, Rect target_box, Rect model_box, Rect enviroment_box, Feature_Color_HSV& s_feature_cells, Feature_Color_HSV& l_feature_cells, bool do_need_check);

	bool Color_Tracker::Update(Mat frame, Mat& target_marsk);
	bool Color_Tracker::Update_shape(Mat frame, Rect& target_box, Mat& target_marsk);
	bool Color_Tracker::NormalizeEnviroment(Mat& enviroment_img, Mat& normalized_img, float& ratio);

	float Color_Tracker::TargetSegmentation_morphological(Mat frame, Rect& target_box, Mat& shape_marsk);
	/////////////////////////////////////

	Point2f Color_Tracker::BoxCenter(Rect box)
	{
		return Point2f((box.x + box.width / 2.0), box.y + box.height / 2.0);
	}
	double Color_Tracker::GaussianFunction(double u, double s, double x)
	{
		double ee = (1 / (s * sqrt(2 * M_PI))) * exp((-0.5) * pow((x - u) / s, 2.0));
		return ee;
	}
	double Color_Tracker::RectOverlap(const Rect& box1, const Rect& box2)
	{
		if (box1.x > box2.x + box2.width) { return 0.0; }
		if (box1.y > box2.y + box2.height) { return 0.0; }
		if (box1.x + box1.width < box2.x) { return 0.0; }
		if (box1.y + box1.height < box2.y) { return 0.0; }

		float colInt = min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x);
		float rowInt = min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y);

		float intersection = colInt * rowInt;
		float area1 = box1.width*box1.height;
		float area2 = box2.width*box2.height;
		return intersection / (area1 + area2 - intersection);
	}

	double  Color_Tracker::PointDistance(Point2f p1, Point2f p2)
	{
		return (sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y)));
	}
	double  Color_Tracker::PointDistance(Point p1, Point p2)
	{
		return (sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y)));
	}
	void Color_Tracker::morphOps(Mat &thresh)
	{

		//create structuring element that will be used to "dilate" and "erode" image.
		//the element chosen here is a 3px by 3px rectangle

		Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
		//dilate with larger element so make sure object is nicely visible
		Mat dilateElement = getStructuringElement(MORPH_RECT, Size(4, 4));
		
		erode(thresh, thresh, erodeElement);
		dilate(thresh, thresh, dilateElement);
		erode(thresh, thresh, erodeElement);
		dilate(thresh, thresh, dilateElement);
	//	dilate(thresh, thresh, dilateElement);

		//

	}

};
#endif