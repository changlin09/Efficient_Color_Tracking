#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_
///global variables for the target and background model 
#include "opencv2/core/core.hpp"
#include <vector>
#include <string>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
struct Color_Histogram
{
	int h_bin = 20;//(0-2pai)*180/pai///0-360
	int s_bin = 20;//(0-1)*100///0-100;
	int v_bin = 5;//0-255
	float v_step = 256.0 / v_bin;
	float s_step = 256.0 / s_bin;
	float h_step = 256.0 / h_bin;
	vector<float> color_hist;
};

using namespace std;
using namespace cv;
double const M_PI = 3.14159265358979323846;

bool Bgr2Hsi(Mat src, Mat& hsi_mat);
void MouseBox(int event, int x, int y, int flags, void *param);
bool GetBoxByMouse(Mat& inputimage, Rect& target_rect);
bool GeneratEnviromentBox(Mat& image, Rect& last_target_box, Rect& enviroment_box, float ENVIRO_WINDWO_SIZE);
float TestHistogramBelong(vector<float> h1, vector<float> h2);
Point2f BoxCenter(Rect box);
double GaussianFunction(double u, double s, double x);
double DetectTarget(Mat hsi_image, Color_Histogram& target_hist,  Rect& detected_target_box, int lost_time);
double RectOverlap(const Rect& box1, const Rect& box2);
float CompareTwoHistograms(vector<float> h1, vector<float> h2);
double InitializationColorModels(Mat input_image, Rect target_box, Color_Histogram& long_h, Color_Histogram&  short_h);
void GenerateLongAndShortHistogramModel(Mat hsi, Rect target_box, Color_Histogram& long_h, Color_Histogram&  short_h);


bool Bgr2Hsi(Mat src, Mat& hsi_mat)
{
	if (src.empty())
	{
		cerr << "Error: Loading image" << endl;
		return false;
	}
	Mat hsi(src.rows, src.cols, src.type());

	float r, g, b, h, s, in;

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			b = src.at<Vec3b>(i, j)[0];
			g = src.at<Vec3b>(i, j)[1];
			r = src.at<Vec3b>(i, j)[2];

			in = (b + g + r) / 3;

			int min_val = 0;
			min_val = std::min(r, std::min(b, g));

			s = 1 - 3 * (min_val / (b + g + r));
			if (s < 0.00001)
			{
				s = 0;
			}
			else if (s > 0.99999){
				s = 1;
			}

			if (s != 0)
			{
				h = 0.5 * ((r - g) + (r - b)) / sqrt(((r - g)*(r - g)) + ((r - b)*(g - b)));
				h = acos(h);

				if (b <= g)
				{
					h = h;
				}
				else{
					h = ((360 * 3.14159265) / 180.0) - h;
				}
			}
			else
				h = 0;

			hsi.at<Vec3b>(i, j)[0] = (h * 180) / 3.14159265;
			hsi.at<Vec3b>(i, j)[1] = s * 100;
			hsi.at<Vec3b>(i, j)[2] = in;
		}
	}
	hsi_mat = hsi;
	return true;
}
void MouseBox(int event, int x, int y, int flags, void *param)
{
	Rect* initial_box = (Rect*)param;
	switch (event)
	{
		//case CV_EVENT_MOUSEMOVE:
		//	//if (drawing_box)
		//	{
		//		initial_box->width = x - initial_box->x;
		//		initial_box->height = y - initial_box->y;
		//	}
		//	break;
	case CV_EVENT_LBUTTONDOWN:
		//drawing_box = true;
		initial_box->x = x;
		initial_box->y = y;
		initial_box->width = 0;
		initial_box->height = 0;
		break;
	case CV_EVENT_LBUTTONUP:
		initial_box->width = x - initial_box->x;
		initial_box->height = y - initial_box->y;
		//drawing_box = false;
		if (initial_box->width < 0)
		{
			initial_box->x += initial_box->width;
			initial_box->width *= -1;
		}
		if (initial_box->height < 0)
		{
			initial_box->y += initial_box->height;
			initial_box->height *= -1;
		}
		break;
	}
}
bool GetBoxByMouse(Mat& inputimage, Rect& target_rect)
{
	Rect initial_box = Rect(0, 0, 0, 0);
	int MIN_TARGET_SIZE = 5;
	bool is_get_box = false;
	Mat temporary_image;
	inputimage.copyTo(temporary_image);
	cv::namedWindow("Initial_Window", 1);
	cvSetMouseCallback("Initial_Window", MouseBox, &initial_box);
	while (!is_get_box)
	{
		imshow("Initial_Window", temporary_image);
		cv::waitKey(10);


		if (min(initial_box.width, initial_box.height) > MIN_TARGET_SIZE)
		{
			//cout << "Bounding box too small, try again." << endl;
			is_get_box = true;
		}
	}
	{
		rectangle(temporary_image, initial_box, Scalar(255, 0, 5), 1);
		imshow("Initial_Window", temporary_image);
		cv::waitKey(1);
		cvSetMouseCallback("Initial_Window", NULL, NULL);
		destroyWindow("Initial_Window");
		target_rect = initial_box;
		return true;
	}

}
bool GeneratEnviromentBox(Mat& image, Rect& last_target_box, Rect& enviroment_box, float ENVIRO_WINDWO_SIZE)
{
	if (!image.empty() && last_target_box.x > 0 && 0 && last_target_box.y > 0 && last_target_box.width + last_target_box.x < image.cols && last_target_box.height + last_target_box.y < image.rows)
	{
		printf("wrong inputs in findbestedgenearby\n");
		return false;
	}
	int image_cols = image.cols;
	int image_rows = image.rows;
	enviroment_box.x = max(0, int(last_target_box.x - (ENVIRO_WINDWO_SIZE - 1) / 2 * last_target_box.width));
	enviroment_box.y = max(0, int(last_target_box.y - (ENVIRO_WINDWO_SIZE - 1) / 2 * last_target_box.height));

	if (enviroment_box.x + ENVIRO_WINDWO_SIZE*last_target_box.width > image_cols)
		enviroment_box.width = image_cols - enviroment_box.x;
	else
		enviroment_box.width = int(ENVIRO_WINDWO_SIZE*last_target_box.width);

	if (enviroment_box.y + ENVIRO_WINDWO_SIZE*last_target_box.height > image_rows)
		enviroment_box.height = image_rows - enviroment_box.y;
	else
		enviroment_box.height = int(ENVIRO_WINDWO_SIZE*last_target_box.height);

	Point sp_shift = Point(enviroment_box.x, enviroment_box.y);

	if (enviroment_box.x > 0 && enviroment_box.y > 0 && enviroment_box.width + enviroment_box.x < image_cols && enviroment_box.height + enviroment_box.y < image_rows)
		return true;
	else
		return false;
}
Point2f BoxCenter(Rect box)
{
	return Point2f((box.x + box.width / 2.0), box.y + box.height / 2.0);
}
double GaussianFunction(double u, double s, double x)
{
	double ee = (1 / (s * sqrt(2 * M_PI))) * exp((-0.5) * pow((x - u) / s, 2.0));
	return ee;
}
double RectOverlap(const Rect& box1, const Rect& box2)
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
double InitializationColorModels(Mat input_image, Rect target_box, Color_Histogram& long_h, Color_Histogram&  short_h)
{
	if (input_image.empty() || target_box.x < 0 || target_box.y<0 || target_box.x + target_box.width>input_image.cols || target_box.y + target_box.height>input_image.rows)
	{
		printf("wrong input in InitializationColorModels\n");
		return -1;
	}
	double model_score = 0;
	Mat hsi;
	cvtColor(input_image, hsi, CV_BGR2HSV);
//	Bgr2Hsi(input_image, hsi);
	Color_Histogram hist;
	GenerateLongAndShortHistogramModel(hsi, target_box, long_h, short_h);
	float inside = 0;
	float outside = 0;
	double check_score = 0;
	Mat color_confident_mat = Mat(hsi.size(), CV_8UC1, Scalar(0));
	for (int i = 0; i < hsi.cols; i++)
	{
		for (int j = 0; j < hsi.rows; j++)
		{
			Scalar color = Scalar(hsi.at<Vec3b>(j, i));
			int h = color[0] / long_h.h_step;
			int s = color[1] / long_h.s_step;
			int v = color[2] / long_h.v_step;
			int id = h + s*long_h.h_bin + v*long_h.s_bin*long_h.h_bin;
			double score = short_h.color_hist[id];

			if (score > 0.00 && target_box.contains(Point(j, i)))
			{
				inside++;//-10;//
				
			}
			if (score > 0 && !target_box.contains(Point(j, i)))
			{
				inside--;// = -0.5;//-10;//
				
			}
			if (score > 0)
			color_confident_mat.at<uchar>(Point(j, i)) = 255;
			else
			color_confident_mat.at<uchar>(Point(j, i)) = 0;
		}
	}
	imshow("color_confident_mat", color_confident_mat);
	waitKey(1);
	check_score = inside / target_box.area();
	return check_score;
}
void GenerateLongAndShortHistogramModel(Mat hsi, Rect target_box, Color_Histogram& long_h, Color_Histogram&  short_h)
{
	if (hsi.empty() || target_box.x < 0 || target_box.y<0 || target_box.x + target_box.width>hsi.cols || target_box.y + target_box.height>hsi.rows)
	{
		printf("wrong input in GenerateLongAndShortHistogramModel\n");
		return;
	}
	int count_p = 0;// to normalize positive bins for long time.
	int count = hsi.cols*hsi.rows;// to normalize all bins for short time.
	Color_Histogram l_h;
	Color_Histogram s_h;
	int size = l_h.h_bin * l_h.s_bin * l_h.v_bin;// since long and short models are the same size histogram
	l_h.color_hist.resize(size, 0);
	s_h.color_hist.resize(size, 0);
	for (int i = 0; i < hsi.cols; i++)
		for (int j = 0; j < hsi.rows; j++)
		{
			Point pp = Point(i, j);
			Vec3b color = hsi.at<Vec3b>(pp);
			int h = color[0] / l_h.h_step;
			int s = color[1] / l_h.s_step;
			int v = color[2] / l_h.v_step;
			int id = h + s*l_h.h_bin + v*l_h.s_bin*l_h.h_bin;
			if (target_box.contains(pp))
			{
				s_h.color_hist[id]++;
			}
			else
			{
				s_h.color_hist[id]--;
			}
		}
	for (int i = 0; i < size; i++)
	{
		if (s_h.color_hist[i] > 0)
		{
			l_h.color_hist[i] = s_h.color_hist[i];
			count_p = count_p + s_h.color_hist[i];
		}
	}
	if (count_p == 0)
			count_p = 1; 
	float count_n = count - count_p;
	for (int i = 0; i < size; i++)
	{
		if (s_h.color_hist[i] > 0)
		{
			s_h.color_hist[i] = float(s_h.color_hist[i]) / count_p;
		}
		else	
			s_h.color_hist[i] = float(s_h.color_hist[i]) / count_n;
		l_h.color_hist[i] = float(l_h.color_hist[i]) / count_p;
		
		
	}

	long_h = l_h;
	short_h = s_h;
}
float TestHistogramBelong(vector<float> h1, vector<float> h2)
{
	if (h1.size() != h2.size())
	{
		printf("wrong input in CompareHistogram\n");
		return -1;
	}
	float belong = 0;
	bool is_find = false;
	float count = 0;
	for (int i = 0; i < h2.size(); i++)
	{
		if (h2[i] >= 0.05)
		{
			count = count + h2[i];
			if (h1[i] > 0.0005)
			{
				belong = belong + h2[i];
				is_find = true;
			}
			else
				if (h1[i] <= 0.0005)
					belong = belong - h2[i];;
		}
	}
	//if (!is_find)
	//	return  0;
	return belong / count;
}
float CompareTwoHistograms(vector<float> h1, vector<float> h2)
{
	if (h1.size() != h2.size())
	{
		printf("wrong input in CompareHistogram\n");
		return -1;
	}
	bool is_find = false;
	float target_num = 0;
	float candidacy_num = 0;
	double toto = 0;
	float number = 0;
	float thred1 = 0;
	float thred2 = 0;
	for (int i = 0; i < h1.size(); i++)
	{
		if (h1[i] > 0)
		{
			toto = toto + h1[i];
			number++;
		}
	}
	thred1 = toto / number;
	toto = 0;
	number = 0;
	for (int i = 0; i < h2.size(); i++)
	{
		if (h2[i] > 0)
		{
			toto = toto + h2[i];
			number++;
		}
	}
	thred2 = toto / number;
	for (int i = 0; i < h2.size(); i++)
	{
		if (h1[i] > thred1)
		{
			target_num++;
			if (h2[i] > 0.005)
			{
				candidacy_num++;
				is_find = true;
			}			
			//else //if (h2[i] > 0.005)
			//{
			//	candidacy_num--;
			//	is_find = true;
			//}
		}
	}
	if (!is_find)
		candidacy_num = 0;
	return candidacy_num / target_num;
}
double DetectTarget(Mat hsi_image, Color_Histogram& long_h, Rect& detected_target_box, int lost_time)
{
	Mat check_mat;
	hsi_image.copyTo(check_mat);
	Rect target_box;
	rectangle(hsi_image, detected_target_box, Scalar(225));
	imshow("hsi_image", hsi_image);
	waitKey(0);
	double toto = 0;
	float number = 0;
	for (int i = 0; i < long_h.color_hist.size(); i++)
	{
		if (long_h.color_hist[i] > 0)
		{
			toto = toto + long_h.color_hist[i];
			number++;
		}
	}
	double thred1 = toto / number;
	Mat color_confident_mat_show = Mat(hsi_image.size(), CV_8UC1);
	Mat color_confident_mat = Mat(hsi_image.size(), CV_32FC1, Scalar(0));
	for (int i = 0; i < hsi_image.cols; i++)
	{
		for (int j = 0; j < hsi_image.rows; j++)
		{
			Scalar color = Scalar(hsi_image.at<Vec3b>(j, i));
			int h = color[0] / long_h.h_step;
			int s = color[1] / long_h.s_step;
			int v = color[2] / long_h.v_step;
			int id = h + s*long_h.h_bin + v*long_h.s_bin*long_h.h_bin;
			double score_l = long_h.color_hist[id];
			if (score_l > thred1)
			{
				score_l = 1;//-10;//
				color_confident_mat_show.at<uchar>(j, i) = 255;
			}
			else
			{
				score_l = -0.5;
				color_confident_mat_show.at<uchar>(j, i) = 0;
			}
			color_confident_mat.at<float>(j, i) = score_l;
		}
	}

	Point2f orign_center = BoxCenter(detected_target_box);
	Rect temp_target_box;
	///generate integra_mat
	Rect tracked_box_e_s;
	Rect tracked_box_e_l;
	double normalized_score = 0;
	double max_n_score = 0;
	double scaled_score = 0;
	Rect best_boxes;

	////////////////////////
	/////tracking target by  color
	///////////////////////////////
	Mat color_integre_mat_l = Mat(color_confident_mat.size(), CV_32FC1, Scalar(0));
	double length = (detected_target_box.width + detected_target_box.height) / 2.0;
	Mat color_integre_mat = Mat(hsi_image.size(), CV_32FC1, Scalar(0));
	integral(color_confident_mat, color_integre_mat);
	float scale[9] = { 0.5, 0.650, 0.75, 0.86, 1, 1.1, 1.2, 1.35, 1.5 };
	Rect bb_box;
	double bb = -DBL_MAX;
	Point2f p1 = BoxCenter(detected_target_box);
	for (int x_s = 0; x_s < 9; x_s++)
		for (int y_s = 0; y_s < 9; y_s++)
		{
			double area_tation = scale[x_s] * scale[y_s];
			float width = detected_target_box.width*scale[x_s];
			float height = detected_target_box.height*scale[y_s];
			for (int x = 0; x < hsi_image.cols - width; x = x + 2)
				for (int y = 0; y < hsi_image.rows - height; y = y + 2)
				{
					Point2f p2 = Point2f(x + width / 2, y + height / 2);
					float dist = norm(p1 - p2);
					if (lost_time != 0)
					{
						if (dist > length*(lost_time + 7))
							continue;
					}
					Point point1 = Point(x, y);
					Point point2 = Point(x + width, y);
					Point point3 = Point(x, y + height);
					Point point4 = Point(x + width, y + height);

					double score = color_integre_mat.at<double>(point1)+color_integre_mat.at<double>(point4)-color_integre_mat.at<double>(point2)-color_integre_mat.at<double>(point3);
					double distance;
					if (lost_time == 0)
						distance = 1;

					else
						distance = exp((dist / (length*(lost_time*0.5 + 1))) * (-2));
					if (score > 0)
						score = distance*score;
					else
						score = score / distance;
					if (score > bb)
					{
						bb = score;
						bb_box = Rect(x, y, width, height);
					}
				}

		}


	rectangle(color_confident_mat_show, detected_target_box, Scalar(225));
	detected_target_box = bb_box;
	//	rectangle(color_confident_mat, bb_box, Scalar(125));
	imshow("d_color_confident_mat", color_confident_mat_show);
	waitKey(1);
	rectangle(color_confident_mat, detected_target_box, Scalar(225));
	imshow("c_ident_mat", color_confident_mat);
	waitKey(1);
 	Color_Histogram current_hist_s, current_hist_l;
	Size normalized_enviroment_size = Size(150, 150);
	float window_resize_ration = 2.5;// surrounding image size
	Rect enviroment_box;
	GeneratEnviromentBox(hsi_image, detected_target_box, enviroment_box, window_resize_ration);
	Point shift = Point(enviroment_box.x, enviroment_box.y);
	Mat enviroment_image = hsi_image(enviroment_box);
	float x_resize_ration, y_resize_ration;
	x_resize_ration = float(enviroment_image.cols) / normalized_enviroment_size.width;
	y_resize_ration = float(enviroment_image.rows) / normalized_enviroment_size.height;
	resize(enviroment_image, enviroment_image, normalized_enviroment_size);
	Rect last_target_box_e = Rect((detected_target_box.x - shift.x) / x_resize_ration, (detected_target_box.y - shift.y) / y_resize_ration, detected_target_box.width / x_resize_ration, detected_target_box.height / y_resize_ration);

	//generate target and background model

	InitializationColorModels(enviroment_image, last_target_box_e, current_hist_l, current_hist_s);

	double score = CompareTwoHistograms(long_h.color_hist, current_hist_l.color_hist);
	return score;
}




#endif