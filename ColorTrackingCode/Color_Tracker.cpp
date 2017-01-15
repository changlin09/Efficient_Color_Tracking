#include "Color_Tracker.h"
#include < opencv2\opencv.hpp>
using namespace std;
using namespace cv;
Color_Tracker::Color_Tracker()
{
}


Color_Tracker::~Color_Tracker()
{
}
void Color_Tracker::SetColorHistogram(int color_format, int bin1, int bin2, int bin3)
{
	bin1_ = bin1;
	bin2_ = bin2;
	bin3_ = bin3;
	size_ = bin1*bin2*bin3;
	if (color_format == 1)
		color_format_ = BGR;
	else
		if (color_format == 2)
			color_format_ = HSV;
	{
		feature_space_s_.bin1 = bin1;
		feature_space_s_.bin2 = bin2;
		feature_space_s_.bin3 = bin3;

		feature_space_l_.bin1 = bin1;
		feature_space_l_.bin2 = bin2;
		feature_space_l_.bin3 = bin3;

		feature_space_d_.bin1 = bin1;
		feature_space_d_.bin2 = bin2;
		feature_space_d_.bin3 = bin3;
	}
	{
		step1_ = feature_space_s_.step1;
		step2_ = feature_space_s_.step2;
		step3_ = feature_space_s_.step3;
	}
}
double Color_Tracker::Initial(Mat frame, Rect target_box)
{
	if (frame.empty() || target_box.area() <= 0)
	{
		cout << "wrong input in Initial" << endl;
		return false;
	}
	current_rect_ = target_box;
	current_frame_ = frame;
	///initial parameters
	{
		bin1_ = feature_space_s_.bin1;
		bin2_ = feature_space_s_.bin2;
		bin3_ = feature_space_s_.bin3;
	}

	{
		step1_ = feature_space_s_.step1;
		step2_ = feature_space_s_.step2;
		step3_ = feature_space_s_.step3;
	}

	size_ = bin1_*bin2_*bin3_;
	GeneratEnviromentBox(frame, target_box, model_box_, surround_window_size_);
	GeneratEnviromentBox(frame, target_box, enviroment_box_, search_window_size_);

	double DistributionEstimation = InitColorHist(frame, target_box, model_box_, enviroment_box_, feature_space_s_, feature_space_l_);

                  	destroyAllWindows();
	return DistributionEstimation;

}

bool Color_Tracker::GeneratEnviromentBox(Mat& image, Rect& last_target_box, Rect& enviroment_box, float ENVIRO_WINDWO_SIZE)
{
	if (image.empty() || last_target_box.area() < 1)
	{
		printf("wrong input in InitializationColorModels\n");
		return -1;
	}
	int image_cols = image.cols;
	int image_rows = image.rows;
	int rang = (last_target_box.width + last_target_box.height) / 4.0 + 0.5;
	enviroment_box.x = max(0, int(last_target_box.x - (ENVIRO_WINDWO_SIZE)* rang));
	enviroment_box.y = max(0, int(last_target_box.y - (ENVIRO_WINDWO_SIZE)*rang));

	if (enviroment_box.x + ENVIRO_WINDWO_SIZE * 2 * rang + last_target_box.width > image_cols)
		enviroment_box.width = image_cols - enviroment_box.x;
	else
		enviroment_box.width = int(ENVIRO_WINDWO_SIZE * 2 * rang + last_target_box.width);

	if (enviroment_box.y + ENVIRO_WINDWO_SIZE * 2 * rang + last_target_box.height > image_rows)
		enviroment_box.height = image_rows - enviroment_box.y;
	else
		enviroment_box.height = int(ENVIRO_WINDWO_SIZE * 2 * rang + last_target_box.height);

	if (enviroment_box.x > 0 && enviroment_box.y > 0 && enviroment_box.width + enviroment_box.x < image_cols && enviroment_box.height + enviroment_box.y < image_rows)
		return true;
	else
		return false;
}

double Color_Tracker::InitColorHist(Mat& image, Rect target_box, Rect model_box, Rect enviroment_box, Feature_Color_HSV& s_feature_cells, Feature_Color_HSV& l_feature_cells)
{
	double t = (double)getTickCount();
	if (image.empty() || enviroment_box.x < 0 || enviroment_box.y<0 || enviroment_box.x + enviroment_box.width>image.cols || enviroment_box.y + enviroment_box.height>image.rows)
	{
		printf("wrong input in InitFeatureSpace\n");
		return -1;
	}
	if (target_box.area() < 1)
	{
		printf("wrong input in InitFeatureSpace\n");
		return -1;
	}

	Mat nor_image;
	float ratio;
	NormalizeEnviroment(image(enviroment_box), nor_image, ratio);
	ratio_ = ratio;
	///////////////

	Rect sub_model_box;
	sub_model_box.x = (model_box.x - enviroment_box.x)*ratio;
	sub_model_box.y = (model_box.y - enviroment_box.y)*ratio;
	sub_model_box.width = model_box.width*ratio;
	sub_model_box.height = model_box.height*ratio;

	Rect sub_target_box, sub_sub_target_box;
	sub_target_box.x = (target_box.x - enviroment_box.x)*ratio;
	sub_target_box.y = (target_box.y - enviroment_box.y)*ratio;
	sub_target_box.width = target_box.width*ratio;
	sub_target_box.height = target_box.height*ratio;

	sub_sub_target_box = Rect(sub_target_box.x - sub_model_box.x, sub_target_box.y - sub_model_box.y, sub_target_box.width, sub_target_box.height);
	Mat sub_img;
	if (color_format_ == HSV)
	{
		cvtColor(nor_image, nor_image, CV_BGR2HSV);
		sub_img = nor_image(sub_model_box);
	}
	else
		nor_image(sub_model_box).copyTo(sub_img);
	if (CHECK_PINPUT)
	{
		Mat sub_show;
		sub_img.copyTo(sub_show);
		rectangle(sub_show, sub_sub_target_box, Scalar(0, 255, 0));
		imshow("sub_show", sub_show);
		imshow("nor_image", nor_image);
	}
	int size = size_;
	l_feature_cells.color_hist.resize(size, 0);
	l_feature_cells.color_number.resize(size, 0);
	int channel = sub_img.channels();
	int elemet = sub_img.elemSize();
	int j_end = sub_img.cols*channel;
	for (int i = 0; i < sub_img.rows; i++)
	{
		uchar* data = sub_img.ptr<uchar>(i);
		int y = 0;
		for (int j = 0; j < j_end; j = j + elemet)
		{
			Point p = Point(y, i);
			int h = data[j] / step1_;
			int s = data[j + 1] / step2_;
			int v = data[j + 2] / step3_;
			int id = h + s*bin1_ + v*bin2_*bin1_;
			l_feature_cells.color_number[id]++;
			if (sub_sub_target_box.contains(p))
			{
				l_feature_cells.color_hist[id]++;
			}
			y++;
		}
	}

	for (int i = 0; i < size; i++)
	{
			float number2 = l_feature_cells.color_number[i];
			if (number2 <= 0)
			{
				l_feature_cells.color_hist[i] = 0;
			}
			else 
			{
				float mm = l_feature_cells.color_hist[i] / number2;
				l_feature_cells.color_hist[i] = mm;
			}
	}

	//////////////////
	///check
	////////
	float check_score=0;
	{
		j_end = nor_image.cols*channel;
		Mat color_confident_mat_l = Mat(nor_image.size(), CV_32FC1, Scalar(0));
		for (int i = 0; i < nor_image.rows; i++)
		{
			uchar* data = nor_image.ptr<uchar>(i);
			int y = 0;
			for (int j = 0; j < j_end; j = j + elemet)
			{
				int h = data[j] / step1_;
				int s = data[j + 1] / step2_;
				int v = data[j + 2] / step3_;
				int id = h + s*bin1_ + v*bin2_*bin1_;
				double score = l_feature_cells.color_hist[id];
				color_confident_mat_l.at<float>(i, y) = score;
				y++;
			}
		}
		if (bool aaiafdss_need_check = true)
		{
			double min, max;
			minMaxLoc(color_confident_mat_l, &min, &max);
			Mat show_mat = Mat(color_confident_mat_l.size(), CV_8U, Scalar(0));
			for (int i = 0; i < color_confident_mat_l.cols; i++)
				for (int j = 0; j < color_confident_mat_l.rows; j++)
				{
					float bbb = color_confident_mat_l.at<float>(j, i);
					if (bbb > 0.5)
					{
						show_mat.at<uchar>(j, i) = bbb / (max)* 255;
					}
				}
			imshow("l_first_appear", show_mat);
			waitKey(1);
		}
		
		{
			TargetSegmentation_morphological(color_confident_mat_l, sub_target_box, shpae_mat_);
			GaussianBlur(shpae_mat_, shpae_mat_, Size(5, 5), 0, 0);
			morphOps(shpae_mat_);
			//Mat dilateElement = getStructuringElement(MORPH_RECT, Size(5, 5));

			//dilate(shpae_mat_, shpae_mat_, dilateElement);

			Mat showe_;
			threshold(shpae_mat_, showe_, 0.5, 255, THRESH_BINARY);
			//imshow("shpae_mat_", showe_);
			//waitKey(1);
			//	Feature_Color_HSV space_s_n, space_l_n;
		
			check_score = InitColorHist_shape(nor_image, showe_, sub_target_box, sub_model_box, s_feature_cells, l_feature_cells, true);
		}
	}
	
	return check_score;

}
double Color_Tracker::InitColorHist_shape(Mat& image, Mat target_marsk, Rect target_box, Rect model_box, Feature_Color_HSV& s_feature_cells, Feature_Color_HSV& l_feature_cells, bool do_need_check, Point move)
{
	Mat sub_img = image;
	Rect sub_target_box = target_box;
	sub_img_ = sub_img;
	sub_target_box_ = sub_target_box;
	int size = size_;
	//	s_feature_cells.color_hist.clear();
	l_feature_cells.color_hist.clear();
	//	s_feature_cells.color_number.clear();
	l_feature_cells.color_number.clear();

	s_feature_cells.color_hist.resize(size, 0);
	l_feature_cells.color_hist.resize(size, 0);
	s_feature_cells.color_number.resize(size, 0);
	l_feature_cells.color_number.resize(size, 0);
	int channel = sub_img.channels();
	int elemet = sub_img.elemSize();
	int j_end = sub_img.cols*channel;
	if (CHECK_PINPUT)
	{
		Mat sub_show;
		sub_img.copyTo(sub_show);
		rectangle(sub_show, target_box, Scalar(0, 233, 23));
		rectangle(sub_show, model_box, Scalar(233, 233, 23));
		imshow("target_input", sub_show);
		imshow("target_marsk", target_marsk);
		waitKey(1);
		if (bool is_checking = true)
		{
			Mat sss;
			sub_img.copyTo(sss);
			for (int i = 0; i < sub_img.rows; i++)
			{
				uchar* data = sub_img.ptr<uchar>(i);
				int y = 0;
				for (int j = 0; j < j_end; j = j + elemet)
				{
					Point p = Point(y, i);
					Point c_p = p + move;
					y++;
					if (c_p.x<0 || c_p.y<0 || c_p.x>target_marsk.cols - 1 || c_p.y>target_marsk.rows - 1)
						continue;
					if (target_marsk.at<uchar>(c_p)>0)
					{
						Vec3b& color = sss.at<Vec3b>(p);
						color[0] = 0;
						color[1] = 0;
						color[2] = 255;;
					}

				}
			}
			imshow("ss", sss);
			waitKey(1);
		}

	}
	//	sub_image.copyTo(hsv_image);

	//generate feature sapce

	//	Feature_Color_HSV feature_space;

	for (int i = 0; i < sub_img.rows; i++)
	{
		uchar* data = sub_img.ptr<uchar>(i);
		int y = 0;
		for (int j = 0; j < j_end; j = j + elemet)
		{
			Point p = Point(y, i);
			y++;
			int h = data[j] / step1_;
			int s = data[j + 1] / step2_;
			int v = data[j + 2] / step3_;
			int id = h + s*bin1_ + v*bin2_*bin1_;
			s_feature_cells.color_number[id]++;
			if (model_box.contains(p))
			{
				l_feature_cells.color_number[id]++;
			}
			Point c_p = p + move;
			if (c_p.x<0 || c_p.y<0 || c_p.x>target_marsk.cols - 1 || c_p.y>target_marsk.rows - 1)
				continue;
			if (target_marsk.at<uchar>(c_p)>0)
			{
				s_feature_cells.color_hist[id]++;
				l_feature_cells.color_hist[id]++;
			}
			
		}
	}
	for (int i = 0; i < size; i++)
	{
		{
			float number = s_feature_cells.color_number[i];
			if (number != 0)
			{
				float mm = s_feature_cells.color_hist[i] / number;
				s_feature_cells.color_hist[i] = mm;
			}
			float number2 = l_feature_cells.color_number[i];
			if (number2 != 0)
			{
				float mm = l_feature_cells.color_hist[i] / number2;
				l_feature_cells.color_hist[i] = mm;
			}
			if (l_feature_cells.color_hist[i] > 1)
				l_feature_cells.color_hist[i] = 1;
		}
	}

	float check_score = 0;

	if (do_need_check)
	{
		float inside = 0;
		Mat color_confident_mat_s = Mat(sub_img.size(), CV_32FC1, Scalar(0));
		for (int i = 0; i < sub_img.rows; i++)
		{
			uchar* data = sub_img.ptr<uchar>(i);
			int y = 0;
			for (int j = 0; j < j_end; j = j + elemet)
			{
				int h = data[j] / step1_;
				int s = data[j + 1] / step2_;
				int v = data[j + 2] / step3_;
				int id = h + s*bin1_ + v*bin2_*bin1_;
				double score = s_feature_cells.color_hist[id];
				if (score > 0.50 && sub_target_box.contains(Point(y, i)))
				{
					inside++;//-10;//
				}
				if (score > 0.5 && !sub_target_box.contains(Point(y, i)))
				{
					inside--;// = -0.5;//-10;//
				}
				color_confident_mat_s.at<float>(i, y) = score;
				y++;
			}
		}
		if (bool aaiafdss_need_check = false)
		{
			double min, max;
			minMaxLoc(color_confident_mat_s, &min, &max);
			Mat show_mat = Mat(color_confident_mat_s.size(), CV_8U, Scalar(0));
			for (int i = 0; i < color_confident_mat_s.cols; i++)
				for (int j = 0; j < color_confident_mat_s.rows; j++)
				{
					float bbb = color_confident_mat_s.at<float>(j, i);
					if (bbb > 0.5)
					{
						show_mat.at<uchar>(j, i) = bbb / (max)* 255;
					}
				}
			imshow("s_initial_check_appear", show_mat);
			waitKey(1);
		}
		check_score = float(inside) / sub_target_box.area();
	}
	if (do_need_check)
	{
		Mat color_confident_mat_l = Mat(sub_img.size(), CV_32FC1, Scalar(0));
		for (int i = 0; i < sub_img.rows; i++)
		{
			uchar* data = sub_img.ptr<uchar>(i);
			int y = 0;
			for (int j = 0; j < j_end; j = j + elemet)
			{
				int h = data[j] / step1_;
				int s = data[j + 1] / step2_;
				int v = data[j + 2] / step3_;
				int id = h + s*bin1_ + v*bin2_*bin1_;
				double score = l_feature_cells.color_hist[id];
				color_confident_mat_l.at<float>(i, y) = score;
				y++;
			}
		}
		if (bool aaiafdss_need_check = false)
		{
			double min, max;
			minMaxLoc(color_confident_mat_l, &min, &max);
			Mat show_mat = Mat(color_confident_mat_l.size(), CV_8U, Scalar(0));
			for (int i = 0; i < color_confident_mat_l.cols; i++)
				for (int j = 0; j < color_confident_mat_l.rows; j++)
				{
					float bbb = color_confident_mat_l.at<float>(j, i);
					if (bbb > 0.5)
					{
						show_mat.at<uchar>(j, i) = bbb / (max)* 255;
					}
				}
			rectangle(show_mat, sub_target_box, Scalar(255, 255, 255));

			imshow("l_initial", show_mat);
			waitKey(1);
		}	
	}
	return check_score;
}
double Color_Tracker::InitColorHist_rect(Mat& image, Rect target_box, Rect model_box, Rect enviroment_box, Feature_Color_HSV& s_feature_cells, Feature_Color_HSV& l_feature_cells,  bool do_need_check)
{

	Mat nor_image;
	float ratio;
	NormalizeEnviroment(image(enviroment_box), nor_image, ratio);
	//	unique_ids_.clear();
	Rect sub_target_box;
	sub_target_box.x = (target_box.x - enviroment_box.x)*ratio;
	sub_target_box.y = (target_box.y - enviroment_box.y)*ratio;
	sub_target_box.width = target_box.width*ratio;
	sub_target_box.height = target_box.height*ratio;
	Rect sub_model_box;
	sub_model_box.x = (model_box.x - enviroment_box.x)*ratio;
	sub_model_box.y = (model_box.y - enviroment_box.y)*ratio;
	sub_model_box.width = model_box.width*ratio;
	sub_model_box.height = model_box.height*ratio;

	Mat sub_img;
	if (color_format_ == HSV)
	{
		cvtColor(nor_image, sub_img, CV_BGR2HSV);
	}
	else
		nor_image.copyTo(sub_img);

	sub_img_ = sub_img;
	sub_target_box_ = sub_target_box;
	int size = size_;

	if (CHECK_PINPUT)
	{
		Mat sub_show;
		sub_img.copyTo(sub_show);
		imshow("sub_show", sub_show);
	}

	s_feature_cells.color_hist.resize(size, 0);
	l_feature_cells.color_hist.resize(size, 0);
	s_feature_cells.color_number.resize(size, 0);
	l_feature_cells.color_number.resize(size, 0);
	int channel = sub_img.channels();
	int elemet = sub_img.elemSize();
	int j_end = sub_img.cols*channel;
	for (int i = 0; i < sub_img.rows; i++)
	{
		uchar* data = sub_img.ptr<uchar>(i);
		int y = 0;
		for (int j = 0; j < j_end; j = j + elemet)
		{
			Point p = Point(y, i);
			int h = data[j] / step1_;
			int s = data[j + 1] / step2_;
			int v = data[j + 2] / step3_;
			int id = h + s*bin1_ + v*bin2_*bin1_;
			s_feature_cells.color_number[id]++;
			if (sub_model_box.contains(p))
			{
				l_feature_cells.color_number[id]++;
			}
			if (sub_target_box.contains(p))
			{
				s_feature_cells.color_hist[id]++;
				l_feature_cells.color_hist[id]++;
			}
			y++;
		}
	}

	for (int i = 0; i < size; i++)
	{
		{
			float number = s_feature_cells.color_number[i];
			if (number != 0)
			{
				float mm = s_feature_cells.color_hist[i] / number;
				s_feature_cells.color_hist[i] = mm;
			}
			float number2 = l_feature_cells.color_number[i];
			if (number2 != 0)
			{
				float mm = l_feature_cells.color_hist[i] / number2;
				l_feature_cells.color_hist[i] = mm;
			}
		}
	}

	//////////////////
	///check
	////////

	float inside = 0;
	if (do_need_check)
	{
		//		Mat color_confident_mat = Mat(nor_image.size(), CV_8UC3);
		Mat color_confident_mat_s = Mat(nor_image.size(), CV_32FC1, Scalar(0));
		for (int i = 0; i < sub_img.rows; i++)
		{
			uchar* data = sub_img.ptr<uchar>(i);
			int y = 0;
			for (int j = 0; j < j_end; j = j + elemet)
			{
				int h = data[j] / step1_;
				int s = data[j + 1] / step2_;
				int v = data[j + 2] / step3_;
				int id = h + s*bin1_ + v*bin2_*bin1_;
				double score = s_feature_cells.color_hist[id];
				if (score > 0.50 && sub_target_box.contains(Point(y, i)))
				{
					inside++;//-10;//
				}
				if (score > 0.5 && !sub_target_box.contains(Point(y, i)))
				{
					inside--;// = -0.5;//-10;//
				}
				color_confident_mat_s.at<float>(i, y) = score;
				y++;
			}
		}
		if (bool aaiafdss_need_check = true)
		{
			double min, max;
			minMaxLoc(color_confident_mat_s, &min, &max);
			Mat show_mat = Mat(color_confident_mat_s.size(), CV_8U, Scalar(0));
			for (int i = 0; i < color_confident_mat_s.cols; i++)
				for (int j = 0; j < color_confident_mat_s.rows; j++)
				{
					float bbb = color_confident_mat_s.at<float>(j, i);
					if (bbb > 0.5)
					{
						show_mat.at<uchar>(j, i) = bbb / (max)* 255;
					}
				}
			rectangle(show_mat, sub_target_box, Scalar(255, 255, 255));
			imshow("t_check", show_mat);
			waitKey(1);
		}
	}
	if (do_need_check)
	{
		//		Mat color_confident_mat = Mat(nor_image.size(), CV_8UC3);
		Mat color_confident_mat_l = Mat(nor_image.size(), CV_32FC1, Scalar(0));
		for (int i = 0; i < sub_img.rows; i++)
		{
			uchar* data = sub_img.ptr<uchar>(i);
			int y = 0;
			for (int j = 0; j < j_end; j = j + elemet)
			{
				int h = data[j] / step1_;
				int s = data[j + 1] / step2_;
				int v = data[j + 2] / step3_;
				int id = h + s*bin1_ + v*bin2_*bin1_;
				//				double score = l_feature_cells.color_hist[id];
				double score = l_feature_cells.color_hist[id];
				color_confident_mat_l.at<float>(i, y) = score;

				y++;
			}
		}
		if (bool aaiafdss_need_check = true)
		{
			double min, max;
			minMaxLoc(color_confident_mat_l, &min, &max);
			Mat show_mat = Mat(color_confident_mat_l.size(), CV_8U, Scalar(0));
			for (int i = 0; i < color_confident_mat_l.cols; i++)
				for (int j = 0; j < color_confident_mat_l.rows; j++)
				{
					float bbb = color_confident_mat_l.at<float>(j, i);
					if (bbb > 0.5)
					{
						show_mat.at<uchar>(j, i) = bbb / (max)* 255;
					}
				}
			imshow("1_check_appear", show_mat);
			waitKey(1);
		}
		
	}
	float check_score = float(inside) / sub_target_box.area();
	return check_score;

}

double Color_Tracker::Track(Mat frame, Rect& result_box, Rect true_box, Mat& show_r)
{
	Rect previouse_target = current_rect_;
	if (frame.empty())
	{
		cout << "wrong input in Update" << endl;
		return false;
	}
	Mat sub_img = sub_img_;
	Rect sub_target_box = sub_target_box_;
	if (CHECK_PINPUT)
	{
		Mat sub_show;
		sub_img.copyTo(sub_show);
		rectangle(sub_show, sub_target_box, Scalar(0, 244, 0));
		imshow("enviroment_image", sub_show);
	}

	//////////////////////////////////////////////////////
	Mat appea_confident_mat = Mat(sub_img.size(), CV_32FC1, Scalar(0.0));
	Mat appea_confident_mat_l = Mat(sub_img.size(), CV_32FC1, Scalar(0.0));
	int channel = sub_img.channels();
	int elemet = sub_img.elemSize();
	int j_end = sub_img.cols*channel;
	for (int i = 0; i < sub_img.rows; i++)
	{
		uchar* data = sub_img.ptr<uchar>(i);
		int y = 0;
		for (int j = 0; j < j_end; j = j + elemet)
		{
			int h = data[j] / step1_;
			int s = data[j + 1] / step2_;
			int v = data[j + 2] / step3_;
			int id = h + s*bin1_ + v*bin2_*bin1_;
			double score = feature_space_s_.color_hist[id];
			double l_score = feature_space_l_.color_hist[id];
			appea_confident_mat_l.at<float>(i, y) = (l_score + score)/2.0;
			if (l_score <= 0)
				score = 0;			
			appea_confident_mat.at<float>(i, y) = score;
			y++;
		}
	}

	if (bool aaiafdss_need_check = true)
	{
		double min, max;
		minMaxLoc(appea_confident_mat_l, &min, &max);
		Mat show_mat = Mat(appea_confident_mat_l.size(), CV_8U, Scalar(0));
		for (int i = 0; i < appea_confident_mat_l.cols; i++)
			for (int j = 0; j < appea_confident_mat_l.rows; j++)
			{
				float bbb = appea_confident_mat_l.at<float>(j, i);
				if (bbb > 0.5)
				{
					show_mat.at<uchar>(j, i) = bbb / (max)* 255;
				}
			}
		imshow("l_appear", show_mat);
		waitKey(1);
	}
	if (bool aaiafdss_need_che0u09ck = true)
	{
		double min, max;
		minMaxLoc(appea_confident_mat, &min, &max);
		Mat show_mat = Mat(appea_confident_mat.size(), CV_8U, Scalar(0));
		for (int i = 0; i < appea_confident_mat.cols; i++)
			for (int j = 0; j < appea_confident_mat.rows; j++)
			{
				float bbb = appea_confident_mat.at<float>(j, i);
				if (bbb > 0.5)
				{
					show_mat.at<uchar>(j, i) = bbb / (max)* 255;
				}
			}
		imshow("s_appear", show_mat);
		waitKey(0);
	}

	////////////////////////
	/////tracking target by  color
	Rect short_box;
	if (bool is_need_short_map = true)
	{
		Point2f orign_center = BoxCenter(sub_target_box);

		Mat color_integre_mat = Mat(appea_confident_mat.size(), CV_32FC1, Scalar(0));
		integral(appea_confident_mat, color_integre_mat);
		double bb = -DBL_MAX;
		Rect best_box;
		{
			float width = sub_target_box.width*0.8;
			float height = sub_target_box.height*0.8;
			for (int x = 0; x < color_integre_mat.cols - width; x = x + 3)
				for (int y = 0; y < color_integre_mat.rows - height; y = y + 3)
				{
					Point point1 = Point(x, y);
					Point point2 = Point(x + width, y);
					Point point3 = Point(x, y + height);
					Point point4 = Point(x + width, y + height);
					double s_score = color_integre_mat.at<double>(point1)+color_integre_mat.at<double>(point4)-color_integre_mat.at<double>(point2)-color_integre_mat.at<double>(point3);
					Point2f current_center = Point2f(x + width / 2.0, y + height / 2.0);
					double move_dis = norm(current_center - orign_center);// *resize_ration;

					double aa = GaussianFunction(0, stader_, move_dis);
					s_score = s_score*aa;
					if (s_score > bb)
					{
						bb = s_score;
						best_box = Rect(x, y, width, height);
					}
				}
		}
		short_box = best_box;
	}
	float segmentation_s;
	Rect final_box = Rect(short_box.x - 0.1*sub_target_box.width, short_box.y - 0.1*sub_target_box.height, sub_target_box.width, sub_target_box.height);
	Mat marks;
	segmentation_s=TargetSegmentation_morphological(appea_confident_mat_l, final_box, marks);

	///////////////////////////
	if (0)
	{
		Mat results;
		sub_img.copyTo(results);
		rectangle(results, final_box, Scalar(233, 233, 0));
		imshow("results", results);
		waitKey(1);
	}

	Mat showe_;
	threshold(marks, showe_, 0.5, 255, THRESH_BINARY);
	//			imshow("shpae_mat_", showe_);
	////////////////////////////////////////////////////////
	if (0)
	{
		Rect enviroment_box;
		Rect model_box;
		GeneratEnviromentBox(frame, final_box, enviroment_box, search_window_size_);
		GeneratEnviromentBox(frame, final_box, model_box, surround_window_size_);
		Point move = Point(enviroment_box.x - enviroment_box_.x, enviroment_box.y - enviroment_box_.y);
		Mat nor_image;
		float ratio;
		NormalizeEnviroment(frame(enviroment_box), nor_image, ratio);
		Rect sub_target_box;
		sub_target_box.width = final_box.width*ratio + 0.5;
		sub_target_box.height = final_box.height*ratio + 0.5;
		sub_target_box.x = (final_box.x - enviroment_box.x)*ratio + 0.5;
		sub_target_box.y = (final_box.y - enviroment_box.y)*ratio + 0.5;
		Rect sub_model_box;
		sub_model_box.width = model_box.width*ratio + 0.5;
		sub_model_box.height = model_box.height*ratio + 0.5;
		sub_model_box.x = (model_box.x - enviroment_box.x)*ratio + 0.5;
		sub_model_box.y = (model_box.y - enviroment_box.y)*ratio + 0.5;
		Feature_Color_HSV space_s_n, space_l_n;
		float	check_score = InitColorHist_shape(sub_img, showe_, sub_target_box, sub_model_box, space_s_n, space_l_n, true);
	}
	////////////////////////////////////////////
	if (bool is_show_marsk =1)
	{
		Mat show_marsk;
		showe_.copyTo(show_marsk);
		rectangle(show_marsk, final_box, Scalar(255, 255, 255));
		imshow("marks", show_marsk);
		waitKey(1);
	}
      	resize(showe_, showe_, Size(showe_.cols / ratio_ + 0.5, showe_.rows / ratio_ + 0.5));
	//////////////////////////////
	if (bool is_checking = true)
	{
		int channel = frame.channels();
		int elemet = frame.elemSize();
		int j_end = frame.cols*channel;
		Point m = Point(enviroment_box_.x, enviroment_box_.y);
		Mat sss;
		frame.copyTo(sss);
		for (int i = 0; i < frame.rows; i++)
		{
			uchar* data = frame.ptr<uchar>(i);
			int y = 0;
			for (int j = 0; j < j_end; j = j + elemet)
			{
				Point p = Point(y, i);
				Point c_p = p - m;
				y++;
				if (c_p.x<0 || c_p.y<0 || c_p.x>showe_.cols - 1 || c_p.y>showe_.rows - 1)
					continue;

				if (showe_.at<uchar>(c_p)>0)
				{
					Vec3b& color = sss.at<Vec3b>(p);
					color[0] = 0;
					color[1] = 0;
					color[2] = 255;;
				}

			}
		}
		show_r = sss;
		//imshow("ss", sss);
		//waitKey(1);
	}
	Rect final_target_box;
	final_target_box.width = final_box.width / ratio_ + 0.5;
	final_target_box.height = final_box.height / ratio_ + 0.5;
	final_target_box.x = final_box.x / ratio_ + enviroment_box_.x + 0.5;
	final_target_box.y = final_box.y / ratio_ + enviroment_box_.y + 0.5;
	if (0)
	{
		Mat results;
		frame.copyTo(results);
		rectangle(results, final_target_box, Scalar(233, 233, 0));
		imshow("final_target_box", results);
		waitKey(0);
	}
	cout << "segmentation_s: "<<segmentation_s << endl;
	if (segmentation_s < 0.3)
	{
		Update(frame, final_target_box);
	}
	else
		Update_shape(frame, final_target_box, showe_);
	

	result_box = final_target_box;

	return 0;
}


float Color_Tracker::CompareTwoHistograms(vector<float> h1, vector<float> h2)
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
	//for (int i = 0; i < h1.size(); i++)
	//{
	//	if (h1[i] > 0)
	//	{
	//		toto = toto + h1[i];
	//		number++;
	//	}
	//}
	//thred1 = toto / number;
	thred1 = 0.3;
	//toto = 0;
	//number = 0;
	//for (int i = 0; i < h2.size(); i++)
	//{
	//	if (h2[i] > 0)
	//	{
	//		toto = toto + h2[i];
	//		number++;
	//	}
	//}
	//thred2 = toto / number;
	thred2 = 0.3;
	for (int i = 0; i < h2.size(); i++)
	{
		if (h1[i] > thred1)
		{
			target_num++;
			if (h2[i] > thred2)
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
float Color_Tracker::CompareTwoHistograms_r(vector<float> h1, vector<float> h2)
{
	if (h1.size() != h2.size())
	{
		printf("wrong input in CompareHistogram\n");
		return -1;
	}
	float aa = 0;
	float cc;
	for (int i = 0; i < h1.size(); i++)
	{
		cc = h1[i] * h2[i];
		if (cc == 0)
			continue;
		else
			aa = aa + sqrt(cc);
	}
	return aa;
}

bool Color_Tracker::NormalizeEnviroment(Mat& enviroment_img, Mat& normalized_img, float& ratio)
{
	Mat norm_img;
	int l_side = max(enviroment_img.cols, enviroment_img.rows);
	ratio = larger_size_noml / l_side;
	normalized_enviroment_size_ = Size(enviroment_img.cols*ratio, enviroment_img.rows*ratio);
	resize(enviroment_img, norm_img, normalized_enviroment_size_);
	normalized_img = norm_img;
	return 0;
}


bool Color_Tracker::Update_shape(Mat frame, Rect& target_box, Mat& target_marsk)
{
	bool is_occlused = true;
	bool is_lost = true;
	Rect final_box;
	float tracking_score;
	Feature_Color_HSV space_s_n, space_l_n;
	float represen_s;
	if (bool test_by_target_histogram = true)
	{
		if (target_box.width > 5 & target_box.height > 5)
		{
			//			float old_ratio = ratio_;
			Rect enviroment_box;
			Rect model_box;
			GeneratEnviromentBox(frame, target_box, enviroment_box, search_window_size_);
			GeneratEnviromentBox(frame, target_box, model_box, surround_window_size_);
			Point move = Point(enviroment_box.x - enviroment_box_.x, enviroment_box.y - enviroment_box_.y);
			Mat nor_image;
			float ratio;
			NormalizeEnviroment(frame(enviroment_box), nor_image, ratio);
			Rect sub_target_box;
			sub_target_box.width = target_box.width*ratio + 0.5;
			sub_target_box.height = target_box.height*ratio + 0.5;
			sub_target_box.x = (target_box.x - enviroment_box.x)*ratio + 0.5;
			sub_target_box.y = (target_box.y - enviroment_box.y)*ratio + 0.5;
			Rect sub_model_box;
			sub_model_box.width = model_box.width*ratio + 0.5;
			sub_model_box.height = model_box.height*ratio + 0.5;
			sub_model_box.x = (model_box.x - enviroment_box.x)*ratio + 0.5;
			sub_model_box.y = (model_box.y - enviroment_box.y)*ratio + 0.5;
			move = Point(move.x*ratio + 0.5, move.y*ratio + 0.5);
			if (color_format_ == HSV)
			{
				cvtColor(nor_image, nor_image, CV_BGR2HSV);
			}
			resize(target_marsk, target_marsk, Size(target_marsk.cols * ratio + 0.5, target_marsk.rows * ratio + 0.5));
			double DistributionEstimation = InitColorHist_shape(nor_image, target_marsk, sub_target_box, sub_model_box, space_s_n, space_l_n, false, move);
			tracking_score = CompareTwoHistograms(feature_space_l_.color_hist, space_l_n.color_hist);
			//			double size_score = 1 - abs((mean_characters[1] - current_target_box.area())) / mean_characters[1];
			cout << "tracking_score" << tracking_score << endl;
			if (tracking_score > mean_characters_[3] * 3 / 5.0)
			{
				is_lost = false;
				is_occlused = false;
				final_box = target_box;
			}
			else
				if (tracking_score > mean_characters_[3] * 2 / 5.0)
				{
					is_occlused = true;
					is_lost = false;
					Point2f move_e = BoxCenter(target_box) - BoxCenter(current_rect_);
					final_box.x = current_rect_.x + move_e.x / 2;
					final_box.y = current_rect_.y + move_e.y / 2;
					final_box.width = current_rect_.width;
					final_box.height = current_rect_.height;
					//	final_box = target_box;
				}
				else
				{
					is_lost = true;
					is_occlused = false;
					final_box = current_rect_;

				}
		}
		else
		{
			is_lost = true;
			is_occlused = false;
			final_box.width = current_rect_.width;// *ratio;
			final_box.height = current_rect_.height;// *ratio;
			final_box.x = BoxCenter(target_box).x - target_box.width / 2.0;
			final_box.y = BoxCenter(target_box).y - target_box.height / 2.0;

			//			final_box = target_box;
		}
	}
	//if (!is_lost && !is_occlused)
	//	cout << "updated is true" << endl;
	//else
	//	cout << "updated not true" << endl;
	//unique_ids_.clear();
	if (!is_lost && !is_occlused)
	{

		double r = alpha_*tracking_score;
		double sl = gama_ * tracking_score;
		//		r = 1;
		for (int i = 0; i < size_; i++)///////get the color's center point of each bin 
		{
			feature_space_l_.color_hist[i] = (1 - sl)*feature_space_l_.color_hist[i] + sl* space_l_n.color_hist[i];
			feature_space_s_.color_hist[i] = (1 - r)*feature_space_s_.color_hist[i] + r* space_s_n.color_hist[i];
		}
		if (characters_.size() == H)
		{
			vector<Scalar>::iterator  it;
			it = characters_.begin();
			characters_.erase(it);
		}
		Scalar character;
		//Point2f p1 = BoxCenter(current_rect_);
		//Point2f p2 = BoxCenter(target_box);

		//double move_dis = norm(p1 - p2);// *resize_ration;
		//character[0] = float(current_rect_.width) / target_box.width;
		//character[1] = float(current_rect_.height) / target_box.height;
		//character[2] = move_dis;
		character[3] = tracking_score;
		characters_.push_back(character);
		meanStdDev(characters_, mean_characters_, stader_character_aviation_);
		//stader_character_aviation_[0] = stader_character_aviation_[0] * 5;
		//stader_character_aviation_[1] = stader_character_aviation_[1] * 5;
		//stader_character_aviation_[2] = stader_character_aviation_[2] * 5;
	}
	//	final_box = target_box;
	GeneratEnviromentBox(frame, final_box, enviroment_box_, search_window_size_);
	int l_side = max(enviroment_box_.width, enviroment_box_.height);
	ratio_ = larger_size_noml / l_side;
	target_box = final_box;
	current_rect_ = final_box;
	//	previouse_target_ = target_box;
	//	
	return false;
}
bool Color_Tracker::Update(Mat frame, Rect& target_box)
{
	bool is_occlused = true;
	bool is_lost = true;
	Rect final_box;
	float tracking_score;
	Feature_Color_HSV space_s_n, space_l_n;
	float represen_s;
	if (bool test_by_target_histogram = true)
	{
		if (target_box.width > 5 & target_box.height > 5)
		{
			//			float old_ratio = ratio_;
			Rect enviroment_box;
			Rect model_box;
			GeneratEnviromentBox(frame, target_box, enviroment_box, search_window_size_);
			GeneratEnviromentBox(frame, target_box, model_box, surround_window_size_);
			double DistributionEstimation = InitColorHist_rect(frame, target_box, model_box, enviroment_box, space_s_n, space_l_n, true);
			tracking_score = CompareTwoHistograms(feature_space_l_.color_hist, space_l_n.color_hist);
			//			double size_score = 1 - abs((mean_characters[1] - current_target_box.area())) / mean_characters[1];
			cout << "tracking_score" << tracking_score << endl;
			if (tracking_score > mean_characters_[3] * 2 / 5.0)
			{
				is_lost = false;
				is_occlused = false;
				final_box = target_box;
			}
			else
				if (tracking_score > mean_characters_[3] * 1 / 4.0)
				{
					is_occlused = true;
					is_lost = false;
					Point2f move_e = BoxCenter(target_box) - BoxCenter(current_rect_);
					final_box.x = current_rect_.x + move_e.x / 2;
					final_box.y = current_rect_.y + move_e.y / 2;
					final_box.width = current_rect_.width;
					final_box.height = current_rect_.height;
					//	final_box = target_box;
				}
				else
				{
					is_lost = true;
					is_occlused = false;
					final_box = current_rect_;

				}
		}
		else
		{
			is_lost = true;
			is_occlused = false;
			final_box.width = current_rect_.width;// *ratio;
			final_box.height = current_rect_.height;// *ratio;
			final_box.x = BoxCenter(target_box).x - target_box.width / 2.0;
			final_box.y = BoxCenter(target_box).y - target_box.height / 2.0;

			//			final_box = target_box;
		}
	}
	//unique_ids_.clear();
	if (!is_lost && !is_occlused)
		cout << "updated is true" << endl;
	else
		cout << "updated not true" << endl;

	if (!is_lost && !is_occlused)
	{

		double r = alpha_*tracking_score;
		double sl = gama_ * tracking_score;
		//		r = 1;
		for (int i = 0; i < size_; i++)///////get the color's center point of each bin 
		{
			feature_space_l_.color_hist[i] = (1 - sl)*feature_space_l_.color_hist[i] + sl* space_l_n.color_hist[i];
			feature_space_s_.color_hist[i] = (1 - r)*feature_space_s_.color_hist[i] + r* space_s_n.color_hist[i];
		}
		//		SearchSimilarObjects(frame, target_box, search_box_, feature_space_s_, feature_space_l_, feature_space_d_,object_locations_);
		//	feature_space_s_ = space_s_n;
		if (characters_.size() == H)
		{
			vector<Scalar>::iterator  it;
			it = characters_.begin();
			characters_.erase(it);
		}
		Scalar character;
		character[3] = tracking_score;
		characters_.push_back(character);
		meanStdDev(characters_, mean_characters_, stader_character_aviation_);

	}
	//	final_box = target_box;
	GeneratEnviromentBox(frame, final_box, enviroment_box_, search_window_size_);
	int l_side = max(enviroment_box_.width, enviroment_box_.height);
	ratio_ = larger_size_noml / l_side;
	target_box = final_box;
	current_rect_ = final_box;
	//	previouse_target_ = target_box;
	//	
	return false;
}
float Color_Tracker::TargetSegmentation_morphological(Mat frame, Rect& target_box, Mat& shape_marsk)
{


	Mat thresholdImage;
	threshold(frame, thresholdImage, 0.5, 255, THRESH_BINARY);
	thresholdImage.convertTo(thresholdImage, CV_8U);
	morphOps(thresholdImage);
	//	rectangle(thresholdImage, target_box, Scalar(255));
//	imshow("thresholdImage", thresholdImage);
	Mat lable_mat;
	int n = connectedComponents(thresholdImage, lable_mat);
	vector<float> ratios(n, 0);
	vector<int>numbers(n, 0);

	for (int i = 0; i < lable_mat.rows; i++)
	{
		for (int j = 0; j < lable_mat.cols; j++)
		{

			int l = lable_mat.at<int>(i, j);

			numbers[l]++;
			if (target_box.contains(Point(j, i)))
			{
				ratios[l]++;
			}

		}
	}
	vector<int> target_labs;
	for (int i = 0; i < n; i++)
	{
		float aa = ratios[i] / numbers[i];
		if (aa > 0.2)
			target_labs.push_back(i);
	}

	Mat mat_f = Mat(thresholdImage.size(), CV_8UC1, Scalar(0));
	///////////////////////////
	Rect refined_seg_box;
	Point p1, p2, p3, p4;
	int p1_min_x = target_box.x - target_box.width * 10/ 100.0;
	int p1_min_y = target_box.y - target_box.height * 10 / 100.0;
	int p1_max_x = target_box.x + target_box.width * 10 / 100.0;
	int p1_max_y = target_box.y + target_box.height * 10 / 100.0;
	int p2_min_x = target_box.x + target_box.width * 90 / 100.0;
	int p2_min_y = target_box.y + target_box.height * 90 / 100.0;
	int p2_max_x = target_box.x + target_box.width * 110 / 100.0;
	int p2_max_y = target_box.y + target_box.height * 110/ 100.0;
	//int p1_min_x = target_box.x - 1;
	//int p1_min_y = target_box.y - 1;
	//int p1_max_x = target_box.x + 1;
	//int p1_max_y = target_box.y + 1;
	//int p2_min_x = target_box.x + target_box.width -1;
	//int p2_min_y = target_box.y + target_box.height -1;
	//int p2_max_x = target_box.x + target_box.width + 1;
	//int p2_max_y = target_box.y + target_box.height + 1;


	int min_x, min_y, max_x, max_y;
	min_x = p1_max_x;
	min_y = p1_max_y;
	max_x = p2_min_x;
	max_y = p2_min_y;
	int positive_number = 0;
	for (int i = 0; i < lable_mat.rows; i++)
	{
		for (int j = 0; j < lable_mat.cols; j++)
		{

			int l = lable_mat.at<int>(i, j);

			bool is_target = false;
			for (int mm = 0; mm < target_labs.size(); mm++)
			{
				if (l == target_labs[mm])
				{
					is_target = true;
					break;
				}
			}
			if (is_target)
			{
				if (i > p1_min_y && i< p2_max_y && j > p1_min_x && j < p2_max_x)
				{
					positive_number++;
					mat_f.at<uchar>(Point(j, i)) = 255;
					if (i < min_y)
						min_y = i;
					if (j < min_x)
						min_x = j;
					if (j > max_x)
						max_x = j;
					if (i > max_y)
						max_y = i;

				}
			}


		}
	}

	shape_marsk = mat_f;
	refined_seg_box = Rect(Point(min_x, min_y), Point(max_x, max_y));
	target_box = refined_seg_box;
	return float (positive_number) / refined_seg_box.area();
}
