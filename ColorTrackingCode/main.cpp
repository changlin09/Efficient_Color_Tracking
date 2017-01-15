#include "opencv2/highgui/highgui.hpp"
#include "imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp>
#include "Functions.h"
#include <iostream>
#include <fstream>

#include"Color_Tracker.h"
#include "DataInput.h"
using namespace cv;
using namespace std;

void DistinctiveColorTracking();
bool DEBUG = true;
void EfficentColorTracking_image_sequence_test();
void main()
{
	EfficentColorTracking_image_sequence_test();
}


void EfficentColorTracking_image_sequence_test()
{
	Color_Tracker m_tracker;
	DataInput m_data_loader;
	//	m_data_loader.LoadDataSet("ALL_sequences.txt");
	m_data_loader.LoadDataSet();
	int num_sequence = m_data_loader.sequences.size();
	float initial_values;

	string video_path = "C:/Request/bird_tracking/20160324_055427.464695export_99fps.mp4";
	VideoCapture m_video_capture;
	m_video_capture.open(video_path);
	Mat first_img;
	m_video_capture >> first_img;
	if (first_img.empty())
	{
		cout << "no video found";
		return;
	}
	else{
		while (1)
		{
			resize(first_img, first_img, Size(500, 500));
		imshow("input_video",first_img);
		char c = waitKey(0);

		if ((char)c == 32)	// Space
		{
			m_video_capture >> first_img;
		}
		if ((char)c == 102) //f
		{
			destroyWindow("input_video");
			break;
		}
	}
	Rect initial_box;
	vector<Rect> ture_boxes;
	vector<Rect> detected_boxes;
	if (1)
	{
		while (!GetBoxByMouse(first_img, initial_box));
		ture_boxes.push_back(initial_box);
		detected_boxes.push_back(initial_box);
	}
	double initial_value = m_tracker.Initial(first_img, initial_box);
	int i_count = 0;
	Mat img;
	Rect result;
	while (1)
	{
		i_count++;

		m_video_capture >> img;
		resize(img, img, Size(500, 500));
		if (img.empty())
		{
			printf("no image input\n");
			break;
		}

		m_tracker.Track(img, result, initial_box);
		detected_boxes.push_back(result);
		rectangle(img, result, Scalar(0, 0, 255));
		imshow("result", img);
		waitKey(0);
		}

		ofstream fout;
		string txt_filename = "tracked_boxes.txt";
		fout.open(txt_filename);
		//fout << initial_value << endl;
		for (int i = 0; i < detected_boxes.size(); i++)
		{
			fout << detected_boxes[i].x << "," << detected_boxes[i].y << " ," << detected_boxes[i].width << "," << detected_boxes[i].height << endl;
		}
		fout << flush;
		fout.close();
	}

}
void DistinctiveColorTracking()
{
	Color_Tracker m_tracker;
	DataInput m_data_loader;
	//	m_data_loader.LoadDataSet("ALL_sequences.txt");
	m_data_loader.LoadDataSet();
	int num_sequence = m_data_loader.sequences.size();
	vector<float> initial_values;
	for (int id = 0; id < num_sequence; id++)
	{
		vector<Mat> images;
		vector<String> image_file;
		vector<Rect> true_boxes;
		vector<Rect> detected_boxes;
		Mat first_img;
		int num_imgs;
		if (DEBUG)
		{
			m_data_loader.LoadImagesAndBoxs_D(id, image_file, true_boxes);
			first_img = imread(image_file[0]);
			num_imgs = image_file.size();
		}
		else
		{
			m_data_loader.LoadImagesAndBoxs(id, images, true_boxes);
			first_img = images[0];
			num_imgs = images.size();
		}
		Rect first_box = true_boxes[0];
		detected_boxes.push_back(first_box);
		//.............
		//first_img = imread("1.bmp");
		//first_box = Rect(Point(251,86),Point(352,254));
		//..........
		double initial_value = m_tracker.Initial(first_img, first_box);
		for (int i = 1; i < num_imgs; i++)
		{
			double t = (double)getTickCount();
			Mat img;
			if (DEBUG)
			{
				img = imread(image_file[i]);
			}
			else
			{
				img = images[i];
			}

			Mat input_img;
			img.copyTo(input_img);
			rectangle(input_img, true_boxes[i], Scalar(255, 0, 0));
			imshow("input_img", input_img);
			Rect result;
			m_tracker.Track(img, result, true_boxes[i]);
			//m_tracker.Track_s(img, result, true_boxes[i]);
			//			m_tracker.Track(img, result);
			t = ((double)getTickCount() - t) / getTickFrequency();
			cout << 1 / t << endl;
			detected_boxes.push_back(result);
			rectangle(img, result, Scalar(0, 0, 255));
			imshow("result", img);
			waitKey(0);
		}

		ofstream fout;
		string txt_filename = m_data_loader.sequences[id];
		txt_filename = txt_filename + ".txt";
		fout.open(txt_filename);
		fout << initial_value << endl;
		for (int i = 0; i < detected_boxes.size(); i++)
		{
			fout << detected_boxes[i].x << "," << detected_boxes[i].y << " ," << detected_boxes[i].width << "," << detected_boxes[i].height << endl;
		}
		fout << flush;
		fout.close();
		initial_values.push_back(initial_value);
	}
	ofstream fout;
	string txt_filename = "initial_values.txt";

	fout.open(txt_filename);

	for (int i = 0; i < initial_values.size(); i++)
	{
		fout << initial_values[i] << endl;
	}
	fout << flush;
	fout.close();


}
