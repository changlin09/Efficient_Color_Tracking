#include "opencv2/highgui/highgui.hpp"
#include "imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
class DataInput
{
public:
	DataInput();
	DataInput(string data_set_path);
	~DataInput();
public:
	bool DataInput::SetDataSet(string data_set_path);
	bool DataInput::LoadDataSet(string specified_txt_file = "1");
	bool DataInput::LoadDataSet_VOT(string specified_txt_file = "1");
	bool DataInput::LoadImagesAndBoxs(int sequences_id, vector<Mat>& imgs, vector<Rect>& targer_true_box);
	bool DataInput::LoadImagesAndBoxs(string sequence_data_path, vector<Mat>& imgs, vector<Rect>& targer_true_box);
	bool DataInput::LoadImagesAndBoxs_D(int sequences_id, vector<String>& imgs, vector<Rect>& targer_true_box);
	bool DataInput::LoadImagesAndBoxs_VOT(int sequences_id, vector<Mat>& imgs, vector<Rect>& targer_true_box);
public:
	vector<string> sequences;
	int num_sequences;

private:
	string data_set_file_path = "C:/TB_50/";
//	string data_set_file_path = "C:/vot2015/";
	bool is_add_Path = true;
	bool is_mouse_box = false;
	bool is_file_box = !is_mouse_box;

	string file_box_path;

	string file_path_here;
	string name;
private:
	
};

