#include "DataInput.h"
using namespace std;
using namespace cv;

DataInput::DataInput()
{
//	LoadDataSet();
//	LoadDataSet_VOT();
}

DataInput::DataInput(string data_set_path)
{
	data_set_file_path = data_set_path;
	LoadDataSet();
}
DataInput::~DataInput()
{
}
bool DataInput::LoadDataSet(string specified_txt_file)
{
	int input_file_format = 0;
	if (specified_txt_file != "1")
	{
		input_file_format = 1;
	}

	if (input_file_format == 0)
	{
		string sequence_name;
		int video_name = 1;
		switch (video_name)
		{
		case 0:
			sequence_name = "Bolt";
			break;
		case 1:
			sequence_name = "Basketball";
			break;
		case 2:
			sequence_name = "Tiger1";
			break;
		case 3:
			sequence_name = "Liquor";
			break;
		case 4:
			sequence_name = "CarScale";
			break;
		case 5:
			sequence_name = "DragonBaby";
			break;
		case 6:
			sequence_name = "Skiing";
			break;
		case 7:
			sequence_name = "bird1";
			break;
			
	


		}
		sequences.push_back(sequence_name);
	}
	////////////test new input way/////////////
	////////////
	if (input_file_format == 1)
	{
		//		ifstream sequences_names("C:/TB_50/BC_sequences.txt");
		ifstream sequences_names(data_set_file_path + specified_txt_file);

		string str;
		while (getline(sequences_names, str))
		{
			string s;
			for (int i = 0; i < str.size(); i++)
			{
				if (str[i] == ' ')
					continue;
				char a = str[i];
				if (a != ',')
					s.push_back(a);
				else
				{
					cout << s;
					sequences.push_back(s);
					s.clear();
				}
			}
			sequences.push_back(s);
		}
	}
	return true;
}
bool DataInput::LoadImagesAndBoxs(int sequences_id, vector<Mat>& imgs, vector<Rect>& targer_true_box)
{
	imgs.clear();
	string path_here = data_set_file_path + sequences[sequences_id] + "/";
	/////////////////////
	////////////////////////////////
	string file_path_here = path_here + "img";
	vector<String> image_file;
	glob(file_path_here, image_file);


	//Directory dir;
	//string file_exten_here = "*.jpg";
	//
	////		string file_path_here = "C:/TB_50/Box/img";
	//image_file = dir.GetListFiles(file_path_here, file_exten_here, is_add_Path);

	if (image_file.size() == 0)
	{
		cout << "there is no such image in this path." << endl;
		return false;
	}
	Mat first_image = imread(image_file[0]);
	if (first_image.empty())
	{
		cout << "cannot read this image." << endl;
		return false;
	}
	file_box_path = path_here + "groundtruth_rect.txt";
	Rect initial_box;
	vector<Rect> true_boxes;
	{
		ifstream box_file(file_box_path);
		string str;
		int count = 0;
		Rect box;
		int x, y, h, w;
		bool is_dot = true;
		while (getline(box_file, str))
		{
			char x_c[20], y_c[20], h_c[20], w_c[20];
			const char *c = str.c_str();
			if (is_dot)
			{
				sscanf(c, "%d,  %d, %d,  %d", &x, &y, &w, &h);
			}
			if (w <= 0 || h <= 0)
			{
				is_dot = false;			
			}
			if (!is_dot)
				sscanf(c, "%d  %d %d  %d", &x, &y, &w, &h);
			if (x < 0)
				x = 0;
			if (y < 0)
				y = 0;
			box.x = x;
			box.y = y;
			box.width = w;
			box.height = h;
			true_boxes.push_back(box);
		}
	}
	int i_count = 0;
	while (1)
	{
		if (i_count >= image_file.size())
			break;
		imgs.push_back(imread(image_file[i_count]));
		i_count++;
	}
	targer_true_box = true_boxes;
}
bool DataInput::LoadImagesAndBoxs_D(int sequences_id, vector<String>& imgs, vector<Rect>& targer_true_box)
{
	imgs.clear();
	string path_here = data_set_file_path + sequences[sequences_id] + "/";
	/////////////////////
	////////////////////////////////
	string file_path_here = path_here + "img";
	vector<String> image_file;
	glob(file_path_here, image_file);
	imgs = image_file;

	//Directory dir;
	//string file_exten_here = "*.jpg";
	//
	////		string file_path_here = "C:/TB_50/Box/img";
	//image_file = dir.GetListFiles(file_path_here, file_exten_here, is_add_Path);

	if (image_file.size() == 0)
	{
		cout << "there is no such image in this path." << endl;
		return false;
	}
	Mat first_image = imread(image_file[0]);
	if (first_image.empty())
	{
		cout << "cannot read this image." << endl;
		return false;
	}
	file_box_path = path_here + "groundtruth_rect.txt";
	Rect initial_box;
	vector<Rect> true_boxes;
	{
		ifstream box_file(file_box_path);
		string str;
		int count = 0;
		Rect box;
		int x, y, h, w;
		bool is_dot = true;
		while (getline(box_file, str))
		{
			char x_c[20], y_c[20], h_c[20], w_c[20];
			const char *c = str.c_str();
			if (is_dot)
			{
				sscanf(c, "%d,  %d, %d,  %d", &x, &y, &w, &h);
			}
			if (w <= 0 || h <= 0)
			{
				is_dot = false;
			}
			if (!is_dot)
				sscanf(c, "%d  %d %d  %d", &x, &y, &w, &h);
			if (x < 0)
				x = 0;
			if (y < 0)
				y = 0;
			box.x = x;
			box.y = y;
			box.width = w;
			box.height = h;
			true_boxes.push_back(box);
		}
	}
	targer_true_box = true_boxes;
}
bool DataInput::LoadDataSet_VOT(string specified_txt_file)
{
	data_set_file_path = "C:/vot2015/";
	int input_file_format = 0;
	if (specified_txt_file != "1")
	{
		input_file_format = 1;
	}

	if (input_file_format == 0)
	{
		string sequence_name;
		int video_name = 0;
		switch (video_name)
		{
		case 0:
			sequence_name = "bag";
			break;
		case 1:
			sequence_name = "Basketball";
			break;
		case 2:
			sequence_name = "Tiger2";
			break;
		case 3:
			sequence_name = "Liquor";
			break;
		case 4:
			sequence_name = "CarScale";
			break;
		case 5:
			sequence_name = "DragonBaby";
			break;
		case 6:
			sequence_name = "Bird1";
			break;
		case 7:
			sequence_name = "Woman";
			break;




		}
		sequences.push_back(sequence_name);
	}
	////////////test new input way/////////////
	////////////
	if (input_file_format == 1)
	{
		//		ifstream sequences_names("C:/TB_50/BC_sequences.txt");
		ifstream sequences_names(data_set_file_path + specified_txt_file);

		string str;
		while (getline(sequences_names, str))
		{
			string s;
			for (int i = 0; i < str.size(); i++)
			{
				if (str[i] == ' ')
					continue;
				char a = str[i];
				if (a != ',')
					s.push_back(a);
				else
				{
					cout << s;
					sequences.push_back(s);
					s.clear();
				}
			}
			sequences.push_back(s);
		}
	}
	return true;
}

bool DataInput::LoadImagesAndBoxs_VOT(int sequences_id, vector<Mat>& imgs, vector<Rect>& targer_true_box)
{
	imgs.clear();
	string path_here = data_set_file_path + sequences[sequences_id] + "/";
	/////////////////////
	////////////////////////////////
	string file_path_here = path_here + "img";
	vector<String> image_file;
	glob(file_path_here, image_file);

	//Directory dir;
	//string file_exten_here = "*";//"*.jpg"
	//string file_path_here = path_here;
	////		string file_path_here = "C:/TB_50/Box/img";
	//image_file = dir.GetListFiles(file_path_here, file_exten_here, is_add_Path);

	if (image_file.size() == 0)
	{
		cout << "there is no such image in this path." << endl;
		return false;
	}
	Mat first_image = imread(image_file[0]);
	if (first_image.empty())
	{
		cout << "cannot read this image." << endl;
		return false;
	}
	file_box_path = path_here + "groundtruth.txt";
	Rect initial_box;
	vector<Rect> true_boxes;
	{
		ifstream box_file(file_box_path);
		string str;
		int count = 0;
		Rect box;
		float x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4;
		while (getline(box_file, str))
		{
			/*char x_1[20], y_1[20], x_2[20], y_2[20], x_3[20], y_3[20], x_4[20], y_4[20];*/
			const char *c = str.c_str();
			sscanf(c, "%f,  %f, %f,  %f , %f,  %f, %f,  %f", &x_1, &y_1, &x_2, &y_2, &x_3, &y_3, &x_4, &y_4);
			if (x_1 < 0)
				x_1 = 0;
			if (y_1 < 0)
				y_1 = 0;
			box.x = x_1;
			box.y = y_1;
			box.width = x_2-x_1;
			box.height = y_2-y_1;
			true_boxes.push_back(box);
		}
	}
	int i_count = 0;
	while (1)
	{
		if (i_count >= image_file.size() - 1)
			break;
		imgs.push_back(imread(image_file[i_count]));
		i_count++;
	}
	targer_true_box = true_boxes;
}