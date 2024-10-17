#include <iostream>
#include <Eigen\Dense>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iomanip>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

double get_3d_distance(Point3d pt1, Point3d pt2)
{
	double distance = sqrtf(powf((pt1.x - pt2.x), 2) + powf((pt1.y - pt2.y), 2) + powf((pt1.z - pt2.z), 2));
	return distance;
}

void Get_Coordinate(const Point3d& pt0, const Point3d& pt1, const Point3d& pt2)
{
	Vector3d nx, ny, nz,vpt0,vpt1,vpt2;
	vpt0 = Vector3d(pt0.x,pt0.y,pt0.z);
	vpt1 = Vector3d(pt1.x, pt1.y, pt1.z);
	vpt2 = Vector3d(pt2.x, pt2.y, pt2.z);
	nx = (vpt1-vpt0)/get_3d_distance(pt1,pt0);
	nz=((vpt2-vpt0).cross(nx))/ get_3d_distance(pt2, pt0);
	ny = nx.cross(nz);
	cout << "nx:  " << nx << endl << "ny:  " << ny << endl<<"nz:  "<<nz<<endl;
}

//pts2-->pts1
void pose_estimation_3dto3d(const vector<Point3d>& pts1, const vector<Point3d>& pts2,
	Mat& R, Mat& t, Eigen::Affine3d& transform)
{
	Point3d pt1, pt2;
	int N = pts1.size();
	for (int i = 0; i < N; i++)
	{
		pt1 += pts1[i];
		pt2 += pts2[i];
	}
	pt1 /= N;
	pt2 /= N;
	vector<Point3d> q1(N), q2(N);
	for (int i = 0; i < N; i++)
	{
		q1[i] = pts1[i] - pt1;
		q2[i] = pts2[i] - pt2;
	}
	Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
	for (int i = 0; i < N; i++)
	{
		W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
	}
	//cout << "W: " << W << endl;
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3d U = svd.matrixU();
	Eigen::Matrix3d V = svd.matrixV();
	//cout << "U: " << U << endl;
	// << "V: " << V << endl;
	Eigen::Matrix3d R_ = U * (V.transpose());
	Eigen::Matrix3d rf;
	rf << 1, 0, 0, 0, 1, 0, 0, 0, -1;
	//cout << "R_: " << R_.determinant() <<endl<<R_<< endl;
	double det = R_.determinant();
	if ((det + 1) < 0.001)
	{
		R_ = U * rf * (V.transpose());
		//cout << "R_: " << R_.determinant() << endl<< R_ << endl;
	}
	
	Eigen::Vector3d t_ = Eigen::Vector3d(pt1.x, pt1.y, pt1.z) - R_ * Eigen::Vector3d(pt2.x, pt2.y, pt2.z);
	R = (Mat_<double>(3, 3) <<
		R_(0, 0), R_(0, 1), R_(0, 2),
		R_(1, 0), R_(1, 1), R_(1, 2),
		R_(2, 0), R_(2, 1), R_(2, 2)
		);

	//cout << "R_: " << R_.determinant() << endl;
	t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
	//cout << "R: " <<endl<< R << endl;
	//cout << "t: " << endl<<t << endl;
	Eigen::Affine3d transform_ = Eigen::Affine3d::Identity();
	transform_.translation() << t_(0, 0), t_(1, 0), t_(2, 0);
	transform_.rotate(R_);
	//cout << "T:  "<<endl<<transform_.matrix() << endl;
	//cout << transform_.matrix().inverse() << endl;
	transform = transform_;
}

void read_ptcloud_file(string path,vector<string>& ptnames,vector<Point3d>& pts)
{
	ifstream readfile;
	readfile.open(path, ios::in);
	int ptnum = 0;
	string tmp;
	while (getline(readfile, tmp, '\n'))
	{
		ptnum++;
	}
	readfile.close();
	readfile.open(path, ios::in);
	for (int i = 0; i < ptnum; i++)
	{
		string name;
		double x, y, z;
		readfile >> name;
		readfile >> x;
		readfile >> y;
		readfile >> z;
		ptnames.push_back(name);
		pts.push_back(Point3d(x, y, z));
	}
}

void find_common_markers(const vector<string>& ptnames1, const vector<Point3d>& pts1,
	const vector<string>& ptnames2, const vector<Point3d>& pts2, 
	vector<string>& common_ptnames1,  vector<Point3d>& common_pts1, 
	vector<string>& common_ptnames2, vector<Point3d>& common_pts2)
{
	for (int i = 0; i < ptnames1.size(); i++)
	{
		for (int j = 0; j < ptnames2.size(); j++)
		{
			if (ptnames1[i].compare(ptnames2[j])==0)
			{
				common_ptnames1.push_back(ptnames1[i]);
				common_ptnames2.push_back(ptnames2[j]);
				common_pts1.push_back(pts1[i]);
				common_pts2.push_back(pts2[j]);
				break;
			}
		}
	}
}

void get_transmat(const vector<string>& srcnames,const vector<string>& dstnames,
	const vector<Point3d>& srcpts,const vector<Point3d>& dstpts,
	Mat& R, Mat& t, Eigen::Affine3d& transmat)
{
	vector<Point3d> commonpts_src, commonpts_dst;
	for (int i = 0; i < srcnames.size(); i++)
	{
		for (int j = 0; j < dstnames.size(); j++)
		{
			if (srcnames[i].compare(dstnames[j]) == 0)
			{
				commonpts_src.push_back(srcpts[i]);
				commonpts_dst.push_back(dstpts[j]);
			}
		}
	}
	//cout << "find " << to_string(commonpts_dst.size()) << "common pts" << endl;
	pose_estimation_3dto3d(commonpts_dst, commonpts_src,R, t, transmat);
}

int main()
{
	//标志物球坐标系标定
	Point3d pt0, pt1, pt2,ballpt0,ballpt1,ballpt2;
	vector<Point3d> Ballpoints,Pose0points, Pose1points;
	vector<string> Ballnames;
	vector<vector<Point3d>> PoseNpoints;
	//pt0 = Point3d(-62.831, 73.534, 800.936);
	//pt1 = Point3d(117.385, 13.231, 785.438);
	//pt2 = Point3d(-122.324, -47.434, 804.377);
	pt0 = Point3d(-152.712, 1.689, 492.039);
	pt1 = Point3d(85.752, 61.426, 463.023);
	pt2 = Point3d(24.288, -58.588, 457.849);
	Get_Coordinate(pt0, pt1, pt2);


	//组合标志物球坐标系建立
	//ballpt0 = Point3d(0, 0, 0);
	//ballpt1 = Point3d(190.668,0,0);
	//ballpt2 = Point3d(-18.253, -133.609, 0);
	ballpt0 = Point3d(0, 0, 0);
	ballpt1 = Point3d(247.3141, 0, 0);
	ballpt2 = Point3d(160.1253, 102.4066, 0);
	Ballpoints.push_back(ballpt0);
	Ballpoints.push_back(ballpt1);
	Ballpoints.push_back(ballpt2);
	Ballnames.push_back("Ball0");
	Ballnames.push_back("Ball1");
	Ballnames.push_back("Ball2");

	//编码球笼标定
	int marker_calib_posenum = 3;
	vector<vector<string>> marker_calib_pointnames;
	vector<vector<Point3d>> marker_calib_points;
	for (int i = 0; i < marker_calib_posenum; i++)
	{
		string marker_calib_filename = "resources\\marker_calib\\marker_calib" + to_string(i+1) + ".txt";
		vector<string> pointnames;
		vector<Point3d> points;
		read_ptcloud_file(marker_calib_filename, pointnames, points);
		marker_calib_pointnames.push_back(pointnames);
		marker_calib_points.push_back(points);
	}
	vector<string> common01_names_0, common01_names_1,common12_names_0, common12_names_2,marker_names;
	vector<Point3d> common01_pts_0, common01_pts_1, common12_pts_0,common12_pts_2,marker_pts;

	find_common_markers(marker_calib_pointnames[0], marker_calib_points[0],
		marker_calib_pointnames[1], marker_calib_points[1],
		common01_names_0, common01_pts_0, common01_names_1, common01_pts_1);

	find_common_markers(marker_calib_pointnames[1], marker_calib_points[1],
		marker_calib_pointnames[2], marker_calib_points[2],
		common12_names_0, common12_pts_0, common12_names_2, common12_pts_2);

	//由公共点计算变换矩阵

	Eigen::Affine3d transmat01, transmat02;
	Mat transR01, transt01, transR02, transt02;
	pose_estimation_3dto3d(common01_pts_0, common01_pts_1, transR01, transt01, transmat01);
	pose_estimation_3dto3d(common12_pts_0, common12_pts_2, transR02, transt02, transmat02);

	//求标定后的整体点在摄影测量坐标系下坐标
	vector<vector<string>> marker_regis_pointnames;
	vector<vector<Point3d>> marker_regis_points;
	vector<string> ref_marker_pointnames;	//最终整体编码
	vector<Point3d> ref_marker_points;
	for (int i = 0; i < marker_calib_posenum; i++)
	{
		string marker_regis_filename = "resources\\marker_calib\\marker_calib" + to_string(i+1) + "_regis"+".txt";
		vector<string> pointnames;
		vector<Point3d> points;
		read_ptcloud_file(marker_regis_filename, pointnames, points);
		marker_regis_pointnames.push_back(pointnames);
		marker_regis_points.push_back(points);
	}

	//先把齐次变换后的编码点坐标全部累加
	vector<string> add_names;
	vector<Point3d> add_points;
	for (int i = 0; i < marker_regis_pointnames[0].size(); i++)
	{
		add_names.push_back(marker_regis_pointnames[0][i]);
		add_points.push_back(marker_regis_points[0][i]);
	}
	for (int i = 0; i < marker_regis_pointnames[1].size(); i++)
	{
		add_names.push_back(marker_regis_pointnames[1][i]);
		add_points.push_back(marker_regis_points[1][i]);
	}
	for (int i = 0; i < marker_regis_pointnames[2].size(); i++)
	{
		add_names.push_back(marker_regis_pointnames[2][i]);
		add_points.push_back(marker_regis_points[2][i]);
	}

	//寻找公共点，并使其唯一，得到最终标定坐标点
	vector<int>skip_num;
	Point3d avgPoint;
	int ptnum=0;
	bool common_flag = false;
	for (int i = 0; i < add_names.size(); i++)
	{
		string name;
		Point3d Point;
		name = add_names[i];
		if (find(skip_num.begin(), skip_num.end(), i) != skip_num.end())
		{
			continue;
		}
		for (int j = i+1; j < add_names.size(); j++)
		{
			if (add_names[i].compare(add_names[j]) == 0)
			{
				skip_num.push_back(j);
				Point = (add_points[i] + add_points[j])/2;
				common_flag = true;
			}
		}

		if (!common_flag)
		{
			Point = add_points[i];
		}
		ref_marker_pointnames.push_back(name);
		ref_marker_points.push_back(Point);
		common_flag = false;
	}
	for (int i = 0; i < ref_marker_pointnames.size(); i++)
	{
		cout << fixed << setprecision(3) << ref_marker_pointnames[i] << "  " << ref_marker_points[i].x << "  " << ref_marker_points[i].y << "  " << ref_marker_points[i].z << endl;
		avgPoint += ref_marker_points[i];
		ptnum++;
	}
	avgPoint = avgPoint/ptnum;
	cout <<"coordinate center:  " << avgPoint << endl;
	//建系
	Get_Coordinate(avgPoint, ref_marker_points[4], ref_marker_points[58]);


	//读靶标坐标系下编码点坐标
	vector<Point3d> Markerpoints;
	vector<string> Markerpointnames;
	string Marker_Coordinates_file = "resources\\marker_calib\\coordinates_result.txt";
	read_ptcloud_file(Marker_Coordinates_file, Markerpointnames, Markerpoints);

	//读组合标志物编码点坐标系下编码点坐标
	vector<Point3d> TMarkerpoints;
	vector<string> TMarkerpointnames;
	string TMarker_Coordinates_file = "resources\\targert_marker_calib\\TargetMarkerCoordinates.txt";
	read_ptcloud_file(TMarker_Coordinates_file, TMarkerpointnames,TMarkerpoints);

	//读取实验靶标编码点坐标
	vector<vector<Point3d>> PoseN_Markerpoints;
	vector<vector<string>> PoseN_Markerpointnames;
	int posenum = 10;
	for (int i = 0; i < posenum; i++)
	{
		string PoseN_Marker_file = "resources\\handeye_calib\\Pose" + to_string(i+1) + "Markerpoints.txt";
		vector<Point3d> PoseN_marker;
		vector<string> PoseN_markername;
		read_ptcloud_file(PoseN_Marker_file, PoseN_markername, PoseN_marker);
		PoseN_Markerpointnames.push_back(PoseN_markername);
		PoseN_Markerpoints.push_back(PoseN_marker);
	}

	//读取实验组合标志物编码点坐标
	vector<vector<Point3d>> PoseN_TMarkerpoints;
	vector<vector<string>> PoseN_TMarkerpointnames;
	for (int i = 0; i < posenum; i++)
	{
		string PoseN_TMarker_file = "resources\\handeye_calib\\Pose" + to_string(i) + "TMarkerpoints.txt";
		vector<Point3d> PoseN_Tmarker;
		vector<string> PoseN_Tmarkername;
		read_ptcloud_file(PoseN_TMarker_file, PoseN_Tmarkername, PoseN_Tmarker);
		PoseN_TMarkerpointnames.push_back(PoseN_Tmarkername);
		PoseN_TMarkerpoints.push_back(PoseN_Tmarker);
	}

	//读取实验组合标志物球心坐标
	vector<vector<Point3d>> PoseN_TBallpoints;
	vector<vector<string>> PoseN_TBallpointnames;
	for (int i = 0; i < posenum; i++)
	{
		string PoseN_TBall_file = "resources\\handeye_calib\\Pose" + to_string(i+1) + "TBallpoints.txt";
		vector<Point3d> PoseN_TBall;
		vector<string> PoseN_TBallname;
		read_ptcloud_file(PoseN_TBall_file, PoseN_TBallname, PoseN_TBall);
		PoseN_TBallpointnames.push_back(PoseN_TBallname);
		PoseN_TBallpoints.push_back(PoseN_TBall);
	}

	//计算摄影测量到球笼坐标系的变换矩阵
	vector<Mat> M2W_R, M2W_t;
	vector<Eigen::Affine3d> M2W_transmat;
	for (int i = 0; i < posenum; i++)
	{
		Mat R, t;
		Eigen::Affine3d trans;
		get_transmat(Markerpointnames, PoseN_Markerpointnames[i], Markerpoints, PoseN_Markerpoints[i], R, t, trans);
		M2W_R.push_back(R);
		M2W_t.push_back(t);
		M2W_transmat.push_back(trans);
	}

	//计算摄影测量到组合标志物编码点坐标系的变换矩阵
	vector<Mat> TM2W_R, TM2W_t;
	vector<Eigen::Affine3d> TM2W_transmat;
	for (int i = 0; i < posenum; i++)
	{
		Mat R, t;
		Eigen::Affine3d trans;
		get_transmat(TMarkerpointnames, PoseN_TMarkerpointnames[i], TMarkerpoints, PoseN_TMarkerpoints[i], R, t, trans);
		TM2W_R.push_back(R);
		TM2W_t.push_back(t);
		TM2W_transmat.push_back(trans);
	}

	//计算扫描仪到组合标志物球坐标系的变换矩阵
	vector<Mat> TB2S_R, TB2S_t;
	vector<Eigen::Affine3d> TB2S_transmat;
	for (int i = 0; i < posenum; i++)
	{
		Mat R, t;
		Eigen::Affine3d trans;
		get_transmat(Ballnames, PoseN_TBallpointnames[i], Ballpoints, PoseN_TBallpoints[i],R, t, trans);
		TB2S_R.push_back(R);
		TB2S_t.push_back(t);
		TB2S_transmat.push_back(trans);
	}

	//计算AX=YB
	//M-->gripper,W-->base,S-->camera,M-->gripper,TB-->target
	Mat S2M_R, S2M_t;
	calibrateHandEye(M2W_R, M2W_t, TB2S_R, TB2S_t, S2M_R,S2M_t);
	cout << S2M_R << endl << S2M_t << endl;
	Matrix4d S2M_trans=Matrix4d::Identity();
	Matrix3d transR;
	cv2eigen(S2M_R, transR);
	Vector3d transt;
	cv2eigen(S2M_t, transt);
	S2M_trans.block(0, 0, 3, 3) = transR;
	S2M_trans.block(0, 3, 3, 1) = transt;
	cout <<"scanner to marker: "<<endl<< S2M_trans << endl;
		Matrix4d S2M_trans_opt;
	S2M_trans_opt << -0.60469,-0.79209,0.083264,-74.407,
		0.17006,-0.23054,-0.95809,-70.184,
		0.77809,-0.56519,0.27411,51.556,
		0,0,0,1;

	//球棒拼接
	vector<vector<string>> testbnames;
	vector<vector<Point3d>> testbpoints;
	for (int i = 0; i < 2; i++)
	{
		string Ntest_bfile = "resources\\qbs666\\pose" + to_string(i + 1) + ".txt";
		vector<Point3d> Ntestbpoint;
		vector<string> Ntestbname;
		read_ptcloud_file(Ntest_bfile, Ntestbname, Ntestbpoint);
		testbnames.push_back(Ntestbname);
		testbpoints.push_back(Ntestbpoint);
	}
	Mat R_b1ori, t_b1ori, R_b0ori, t_b0ori;
	Eigen::Affine3d trans_b1ori, trans_b0ori;
	get_transmat(Markerpointnames, testbnames[0], Markerpoints, testbpoints[0], R_b0ori, t_b0ori, trans_b0ori);
	get_transmat(Markerpointnames, testbnames[1], Markerpoints, testbpoints[1], R_b1ori, t_b1ori, trans_b1ori);
	Eigen::Matrix4d transb1, transb2;
	transb1 = trans_b0ori.matrix() * S2M_trans;
	transb2 = trans_b1ori.matrix() * S2M_trans;
	cout << "b1Pose:  " << endl << transb1 << endl;
	cout << "b2Pose:  " << endl << transb2 << endl;

	//转站扫描标准球棒
	vector<vector<string>> testnames;
	vector<vector<Point3d>> testpoints;
	for (int i = 0; i < 10; i++)
	{
		string Ntest_file = "resources\\base_transfer\\240702\\2\\poses\\T1B2-" + to_string(i+1)+".txt";
		vector<Point3d> Ntestpoint;
		vector<string> Ntestname;
		read_ptcloud_file(Ntest_file, Ntestname, Ntestpoint);
		testnames.push_back(Ntestname);
		testpoints.push_back(Ntestpoint);
	}
	//Mat R_1ori, t_1ori,R_0ori,t_0ori;
	//Eigen::Affine3d trans_1ori,trans_0ori;
	//get_transmat(Markerpointnames, testnames[0], Markerpoints, testpoints[0], R_0ori, t_0ori, trans_0ori);
	//get_transmat(Markerpointnames, testnames[1], Markerpoints, testpoints[1], R_1ori, t_1ori, trans_1ori);
	//Eigen::Matrix4d transT1B1, transT1B2;
	//transT1B1 = trans_0ori.matrix() * S2M_trans;
	//transT1B2 = trans_1ori.matrix() * S2M_trans;
	//cout << "T1B1Pose:  " <<endl<< transT1B1 << endl;
	//cout << "T1B2Pose:  " <<endl<< transT1B2 << endl;
	vector<Eigen::Affine3d> PoseN_transmat_ballbar;
	for (int i = 0; i < 10; i++)
	{
		Mat R, t;
		Eigen::Affine3d trans_ballbar;
		get_transmat(Markerpointnames, testnames[i], Markerpoints, testpoints[i], R, t, trans_ballbar);
		trans_ballbar = trans_ballbar * S2M_trans;
		PoseN_transmat_ballbar.push_back(trans_ballbar);
		cout << "Pose1-" << to_string(i + 1) << " : " << endl << trans_ballbar.matrix() << endl;
	}

	//转站初始位姿估计
	vector<vector<string>> est_pose_names;
	vector<vector<Point3d>> est_pose_points;
	for (int i = 0; i < 2; i++)
	{
		string Ntest_file = "resources\\qbs666\\pose" + to_string(i + 1) + ".txt";
		vector<Point3d> Ntestpoint;
		vector<string> Ntestname;
		read_ptcloud_file(Ntest_file, Ntestname, Ntestpoint);
		est_pose_names.push_back(Ntestname);
		est_pose_points.push_back(Ntestpoint);
	}
	Mat R_pose1_base1, t_pose1_base1, R_pose1_base2, t_pose1_base2;
	Eigen::Affine3d trans_pose1_base1, trans_pose1_base2;
	get_transmat(Markerpointnames, est_pose_names[0], Markerpoints, est_pose_points[0], R_pose1_base1, t_pose1_base1, trans_pose1_base1);
	get_transmat(Markerpointnames, est_pose_names[1], Markerpoints, est_pose_points[1], R_pose1_base2, t_pose1_base2, trans_pose1_base2);
	Eigen::Matrix4d trans_p1b1, trans_p1b2,trans_b2b1;
	trans_p1b1 = trans_pose1_base1.matrix() * S2M_trans;
	trans_p1b2 = trans_pose1_base2.matrix() * S2M_trans;
	cout << "pose1 to base1:  " << endl << trans_p1b1 << endl;
	cout << "pose1 to base2:  " << endl << trans_p1b2 << endl;
	trans_b2b1= trans_p1b1*trans_p1b2.inverse();
	cout << "base2 to base1:  " << endl << trans_b2b1 << endl;
	//转站
	vector<vector<string>> base_transnames;		//储存顺序为1-1，2-1，2-2，2-3
	vector<vector<Point3d>> base_transpoints;
	for (int i = 0; i < 4; i++)
	{
		string Trans1_file = "resources\\base_transfer\\240702\\2\\poses\\T1-" + to_string(i + 1) + ".txt";
		vector<Point3d> base_transpoint;
		vector<string> base_transname;
		read_ptcloud_file(Trans1_file, base_transname, base_transpoint);
		base_transnames.push_back(base_transname);
		base_transpoints.push_back(base_transpoint);
	}
	//分别表示扫描位姿1到基站1，扫描位姿1到基站2，扫描位姿2到基站2，扫描位姿3到基站2
	Mat R_1b1, t_1b1, R_1b2, t_1b2,R_2b2,t_2b2,R_3b2,t_3b2;		
	Eigen::Affine3d trans_1b1, trans_1b2,trans_2b2,trans_3b2;
	get_transmat(Markerpointnames, base_transnames[0], Markerpoints, base_transpoints[0], R_1b1, t_1b1, trans_1b1);
	get_transmat(Markerpointnames, base_transnames[1], Markerpoints, base_transpoints[1], R_1b2, t_1b2, trans_1b2);
	get_transmat(Markerpointnames, base_transnames[2], Markerpoints, base_transpoints[2], R_2b2, t_2b2, trans_2b2);
	get_transmat(Markerpointnames, base_transnames[3], Markerpoints, base_transpoints[3], R_3b2, t_3b2, trans_3b2);
	//1,2,3,4分别代表扫描位姿1，转站位姿，扫描位姿2，扫描位姿3
	Eigen::Matrix4d T1,T2,T3,T4,T12,T23,T24;
	T1 = trans_1b1.matrix() * S2M_trans;		//1到base记为T1
	T12 = (trans_1b2.matrix() * S2M_trans).inverse();
	T23= trans_2b2.matrix() * S2M_trans;
	T24 = trans_3b2.matrix() * S2M_trans;
	T2 = T1 * T12;
	cout << "T1:  " << endl << fixed << setprecision(5)<<T1 << endl;
	cout << "T12:  " << endl << T12 << endl;
	cout << "T23:  " << endl << T23 << endl;
	cout << "T24:  " << endl << T24 << endl;
	cout << "T2:  " << endl << T2 << endl;

	//转站2扫描标准球棒
	vector<vector<string>> testnames2;
	vector<vector<Point3d>> testpoints2;
	for (int i = 0; i < 2; i++)
	{
		string Ntest_file2 = "resources\\base_transfer\\240129\\poses\\T2B" + to_string(i + 1) + ".txt";
		vector<Point3d> Ntestpoint2;
		vector<string> Ntestname2;
		read_ptcloud_file(Ntest_file2, Ntestname2, Ntestpoint2);
		testnames2.push_back(Ntestname2);
		testpoints2.push_back(Ntestpoint2);
	}
	Mat R2_1ori, t2_1ori, R2_0ori, t2_0ori;
	Eigen::Affine3d trans2_1ori, trans2_0ori;
	get_transmat(Markerpointnames, testnames2[0], Markerpoints, testpoints2[0], R2_0ori, t2_0ori, trans2_0ori);
	get_transmat(Markerpointnames, testnames2[1], Markerpoints, testpoints2[1], R2_1ori, t2_1ori, trans2_1ori);
	Eigen::Matrix4d transT2B1, transT2B2;
	transT2B1 = trans2_0ori.matrix() * S2M_trans;
	transT2B2 = trans2_1ori.matrix() * S2M_trans;
	cout << "T2B1Pose:  " << endl << transT2B1 << endl;
	cout << "T2B2Pose:  " << endl << transT2B2<< endl;

	//转站2
	vector<vector<string>> base2_transnames;		//储存顺序为1-1，2-1，2-2，2-3
	vector<vector<Point3d>> base2_transpoints;
	for (int i = 0; i < 4; i++)
	{
		string Trans2_file = "resources\\base_transfer\\240129\\poses\\T2-" + to_string(i + 1) + ".txt";
		vector<Point3d> base2_transpoint;
		vector<string> base2_transname;
		read_ptcloud_file(Trans2_file, base2_transname, base2_transpoint);
		base2_transnames.push_back(base2_transname);
		base2_transpoints.push_back(base2_transpoint);
	}
	//分别表示扫描位姿4到基站1，扫描位姿4到基站2，扫描位姿5到基站2，扫描位姿6到基站2
	Mat R_4b1, t_4b1, R_4b2, t_4b2, R_5b2, t_5b2, R_6b2, t_6b2;
	Eigen::Affine3d trans_4b1, trans_4b2, trans_5b2, trans_6b2;
	get_transmat(Markerpointnames, base2_transnames[0], Markerpoints, base2_transpoints[0], R_4b1, t_4b1, trans_4b1);
	get_transmat(Markerpointnames, base2_transnames[1], Markerpoints, base2_transpoints[1], R_4b2, t_4b2, trans_4b2);
	get_transmat(Markerpointnames, base2_transnames[2], Markerpoints, base2_transpoints[2], R_5b2, t_5b2, trans_5b2);
	get_transmat(Markerpointnames, base2_transnames[3], Markerpoints, base2_transpoints[3], R_6b2, t_6b2, trans_6b2);
	//5,6,7,8分别代表扫描位姿4，转站位姿2，扫描位姿5，扫描位姿6
	Eigen::Matrix4d T5, T6, T7, T8, T56, T67, T68;
	T5 = trans_4b1.matrix() * S2M_trans;		//实际上为扫描位姿4在第一次转站后基站坐标系下的坐标，要转换到世界坐标要再乘T2
	T56 = (trans_4b2.matrix() * S2M_trans).inverse();
	T67 = trans_5b2.matrix() * S2M_trans;
	T68 = trans_6b2.matrix() * S2M_trans;
	T6 = T5 * T56;
	cout << "T5:  " << endl << fixed << setprecision(5) << T5 << endl;
	cout << "T56:  " << endl << T56 << endl;
	cout << "T67:  " << endl << T67 << endl;
	cout << "T68:  " << endl << T68 << endl;
	cout << "T6:  " << endl << T6 << endl;

	//扫描大型件
	vector<vector<string>> PoseN_names;
	vector<vector<Point3d>> PoseN_points;
	int scanpos_num = 25;
	for (int i = 0; i < scanpos_num; i++)
	{
		string PoseN_file = "resources\\240719\\poses\\S" + to_string(i+1) + ".txt";
		vector<string> PoseN_name;
		vector<Point3d> PoseN_point;
		read_ptcloud_file(PoseN_file, PoseN_name, PoseN_point);
		PoseN_names.push_back(PoseN_name);
		PoseN_points.push_back(PoseN_point);
	}
	vector<Mat> PoseN_R, PoseN_t;
	vector<Eigen::Affine3d> PoseN_transmat;
	for (int i = 0; i < scanpos_num; i++)
	{
		Mat R, t;
		Eigen::Affine3d trans;
		get_transmat(Markerpointnames, PoseN_names[i], Markerpoints, PoseN_points[i], R, t, trans);
		trans = trans * S2M_trans;
		PoseN_transmat.push_back(trans);
		cout << "Pose1-" << to_string(i+1) << " : " << endl << trans.matrix() << endl;
	}
	//for (int i = 0; i < scanpos_num; i++)
	//{
	//	ofstream outfile;
	//	outfile.open("resources\\base_transfer\\240702\\2\\poses\\qbs\\poses\\A" + to_string(i+1) + "-1.txt");
	//	outfile << fixed << setprecision(5) << PoseN_transmat[i].matrix() << endl;
	//	outfile.close();
	//}

	//第一次转站
	vector<vector<string>> PoseN_names2;
	vector<vector<Point3d>> PoseN_points2;
	int scanpos_num2 = 5;
	for (int i = 0; i < scanpos_num2; i++)
	{
		string PoseN_file2 = "resources\\base_transfer\\240702\\2\\poses\\S2-" + to_string(i+1) + ".txt";
		vector<string> PoseN_name2;
		vector<Point3d> PoseN_point2;
		read_ptcloud_file(PoseN_file2, PoseN_name2, PoseN_point2);
		PoseN_names2.push_back(PoseN_name2);
		PoseN_points2.push_back(PoseN_point2);
	}
	vector<Mat> PoseN_R2, PoseN_t2;
	vector<Eigen::Affine3d> PoseN_transmat2;
	Eigen::Matrix4d T2_mat;
	T2_mat << 0.986703365972317, -0.0490845610732438, 0.154924604253293, 1825.84229213453,
		0.0486987123935743, 0.998792767648395, 0.00627954664737309, -181.273377121055,
		-0.155059974628768, 0.00134836521812349, 0.987907850146439, 488.194259070893,
		0, 0, 0, 1;
	Eigen::Affine3d T2_opt(T2_mat);
	for (int i = 0; i < scanpos_num2; i++)
	{
		Mat R, t;
		Eigen::Affine3d trans;
		get_transmat(Markerpointnames, PoseN_names2[i], Markerpoints, PoseN_points2[i], R, t, trans);
		trans = trans * S2M_trans;
		PoseN_transmat2.push_back(trans);
		cout << "Pose2-" << to_string(i+1) << " : " << endl << trans.matrix() << endl;
	}
	//第二次转站
	vector<vector<string>> PoseN_names3;
	vector<vector<Point3d>> PoseN_points3;
	int scanpos_num3 = 12;
	for (int i = 0; i < scanpos_num3; i++)
	{
		string PoseN_file3 = "resources\\base_transfer\\240129\\poses\\S3-" + to_string(i) + ".txt";
		vector<string> PoseN_name3;
		vector<Point3d> PoseN_point3;
		read_ptcloud_file(PoseN_file3, PoseN_name3, PoseN_point3);
		PoseN_names3.push_back(PoseN_name3);
		PoseN_points3.push_back(PoseN_point3);
	}
	vector<Mat> PoseN_R3, PoseN_t3;
	vector<Eigen::Affine3d> PoseN_transmat3;
	Eigen::Matrix4d T3_mat;
	T3_mat << 0.947644015493517,-0.101583644429256,0.302732425865960,2398.24956526485,
		0.102281367001972,0.994662847688652,0.0135691410249679,-231.917221585127,
		-0.302494565822715,0.0181014251871836,0.952979157786304,666.166612892724,
		0,0,0,1;
	Eigen::Affine3d T3_opt(T3_mat);
	for (int i = 0; i < scanpos_num3; i++)
	{
		Mat R, t;
		Eigen::Affine3d trans;
		get_transmat(Markerpointnames, PoseN_names3[i], Markerpoints, PoseN_points3[i], R, t, trans);
		trans = trans * S2M_trans;
		PoseN_transmat2.push_back(trans);
		cout << "Pose3-" << to_string(i) << " : " << endl <<  trans.matrix() << endl;
	}
	return 0;
}