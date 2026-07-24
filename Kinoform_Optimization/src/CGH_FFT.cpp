// Copyright (C) 1991-93 2004-26 by SIT & 佐藤甲癸・戸塚真隆・木内健次
// CGH_FFT.cppのファイル（ファイル名：Ｃ＋＋仕様）
// ●　本ソフトは、ｘ６４（ＣＰＵの仕様：６４ビット）専用プログラム
// CV_PI;// 6.28318530717958647692  //CV_PI*2.0←にて２πを作成した。
// 
// ○ コンソールアプリケーションで作成すること
// ○
// ● ＯｐｅｎＣＶ4110シリーズとMicrosoft Visual Studio2026版を使用してください。
// ※

#include <omp.h>

#include <string.h>
#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <time.h>     // for clock()

#include <iostream>
#include <iomanip>

#include <fstream>
#include <string>
#include <sstream>

#include <vector>

#include <ppl.h>
#include <algorithm>
#include <windows.h>
#include <dshow.h>
#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <atomic>
#include <mmsystem.h>
#pragma comment (lib, "winmm.lib") 

using namespace Concurrency;

#define M_MAT 256
#define H_MAT 128

//#include "C:\\opencv\\opencv\\cv.hpp"
#include "C:\\opencv\\opencv2\\highgui\\highgui.hpp"
#include "C:\\opencv\\opencv2\\core\\core.hpp"
#include "C:\\opencv\\opencv2\\imgproc\\imgproc.hpp"
#include "C:\\opencv\\opencv2\\world.hpp"
#include "C:\\opencv\\opencv2\\opencv.hpp"
//#include "C:\\tools\\bm4d\\opencv2/quality/qualitypsnr.hpp"
//Releaseモードの場合
#pragma comment(lib,"C:\\opencv\\opencv_world4110.lib")

using namespace cv;
using namespace std;

//０：評価用コマンドをオンにする
//１：評価用コマンドをオフにする
#define MODE_CUT 0
//０：表示用コマンドをオンにする
//１：表示用コマンドをオフにする
#define DISP_CUT 0

//０：キノフォーム実像（-）
//１：キノフォーム虚像（+）
#define KINO_ReIm 0

static string FileNameBMP[10];


// ==========================================================
// 標準SSIM（AI論文比較用: K1=0.01, K2=0.03）の計算関数
// 引数 i1, i2 は 0-255の 8bit画像(CV_8U) または 64F画像 を想定
// ==========================================================
// ==========================================================
// SSIM 共通計算関数
// C1, C2 を引数で指定できる（既定値は標準SSIM: L=255, K1=0.01, K2=0.03）
//   ・標準SSIM（8bit 0-255画像）: C1=6.5025, C2=58.5225
//   ・吉川先生の定数（0-1画像） : C1=0.01,   C2=0.03
// depth は内部演算精度（CV_32F または CV_64F）
// ==========================================================
cv::Scalar computeSSIM(const cv::Mat& i1, const cv::Mat& i2,
	double C1 = 6.5025, double C2 = 58.5225, int depth = CV_64F)
{
	cv::Mat I1, I2;
	i1.convertTo(I1, depth);
	i2.convertTo(I2, depth);

	cv::Mat I2_2 = I2.mul(I2);
	cv::Mat I1_2 = I1.mul(I1);
	cv::Mat I1_I2 = I1.mul(I2);

	cv::Mat mu1, mu2;
	cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
	cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

	cv::Mat mu1_2 = mu1.mul(mu1);
	cv::Mat mu2_2 = mu2.mul(mu2);
	cv::Mat mu1_mu2 = mu1.mul(mu2);

	cv::Mat sigma1_2, sigma2_2, sigma12;
	cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;

	cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;

	cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;

	cv::Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);

	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);

	cv::Mat ssim_map;
	cv::divide(t3, t1, ssim_map);

	return cv::mean(ssim_map);
}

// ==========================================================
// 標準SSIM（AI論文比較用: K1=0.01, K2=0.03, L=255）
// 引数 i1, i2 は 0-255の 8bit画像(CV_8U) または 64F画像 を想定
// ==========================================================
double getStandardSSIM(const cv::Mat& i1, const cv::Mat& i2)
{
	return computeSSIM(i1, i2, 6.5025, 58.5225, CV_32F)[0];
}
// ==========================================================
// 吉川先生の定数版SSIM（0-1に正規化した画像用: C1=0.01, C2=0.03）
// ==========================================================
cv::Scalar getSSIM(const cv::Mat& i1, const cv::Mat& i2) {
	return computeSSIM(i1, i2, 0.01, 0.03, CV_64F);
}

// HVS用の簡易重み行列（例として定義）
cv::Mat generateHVSWeight(const cv::Size& blockSize) {
	cv::Mat weights = (cv::Mat_<double>(blockSize) <<
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5,
		0.5, 0.6, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5,
		0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.5,
		0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.5,
		0.5, 0.6, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5,
		0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
		);
	return weights;
}

// 画像を8x8ブロックに分割し、DCTと重み付けを適用
cv::Mat applyDCTandWeight(const cv::Mat& image, const cv::Mat& weights) {
	cv::Mat processed = cv::Mat::zeros(image.size(), CV_64F);
	for (int i = 0; i < image.rows; i += 8) {
		for (int j = 0; j < image.cols; j += 8) {
			cv::Rect blockRect(j, i, 8, 8);
			cv::Mat block = image(blockRect);
			cv::Mat blockdouble;
			block.convertTo(blockdouble, CV_64F);

			// DCTを適用
			cv::Mat dctBlock;
			cv::dct(blockdouble, dctBlock);

			// 重み行列を適用
			cv::Mat weightedBlock = dctBlock.mul(weights);

			// 結果を保存
			weightedBlock.copyTo(processed(blockRect));
		}
	}
	return processed;
}
// additional functions/////////////////////////////////////
void addNoiseSoltPepperMono(Mat& src, Mat& dest, double per)
{
	cv::RNG rng;
#pragma omp parallel for
	for (int j = 0; j < src.rows; j++)
	{
		uchar* s = src.ptr(j);
		uchar* d = dest.ptr(j);
		for (int i = 0; i < src.cols; i++)
		{
			double a1 = rng.uniform((double)0, (double)1);

			if (a1 > per)
				d[i] = s[i];
			else
			{
				double a2 = rng.uniform((double)0, (double)1);
				if (a2 > 0.5)d[i] = 0;
				else d[i] = 255;
			}
		}
	}
}
void addNoiseMono(Mat& src, Mat& dest, double sigma)
{
	Mat s;
	src.convertTo(s, CV_16S);
	Mat n(s.size(), CV_16S);
	randn(n, 0, sigma);
	Mat temp = s + n;
	temp.convertTo(dest, CV_8U);
}

double GetPSNR(const cv::Mat& I1, const cv::Mat& I2) {
	cv::Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	s1.convertTo(s1, CV_64F);  // Convert to double
	s1 = s1.mul(s1);           // Square each element

	cv::Scalar s = sum(s1);    // Sum elements per channel

	double sse = s.val[0] + s.val[1] + s.val[2]; // Sum channels

	if (sse <= 1e-10) { // For small values return zero
		return 0;
	}
	else {
		double mse = sse / (double)(I1.channels() * I1.total());
		double psnr = 10.0 * log10((255.0 * 255.0) / mse);
		return psnr;
	}
}

double calcPSNR(Mat& src, Mat& dest)
{
	Mat ssrc;
	Mat ddest;
	if (src.channels() == 1)
	{
		src.convertTo(ssrc, CV_64F);
		dest.convertTo(ddest, CV_64F);
	}
	else
	{
		cvtColor(src, ssrc, cv::COLOR_BGR2YUV);
		cvtColor(dest, ddest, cv::COLOR_BGR2YUV);
		//cvtColor(src, ssrc, COLOR_YUV2BGR_NV12);
		//cvtColor(dest, ddest, COLOR_YUV2BGR_NV12);
	}
	double sn = GetPSNR(ssrc, ddest);
	return sn;
}

// PSNR-HVSを計算
double calculatePSNRHVS(const cv::Mat& original, const cv::Mat& distorted, const cv::Mat& weights) {
	cv::Mat originalProcessed = applyDCTandWeight(original, weights);
	cv::Mat distortedProcessed = applyDCTandWeight(distorted, weights);

	// 平均二乗誤差（MSE）を計算
	cv::Mat diff = originalProcessed - distortedProcessed;
	cv::Mat squaredError;
	cv::multiply(diff, diff, squaredError);

	double mse = cv::mean(squaredError)[0];
	double maxPixel = 255.0;

	// PSNR-HVSを計算
	double psnrHVS = 10.0 * std::log10((maxPixel * maxPixel) / mse);
	return psnrHVS;
}
/**************************/
// 簡易HVS重み行列生成
cv::Mat M_generateHVSWeight(const cv::Size& blockSize) {
	cv::Mat weights = (cv::Mat_<double>(blockSize) <<
		0.5, 0.6, 0.6, 0.7, 0.7, 0.6, 0.6, 0.5,
		0.6, 0.7, 0.8, 0.8, 0.8, 0.8, 0.7, 0.6,
		0.6, 0.8, 0.9, 0.9, 0.9, 0.9, 0.8, 0.6,
		0.7, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.7,
		0.7, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.7,
		0.6, 0.8, 0.9, 0.9, 0.9, 0.9, 0.8, 0.6,
		0.6, 0.7, 0.8, 0.8, 0.8, 0.8, 0.7, 0.6,
		0.5, 0.6, 0.6, 0.7, 0.7, 0.6, 0.6, 0.5
		);
	return weights;
}

// マスキング係数の計算
cv::Mat calculateMasking(const cv::Mat& dctBlock) {
	cv::Mat masking = cv::abs(dctBlock); // DCT係数の絶対値を利用
	return cv::min(masking, cv::Mat::ones(masking.size(), CV_64F) * 255); // 範囲制限
}

// 画像を8x8ブロックに分割し、DCTと重み付けとマスキングを適用
cv::Mat applyDCTWeightMasking(const cv::Mat& image, const cv::Mat& weights) {
	cv::Mat processed = cv::Mat::zeros(image.size(), CV_64F);

	for (int i = 0; i < image.rows; i += 8) {
		for (int j = 0; j < image.cols; j += 8) {
			cv::Rect blockRect(j, i, 8, 8);
			cv::Mat block = image(blockRect);
			cv::Mat blockdouble;
			block.convertTo(blockdouble, CV_64F);

			// DCTを適用
			cv::Mat dctBlock;
			cv::dct(blockdouble, dctBlock);

			// HVS重みを適用
			cv::Mat weightedBlock = dctBlock.mul(weights);

			// マスキングを適用
			cv::Mat masking = calculateMasking(dctBlock);
			cv::Mat maskedBlock = weightedBlock.mul(masking);

			// 結果を保存
			maskedBlock.copyTo(processed(blockRect));
		}
	}
	return processed;
}

// PSNR-HVS-Mを計算
double calculatePSNRHVSM(const cv::Mat& original, const cv::Mat& distorted, const cv::Mat& weights) {
	cv::Mat originalProcessed = applyDCTWeightMasking(original, weights);
	cv::Mat distortedProcessed = applyDCTWeightMasking(distorted, weights);

	// 平均二乗誤差（MSE）を計算
	cv::Mat diff = originalProcessed - distortedProcessed;
	cv::Mat squaredError;
	cv::multiply(diff, diff, squaredError);

	double mse = cv::mean(squaredError)[0];
	double maxPixel = 255.0;

	// PSNR-HVS-Mを計算
	double psnrHVSM = 10.0 * std::log10((maxPixel * maxPixel) / mse);
	return psnrHVSM;
}
/**************************/
void sinF(const Mat& mat, Mat& sinMat) {
#pragma omp parallel for collapse(2)
	// 各要素に対してsin関数を適用
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			sinMat.at<double>(i, j) = std::sinf(mat.at<double>(i, j));
		}
	}
}
void cosF(const Mat& mat, Mat& cosMat) {
#pragma omp parallel for collapse(2)
	// 各要素に対してcos関数を適用
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			cosMat.at<double>(i, j) = std::cosf(mat.at<double>(i, j));
		}
	}
}
void arctan2F(const Mat& mat1, const Mat& mat0, Mat& atanMat) {
#pragma omp parallel for collapse(2)
	// 各要素に対してsin関数を適用
	for (int i = 0; i < mat0.rows; ++i) {
		for (int j = 0; j < mat0.cols; ++j) {
			atanMat.at<double>(i, j) = std::atan2f(mat1.at<double>(i, j), mat0.at<double>(i, j));
		}
	}
}
void writeMatToCSV(const Mat& mat, const string& filename) {
	ofstream file(filename);

	if (!file.is_open()) {
		cerr << "Error opening file: " << filename << endl;
		return;
	}

	for (int i = 0; i < mat.rows; ++i) {
	    for (int j = mat.cols - 1; j > 0 ; --j) {
		//for (int j = 0; j < mat.cols; ++j) {
			file << mat.at<double>(i, j);
			if (j > 0) {
			//if (j < mat.cols - 1) {
				file << ",";
			}
		}
		file << endl;
	}

	file.close();
}

string Img_Read(string FILE_DTOP, int NsLoop, Mat& grayImage, Mat& target_mat, int curSize, int BLK)
{
	std::string FileNameBMP[] = { 
		"test1.bmp",
		"test2.bmp",
		"test3.bmp",
		"test4.bmp",
		"test5.bmp",
		"test6.bmp",
		"test7.bmp",
		"test8.bmp",
		"test9.bmp",
		"test10.bmp" };

	string FName_END = "NULL";
	FName_END = FILE_DTOP + "data\\" + FileNameBMP[NsLoop];
	target_mat = cv::imread(FName_END, IMREAD_GRAYSCALE);
	if (target_mat.empty())
	{
		std::cerr << "The image file could not be loaded." << std::endl;
		return FName_END;  // ← return を追加
	}
	// ② target_mat のサイズが grayImage より小さくないか確認
	if (target_mat.cols < grayImage.cols || target_mat.rows < grayImage.rows)
	{
		std::cerr << "The loaded image is smaller than the reference grayImage.:target="
			<< target_mat.cols << "x" << target_mat.rows
			<< " gray=" << grayImage.cols << "x" << grayImage.rows << std::endl;
		return FName_END;
	}
	target_mat = target_mat(cv::Rect(1, 1, grayImage.cols - 2, grayImage.rows - 2)).clone();


	cv::copyMakeBorder(target_mat, target_mat, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255));//画像の周囲を削る


	target_mat.rowRange(0, (M_MAT / 2)).copyTo(target_mat);//上半分を基準像にする
	target_mat.convertTo(target_mat, CV_8U);//グレースケール画像に変換
	int border = BLK; // 削りたいピクセル数
	if (BLK != 1)
	{
		// --- 1. target_mat をボーダー分だけ削る ---
		if (target_mat.cols <= 2 * border || target_mat.rows <= 2 * border)
		{
			std::cerr << "BLK is too large to be trimmed from target_mat." << std::endl;
			return FName_END;
		}
		target_mat = target_mat(
			cv::Rect(border, border,
				target_mat.cols - 2 * border,
				target_mat.rows - 2 * border)).clone();
		// --- 2. grayImage を target_mat と同じ横幅、縦 2 倍で作成 ---
		int width = target_mat.cols;
		int height = target_mat.rows - BLK * 2;// (target_mat.rows) * 2;

		grayImage = cv::Mat::zeros(cv::Size(width, height), CV_8U);

		// 上半分に target_mat
		cv::Mat upper = grayImage(cv::Rect(0, 0, width, target_mat.rows));
		target_mat.copyTo(upper);

		//Mat COPYgrayImage = grayImage(cv::Rect(1, 1, grayImage.cols - 2 * 1, grayImage.rows - 2 * 1)).clone();
		//cv::copyMakeBorder(COPYgrayImage, grayImage, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255));


	}
	else
	{
		//cv::Mat COPYtarget_mat = target_mat(cv::Rect(border, border, target_mat.cols - 2 * border, target_mat.rows - 2 * border)).clone();
		//cv::copyMakeBorder(COPYtarget_mat, target_mat, BLK, BLK, BLK, BLK, cv::BORDER_CONSTANT, cv::Scalar(255));
		// gray_image を M_MATxM_MAT の黒背景に左上角合わせで合成
		grayImage = cv::Mat::zeros(cv::Size(curSize, curSize), CV_8U);//M_MAT x M_MAT の黒背景を作成
		const int copy_w0 = curSize;//std::min(target_mat.cols, curSize);
		const int copy_h0 = std::min(target_mat.rows, curSize);
		target_mat(cv::Rect(0, 0, copy_w0, copy_h0)).copyTo(grayImage(cv::Rect(0, 0, copy_w0, copy_h0)));// target_mat を左上にコピー
	}


	//int big_width = target_mat.cols > target_mat.rows ? target_mat.cols : target_mat.rows;//縦横どっちか長い方は？
	//double ratio = ((double)curSize / (double)big_width);//割合
	//cv::resize(target_mat, grayImage, cv::Size(), ratio, ratio, cv::INTER_NEAREST);//リサイズ

	return  FName_END;
}
void Img_FFT(const int NbannMe, Mat& WkQ, Mat& re, Mat& im,
	Mat& complexImage, Mat* planes,
	Mat& Trans, int curSize)
{
	Mat WkP(curSize, curSize, CV_64F);
	Mat WkR(curSize, curSize, CV_64F);

	cosF(Trans, WkP);
	planes[0] = WkQ.mul(WkP);//乱数のサイン・コサインをテーブルにした。//b[i][MAX_MAT - 1 - j]*exp(0.0)*cos(Trans[i][j]);
	sinF(Trans, WkR);
	planes[1] = WkQ.mul(WkR);//乱数のサイン・コサインをテーブルにした。//b[i][MAX_MAT - 1 - j]*exp(0.0)*sin(Trans[i][j]);

	cv::merge(planes, 2, complexImage);
	cv::dft(complexImage, complexImage);
	cv::split(complexImage, planes);

	re = planes[0];// 　実数部の代入
	im = planes[1];// 　虚数部の代入
}

// コンソールの文字色変更関数
void SetConsoleColor(WORD color) {
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(hConsole, color);
}
// コンピューター名取得関数
std::string getComputerName() {
	char name[256];
	DWORD size = sizeof(name);
	//if (GetComputerNameW(name, &size)) {// Unicode版の場合
	if (GetComputerNameA(name, &size)) {// ANSI版の場合
		return std::string(name);
	}
	return "Unknown";
}

// 2D 大津法本体：jointHist を使って最適 (s,t) を求める。
// 戻り値: bestBetween (最大の between-class variance)
// best_s, best_t にそれぞれしきい値を格納
static double compute2DOtsuFromJoint(const cv::Mat& jointHist, int& best_s, int& best_t)
{
	// jointHist: CV_64F 256x256 (生カウント)
	const int L = 256;
	double total = cv::sum(jointHist)[0];
	if (total <= 0.0) { best_s = best_t = 0; return 0.0; }

	// 正規化確率 p(i,j)
	cv::Mat P(L, L, CV_64F);
	P = jointHist / total;

	// 累積和 (size = (L+1)x(L+1)) を作る (S, SX, SY)
	std::vector<std::vector<double>> S(L + 1, std::vector<double>(L + 1, 0.0));
	std::vector<std::vector<double>> SX(L + 1, std::vector<double>(L + 1, 0.0));
	std::vector<std::vector<double>> SY(L + 1, std::vector<double>(L + 1, 0.0));

	for (int i = 0; i < L; ++i) {
		double rowSumS = 0.0, rowSumSX = 0.0, rowSumSY = 0.0;
		for (int j = 0; j < L; ++j) {
			double pij = P.at<double>(i, j);
			rowSumS += pij;
			rowSumSX += pij * static_cast<double>(i);
			rowSumSY += pij * static_cast<double>(j);
			S[i + 1][j + 1] = S[i][j + 1] + rowSumS;
			SX[i + 1][j + 1] = SX[i][j + 1] + rowSumSX;
			SY[i + 1][j + 1] = SY[i][j + 1] + rowSumSY;
		}
	}

	// 全体の一次モーメント
	double SX_total = SX[L][L];
	double SY_total = SY[L][L];

	// 全探索 (s,t) を評価
	double bestBetween = -1.0;
	best_s = best_t = 0;
	for (int s = 0; s < L; ++s) {
		for (int t = 0; t < L; ++t) {
			// C0: i <= s, j <= t  -> access cumulative at (s+1,t+1)
			double w0 = S[s + 1][t + 1];
			if (w0 <= 0.0 || w0 >= 1.0) continue;
			double sx0 = SX[s + 1][t + 1];
			double sy0 = SY[s + 1][t + 1];
			double mu0x = sx0 / w0;
			double mu0y = sy0 / w0;

			double w1 = 1.0 - w0;
			double sx1 = SX_total - sx0;
			double sy1 = SY_total - sy0;
			double mu1x = sx1 / w1;
			double mu1y = sy1 / w1;

			double dx = mu0x - mu1x;
			double dy = mu0y - mu1y;
			double between = w0 * w1 * (dx * dx + dy * dy);

			if (between > bestBetween) {
				bestBetween = between;
				best_s = s;
				best_t = t;
			}
		}
	}
	return bestBetween;
}
// approx2D Otsu (Srv_hist のみを使った近似 2D 大津法)
// 説明:
//   入力は Srv_hist（1D ヒストグラム、256 ビン）だけで、画像データを使わずに近似的な 2D 大津法のしきい値
//   (s,t) を求めます。内部では「強度 i に対して局所平均 j が i の周りに分布する」という仮定のもとに
//   ガウスで平滑化して 256x256 の擬似 joint-histogram を作成し、既存の
//   compute2DOtsuFromJoint を用いて (s,t) を探索します。最終的に「白比率」「黒比率」も返します。
//
// 引数:
//   - const cv::Mat& Srv_hist
//       1D ヒストグラム（長さ 256 の CV_64F または CV_64F、あるいは 1x256 / 256x1 行ベクトル）
//   - int& out_s
//       出力: 強度側の閾値 s (0..255)
//   - int& out_t
//       出力: 局所平均側の閾値 t (0..255)
//   - double& whiteRatioPercent
//       出力: 「白」と判定された画素（ピクセル）の割合（%）
//   - double& blackRatioPercent
//       出力: 「黒」と判定された画素（ピクセル）の割合（%）
//   - double sigma = 5.0
//       オプション: 擬似 joint-histogram を作る際のガウス幅（標準偏差）。小さくすると mean=j に近い鋭い分布。
// 戻り値:
//   なし（出力は参照引数 out_s, out_t, whiteRatioPercent, blackRatioPercent に設定される）
//
// 注意:
//   - ヒストグラムに全要素 0 が含まれる場合は out_s=out_t=0、白比率=0、黒比率=100 を返します。
//   - 本手法は「近似」です。真の 2D 大津（画像の joint-histogram）と結果が異なる場合があります。
//   - compute2DOtsuFromJoint（ファイル内既存関数）を利用します。
static void approx2DOtsuFromSrvHist_only(
	const cv::Mat& Srv_hist,
	int& out_s,
	int& out_t,
	double& whiteRatioPercent,
	double& blackRatioPercent,
	double sigma = 5.0)
{
	// 初期化
	out_s = 0; out_t = 0;
	whiteRatioPercent = 0.0; blackRatioPercent = 100.0;

	if (Srv_hist.empty()) return;

	// ヒストグラムを CV_64F のベクトルに変換（形状 256 要素）
	cv::Mat hist64;
	if (Srv_hist.type() == CV_64F) {
		hist64 = Srv_hist.clone();
	}
	else {
		Srv_hist.convertTo(hist64, CV_64F);
	}

	// 形状チェック：1x256 または 256x1 を 256 長さのベクトルにする
	cv::Mat histVec;
	if (hist64.rows == 1 && hist64.cols == 256) histVec = hist64.reshape(1, 256);
	else if (hist64.rows == 256 && hist64.cols == 1) histVec = hist64;
	else if (hist64.total() == 256) histVec = hist64.reshape(1, 256);
	else {
		// 期待外の形状
		return;
	}

	// 合計を取得
	double totalCount = cv::sum(histVec)[0];
	if (totalCount <= 0.0) {
		// 全てゼロ
		out_s = out_t = 0;
		whiteRatioPercent = 0.0;
		blackRatioPercent = 100.0;
		return;
	}

	// 擬似 joint-histogram (256 x 256) を作成
	cv::Mat jointHist = cv::Mat::zeros(256, 256, CV_64F);

	// ガウスカーネル作成（中心差分だけ利用するため、各中心 i に対して j の重みを計算）
	const double twoSigma2 = 2.0 * sigma * sigma;
	for (int i = 0; i < 256; ++i) {
		double hi = histVec.at<double>(i, 0);
		if (hi <= 0.0) continue;
		// j 軸での重み（0..255）
		double sumW = 0.0;
		double wbuf[256];
		for (int j = 0; j < 256; ++j) {
			double d = static_cast<double>(j - i);
			double w = std::exp(-(d * d) / twoSigma2);
			wbuf[j] = w;
			sumW += w;
		}
		// 正規化して jointHist に加算
		if (sumW <= 0.0) continue;
		double factor = hi / sumW;
		for (int j = 0; j < 256; ++j) {
			jointHist.at<double>(i, j) += factor * wbuf[j];
		}
	}

	// jointHist が作成されたので既存の compute2DOtsuFromJoint を利用して (s,t) を求める
	int best_s = 0, best_t = 0;
	compute2DOtsuFromJoint(jointHist, best_s, best_t);

	// C0 を (i <= s && j <= t)、C1 をその補集合とする
	double total = cv::sum(jointHist)[0];
	if (total <= 0.0) {
		out_s = best_s; out_t = best_t;
		whiteRatioPercent = 0.0; blackRatioPercent = 100.0;
		return;
	}

	double cntC0 = 0.0, cntC1 = 0.0;
	double sumI_C0 = 0.0, sumI_C1 = 0.0; // 各クラスの intensity の合計（重み付き）
	for (int i = 0; i < 256; ++i) {
		for (int j = 0; j < 256; ++j) {
			double v = jointHist.at<double>(i, j);
			if (i <= best_s && j <= best_t) {
				cntC0 += v;
				sumI_C0 += v * static_cast<double>(i);
			}
			else {
				cntC1 += v;
				sumI_C1 += v * static_cast<double>(i);
			}
		}
	}

	// 各クラスの平均強度
	double meanC0 = (cntC0 > 0.0) ? (sumI_C0 / cntC0) : 0.0;
	double meanC1 = (cntC1 > 0.0) ? (sumI_C1 / cntC1) : 0.0;

	// 白は平均強度が大きい方のクラスとする
	double whiteCount = 0.0;
	if (meanC1 >= meanC0) whiteCount = cntC1;
	else whiteCount = cntC0;

	whiteRatioPercent = 100.0 * (whiteCount / total);
	blackRatioPercent = 100.0 - whiteRatioPercent;

	// 出力
	out_s = best_s;
	out_t = best_t;
}



// --- drawHistogram の中の 2D 大津部分を真の2D大津に置き換え ---
// （既存の drawHistogram 関数内で meanImg を作っている箇所をこの実装に置換してください）
int drawHistogram(const cv::Mat& Mst_hist, const cv::Mat& Srv_hist, cv::Mat& histImage, int& Deff_median_OUT, Mat& img_Mst, Mat& img_Srv)
{
	int histSize = 256;
	int hist_w = 768; // ヒストグラムの幅
	int hist_h = 250; // ヒストグラムの基準高さ
	int scale_bar_height = 12;
	int label_height = 40;
	int text_height = 150; // 評価指標表示用に拡張

	int total_height = hist_h + scale_bar_height + label_height + text_height;

	// 背景色を白に
	histImage = cv::Mat(total_height, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));
	// 正規化（縦軸を固定値に基づいてスケーリング）
	//cv::Mat normMstHist, normSrvHist;
	//cv::normalize(Mst_hist, normMstHist, 0, hist_h, cv::NORM_MINMAX);
	//cv::normalize(Srv_hist, normSrvHist, 0, hist_h, cv::NORM_MINMAX);


	// 2本のヒストグラムを同一縦軸スケールで描く（共通maxでスケール）
	const double mstMaxCount = cv::norm(Mst_hist, cv::NORM_INF); // max value
	const double srvMaxCount = cv::norm(Srv_hist, cv::NORM_INF); // max value
	const double maxCount = std::max(mstMaxCount, srvMaxCount);

	cv::Mat normMstHist, normSrvHist;
	if (maxCount > 0.0) {
		const double scale = static_cast<double>(hist_h) / maxCount;
		Mst_hist.convertTo(normMstHist, CV_32F, scale);
		Srv_hist.convertTo(normSrvHist, CV_32F, scale);
	}
	else {
		normMstHist = cv::Mat::zeros(Mst_hist.size(), CV_32F);
		normSrvHist = cv::Mat::zeros(Srv_hist.size(), CV_32F);
	}
	
	int bin_w = cvRound((double)hist_w / histSize);

	// Srv_histを黒線で描画（元は白線）
	for (int i = 1; i < histSize; i++) {
		cv::line(histImage,
			cv::Point(bin_w * (i - 1), hist_h - cvRound(normSrvHist.at<float>(i - 1))),
			cv::Point(bin_w * i, hist_h - cvRound(normSrvHist.at<float>(i))),
			cv::Scalar(0, 0, 0), 2, 8, 0); // 黒色
	}

	// Mst_histを赤線で描画（そのまま）
	for (int i = 1; i < histSize; i++) {
		cv::line(histImage,
			cv::Point(bin_w * (i - 1), hist_h - cvRound(normMstHist.at<float>(i - 1))),
			cv::Point(bin_w * i, hist_h - cvRound(normMstHist.at<float>(i))),
			cv::Scalar(0, 0, 255), 2, 8, 0); // 赤色
	}

	// グレースケールバーの描画（そのまま）
	for (int i = 0; i < hist_w; i++) {
		int v = cvRound((double)i / hist_w * 255.0);
		cv::line(histImage,
			cv::Point(i, hist_h),
			cv::Point(i, hist_h + scale_bar_height - 1),
			cv::Scalar(v, v, v), 1);
	}

	// 目盛り線とラベル（色を黒に）
	int scale_y = hist_h + scale_bar_height;
	for (int i = 0; i <= histSize; i += 10) {
		int x = bin_w * i;
		cv::line(histImage, cv::Point(x, hist_h + scale_bar_height), cv::Point(x, scale_y + 6), cv::Scalar(0, 0, 0), 1);
		if (i == 0 || i == 128 || i == 255) {
			cv::putText(histImage, std::to_string(i), cv::Point(x - 12, scale_y + 24),
				cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
		}
	}

	// 軸ラベル（色を黒に）
	cv::putText(histImage, "Intensity (Grayscale 0-255)", cv::Point(425, total_height - text_height - 25),
		cv::FONT_HERSHEY_SIMPLEX, 0.73, cv::Scalar(0, 0, 0), 2);

	// --- 画像エントロピー計算 ---
	auto calcEntropy = [](const cv::Mat& img) -> double {
		cv::Mat hist;
		int histSize = 256;
		float range[] = { 0, 256 };
		const float* histRange = { range };
		cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
		double total = cv::sum(hist)[0];
		if (total <= 0.0) return 0.0;
		double entropy = 0.0;
		for (int i = 0; i < histSize; ++i) {
			double p = hist.at<float>(i) / total;
			if (p > 0.0) entropy -= p * std::log2(p);
		}
		return entropy;
		};

	// --- 空間周波数計算 ---
	auto calcSpatialFrequency = [](const cv::Mat& img) -> double {
		cv::Mat imgF;
		img.convertTo(imgF, CV_64F);
		double rf = 0.0, cf = 0.0;
		for (int y = 0; y < imgF.rows; ++y) {
			for (int x = 1; x < imgF.cols; ++x) {
				double diff = imgF.at<double>(y, x) - imgF.at<double>(y, x - 1);
				rf += diff * diff;
			}
		}
		rf = std::sqrt(rf / (imgF.rows * imgF.cols));
		for (int y = 1; y < imgF.rows; ++y) {
			for (int x = 0; x < imgF.cols; ++x) {
				double diff = imgF.at<double>(y, x) - imgF.at<double>(y - 1, x);
				cf += diff * diff;
			}
		}
		cf = std::sqrt(cf / (imgF.rows * imgF.cols));
		return std::sqrt(rf * rf + cf * cf);
		};

	// --- エントロピー・空間周波数を元画像から計算 ---
	double entropy_Mst = 0.0, entropy_Srv = 0.0, sf_Mst = 0.0, sf_Srv = 0.0;
	if (!img_Mst.empty()) {
		entropy_Mst = calcEntropy(img_Mst);
		sf_Mst = calcSpatialFrequency(img_Mst);
	}
	if (!img_Srv.empty()) {
		entropy_Srv = calcEntropy(img_Srv);
		sf_Srv = calcSpatialFrequency(img_Srv);
	}

	// --- ヒストグラム下部に数値を描画 ---
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(3);

	int out_s = 0, out_t = 0;
	double whiteRatioPercent;
	double blackRatioPercent;

	approx2DOtsuFromSrvHist_only(Srv_hist, out_s, out_t, whiteRatioPercent, blackRatioPercent, 5.0);
	// 表示（2D Otsu）
	std::ostringstream otsuStream, otsuPercStream;
	otsuStream << "Otsu2D(approx) s=" << out_s << " t=" << out_t;
	otsuPercStream << " W:" << std::fixed << std::setprecision(2) << whiteRatioPercent << "% B:" << blackRatioPercent << "%";
	cv::putText(histImage, otsuStream.str() + otsuPercStream.str(), cv::Point(10, total_height - text_height + 25 * 0),
		cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 0, 0), 2);

	// --- 続き: 平均・中央値・標準偏差の描画（元のコードをそのまま流用） ---
	auto calculateStats = [](const cv::Mat& hist) {
		double mean = 0.0, stddev = 0.0, total = 0.0;
		for (int i = 0; i < hist.rows; i++) {
			double value = hist.at<float>(i);
			mean += i * value;
			total += value;
		}
		if (total == 0.0) return std::make_pair(0.0, 0.0);
		mean /= total;

		for (int i = 0; i < hist.rows; i++) {
			double value = hist.at<float>(i);
			stddev += (i - mean) * (i - mean) * value;
		}
		stddev = std::sqrt(stddev / total);
		return std::make_pair(mean, stddev);
		};

	auto calculateMedian = [](const cv::Mat& hist) {
		double total = 0.0;
		for (int i = 0; i < hist.rows; ++i) total += hist.at<float>(i);
		if (total <= 0.0) return 0.0;
		double half = total * 0.5;
		double cum = 0.0;
		for (int i = 0; i < hist.rows; ++i) {
			cum += hist.at<float>(i);
			if (cum >= half) return static_cast<double>(i);
		}
		return static_cast<double>(hist.rows - 1);
		};

	auto [Mst_mean, Mst_stddev] = calculateStats(Mst_hist);
	auto [Srv_mean, Srv_stddev] = calculateStats(Srv_hist);
	double Mst_median = calculateMedian(Mst_hist);
	double Srv_median = calculateMedian(Srv_hist);

	// Deff の統計
	double Deff_mean = 0.0, Deff_std = 0.0, Deff_median = 0.0;
	Deff_mean = (Srv_mean - Mst_mean);
	Deff_std = (Srv_stddev - Mst_stddev);
	Deff_median = (Srv_median - Mst_median);

	std::ostringstream SrvMeanStream, SrvMedianStream, SrvStddevStream;
	SrvMeanStream << std::fixed << std::setprecision(1) << Srv_mean;
	if (Srv_median <= 100)
	{
		SrvMedianStream << std::fixed << std::setprecision(1) << Srv_median;
	}
	else
	{
		SrvMedianStream << std::fixed << std::setprecision(1) << Srv_median;
	}
	SrvStddevStream << std::fixed << std::setprecision(1) << Srv_stddev;
	double fontsize = 1.1, linewith = 32;
	oss.str(""); oss.clear();
	oss << "Mean.:  " << SrvMeanStream.str();
	cv::putText(histImage, oss.str(), cv::Point(10, total_height - text_height + linewith * 2),
		cv::FONT_HERSHEY_SIMPLEX, fontsize, cv::Scalar(0, 0, 0), 2);
	oss.str(""); oss.clear();
	oss << "Median: " << SrvMedianStream.str();
	cv::putText(histImage, oss.str(), cv::Point(10, total_height - text_height + linewith * 3),
		cv::FONT_HERSHEY_SIMPLEX, fontsize, cv::Scalar(0, 0, 0), 2);
	oss.str(""); oss.clear();
	oss << "StdDev: " << SrvStddevStream.str();
	cv::putText(histImage, oss.str(), cv::Point(10, total_height - text_height + linewith * 4),
		cv::FONT_HERSHEY_SIMPLEX, fontsize, cv::Scalar(0, 0, 0), 2);
	cv::putText(histImage, "output", cv::Point(5, total_height - text_height + linewith * 1),
		cv::FONT_HERSHEY_SIMPLEX, fontsize, cv::Scalar(0, 0, 0), 2);

	// Mst_hist側の描画
	std::ostringstream MstMeanStream, MstMedianStream, MstStddevStream;
	MstMeanStream << std::fixed << std::setprecision(1) << Mst_mean;
	if (Mst_median <= 100)
	{
		MstMedianStream << std::fixed << std::setprecision(1) << Mst_median;
	}
	else
	{
		MstMedianStream << std::fixed << std::setprecision(1) << Mst_median;
	}
	MstStddevStream << std::fixed << std::setprecision(1) << Mst_stddev;

	int Mst_x = 270; // 中央寄せ位置（必要に応じて調整）
	oss.str(""); oss.clear();
	oss << "Mean.:  " << MstMeanStream.str();
	cv::putText(histImage, oss.str(), cv::Point(Mst_x, total_height - text_height + linewith * 2),
		cv::FONT_HERSHEY_SIMPLEX, fontsize, cv::Scalar(0, 0, 255), 2);
	oss.str(""); oss.clear();
	oss << "Median: " << MstMedianStream.str();
	cv::putText(histImage, oss.str(), cv::Point(Mst_x, total_height - text_height + linewith * 3),
		cv::FONT_HERSHEY_SIMPLEX, fontsize, cv::Scalar(0, 0, 255), 2);
	oss.str(""); oss.clear();
	oss << "StdDev: " << MstStddevStream.str();
	cv::putText(histImage, oss.str(), cv::Point(Mst_x, total_height - text_height + linewith * 4),
		cv::FONT_HERSHEY_SIMPLEX, fontsize, cv::Scalar(0, 0, 255), 2);
	cv::putText(histImage, "input", cv::Point(Mst_x - 5, total_height - text_height + linewith * 1),
		cv::FONT_HERSHEY_SIMPLEX, fontsize, cv::Scalar(0, 0, 255), 2);

	// Deff 側の描画（中央寄せ）
	std::ostringstream DeffMeanStream, DeffMedianStream, DeffStddevStream;
	DeffMeanStream << std::fixed << std::setprecision(1) << Deff_mean;
	DeffMedianStream << std::fixed << std::setprecision(1) << Deff_median;
	DeffStddevStream << std::fixed << std::setprecision(1) << Deff_std;
	int deff_x = 530; // 中央寄せ位置（必要に応じて調整）
	oss.str(""); oss.clear();
	oss << "Mean.:  " << DeffMeanStream.str();
	cv::putText(histImage, oss.str(), cv::Point(deff_x, total_height - text_height + linewith * 2),
		cv::FONT_HERSHEY_SIMPLEX, fontsize, cv::Scalar(0, 160, 0), 2);
	oss.str(""); oss.clear();
	oss << "Median: " << DeffMedianStream.str();
	cv::putText(histImage, oss.str(), cv::Point(deff_x, total_height - text_height + linewith * 3),
		cv::FONT_HERSHEY_SIMPLEX, fontsize, cv::Scalar(0, 160, 0), 2);
	oss.str(""); oss.clear();
	oss << "StdDev: " << DeffStddevStream.str();
	cv::putText(histImage, oss.str(), cv::Point(deff_x, total_height - text_height + linewith * 4),
		cv::FONT_HERSHEY_SIMPLEX, fontsize, cv::Scalar(0, 160, 0), 2);
	cv::putText(histImage, "Deff", cv::Point(deff_x - 5, total_height - text_height + linewith * 1),
		cv::FONT_HERSHEY_SIMPLEX, fontsize, cv::Scalar(0, 160, 0), 2);

	Deff_median_OUT = static_cast<int>(Srv_median - Mst_median);

	int x_median_inp = cvRound((double)Mst_median * bin_w);
	cv::line(histImage, cv::Point(x_median_inp, 0), cv::Point(x_median_inp, hist_h), cv::Scalar(0, 0, 255), 2); // 緑（2D）
	// ラベルを線の上側右わきに表示（ヒストグラム上部に収まるよう位置を調整）
	{
		std::string label2 = "median_inp";
		int label2_x = std::min(x_median_inp + 6, hist_w - 1 - 80); // 右端オーバー防止（文字幅を考慮）
		int label2_y = std::max(12, 40);
		cv::putText(histImage, label2, cv::Point(label2_x, label2_y),
			cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
	}

	int x_median_out = cvRound((double)Srv_median * bin_w);
	cv::line(histImage, cv::Point(x_median_out, 0), cv::Point(x_median_out, hist_h), cv::Scalar(0, 0, 0), 2); // 緑（2D）
	// ラベルを線の上側右わきに表示（ヒストグラム上部に収まるよう位置を調整）
	{
		std::string label = "median_out";
		int label_x = std::min(x_median_out + 6, hist_w - 1 - 60); // 右端オーバー防止（文字幅を考慮）
		int label_y = std::max(12, 18);
		cv::putText(histImage, label, cv::Point(label_x, label_y),
			cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
	}

	return total_height;
}

/**
 * @brief 物理モデル(Parsevalの定理)に基づく厳密な回折効率計算
 * @param reconComplex 物理モデル(a=1)から得られた複素再生像 (CV_64FC2)
 * @param DEmask 信号領域(ROI)を指定するマスク (CV_64F, 0.0 or 1.0)
 */
double calculateDiffractionEfficiencyStrict(const cv::Mat& reconComplex, const cv::Mat& DEmask) {
	// 1. 各画素の強度（振幅の二乗）を計算
	cv::Mat planes[2];
	cv::split(reconComplex, planes);
	cv::Mat intensityMat;
	cv::magnitude(planes[0], planes[1], intensityMat);
	cv::multiply(intensityMat, intensityMat, intensityMat); // Intensity I = |U|^2

	// 2. 分子：信号領域(ROI)内のエネルギー総和
	cv::Mat maskedIntensity;
	cv::multiply(intensityMat, DEmask, maskedIntensity);
	double energyROI = cv::sum(maskedIntensity)[0];

	// 3. 分母：再生像全体のエネルギー総和（パーセバルの定理によりSLM反射総光量に等しい）
	double energyTotal = cv::sum(intensityMat)[0];

	if (energyTotal <= 0) return 0.0;

	// 4. 比率を算出（数学的に100%を超えない）
	return (energyROI / energyTotal) * 100.0;
}

#include <shlobj.h> // Add this include to define CSIDL_DESKTOPDIRECTORY
std::string getDesktopWKCGHFolder() {
	char desktopPath[MAX_PATH];
	// デスクトップパス取得
	if (SUCCEEDED(SHGetFolderPathA(NULL, CSIDL_DESKTOPDIRECTORY, NULL, 0, desktopPath))) {
		std::string wkCghPath = std::string(desktopPath) + "\\Kinoform_Optimization\\";
		// フォルダが存在しなければ作成
		DWORD attrib = GetFileAttributesA(wkCghPath.c_str());
		if (attrib == INVALID_FILE_ATTRIBUTES || !(attrib & FILE_ATTRIBUTE_DIRECTORY)) {
			CreateDirectoryA(wkCghPath.c_str(), NULL);
		}
		return wkCghPath;
	}
	return "";
}

double calculateDiffractionEfficiencyStrictMAIN(const cv::Mat& WkP, const cv::Mat& inputImage, int curSize) {

	cv::Mat img_ROI;//対象画像

	// --- 2. マスク画像の作成 ---
	cv::Mat mask_8u;
	// 1. マスク画像を作成（0以外を1、0は0）
	mask_8u = (inputImage != 0.0);
	int borderSize = 1; // マスクの境界サイズ
	mask_8u = mask_8u(cv::Rect(borderSize, borderSize, mask_8u.cols - 2 * borderSize, mask_8u.rows - 2 * borderSize)).clone();
	cv::copyMakeBorder(mask_8u, mask_8u, borderSize, borderSize, borderSize, borderSize, cv::BORDER_CONSTANT, cv::Scalar(255));

	// 2. 対象画像にマスクを掛ける
	mask_8u.convertTo(img_ROI, CV_8U, 255);//対象画像をコピー
#if  MODE_CUT == 0
#if DISP_CUT == 0
#pragma omp critical(gui)           // ★ 追加
	cv::imshow("MASK", img_ROI);
#endif // DISP_CUT
#endif //  MODE_CUT	
	cv::Mat mask_64F;
	img_ROI.convertTo(mask_64F, CV_64F, 1.0 / 255.0);

	// --- [メインループ内：ホログラム生成後から再生処理] ---

	// 1. 干渉縞 WkP を位相分布 φ(u,v) へ写像する
	// デバイスの最大位相変調度(例: 2.19π)に合わせてスケールを調整
	double phaseModulationDepth = 2.0 * CV_PI;
	cv::Mat phi = WkP.clone();
	cv::normalize(phi, phi, 0, phaseModulationDepth, cv::NORM_MINMAX);

	// 2. 物理モデル(a=1)に基づく複素ホログラム面の作成
	cv::Mat holoComplex(curSize, curSize, CV_64FC2);
	cv::Mat holoPlanes[2];
	holoPlanes[0] = cv::Mat::ones(curSize, curSize, CV_64F); // 振幅反射率 a = 1.0
	holoPlanes[1] = cv::Mat::zeros(curSize, curSize, CV_64F);

	// 複素指数関数 exp(-j * phi) の計算
	for (int y = 0; y < curSize; y++) {
		for (int x = 0; x < curSize; x++) {
			double p = phi.at<double>(y, x);
			holoPlanes[0].at<double>(y, x) = cos(p);  // 実部
			holoPlanes[1].at<double>(y, x) = -sin(p); // 虚部 (共役像の方向に合わせる)
		}
	}
	cv::merge(holoPlanes, 2, holoComplex);

	// 3. 物理的IFFTの実行 (エネルギー保存のためスケールを適用)
	cv::Mat reconComplex;
	cv::dft(holoComplex, reconComplex, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);

	// 4. 回折効率(DE)の算出 (輝度補正前の生データを使用)
	double Mde_strict = calculateDiffractionEfficiencyStrict(reconComplex, mask_64F);

	return Mde_strict;
}


string MITImg_Read(string FILE_DTOP, string FileName, Mat& grayImage, int curSize, int BLK)
{
	string FName_END = "NULL";
	// 1. ファイルの読み込み
	// IMREAD_ANYCOLOR | IMREAD_ANYDEPTH を指定することで、
	// EXR特有の浮動小数点（32bit/16bit）情報を保持したまま読み込めます。
	std::string filename = FILE_DTOP + "data\\" +  FileName + ".exr";//test
#if MODE_CUT == 0
	std::cout << "Name of the loaded file: " << filename << std::endl;
#endif //MODE_CUT
	// 日本語(マルチバイト)パス対策として、std::ifstream + imdecode でファイルを読み込む
	cv::Mat image;
	{
		std::ifstream ifs(filename, std::ios::binary);
		if (!ifs) {
			std::cerr << "Error: The file could not be opened. (" << filename << ")" << std::endl;
			return "-1";
		}
		std::vector<char> buf((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
		image = cv::imdecode(buf, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
	}

	// 読み込みチェック
	if (image.empty()) {
		std::cerr << "Error: The file could not be loaded." << std::endl;
		return "-1";
	}
	// 2. ビットマップ(8bit)への変換
	// EXRは通常32bit浮動小数点ですが、表示や一般的な保存のためには8bitに変換する必要があります。
	cv::Mat display_image;

	// 単純な変換だと白飛びや黒潰れが起きやすいため、
	// 輝度値を0.0〜1.0の範囲に収めてから255倍するのが一般的です。

	// スケーリング（最大値を255にする例）
	normalize(image, image, 0.0, 1.0, NORM_MINMAX);// 0.0〜1.0に正規化
	image.convertTo(display_image, CV_8UC3, 255);// 0〜255に変換
	// --- ここから白黒変換の処理 ---

	// 3. カラー(BGR)からグレースケールに変換
	cv::Mat gray_image;
	cv::cvtColor(display_image, gray_image, cv::COLOR_BGR2GRAY);

	// gray_image を 384x384 の黒背景に左上角合わせで合成
	cv::Mat comb_image = cv::Mat::zeros(cv::Size(curSize, curSize), CV_8U);
	const int copy_w0 = std::min(gray_image.cols, curSize);
	const int copy_h0 = std::min(gray_image.rows, curSize);
	gray_image(cv::Rect(0, 0, copy_w0, copy_h0)).copyTo(comb_image(cv::Rect(0, 0, copy_w0, copy_h0)));

	const int dst_x = curSize - copy_w0; // 右寄せ
	const int dst_y = 0;              // 上寄せ

	gray_image(cv::Rect(0, 0, copy_w0, copy_h0))
		.copyTo(comb_image(cv::Rect(dst_x, dst_y, copy_w0, copy_h0)));
	//BLK = 1;
	int border = BLK; // 削りたいピクセル数
	cv::Mat COPYtarget_mat = comb_image(cv::Rect(border, border, comb_image.cols - 2 * border, comb_image.rows - 2 * border)).clone();
	cv::copyMakeBorder(COPYtarget_mat, grayImage, BLK, BLK, BLK, BLK, cv::BORDER_CONSTANT, cv::Scalar(255));
	//int big_width = target_mat.cols > target_mat.rows ? target_mat.cols : target_mat.rows;//縦横どっちか長い方は？
	//double ratio = ((double)M_MAT / (double)big_width);//割合
	//cv::resize(target_mat, grayImage, cv::Size(), ratio, ratio, cv::INTER_NEAREST);//リサイズ

	return  FName_END;
}

// FFTの象限を入れ替える関数 (Pythonのnp.fft.fftshift相当)
void fftShift(Mat& magI) {
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // 左上
	Mat q1(magI, Rect(cx, 0, cx, cy));  // 右上
	Mat q2(magI, Rect(0, cy, cx, cy));  // 左下
	Mat q3(magI, Rect(cx, cy, cx, cy)); // 右下

	Mat tmp;
	q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
	q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
}

int Diffmain(
	Mat& imgRef,
	Mat& imgTest,
	string FILE_DTOP,
	int NsLoop
) {
	// 1. 画像の読み込み（グレースケール, CV_32Fとして扱う）
	//Mat imgRef = imread("reference.png", IMREAD_GRAYSCALE);
	//Mat imgTest = imread("test.png", IMREAD_GRAYSCALE);

	if (imgRef.empty() || imgTest.empty()) {
		cout << "The image could not be loaded." << endl;
		return -1;
	}


	// HVSの重みを準備
	cv::Mat weights = generateHVSWeight(cv::Size(8, 8));
	// PSNR-HVS-Mを計算
	double psnrHVSM;//
	psnrHVSM = calculatePSNRHVSM(imgRef, imgTest, weights);

	Mat ref32, test32;
	imgRef.convertTo(ref32, CV_32F);
	imgTest.convertTo(test32, CV_32F);

	// 2. 差分画像の計算
	Mat diff = test32 - ref32;

	// 3. 窓関数の適用 (Hanning Window)
	// 画像端の不連続性による十字ノイズを抑制します
	Mat hann;
	createHanningWindow(hann, diff.size(), CV_32F);
	Mat diffWindowed = diff.mul(hann);// 窓関数を掛ける場合

	// 4. ゼロパディング（最適なDFTサイズへ拡張）
	// 周波数分解能を高めるため、あえて大きなサイズ（2の累乗など）に設定することも可能です
	int m = 2048;// getOptimalDFTSize(diff.rows);
	int n = 2048;// getOptimalDFTSize(diff.cols);
	Mat padded;
	copyMakeBorder(diffWindowed, padded, 0, m - diff.rows, 0, n - diff.cols, BORDER_CONSTANT, Scalar::all(0));

	// 5. FFTの実行（複素数平面への展開）
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);

	// 6. 振幅（Magnitude）の計算
	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magI = planes[0];

	// 7. 対数スケールによる可視化
	magI += Scalar::all(1);
	log(magI, magI);

	// 8. 象限の入れ替え
	fftShift(magI);

	// 9. 正規化と表示（0.0～1.0にスケーリング）
	normalize(magI, magI, 0, 1, NORM_MINMAX);
#if  DISP_CUT == 0
	normalize(diffWindowed, diffWindowed, 0, 1, NORM_MINMAX);
	//imshow("Input: Difference (Windowed)", diffWindowed);// 表示用にスケーリング
	//imshow("Spectrum: Error Analysis", magI);
#endif //  DISP_CUT == 0

	// PSNR-HVS-M を magI に表記（文字描画用に 8bit / BGR に変換）
	cv::Mat magI_u8;
	magI.convertTo(magI_u8, CV_8U, 255.0);

	// magI_u8: 諧調170以下を黒(0)、171以上はそのまま通す
	cv::Mat magI_u8_masked;
	cv::threshold(magI_u8, magI_u8_masked, 170, 255, cv::THRESH_TOZERO);

	// --- 周波数マスクで complexI をフィルタして逆FFTし、画像化する ---
// 前提:
//  - complexI : dft後の複素スペクトル(CV_32FC2), サイズ = padded.size()
//  - magI_u8_masked : fftShift(magI)後の表示画像に対するマスク(CV_8U), サイズ = padded.size()
// 目的:
//  - マスク領域だけ周波数を残して逆FFTした空間画像を得る

// 1) magI_u8_masked -> マスク(0/1 float) にする
	cv::Mat freqMask_u8;
	cv::compare(magI_u8_masked, 0, freqMask_u8, cv::CMP_GT); // 255 or 0

	cv::Mat freqMaskShifted_f;
	freqMask_u8.convertTo(freqMaskShifted_f, CV_32F, 1.0 / 255.0); // 1.0 or 0.0

	// 2) fftShift されたマスクを「元のDFT配置」に戻す（ifftshift相当）
	cv::Mat freqMask_f = freqMaskShifted_f.clone();
	fftShift(freqMask_f); // fftShiftは自己逆なので、もう一度呼ぶと元に戻る

	// 3) 複素スペクトルにマスクを適用（2chの両方に掛ける）
	std::vector<cv::Mat> cplxPlanes(2);
	cv::split(complexI, cplxPlanes);
	cplxPlanes[0] = cplxPlanes[0].mul(freqMask_f);
	cplxPlanes[1] = cplxPlanes[1].mul(freqMask_f);

	cv::Mat complexMasked;
	cv::merge(cplxPlanes, complexMasked);

	// 4) 逆DFT（空間領域へ）
	cv::Mat inv;
	cv::idft(complexMasked, inv, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

	// 5) padded -> 元の diff サイズにクロップ
	cv::Mat invCropped = inv(cv::Rect(0, 0, diff.cols, diff.rows)).clone();

	// 6) 表示用に正規化して 8bit化（必要なら）
	cv::Mat invVis;
	cv::normalize(invCropped, invVis, 0, 255, cv::NORM_MINMAX);
	invVis.convertTo(invVis, CV_8U);

	cv::resize(invVis, invVis, cv::Size(256, 256), 0.0, 0.0, cv::INTER_NEAREST);

	//cv::imshow("iFFT (Masked Spectrum)", invVis);
	string Wstr = std::to_string(NsLoop + 1);// C:\Users\tozu3\OneDrive\デスクトップ
	string FName_0 = FILE_DTOP + "output\\" + Wstr + "_iFFT (Masked Spectrum).bmp";
	cv::imwrite(FName_0, invVis);

	// magI_u8_masked を 256x256 にリサイズ
	cv::Mat magI_u8_256;
	cv::resize(magI_u8_masked, magI_u8_256, cv::Size(256, 256), 0.0, 0.0, cv::INTER_NEAREST);

	//imshow("MaskDiff", magI_u8_256);//表示用
	string Wstri = std::to_string(NsLoop + 1);// C:\Users\tozu3\OneDrive\デスクトップ
	string FName_1 = FILE_DTOP + "output\\" + Wstri + "_MASK (Masked Spectrum).bmp";
	cv::imwrite(FName_1, magI_u8_256);

	string Ws = std::to_string(NsLoop + 1);// C:\Users\tozu3\OneDrive\デスクトップ
	string FName = FILE_DTOP + "output\\" + Ws + "(Original) Difference Image FFT.bmp";
	cv::imwrite(FName, magI_u8);

	// imgRef を magI_u8 の左上に重ねる（サイズははみ出さないようにクリップ）
	{
		const int overlayW = std::min(imgRef.cols, magI_u8.cols);
		const int overlayH = std::min(imgRef.rows, magI_u8.rows);

		if (overlayW > 0 && overlayH > 0) {
			cv::Mat roi = magI_u8(cv::Rect(0, 0, overlayW, overlayH));
			imgRef(cv::Rect(0, 0, overlayW, overlayH)).copyTo(roi);
		}
	}

	cv::Mat magI_bgr;
	cv::cvtColor(magI_u8, magI_bgr, cv::COLOR_GRAY2BGR);

	std::ostringstream psnrText;
	psnrText << std::fixed << std::setprecision(2) << "PSNR-HVS-M = " << psnrHVSM << " dB";

	// 背景（視認性用）
	const int x = 10;
	const int y = n - 30;
	const int p = 4;
	cv::putText(magI_bgr, psnrText.str(), cv::Point(x, y),
		cv::FONT_HERSHEY_SIMPLEX, p, cv::Scalar(0, 255, 255), 11, cv::LINE_AA);
	// 前景
	cv::putText(magI_bgr, psnrText.str(), cv::Point(x, y),
		cv::FONT_HERSHEY_SIMPLEX, p, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);

	// 以降の表示は BGR側を使用
	//cv::imshow("Spectrum: Error Analysis", magI_bgr);

	string Wst = std::to_string(NsLoop + 1);// C:\Users\tozu3\OneDrive\デスクトップ
	string FName_ = FILE_DTOP + "output\\" + Wst + "_Difference Image FFT.bmp";
	cv::imwrite(FName_, magI_bgr);

	//waitKey();
	//std::cin.get(); // キー入力待ち

	return 0;
}

#define NloopMIT 4000//MIT処理のループ回数

#define Image0to8 8 //基準画像分のループ回数追加

static double Hdata[7][NloopMIT + Image0to8];//初期化(履歴データ保存用)
static double N_HIS_PSNR[3][NloopMIT + Image0to8];//初期化(履歴データ保存用)
static double N_HIS_SSIM_val[3][NloopMIT + Image0to8];//初期化(履歴データ保存用)
static double N_HIS_SSIM_sta[3][NloopMIT + Image0to8];//初期化(履歴データ保存用)
static double MAX_His_PSNR[NloopMIT + Image0to8] = {};//初期化(MAX 履歴データ保存用)
static double MAX_His_SSIM_val[NloopMIT + Image0to8] = {};//初期化(MAX 履歴データ保存用)
static double MAX_His_SSIM_sta[NloopMIT + Image0to8] = {};//初期化(MAX 履歴データ保存用)
struct EvalResults { double psnr=0.0, ssim=0.0, de=0.0; Mat reconImg; };

// ============================================================
// (A) 修正版 evaluateHologram
//     ・指標計算のみ（集計ブロックは削除）
//     ・Hdata書き込み → critical(eval)
//     ・コンソール出力 → critical(console)
// ============================================================
EvalResults evaluateHologram(
	Mat& COPYtarget_mat_M_256,
	Mat& COPYtarget_mat_S_256,
	int NsLoop,
	double Mde_physics,
	double Mde_strict,
	string FILE_DTOP,
	int WKKK,
	string inputCom)
{
	EvalResults res;

	// ★ 重み行列はスレッドローカルで計算（共有なし）
	cv::Mat weights = generateHVSWeight(cv::Size(8, 8));
	cv::Mat M_weights = M_generateHVSWeight(cv::Size(8, 8));

	// ===== 各指標をスレッドローカルに計算（critical 不要） =====
	double w = calcPSNR(COPYtarget_mat_M_256, COPYtarget_mat_S_256);

	double psnrHVS = calculatePSNRHVS(COPYtarget_mat_M_256, COPYtarget_mat_S_256, weights);
	double w_hvs = std::isinf(psnrHVS) ? 0.0 : psnrHVS;

	double psnrHVSM = calculatePSNRHVSM(COPYtarget_mat_M_256, COPYtarget_mat_S_256, weights);
	double w_hvsm = std::isinf(psnrHVSM) ? 0.0 : psnrHVSM;

	double ssim_standard = getStandardSSIM(COPYtarget_mat_M_256, COPYtarget_mat_S_256);
	cv::Scalar ssim = getSSIM(COPYtarget_mat_M_256 / 255.0, COPYtarget_mat_S_256 / 255.0);

	// MSE=0のときDEを100%に補正
	double Mde_out = (w_hvsm == 0.0) ? 100.0 : Mde_physics;

	// ===== Hdata への書き込み：critical(eval) で保護 =====
#pragma omp critical(eval)
	{
		Hdata[0][NsLoop + 1] = w;
		Hdata[1][NsLoop + 1] = w_hvs;
		Hdata[2][NsLoop + 1] = w_hvsm;
		Hdata[3][NsLoop + 1] = ssim[0];
		Hdata[6][NsLoop + 1] = ssim_standard;
		Hdata[4][NsLoop + 1] = Mde_out;
		Hdata[5][NsLoop + 1] = Mde_strict;
	}

	// ===== コンソール出力：critical(console) で保護 =====
#pragma omp critical(console)
	{
		SetConsoleColor(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
		std::cout << std::fixed << std::setprecision(2);
#if MODE_CUT == 0
		if (w == 0.0)
			std::cout << "PSNR(MSE=0) for Exact Match";
		else
			std::cout << "PSNR=" << w << "dB";

		if (std::isinf(psnrHVS))
			std::cout << " PSNR-HVS(MSE=0) for Exact Match";
		else
			std::cout << " PSNR-HVS=" << psnrHVS << "dB";

		if (std::isinf(psnrHVSM))
			std::cout << " PSNR-HVSM(MSE=0) for Exact Match";
		else
			std::cout << " PSNR-HVSM=" << psnrHVSM << "dB";

		std::cout << std::fixed << std::setprecision(4);
		std::cout << " Yoshikawa's Criterion SSIM=" << ssim[0]<<"("<< ssim_standard<<")";

		std::cout << std::fixed << std::setprecision(2);
		std::cout << " DE " << Mde_out << "(" << Mde_strict << ")%" << std::endl;
		SetConsoleColor(FOREGROUND_RED | FOREGROUND_INTENSITY);
#endif
	}

	res.de = Mde_strict;
	res.psnr = w;
	res.ssim = ssim[0];
	return res;
}


// ============================================================
// (B) 写真8枚の平均集計
//     ★ NsLoop並列ループ終了後にメインスレッドから呼ぶ
//     （従来 NsLoop==6 のブロック）
//     ・static 変数 → ローカル変数に変更
//     ・cin.get() → 削除（デッドロック防止）
// ============================================================
void aggregatePhotoResults()
{
	// ★ static を外してローカル変数化（スレッド終了後の単独呼び出しなので問題なし）
	double avg[7] = {};
	int    count[7] = {};
	int    ZEROcount[7] = {};

	const int startIdx = 1;  // NsLoop+1: NsLoop=0 → index 1
	const int endIdx = 8;    // NsLoop+1: NsLoop=7 → index 8

	for (int i = 0; i <= 6; ++i)
	{
		double sum = 0.0;
		for (int k = startIdx; k <= endIdx; ++k) {
			sum += Hdata[i][k];
			if (Hdata[i][k] != 0.0) ++count[i];
			else                     ++ZEROcount[i];
		}
		avg[i] = (i <= 2)
			? (count[i] == 0 ? 0.0 : sum / count[i])
			: sum / 8.0;
	}

	std::cout << " PSNR:" << count[0] << ",Z " << ZEROcount[0]
		<< " PSNR-HVS:" << count[1] << ",Z " << ZEROcount[1]
		<< " PSNR-HVSM:" << count[2] << ",Z " << ZEROcount[2]
		<< " SSIM,DE:" << count[2] << " \n";

	if (avg[0] == 0.0) std::cout << " PSNR(MSE=0) for Exact Match";
	else               std::cout << " PSNR= " << avg[0] << "dB.";

	if (avg[1] == 0.0) std::cout << " PSNR-HVS(MSE=0) for Exact Match";
	else               std::cout << " PSNR-HVS= " << avg[1] << " dB ";

	if (avg[2] == 0.0) std::cout << " PSNR-HVSM(MSE=0) for Exact Match";
	else               std::cout << " PSNR-HVSM= " << avg[2] << " dB ";

	std::cout << std::fixed << std::setprecision(4);
	std::cout << "  SSIM(Yoshikawa's Criterion)= " << avg[3] << "  SSIM(Standard AI Criterion)= " << avg[6];
	std::cout << std::fixed << std::setprecision(2);
	if (avg[4] == 100) std::cout << "   DE  100 (" << avg[5] << ")% ";
	else               std::cout << "   DE  " << avg[4] << " (" << avg[5] << ")% ";
	std::cout << "Average" << std::endl;

	// ★ cin.get() は削除（並列環境ではデッドロックになるため）
}


// ============================================================
// (C) MIT画像の平均集計 + imwrite
//     ★ NsLoop並列ループ終了後にメインスレッドから呼ぶ
//     （従来 NsLoop==NloopMIT-1+Image0to8 のブロック）
//     ・static 変数 → ローカル変数 / std::vector に変更
// ============================================================
void aggregateMITResults(int WKKK, const string& FILE_DTOP, const string& inputCom)
{
	// NloopMIT, Image0to8 はグローバルマクロを直接使用
	const int startIdx = Image0to8;
	//const int endIdx = NloopMIT - 1 + Image0to8;
	//const int sampleCount = endIdx - startIdx + 1;
	const int endIdx = NloopMIT + Image0to8;
	const int sampleCount = endIdx - startIdx;
	// ★ static を外してローカル変数化
	double avg[7] = {};
	int    count[7] = {};
	int    ZEROcount[7] = {};
	
	for (int i = 0; i <= 6; ++i)
	{
		double sum = 0.0;
		for (int k = startIdx; k <= endIdx; ++k) {
			sum += Hdata[i][k];
			if (Hdata[i][k] != 0.0) ++count[i];
			else                     ++ZEROcount[i];
		}
		avg[i] = (i <= 2)
			? (count[i] == 0 ? 0.0 : sum / count[i])
			: sum / NloopMIT;
	}

	std::cout << " PSNR:" << count[0] << ",Z " << ZEROcount[0]
		<< " PSNR-HVS:" << count[1] << ",Z " << ZEROcount[1]
		<< " PSNR-HVSM:" << count[2] << ",Z " << ZEROcount[2]
		<< " SSIM,DE:" << count[2] << " \n";

	for (int i = 0; i < 7; i++) { count[i] = 0; ZEROcount[i] = 0; }

	if (avg[0] == 0.0) std::cout << " PSNR(MSE=0) for Exact Match";
	else               std::cout << " PSNR= " << avg[0] << "dB.";
	if (avg[1] == 0.0) std::cout << " PSNR-HVS(MSE=0) for Exact Match";
	else               std::cout << " PSNR-HVS= " << avg[1] << " dB ";
	if (avg[2] == 0.0) std::cout << " PSNR-HVSM(MSE=0) for Exact Match";
	else               std::cout << " PSNR-HVSM= " << avg[2] << " dB ";
	std::cout << std::fixed << std::setprecision(4);
	std::cout << "  SSIM(Yoshikawa's Criterion)= " << avg[3] << "  SSIM(Standard AI Criterion)= " << avg[6];
	std::cout << std::fixed << std::setprecision(2);
	if (avg[4] == 100) std::cout << "   DE  100 (" << avg[5] << ")% ";
	else               std::cout << "   DE  " << avg[4] << " (" << avg[5] << ")% ";
	std::cout << "Average" << std::endl;

	// --- Hdata[0] 単一ヒストグラム ---
	{
		const int binCount = 128;
		double minV = std::numeric_limits<double>::infinity();
		double maxV = -std::numeric_limits<double>::infinity();
		for (int k = startIdx; k <= endIdx; ++k) { minV = std::min(minV, Hdata[0][k]); maxV = std::max(maxV, Hdata[0][k]); }
		if (!std::isfinite(minV) || !std::isfinite(maxV)) { minV = 0.0; maxV = 1.0; }
		if (maxV <= minV) maxV = minV + 1.0;

		std::vector<int> hist(binCount, 0);
		const double range = maxV - minV;
		for (int k = startIdx; k <= endIdx; ++k) {
			double t = std::clamp((Hdata[0][k] - minV) / range, 0.0, 1.0);
			hist[std::clamp((int)(t * (binCount - 1) + 0.5), 0, binCount - 1)]++;
		}
		int maxCount = 1; for (int c : hist) maxCount = std::max(maxCount, c);

		const int graphW = 1400, graphH = 450, marginTop = 35, marginBottom = 70, marginLeft = 80, marginRight = 30;
		cv::Mat histImg(marginTop + graphH + marginBottom, marginLeft + graphW + marginRight, CV_8UC3, cv::Scalar(255, 255, 255));
		cv::rectangle(histImg, { marginLeft,marginTop }, { marginLeft + graphW,marginTop + graphH }, { 0,0,0 }, 1);
		const double binW = (double)graphW / binCount;
		auto yFC = [&](int c) {return marginTop + graphH - (int)((double)c * graphH / maxCount); };
		int px = marginLeft, py = yFC(hist[0]);
		for (int b = 0; b < binCount; ++b) {
			int xR = marginLeft + (int)((b + 1) * binW);
			cv::line(histImg, { px,py }, { xR,py }, { 0,0,255 }, 2, cv::LINE_AA);
			if (b + 1 < binCount) { int yN = yFC(hist[b + 1]); cv::line(histImg, { xR,py }, { xR,yN }, { 0,0,255 }, 2, cv::LINE_AA); px = xR; py = yN; }
		}
		double midV = (minV + maxV) * 0.5;
		cv::putText(histImg, cv::format("%.2f", minV), { marginLeft - 10,     marginTop + graphH + 40 }, cv::FONT_HERSHEY_SIMPLEX, 0.6, { 0,0,0 }, 1, cv::LINE_AA);
		cv::putText(histImg, cv::format("%.2f", midV), { marginLeft + graphW / 2 - 30,marginTop + graphH + 40 }, cv::FONT_HERSHEY_SIMPLEX, 0.6, { 0,0,0 }, 1, cv::LINE_AA);
		cv::putText(histImg, cv::format("%.2f", maxV), { marginLeft + graphW - 70,  marginTop + graphH + 40 }, cv::FONT_HERSHEY_SIMPLEX, 0.6, { 0,0,0 }, 1, cv::LINE_AA);
		cv::putText(histImg, std::to_string(maxCount), { 10,marginTop + 12 }, cv::FONT_HERSHEY_SIMPLEX, 0.7, { 0,0,0 }, 2, cv::LINE_AA);
		cv::putText(histImg, "0", { 10,marginTop + graphH }, cv::FONT_HERSHEY_SIMPLEX, 0.7, { 0,0,0 }, 2, cv::LINE_AA);
		cv::putText(histImg,
			cv::format("Histogram of PSNR  samples=%d  bins=%d  range=[%.2f..%.2f]", sampleCount, binCount, minV, maxV),
			{ marginLeft,24 }, cv::FONT_HERSHEY_SIMPLEX, 0.6, { 0,0,0 }, 2, cv::LINE_AA);

		std::ostringstream oss; oss << WKKK;
		cv::imwrite(FILE_DTOP + "output\\" + "Hdata0" + oss.str() + "_hist_0_to_4000_" + inputCom + ".bmp", histImg);
	}
}
int histogramMAIN(Mat& COPYtarget_mat_M_256, Mat& COPYtarget_mat_S_256, Mat& intensity_roi, Mat& img_Mst, Mat& img_Srv, int curSize, Mat& histImage)// ヒストグラム計算と表示
{
	//// ヒストグラム計算と表示
	// histogram計算 for img_Mst
	cv::Mat Mst_hist;
	int Mst_histSize = 256;// ピン数
	float Mst_range[] = { 0, 256 };// 範囲
	const float* Mst_histRange = { Mst_range };
	cv::calcHist(&COPYtarget_mat_M_256, 1, 0, cv::Mat(), Mst_hist, 1, &Mst_histSize, &Mst_histRange, true, false);
	// histogram計算 for img_Srv
	cv::Mat Srv_hist;
	int Srv_histSize = 256;// ピン数
	float Srv_range[] = { 0, 256 };// 範囲
	const float* Srv_histRange = { Srv_range };


	// M: 基準, S: 対象
	double mstMin = 0.0, mstMax = 0.0;
	double srvMin = 0.0, srvMax = 0.0;

	cv::minMaxLoc(COPYtarget_mat_M_256, &mstMin, &mstMax);
	cv::minMaxLoc(intensity_roi, &srvMin, &srvMax);

	// 対象側の max を基準側の max に合わせるスケーリング
	if (srvMax > 0.0) {
		double scale = mstMax / srvMax;
		intensity_roi.convertTo(intensity_roi, intensity_roi.type(), scale);
		intensity_roi.convertTo(COPYtarget_mat_S_256, CV_8U);
	}
	cv::calcHist(&COPYtarget_mat_S_256, 1, 0, cv::Mat(), Srv_hist, 1, &Srv_histSize, &Srv_histRange, true, false);
	intensity_roi.convertTo(img_Srv, CV_8U);//再生像の描画データ

	int Deff_median_OUT = 0;
	int hist_w = 768;// ヒストグラム表示用画像の幅
	int total_height = 0;// ヒストグラム表示用画像の高さ
	total_height = drawHistogram(Mst_hist, Srv_hist, histImage, Deff_median_OUT, img_Mst, img_Srv);// ヒストグラムの描画
#if MODE_CUT == 0
#if DISP_CUT == 0
	cv::resizeWindow("Histogram", hist_w, total_height);// ウィンドウサイズの変更
	cv::imshow("Histogram", histImage);// ヒストグラムの表示
	cv::waitKey(1);
#endif // DISP_CUT				
#endif // MODE_CUT
	return Deff_median_OUT;
}


Mat Img_IFFT(Mat & WkP, int curSize)
{
	Mat re = Mat_<double>(curSize, curSize);
	Mat im = Mat_<double>(curSize, curSize);
	Mat complexImage = cv::Mat::zeros(cv::Size(curSize, curSize), CV_64FC2); // 2-channel double image
	Mat planes[] = { re.clone(), im.clone() };

/***************************************************************************************/
	// --- 振幅型から位相型（Phase-Only）への変更 ---
	// 1. WkP (0.0〜1.0 にスケーリングされている前提) を 0〜2π の位相（ラジアン）に変換
	Mat phi = WkP * 2.0 * CV_PI;
	// 2. 振幅はすべて 1.0（光を吸収せず、100%透過・反射させる）
	Mat mag = Mat::ones(curSize, curSize, CV_64F);
	// 3. 極座標 (mag, phi) から直交座標 (実部 planes[0], 虚部 planes[1]) へ変換
	cv::polarToCart(mag, phi, planes[0], planes[1]);

	// IFFTの実行 For display
	cv::merge(planes, 2, complexImage);
	cv::dft(complexImage, complexImage, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);// IFFTを実行してスケーリングも行う
	cv::split(complexImage, planes);

	cv::Mat intensity_roi = Mat_<double>(curSize, curSize);
	intensity_roi = cv::Mat::zeros(intensity_roi.size(), intensity_roi.type());
	cv::magnitude(planes[0], planes[1], intensity_roi);// magnitude = sqrt(real^2 + imag^2)
	/*sqrt(X^2+Y^2)の自乗の変数のため零より必ず大きな値をとる*/
	/***************************************************************************************/
	// クリッピング値の計算
	cv::Mat CLIP_intensity_roi = Mat_<UCHAR>(curSize, curSize);// = intensity_roi.clone();
	intensity_roi.convertTo(CLIP_intensity_roi, CV_8U, 255);
	// ヒストグラム計算
	cv::Mat CLIP_hist;
	int CLIP_histSize = 256; // ビン数
	float CLIP_range[] = { 0, 256 }; // 範囲
	const float* CLIP_histRange = { CLIP_range };
	cv::calcHist(&CLIP_intensity_roi, 1, 0, cv::Mat(), CLIP_hist, 1, &CLIP_histSize, &CLIP_histRange, true, false);

	// 累積分布を手動で計算 (1D ヒストグラム版)
	cv::Mat cumulative_hist(CLIP_hist.rows, CLIP_hist.cols, CV_64F);

	double running_sum = 0.0;
	if (CLIP_hist.rows == 1) {
		// 1 x N の場合（行ベクトル）
		for (int i = 0; i < CLIP_hist.cols; ++i) {
			double v = static_cast<double>(CLIP_hist.at<float>(0, i));
			running_sum += v;
			cumulative_hist.at<double>(0, i) = running_sum;
		}
	}
	else if (CLIP_hist.cols == 1) {
		// N x 1 の場合（列ベクトル）
		for (int i = 0; i < CLIP_hist.rows; ++i) {
			double v = static_cast<double>(CLIP_hist.at<float>(i, 0));
			running_sum += v;
			cumulative_hist.at<double>(i, 0) = running_sum;
		}
	}
	else {
		// 想定外の形 (安全のためメッセージを出しておく)
		std::cerr << "CLIP_hist must be 1D (1xN or Nx1)." << std::endl;
	}

	// 上位10%の閾値を計算
	double total_sum = cumulative_hist.at<double>(CLIP_histSize - 1);
	double target_value = total_sum * 0.99;//10% // 0.9 -> 1000% // 0.99 -> 100% // 0.999 -> 10% // 0.9999 -> 1%

	int clip_value = 0;
	for (int i = 0; i < CLIP_histSize; i++) {
		if (cumulative_hist.at<double>(i) >= target_value) {
			clip_value = i;
			break;
		}
	}

	if (intensity_roi.empty()) {
		std::cerr << "Error: intensity_roi is empty!" << std::endl;
	}
	// --- CLIP_intensity_roi(CV_8U) からマスク生成 → intensity_roi(CV_64F) のマスク範囲を最小値で埋める ---
	CV_Assert(CLIP_intensity_roi.type() == CV_8U);
	CV_Assert(intensity_roi.type() == CV_64F);
	CV_Assert(CLIP_intensity_roi.size() == intensity_roi.size());

	// 1) マスク作成（例：clip_value を超える画素を対象にする）
	cv::Mat mask; // CV_8U(0/255)
	cv::compare(CLIP_intensity_roi, clip_value, mask, cv::CMP_GT);

	// 2) マスク範囲の最小値を取得（intensity_roi は CV_64F のまま）
	double minValMasked = 0.0;
	double maxValMasked = 0.0;
	cv::minMaxLoc(intensity_roi, &minValMasked, &maxValMasked, nullptr, nullptr, mask);

	// 3) マスク範囲を最小値で埋める（CV_64F のまま）
	if (cv::countNonZero(mask) > 0) {
		cv::Mat fillMat(intensity_roi.size(), CV_64F, cv::Scalar(minValMasked));
		fillMat.copyTo(intensity_roi, mask);
	}

	// 互換のため（元の used_threshold 相当）
	double used_threshold = static_cast<double>(clip_value);

	//double used_threshold = cv::threshold(CLIP_intensity_roi, intensity_roi, clip_value, clip_value, cv::THRESH_TRUNC);
	//std::cout << "Used threshold: " << used_threshold << std::endl;
	return intensity_roi;
}
void Disp_Save_img(string FName_all, Mat& COPYtarget_mat_M_256, Mat& COPYtarget_mat_S_256, Mat& img_CGH, Mat& img_all, Mat& img_Mst, Mat& img_Srv, Mat& histImage,
	string FName_Src, string FName_Mst,string FName_Srv,string FName_cgh,string FName_his, string FILE_DTOP, string inputCom,int NsLoop,int curSize)
{
	Mat WkP(curSize, curSize, CV_64F);
	Mat WkS(curSize, curSize, CV_64F);
	Mat WkT(curSize, curSize, CV_64F);
#if DISP_CUT == 0
	cv::waitKey(1);// 1.0msecの待ち時間にて、干渉縞の表示が出る。
	cv::imshow("Master Image", COPYtarget_mat_M_256);
	cv::imshow("Target Image", COPYtarget_mat_S_256);//対象表示
	cv::waitKey(1);// 1.0msecの待ち時間にて、干渉縞の表示が出る。
	cv::imshow("CGH", img_CGH);
	cv::waitKey(1);// 1.0msecの待ち時間にて、干渉縞の表示が出る。
#endif //DISP_CUT
#if MODE_CUT == 0
#if DISP_CUT == 0
	cv::normalize(img_all, img_all, 0, 255, cv::NORM_MINMAX);
	cv::imshow("Overall Output Image", img_all);
	cv::waitKey(1);// 1.0msecの待ち時間にて、干渉縞の表示が出る。
	//cv::imwrite(FName_all, img_all);
	//std::cout << "Enter any key to continue...img_ALL End" << std::endl;
	//std::cin.get(); // キー入力待ち
#endif // DISP_CUT
				// COPYtarget_mat_M_256,BLKとCOPYtarget_mat_S_256,BLKの差分画像を計算
	cv::Mat diff_img;
	COPYtarget_mat_M_256.convertTo(WkT, CV_64F);
	COPYtarget_mat_S_256.convertTo(WkS, CV_64F);
	cv::absdiff(WkT, WkS, diff_img);

	// 差分画像を正規化して可視化
	cv::Mat diff_img_norm;
	cv::normalize(diff_img, diff_img_norm, 0, 255, cv::NORM_MINMAX);
	diff_img_norm.convertTo(diff_img_norm, CV_8U);

	// 差分画像を表示
#if DISP_CUT == 0
	cv::imshow("Difference Image", diff_img_norm);
	cv::waitKey(1); // 表示のための待機
#endif // DISP_CUT				
	if (NsLoop < 8)// 0-7回までのループで差分計算を行う
	{
		Diffmain(COPYtarget_mat_M_256, COPYtarget_mat_S_256, FILE_DTOP, NsLoop);//差分計算表示
	}

#endif // MODE_CUT

#if MODE_CUT == 0
	if (inputCom.find("save") != std::string::npos) {// "save" を含む場合のみ実行される処理
		if (NsLoop < 8)// 0-7回までのループで保存する
		{
			// 保存処理を行う
			//cv::normalize(img_all, img_all, 0, 255, cv::NORM_MINMAX);
			cv::imwrite(FName_all, img_all);
			cv::imwrite(FName_Src, COPYtarget_mat_S_256);
			cv::imwrite(FName_Mst, COPYtarget_mat_M_256);
			std::cout << "The image has been exported to a file." << FName_Srv << endl;
			cv::imwrite(FName_cgh, img_CGH);
			cv::imwrite(FName_his, histImage);


			// CSVファイルに出力
			/*この後の処理は削除はしないでください*/
			img_Srv.convertTo(WkP, CV_64F);
			string filenames = FILE_DTOP + "output\\" + std::string("Srv.csv");
			writeMatToCSV(WkP, filenames);
			//cout << "CSVファイルに出力しました" << filenames << endl;

			// CSVファイルに出力
			/*この後の処理は削除はしないでください*/
			img_Mst.convertTo(WkP, CV_64F);
			filenames = FILE_DTOP + "output\\" + std::string("Mst.csv");
			writeMatToCSV(WkP, filenames);
			//cout << "CSVファイルに出力しました" << filenames << endl;
		}
	}
	if (inputCom.find("step") != std::string::npos) // "step" を含む場合
	{
		std::cout << "    [step: The stop operation is skipped due to parallelization.]\n";
		//std::cout << "    Enter any other key to continue...Stop >> ";// << std::endl;
		//std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // バッファクリア
		//std::cin.get(); // どれかのキー入力待ち
		//std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // バッファクリア
	}
#endif // MODE_CUT
	return;
}
// 計算方法を選択するためのenumクラス
enum class DE_Method {
	PAPER_METHOD,      // 論文記載の方法（100%を超える可能性あり）
	PHYSICS_BASED,     // 物理ベース（物体光＋参照光）の方法 (推奨)
	PARSEVAL_BASED,    // パーセバルの定理に基づく方法
	PHYSICS_BASED_2    // 参照光のエネルギー (強度が1.0の平行光と仮定)
};


/**
 * @brief マスク画像を用いてホログラムの回折効率を計算します。
 * @param inputObjectImage 元の入力画像（物体光、CV_64F）。
 * @param reconstructedComplexImage フーリエ変換後の複素再生像（CV_64FC2）。
 * @param mask 目的領域を示すマスク画像（CV_64F、値は0.0と1.0）。
 * @param method 計算方法を指定 (DE_Method)。
 * @return 計算された回折効率（%）。
 */
double calculateDiffractionEfficiencyWithMask(
	const cv::Mat& inputObjectImage,
	const cv::Mat& reconstructedComplexImage,
	const cv::Mat& DEmask, // Rectの代わりにmaskを受け取る
	DE_Method method,
	int curSize)
{
	// --- 分子：再生像の「目的領域D」の総強度を計算 ---
	Mat re = Mat_<double>(curSize, curSize);
	Mat im = Mat_<double>(curSize, curSize);
	Mat DEchannels[] = { re.clone(), im.clone() };

	cv::split(reconstructedComplexImage, DEchannels);
	cv::Mat total_intensity_mat;
	cv::magnitude(DEchannels[0], DEchannels[1], total_intensity_mat);
	cv::multiply(total_intensity_mat, total_intensity_mat, total_intensity_mat);

	cv::Mat masked_intensity;
	cv::multiply(total_intensity_mat, DEmask, masked_intensity);

	const double total_intensity_reconstructed_D = cv::sum(masked_intensity)[0];

	// --- 分母の計算 ---
	double total_input_energy = 0.0;

	// ログ用：sum_input = sum(input^2)
	double sum_input = 0.0;
	if (!inputObjectImage.empty()) {
		cv::Mat input_intensity;
		cv::multiply(inputObjectImage, inputObjectImage, input_intensity);
		sum_input = cv::sum(input_intensity)[0];
	}
	// 強度ホログラム用：sum_holo_intensity = sum(I_holo)
	// ※ reconstructedComplexImage は「再生像」だが、ここでは便宜上
	//    main側で I(x)ホログラムをそのまま複素(実部=I,虚部=0)として
	//    渡す運用に切り替える場合に使用できる。
	//    使わない場合でもログ用途として保持しておく。
	double sum_holo_intensity = 0.0;
	{
		std::vector<cv::Mat> ch(2);
		cv::split(reconstructedComplexImage, ch);
		// 強度ホログラム運用の場合、実部に I(x) が入っている前提
		sum_holo_intensity = cv::sum(ch[0])[0];
	}
	// ログ用：sum_total_intensity = sum(|reconstructed|^2)
	const double sum_total_intensity = cv::sum(total_intensity_mat)[0];

	double reference_energy = 0.0;

	switch (method) {
	case DE_Method::PAPER_METHOD:
	{
		total_input_energy = sum_input;
		break;
	}
	
	case DE_Method::PHYSICS_BASED:
	{
		// 修正：入射光の全エネルギー(平面波と仮定)
		// curSize * curSize の面積に強度1.0の光が入射したとする
		//reference_energy = static_cast<double>(inputObjectImage.rows * inputObjectImage.cols);
		total_input_energy = 1.0;// reference_energy;
		break;
	}
	case DE_Method::PARSEVAL_BASED:
	{
		total_input_energy = sum_total_intensity;
		break;
	}
	case DE_Method::PHYSICS_BASED_2:
	{
		reference_energy = static_cast<double>(inputObjectImage.rows * inputObjectImage.cols);
		total_input_energy = reference_energy;
		break;
	}
	}

	if (total_input_energy == 0.0) {
		return 0.0;
	}

	// --- 追加ログ（PHYSICS_BASEDのみ） ---
#if MODE_CUT == 0
#if DISP_CUT == 0 

		// 出力が多すぎる場合は間引き（例：100回に1回）
	static std::atomic<int> s_logCounter(0);
	const bool doLog = ((s_logCounter.fetch_add(1) % 100) == 0);

	if (doLog) {
		std::ios::fmtflags cur = std::cout.flags();
		std::cout << std::fixed << std::setprecision(6);

		std::cout
			<< "[DE-PHYS] sum_input=" << sum_input
			<< " reference_energy=" << reference_energy
			<< " denom(sum_input+ref)=" << total_input_energy
			<< " sum_holo_intensity=" << sum_holo_intensity
			<< " sum_total_intensity=" << sum_total_intensity
			<< " sum_D=" << total_intensity_reconstructed_D
			<< " ratio(sum_total/denom)=" << (sum_total_intensity / total_input_energy)
			<< " ratio(D/denom)=" << (total_intensity_reconstructed_D / total_input_energy)
			<< std::endl;

		std::cout.flags(cur);
	}
	//}
#endif
#endif
	return (total_intensity_reconstructed_D / total_input_energy) * 100.0;
}




double REminVal, REmaxVal;
double IMminVal, IMmaxVal;
std::string inputCom;

int main(int argc, char* argv[])
{
	int WKKK = 0;

	std::string folderPath = getDesktopWKCGHFolder();
	// folderPath + "\\ファイル名" でファイル保存などに利用できます

	// main関数の最初の方に記述
	_putenv_s("OPENCV_IO_ENABLE_OPENEXR", "1");

	std::string FILE_DTOP;
	std::string computerName = getComputerName();
		FILE_DTOP = folderPath;
		std::cout << "This computer is " << computerName << "(" << FILE_DTOP << ")" << std::endl;

#if DISP_CUT == 0
	namedWindow("Input Image", WINDOW_AUTOSIZE);// ウインドウを作成
	cv::moveWindow("Input Image", 10, 100);// ウインドウの位置を指定 (x座標, y座標)
	cv::resizeWindow("Input Image", 260, 260);// ウインドウのサイズを指定 (幅, 高さ
	namedWindow("CGH", WINDOW_AUTOSIZE);// ウインドウを作成
	cv::moveWindow("CGH", 10 + 300 * 1, 100);// ウインドウの位置を指定 (x座標, y座標)
	cv::resizeWindow("CGH", 260, 260);// ウインドウのサイズを指定 (幅, 高さ

	namedWindow("Master Image", WINDOW_AUTOSIZE);// ウインドウを作成
	cv::moveWindow("Master Image", 10, 150 + 270 + 200);// ウインドウの位置を指定 (x座標, y座標)
	cv::resizeWindow("Master Image", 260, 130);// ウインドウのサイズを指定 (幅, 高さ
	namedWindow("Target Image", WINDOW_AUTOSIZE);// ウインドウを作成
	cv::moveWindow("Target Image", 10, 150 + 270);// ウインドウの位置を指定 (x座標, y座標)
	cv::resizeWindow("Target Image", 260, 130);// ウインドウのサイズを指定 (幅, 高さ
#endif //DISP_CUT
#if MODE_CUT == 0
#if DISP_CUT == 0
	namedWindow("Overall Output Image", WINDOW_AUTOSIZE);// ウインドウを作成
	cv::moveWindow("Overall Output Image", 10 + 300, 150 + 270);// ウインドウの位置を指定 (x座標, y座標)
	cv::resizeWindow("Overall Output Image", 260, 260);// ウインドウのサイズを指定 (幅, 高さ
	namedWindow("Histogram", WINDOW_AUTOSIZE);
	cv::moveWindow("Histogram", 10 + 300 * 2, 100);// 150 + 270);
	namedWindow("Difference Image", WINDOW_AUTOSIZE);
	cv::moveWindow("Difference Image", 10, 150 + 270 + 400);
	namedWindow("MASK", WINDOW_AUTOSIZE);
	cv::moveWindow("MASK", 10 + 300, 150 + 270 + 400);
#endif //DISP_CUT
#endif //MODE_CUT

	DWORD elapsedall = 0;

	srand((unsigned)time(NULL)); /*乱数の初期化*/


	while (WKKK <= 99) //++++++++++++++++++++Main Math Loop++++++++++++++++++++++++
	{
		////string FName,
		////	FName_all,
		////	FName_END,
		////	FName_cgh,
		////	FName_Mst,
		////	FName_Srv_,
		////	FName_Srv,
		////	FName_Src,
		////	FName_his;//画像保存用のファイル名
		// 現状保存
		std::ios::fmtflags curret_flag = std::cout.flags();

		//123の頭に5個0を詰めて8桁にする
		std::ostringstream Wss;
		Wss << std::setw(3) << std::setfill('0') << WKKK;// << "\n";
		std::string Ws(Wss.str());
		std::cout << Ws;

		WKKK++;
		//std::cout << "WKKK=" << WKKK << "  \\ \n";
	Retry:

		std::cout << "Enter save(画像保存),step(各停),run(実行:正規化,四捨五入,中央値補正,GS法:ダミー領域有り)>> ";// << std::endl;
		std::cin >> inputCom; // 入力を取得
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // バッファクリア

		if (inputCom.find("run") == std::string::npos) goto Retry;// 'run'が含まれていなければ再入力

		DWORD start = timeGetTime();

		////const std::string csvPath = FILE_DTOP  +"output\\"+ "alpha_sweep.csv";
		////std::ofstream csv(csvPath, std::ios::out | std::ios::trunc);
		////if (!csv.is_open()) {
		////	std::cerr << "[CSV] open failed: " << csvPath << std::endl;
		////	// ここで sweep を中止（または return / continue）
		////	continue;
		////}


// ★① ループ外に移動: static int → std::atomic<int>
		static std::atomic<int> GS_logCounter(0);

		// ★② NsLoop 並列化
#pragma omp parallel for schedule(dynamic)
		for (int NsLoop = 0; NsLoop <= NloopMIT - 1 + Image0to8; NsLoop++)
		{
			int BLK = 1;
			int curSize = M_MAT;
			if (NsLoop >= 8) curSize = 384;

			// ★③ FName系を全てループ内ローカル変数として定義（スレッド競合を排除）
			std::string FName, FName_Mst, FName_all,
				FName_Srv_, FName_Srv, FName_Src,
				FName_cgh, FName_his, FName_END;

			// 以下の Mat 系は元のコードのまま（既にローカル変数）
			Mat target_mat = Mat_<double>(curSize, curSize);
			Mat img_all(cv::Size(curSize, curSize), CV_8U);
			Mat img_CGH(cv::Size(curSize, curSize), CV_8U);
			Mat img_src(cv::Size(curSize, (curSize / 2)), CV_8U);
			Mat img_Mst(cv::Size(curSize, (curSize / 2)), CV_8U);
			Mat img_Srv(cv::Size(curSize, (curSize / 2)), CV_8U);
			Mat Trans = Mat::zeros(cv::Size(curSize, curSize), CV_64F);
			randu(Trans, 0, 2.0 * CV_PI);
			Mat grayImage(cv::Size(curSize, curSize), CV_8U);
			Mat inputImage(cv::Size(curSize, curSize), CV_64F);
			std::string str;
			Mat re = Mat_<double>(curSize, curSize);
			Mat im = Mat_<double>(curSize, curSize);
			Mat complexImage = cv::Mat::zeros(cv::Size(curSize, curSize), CV_64FC2);
			Mat planes[] = { re.clone(), im.clone() };
			Mat WkP = Mat_<double>(curSize, curSize);
			double hMean = 0.0;

			if (NsLoop <= (Image0to8 - 1))
			{
				str = std::to_string(NsLoop);
				// ★③ FName系はローカル変数に代入
				FName = FILE_DTOP +"output\\" + Ws + "_photo " + str + "_" + inputCom + ".bmp";
				FName_Mst = FILE_DTOP + "output\\" + Ws + "_Mst_photo " + str + "_" + inputCom + ".bmp";
				FName_all = FILE_DTOP + "output\\" + Ws + "_ALL_photo " + str + "_" + inputCom + ".bmp";
				FName_Srv_ = FILE_DTOP + "output\\" + Ws + "_Srv_photo " + str + "_" + inputCom + "_";
				FName_Src = FILE_DTOP + "output\\" + Ws + "_Src_photo " + str + "_" + inputCom + ".bmp";
				FName_cgh = FILE_DTOP + "output\\" + Ws + "_CGH_photo " + str + "_" + inputCom + ".bmp";
				FName_his = FILE_DTOP + "output\\" + Ws + "_His_photo " + str + "_" + inputCom + ".bmp";
				// ★④ SetConsoleColor/cout は critical で保護
#pragma omp critical(console)
				{
					SetConsoleColor(FOREGROUND_GREEN | FOREGROUND_INTENSITY);
					std::cout << FName;
					SetConsoleColor(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
				}
			}
			//if (NsLoop >= 0 && NsLoop <= 7) {
			if (NsLoop <= 7) {
				FName_END = Img_Read(FILE_DTOP, NsLoop, grayImage, target_mat, curSize, BLK);
				FName_Srv = FName_Srv_ + FileNameBMP[NsLoop];
#pragma omp critical(console)
				{
					{ cout << "\nFName_END=" << FName_END << std::endl; }
				}
			}
			if (NsLoop >= 8)
			{
				curSize = 384;
				std::ostringstream Wss;
				Wss << std::setw(4) << std::setfill('0') << (NsLoop - 8);
				std::string Ws_loc(Wss.str()); // ★③ 外側 Ws との名前衝突を回避
#if MODE_CUT == 0
				// ★④
#pragma omp critical(console)
				{ std::cout << Ws_loc; }
#endif
				FName_END = MITImg_Read(FILE_DTOP, Ws_loc, grayImage, curSize, 1);
			}

#if DISP_CUT == 0
			// ★⑤ imshow/waitKey は critical で保護
#pragma omp critical(gui)
			{
				cv::waitKey(1);
				cv::imshow("Input Image", grayImage);
			}
#endif

			grayImage.convertTo(inputImage, CV_64F, 1.0 / 255.0);
			WkP = cv::Mat::zeros(cv::Size(curSize, curSize), CV_64F);
			re = cv::Mat::zeros(cv::Size(curSize, curSize), CV_64F);
			im = cv::Mat::zeros(cv::Size(curSize, curSize), CV_64F);

			Img_FFT(NsLoop, inputImage, re, im, complexImage, planes, Trans, curSize);

#if MODE_CUT == 0
#if FILE_OUT == 1
			cv::magnitude(re, im, WkP);
			string filenames = FILE_DTOP + "output\\" + std::string("A_sinpuku.csv");
			// ★④
#pragma omp critical(console)
			{ cout << "CSVファイルに出力しました" << filenames << endl; }
#endif
#endif

			// === GS法 ===
			// Gerchberg-Saxton 反復法による位相限定ホログラム(POH)生成。
			// 物体面(画像)とホログラム面(周波数領域)を FFT/IFFT で往復しながら、
			// それぞれの面で振幅拘束を課して位相分布を収束させる。
			//   ・物体面   : 信号領域の振幅を目標振幅 A_target に強制
			//   ・ホログラム面 : 振幅を 1 に強制（位相のみ = POH 拘束）
			// 画像の下側をダミー領域(ノイズ逃がし)とし、そこでは振幅を自由にすることで
			// 信号領域の再生品質(PSNR)を大幅に向上させる方式。
			int num_gs_iterations = 119;                    // GS反復回数（既定:信号50% / ダミー50%）
			cv::Mat A_target;
			cv::sqrt(inputImage, A_target);                 // 目標振幅 = sqrt(強度)。強度画像→振幅への変換
			cv::Mat current_phase = Trans.clone();          // 初期位相：ランダム位相 Trans からスタート
			cv::Mat current_mag = A_target.clone();         // 初期振幅：目標振幅そのもの
			cv::Mat gs_complex, gs_holo, gs_mag, gs_phase;  // 作業用（複素場・ホログラム・振幅・位相）
			cv::Mat gs_planes[2];                           // 複素場の実部/虚部プレーン
			int signalRows = curSize / 2;                   // 信号領域の行数（上側）。残りがダミー領域(信号50% / ダミー50%)（最良: 平均PSNR約47.8dB）

			// コマンド文字列で信号/ダミー比率と反復回数を切り替える。
			// DUMMYxx = 信号領域が全体の xx%（残りがダミー領域）。
			// 反復回数はダミー領域が狭いほど収束が遅いため多めに設定（実験で決定）。
			if (inputCom.find("DUMMY75") != std::string::npos) {
				num_gs_iterations = 470;            // 信号75% / ダミー25%（収束遅い→470回）
				signalRows = curSize * 3 / 4;
			}
			else if (inputCom.find("DUMMY60") != std::string::npos) {
				num_gs_iterations = 251;            // 信号60% / ダミー40%
				signalRows = curSize * 6 / 10;
			}

			// GS設定の確認ログ（50回に1回だけ出力して冗長化を防ぐ）
			// ★① atomic でカウンタ取得
			int myLogCount = GS_logCounter.fetch_add(1);
			const bool doLog = ((myLogCount % 50) == 0);
			if (doLog) {
				// ★④
#pragma omp critical(console)
				{
					std::cout << "[GS] num_gs_iterations = " << num_gs_iterations
						<< " signalRows=" << signalRows
						<< " dummyRows=" << (curSize - signalRows)
						<< " dummyRatio=" << (100.0 * (curSize - signalRows) / curSize) << "%"
						<< std::endl;
				}
			}

			// --- GS 反復ループ ---
			for (int iter = 0; iter < num_gs_iterations; iter++) {
				// [Step1] 物体面の複素場を構成（振幅×exp(i位相)）し、FFTでホログラム面へ伝搬
				cv::polarToCart(current_mag, current_phase, gs_planes[0], gs_planes[1]);
				cv::merge(gs_planes, 2, gs_complex);
				cv::dft(gs_complex, gs_holo, cv::DFT_COMPLEX_OUTPUT);
				// [Step2] ホログラム面で位相 gs_phase を抽出
				cv::split(gs_holo, gs_planes);
				cv::cartToPolar(gs_planes[0], gs_planes[1], gs_mag, gs_phase);
				// [Step3] POH拘束：ホログラム面の振幅を 1 に強制（位相情報のみ保持）
				cv::Mat mag_ones = cv::Mat::ones(curSize, curSize, CV_64F);
				cv::polarToCart(mag_ones, gs_phase, gs_planes[0], gs_planes[1]);
				cv::merge(gs_planes, 2, gs_holo);
				// [Step4] IDFT（DFT_SCALEで1/N^2正規化）で物体面へ逆伝搬し、再生振幅 gs_mag と位相を取得
				//         位相 current_phase は次反復へそのまま引き継ぐ
				cv::dft(gs_holo, gs_complex, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);
				cv::split(gs_complex, gs_planes);
				cv::cartToPolar(gs_planes[0], gs_planes[1], gs_mag, current_phase);

				// [Step5] エネルギー整合スケールの算出。
				// 信号領域内で目標振幅と再生振幅のエネルギー(振幅^2の和)を比較し、
				// scale = sqrt(energy_A/energy_gs) でダミー領域の振幅レベルを信号側に整合させる。
				// ※scale に余分な倍率(BOOST>1等)を掛けると正帰還で発散するため厳禁。
				double energy_A = 0.0, energy_gs = 0.0;
				// ★⑥ #pragma omp parallel for reduction を削除（外側NsLoopとの二重並列回避）
				for (int y = 0; y < signalRows; y++) {
					const double* pA = A_target.ptr<double>(y);
					const double* pG = gs_mag.ptr<double>(y);
					for (int x = 0; x < curSize; x++) {
						energy_A += pA[x] * pA[x];
						energy_gs += pG[x] * pG[x];
					}
				}
				double scale = (energy_gs > 0.0) ? std::sqrt(energy_A / energy_gs) : 1.0;

				// [Step6] 物体面の振幅拘束（次反復用の current_mag を更新）
				//   信号領域 (y < signalRows) : 目標振幅 A_target に強制（画像を忠実に再生）
				//   ダミー領域 (y >= signalRows): 再生振幅をそのまま許容（scaleで整合のみ）
				//                               → 誤差エネルギーをここへ逃がす「ノイズ逃がし」
				// ★⑥ #pragma omp parallel for を削除（外側NsLoopとの二重並列回避）
				for (int y = 0; y < curSize; y++) {
					for (int x = 0; x < curSize; x++) {
						if (y < signalRows)
							current_mag.at<double>(y, x) = A_target.at<double>(y, x);
						else
							current_mag.at<double>(y, x) = gs_mag.at<double>(y, x) * scale;
					}
				}
			}

			// [Step7] 最終ホログラム位相 gs_phase を [0,2π) に折り畳み、
			//         [0,1] に正規化して位相ホログラム WkP へ格納（元のコードのまま）
			for (int y = 0; y < curSize; y++) {
				for (int x = 0; x < curSize; x++) {
					double p = gs_phase.at<double>(y, x);
					p = std::fmod(p, 2.0 * CV_PI);          // 2πで剰余を取り
					if (p < 0.0) p += 2.0 * CV_PI;          // 負値は +2π して [0,2π) に収める
					WkP.at<double>(y, x) = p / (2.0 * CV_PI); // 0～1 に正規化（位相/2π）
				}
			}
			// === ここまで GS法 ===
			// 以降：WkP を bitDepth に量子化して img_CGH（表示・DE評価用ホログラム）を生成

			double scalingFactor = 1.0;
			double bitDepth = 8.0;
			normalize(WkP, WkP, 0.0, 1.0, NORM_MINMAX);
			double numLevels = std::pow(2.0, bitDepth) - 1.0f;
			WkP.convertTo(WkP, CV_64F, numLevels);
			WkP += 0.5;
			if (bitDepth <= 8.0)
				WkP.convertTo(img_CGH, CV_8U);
			else
				WkP.convertTo(img_CGH, CV_16U);
			scalingFactor = numLevels;

			cv::Mat holoForDE;
			img_CGH.convertTo(holoForDE, CV_64F);
			if (scalingFactor != 1.0 && scalingFactor != 0.0)
				holoForDE /= scalingFactor;
			cv::Mat phiDE = holoForDE * 2.0 * CV_PI;
			cv::Mat magDE = cv::Mat::ones(holoForDE.size(), CV_64F);
			cv::Mat reDE, imDE;
			cv::polarToCart(magDE, phiDE, reDE, imDE);
			cv::Mat holoPlanesDE[] = { reDE, imDE };
			cv::Mat holoComplexForDE;
			cv::merge(holoPlanesDE, 2, holoComplexForDE);
			cv::Mat reconstructedComplexForDE;
			cv::dft(holoComplexForDE, reconstructedComplexForDE, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);
			cv::Mat mask_8u_de = (inputImage != 0.0);
			cv::Mat mask_64F_de;
			mask_8u_de.convertTo(mask_64F_de, CV_64F, 1.0 / 255.0);
#if  MODE_CUT == 0
#if DISP_CUT == 0
#pragma omp critical(gui)
			{
				cv::Mat img_ROI_de;
				mask_8u_de.convertTo(img_ROI_de, CV_8U, 255);
				cv::imshow("MASK", img_ROI_de);
			}
#endif // DISP_CUT
#endif //  MODE_CUT
			double Mde_physics = calculateDiffractionEfficiencyWithMask(inputImage, reconstructedComplexForDE, mask_64F_de, DE_Method::PHYSICS_BASED, curSize);
			double Mde_strict = calculateDiffractionEfficiencyStrictMAIN(holoForDE, inputImage, curSize);

			cv::Mat intensity_roi = Mat_<double>(curSize, curSize);
			intensity_roi = Img_IFFT(holoForDE, curSize);
			cv::multiply(intensity_roi, intensity_roi, intensity_roi);
			cv::normalize(intensity_roi, intensity_roi, 0.0, 1.0, cv::NORM_MINMAX);
			intensity_roi.convertTo(img_all, CV_8U, 255);

			int border = BLK;
			Mat COPYtarget_mat_M_BLK(cv::Size(curSize - BLK * 2, (curSize / 2) - BLK * 2), CV_8U);
			Mat COPYtarget_mat_M_256(cv::Size(curSize, (curSize / 2)), CV_8U);
			grayImage.rowRange(0, (curSize / 2)).copyTo(img_Mst);
			COPYtarget_mat_M_BLK = img_Mst(cv::Rect(border, border, img_Mst.cols - 2 * border, img_Mst.rows - 2 * border)).clone();
			cv::copyMakeBorder(COPYtarget_mat_M_BLK, COPYtarget_mat_M_256, BLK, BLK, BLK, BLK, cv::BORDER_CONSTANT, cv::Scalar(0));

			if (NsLoop >= 8)
			{
				int BLK_size = 2;
				Mat COPYtarget_mat_M_384BLK(cv::Size(curSize - BLK_size * 2, (curSize / 2) - BLK_size * 2), CV_8U);
				COPYtarget_mat_M_384BLK = img_Mst(cv::Rect(BLK_size, BLK_size, img_Mst.cols - 2 * BLK_size, img_Mst.rows - 2 * BLK_size)).clone();
				cv::copyMakeBorder(COPYtarget_mat_M_384BLK, COPYtarget_mat_M_256, BLK_size, BLK_size, BLK_size, BLK_size, cv::BORDER_CONSTANT, cv::Scalar(0));
				intensity_roi.rowRange(0, (curSize / 2)).copyTo(intensity_roi);
				intensity_roi = intensity_roi(cv::Rect(BLK_size, BLK_size, intensity_roi.cols - 2 * BLK_size, intensity_roi.rows - 2 * BLK_size)).clone();
				cv::copyMakeBorder(intensity_roi, intensity_roi, BLK_size, BLK_size, BLK_size, BLK_size, cv::BORDER_CONSTANT, cv::Scalar(0));
			}
			else
			{
				intensity_roi.rowRange(0, (curSize / 2)).copyTo(intensity_roi);
				intensity_roi = intensity_roi(cv::Rect(border, border, intensity_roi.cols - 2 * border, intensity_roi.rows - 2 * border)).clone();
				cv::copyMakeBorder(intensity_roi, intensity_roi, BLK, BLK, BLK, BLK, cv::BORDER_CONSTANT, cv::Scalar(0));
			}

			cv::Mat histImage;
			cv::Mat COPYtarget_mat_S_256(cv::Size(curSize, (curSize / 2)), CV_8U);
			int Deff_median_OUT = 0;
#pragma omp critical(gui)
			{
				Deff_median_OUT = histogramMAIN(
					COPYtarget_mat_M_256, COPYtarget_mat_S_256,
					intensity_roi, img_Mst, img_Srv, curSize, histImage);
			}
			if (Deff_median_OUT != 0)
			{
				cv::subtract(COPYtarget_mat_S_256, cv::Scalar(Deff_median_OUT), COPYtarget_mat_S_256);
			}

			// ★④
#pragma omp critical(console)
			{ std::cout << "Deff_median:" << Deff_median_OUT; }

			// ★ evaluateHologram はそのまま呼ぶ（内部で critical を使用）
			evaluateHologram(COPYtarget_mat_M_256, COPYtarget_mat_S_256,
				NsLoop, Mde_physics, Mde_strict,
				FILE_DTOP, WKKK, inputCom);

			// ★⑦ Disp_Save_img (imshow含む可能性) は critical で保護
#pragma omp critical(filesave)
			{
				SetConsoleColor(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
				Disp_Save_img(FName_all, COPYtarget_mat_M_256, COPYtarget_mat_S_256, img_CGH, img_all, img_Mst, img_Srv, histImage,
					FName_Src, FName_Mst, FName_Srv, FName_cgh, FName_his, FILE_DTOP, inputCom, NsLoop, curSize);
			}

			img_src.release();
			img_Mst.release();
			img_Srv.release();
			img_all.release();
			img_CGH.release();
			grayImage.release();
			inputImage.release();
			WkP.release();
		}
		// ===== 並列ループ終了後にシングルスレッドで集計 =====

// ★(B) 写真8枚の平均集計（従来 NsLoop==6 のブロック）
		aggregatePhotoResults();

		// ★(C) MIT画像の平均集計 + imwrite（従来末尾ブロック）
		aggregateMITResults(WKKK, FILE_DTOP, inputCom);

		DWORD elapsed = (timeGetTime() - start);
		std::cout << "t=" << elapsed << "ms.(" << (elapsed / 1000.0) << "sec. " << (elapsed / 60000.0) << "min.) / 4,008 = " << (elapsed / 1000.0) / (NloopMIT + Image0to8) << "sec[" << 1.0 / ((elapsed / 1000.0) / (NloopMIT + Image0to8)) << "fps]\n";
		////csv.close();
		std::cout << "Enter any key to exit..." << std::endl;
		std::cin.get(); // キー入力待ち

		// ウィンドウを破棄する
		// COMのリリース
		CoUninitialize();
		return -1;
	}
}