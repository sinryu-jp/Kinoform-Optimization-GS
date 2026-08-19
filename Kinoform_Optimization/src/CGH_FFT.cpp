// Copyright (C) 1991-93 2004-26 by SIT & Koki Sato, Masataka Totsuka, Kenji Kiuchi
//
// CGH_FFT.cpp file (C++ version)
// * This program is intended exclusively for x64 (64-bit CPU spec).
//
// CV_PI;// 6.28318530717958647692  //Created 2*pi using CV_PI*2.0.
// 
// Create it as a console application.
// * Please use the OpenCV 4.110 series and Microsoft Visual Studio 2026.

#include <omp.h>

#define _CRT_SECURE_NO_WARNINGS
#include <cmath>
#include <time.h>     // for clock()

#include <iostream>
#include <iomanip>

#include <fstream>
#include <string>
#include <sstream>

#include <vector>

#include <algorithm>
#include <windows.h>
#include <stdlib.h>
#include <atomic>
#pragma comment (lib, "winmm.lib") 

#define M_MAT 256
#define H_MAT 128

#include "C:\\opencv\\opencv2\\highgui\\highgui.hpp"
#include "C:\\opencv\\opencv2\\imgproc\\imgproc.hpp"
//For the Release build
#pragma comment(lib,"C:\\opencv\\opencv_world4110.lib")

using namespace cv;
using namespace std;

//0: Enable the evaluation command.
//1: Disable the evaluation command.
#define MODE_CUT 0
//0: Enable the display command.
//1: Disable the display command.
#define DISP_CUT 1

static string FileNameBMP[10];


// ==========================================================
// Common SSIM Computation Function
//
// Computes the Structural Similarity Index (SSIM) between two images (i1, i2).
// Constants C1 and C2 can be configured based on the evaluation criteria:
//   - Standard SSIM (8-bit [0, 255] scale, L=255, K1=0.01, K2=0.03):
//       C1 = (K1 * L)^2 = 6.5025,  C2 = (K2 * L)^2 = 58.5225
//   - Prof. Yoshikawa's criteria (Normalized [0, 1] scale):
//       C1 = 0.01,  C2 = 0.03
//
// @param i1, i2 Input images (CV_8U or CV_64F).
// @param C1, C2 Stabilization constants (default: Standard SSIM for 8-bit).
// @param depth  Internal computation precision (CV_32F or CV_64F).
// @return cv::Scalar Mean SSIM value across channels.
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
// Standard SSIM (for AI paper comparison: K1=0.01, K2=0.03, L=255)
// Arguments i1, i2 are expected to be 8-bit images (CV_8U, 0-255) or 64F images
// ==========================================================
double getStandardSSIM(const cv::Mat& i1, const cv::Mat& i2)
{
	return computeSSIM(i1, i2, 6.5025, 58.5225, CV_32F)[0];
}
// ==========================================================
// Prof. Yoshikawa's constant-based SSIM (for images normalized to 0-1: C1=0.01, C2=0.03)
// ==========================================================
cv::Scalar getSSIM(const cv::Mat& i1, const cv::Mat& i2) {
	return computeSSIM(i1, i2, 0.01, 0.03, CV_64F);
}

// Simplified HVS weight matrix (defined as an example)
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

// Split the image into 8x8 blocks and apply DCT and weighting
cv::Mat applyDCTandWeight(const cv::Mat& image, const cv::Mat& weights) {
	cv::Mat processed = cv::Mat::zeros(image.size(), CV_64F);
	for (int i = 0; i < image.rows; i += 8) {
		for (int j = 0; j < image.cols; j += 8) {
			cv::Rect blockRect(j, i, 8, 8);
			cv::Mat block = image(blockRect);
			cv::Mat blockdouble;
			block.convertTo(blockdouble, CV_64F);

			// Apply DCT
			cv::Mat dctBlock;
			cv::dct(blockdouble, dctBlock);

			// Apply the weight matrix
			cv::Mat weightedBlock = dctBlock.mul(weights);

			// Save the result
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
	}
	double sn = GetPSNR(ssrc, ddest);
	return sn;
}

// Calculate PSNR-HVS
double calculatePSNRHVS(const cv::Mat& original, const cv::Mat& distorted, const cv::Mat& weights) {
	cv::Mat originalProcessed = applyDCTandWeight(original, weights);
	cv::Mat distortedProcessed = applyDCTandWeight(distorted, weights);

	// Compute mean squared error (MSE)
	cv::Mat diff = originalProcessed - distortedProcessed;
	cv::Mat squaredError;
	cv::multiply(diff, diff, squaredError);

	double mse = cv::mean(squaredError)[0];
	double maxPixel = 255.0;//the constant representing the maximum pixel value used in the numerator of the PSNR formula for 8‑bit images (0–255)

	// Calculate PSNR-HVS
	double psnrHVS = 10.0 * std::log10((maxPixel * maxPixel) / mse);
	return psnrHVS;
}

// Compute masking coefficients
cv::Mat calculateMasking(const cv::Mat& dctBlock) {
	cv::Mat masking = cv::abs(dctBlock); // Use the absolute value of the DCT coefficients
	return cv::min(masking, cv::Mat::ones(masking.size(), CV_64F) * 255); // Range limiting
}

// Split the image into 8x8 blocks and apply DCT, weighting, and masking
cv::Mat applyDCTWeightMasking(const cv::Mat& image, const cv::Mat& weights) {
	cv::Mat processed = cv::Mat::zeros(image.size(), CV_64F);

	for (int i = 0; i < image.rows; i += 8) {
		for (int j = 0; j < image.cols; j += 8) {
			cv::Rect blockRect(j, i, 8, 8);
			cv::Mat block = image(blockRect);
			cv::Mat blockdouble;
			block.convertTo(blockdouble, CV_64F);

			// Apply DCT
			cv::Mat dctBlock;
			cv::dct(blockdouble, dctBlock);

			// Apply HVS weights
			cv::Mat weightedBlock = dctBlock.mul(weights);

			// Apply masking
			cv::Mat masking = calculateMasking(dctBlock);
			cv::Mat maskedBlock = weightedBlock.mul(masking);

			// Save the result
			maskedBlock.copyTo(processed(blockRect));
		}
	}
	return processed;
}

// Calculate PSNR-HVS-M
double calculatePSNRHVSM(const cv::Mat& original, const cv::Mat& distorted, const cv::Mat& weights) {
	cv::Mat originalProcessed = applyDCTWeightMasking(original, weights);
	cv::Mat distortedProcessed = applyDCTWeightMasking(distorted, weights);

	// Compute mean squared error (MSE)
	cv::Mat diff = originalProcessed - distortedProcessed;
	cv::Mat squaredError;
	cv::multiply(diff, diff, squaredError);

	double mse = cv::mean(squaredError)[0];
	double maxPixel = 255.0;//the constant representing the maximum pixel value used in the numerator of the PSNR formula for 8‑bit images (0–255)

	// Calculate PSNR-HVS-M
	double psnrHVSM = 10.0 * std::log10((maxPixel * maxPixel) / mse);
	return psnrHVSM;
}
/**************************/
void sinF(const Mat& mat, Mat& sinMat) {
#pragma omp parallel for collapse(2)
	// Apply the sin function to each element
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			sinMat.at<double>(i, j) = std::sinf(mat.at<double>(i, j));
		}
	}
}
void cosF(const Mat& mat, Mat& cosMat) {
#pragma omp parallel for collapse(2)
	// Apply the cos function to each element
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			cosMat.at<double>(i, j) = std::cosf(mat.at<double>(i, j));
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
		return FName_END;  // Added return
	}
	// (2) Check whether target_mat size is smaller than grayImage
	if (target_mat.cols < grayImage.cols || target_mat.rows < grayImage.rows)
	{
		std::cerr << "The loaded image is smaller than the reference grayImage.:target="
			<< target_mat.cols << "x" << target_mat.rows
			<< " gray=" << grayImage.cols << "x" << grayImage.rows << std::endl;
		return FName_END;
	}
	target_mat = target_mat(cv::Rect(1, 1, grayImage.cols - 2, grayImage.rows - 2)).clone();


	cv::copyMakeBorder(target_mat, target_mat, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255));//Trim the border of the image


	target_mat.rowRange(0, (M_MAT / 2)).copyTo(target_mat);//Use the upper half as the reference image
	target_mat.convertTo(target_mat, CV_8U);//Convert to grayscale image
	int border = BLK; // Number of pixels to trim
	if (BLK != 1)
	{
		// --- 1. Trim target_mat by the border amount ---
		if (target_mat.cols <= 2 * border || target_mat.rows <= 2 * border)
		{
			std::cerr << "BLK is too large to be trimmed from target_mat." << std::endl;
			return FName_END;
		}
		target_mat = target_mat(
			cv::Rect(border, border,
				target_mat.cols - 2 * border,
				target_mat.rows - 2 * border)).clone();
		// --- 2. Create grayImage with the same width as target_mat and double the height ---
		int width = target_mat.cols;
		int height = target_mat.rows - BLK * 2;// (target_mat.rows) * 2;

		grayImage = cv::Mat::zeros(cv::Size(width, height), CV_8U);

		// target_mat in the upper half
		cv::Mat upper = grayImage(cv::Rect(0, 0, width, target_mat.rows));
		target_mat.copyTo(upper);

		//Mat COPYgrayImage = grayImage(cv::Rect(1, 1, grayImage.cols - 2 * 1, grayImage.rows - 2 * 1)).clone();
		//cv::copyMakeBorder(COPYgrayImage, grayImage, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255));


	}
	else
	{
		//cv::Mat COPYtarget_mat = target_mat(cv::Rect(border, border, target_mat.cols - 2 * border, target_mat.rows - 2 * border)).clone();
		//cv::copyMakeBorder(COPYtarget_mat, target_mat, BLK, BLK, BLK, BLK, cv::BORDER_CONSTANT, cv::Scalar(255));
		// Compose gray_image on an M_MATxM_MAT black background, aligned to the top-left corner
		grayImage = cv::Mat::zeros(cv::Size(curSize, curSize), CV_8U);//Create an M_MAT x M_MAT black background
		const int copy_w0 = curSize;//std::min(target_mat.cols, curSize);
		const int copy_h0 = std::min(target_mat.rows, curSize);
		target_mat(cv::Rect(0, 0, copy_w0, copy_h0)).copyTo(grayImage(cv::Rect(0, 0, copy_w0, copy_h0)));// Copy target_mat to the top-left
	}


	//int big_width = target_mat.cols > target_mat.rows ? target_mat.cols : target_mat.rows;//Which is longer, width or height?
	//double ratio = ((double)curSize / (double)big_width);//Ratio
	//cv::resize(target_mat, grayImage, cv::Size(), ratio, ratio, cv::INTER_NEAREST);//Resize

	return  FName_END;
}
void Img_FFT(const int NbannMe, Mat& WkQ, Mat& re, Mat& im,
	Mat& complexImage, Mat* planes,
	Mat& Trans, int curSize)
{
	Mat WkP(curSize, curSize, CV_64F);
	Mat WkR(curSize, curSize, CV_64F);

	cosF(Trans, WkP);
	planes[0] = WkQ.mul(WkP);//Tabulated the random sine/cosine.//b[i][MAX_MAT - 1 - j]*exp(0.0)*cos(Trans[i][j]);
	sinF(Trans, WkR);
	planes[1] = WkQ.mul(WkR);//Tabulated the random sine/cosine.//b[i][MAX_MAT - 1 - j]*exp(0.0)*sin(Trans[i][j]);

	cv::merge(planes, 2, complexImage);
	cv::dft(complexImage, complexImage);
	cv::split(complexImage, planes);

	re = planes[0];// Assign the real part
	im = planes[1];// Assign the imaginary part
}

// Function to change the console text color
void SetConsoleColor(WORD color) {
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(hConsole, color);
}
// Function to get the computer name
std::string getComputerName() {
	char name[256];
	DWORD size = sizeof(name);
	//if (GetComputerNameW(name, &size)) {// For Unicode version
	if (GetComputerNameA(name, &size)) {// For ANSI version
		return std::string(name);
	}
	return "Unknown";
}

// 2D Otsu method core: finds the optimal (s,t) using jointHist.
// Return value: bestBetween (the maximum between-class variance)
// best_s, best_t hold the respective thresholds
static double compute2DOtsuFromJoint(const cv::Mat& jointHist, int& best_s, int& best_t)
{
	// jointHist: CV_64F 256x256 (raw counts)
	const int L = 256;
	double total = cv::sum(jointHist)[0];
	if (total <= 0.0) { best_s = best_t = 0; return 0.0; }

	// Normalized probability p(i,j)
	cv::Mat P(L, L, CV_64F);
	P = jointHist / total;

	// Build cumulative sums (size = (L+1)x(L+1)): S, SX, SY
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

	// Overall first-order moment
	double SX_total = SX[L][L];
	double SY_total = SY[L][L];

	// Evaluate (s,t) by full search
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
// approx2D Otsu (approximate 2D Otsu method using only Srv_hist)
// Description:
//   The input is only Srv_hist (1D histogram, 256 bins); without using image data, this computes
//   approximate 2D Otsu thresholds (s,t). Internally, assuming that for intensity i the local
//   average j is distributed around i, a pseudo 256x256 joint-histogram is created via Gaussian
//   smoothing, and (s,t) is searched for using the existing compute2DOtsuFromJoint. Finally, the
//   "white ratio" and "black ratio" are also returned.
//
// Arguments:
//   - const cv::Mat& Srv_hist
//       1D histogram (CV_64F of length 256, or a 1x256 / 256x1 row/column vector)
//   - int& out_s
//       Output: threshold s on the intensity side (0..255)
//   - int& out_t
//       Output: threshold t on the local-average side (0..255)
//   - double& whiteRatioPercent
//       Output: percentage (%) of pixels classified as "white"
//   - double& blackRatioPercent
//       Output: percentage (%) of pixels classified as "black"
//   - double sigma = 5.0
//       Optional: Gaussian width (standard deviation) used to build the pseudo joint-histogram.
//       Smaller values give a sharper distribution close to mean=j.
// Return value:
//   None (output is set via reference arguments out_s, out_t, whiteRatioPercent, blackRatioPercent)
//
// Notes:
//   - If the histogram contains all zeros, out_s=out_t=0, whiteRatio=0, blackRatio=100 are returned.
//   - This method is an "approximation". Results may differ from the true 2D Otsu (using the image's
//     actual joint-histogram).
//   - Uses compute2DOtsuFromJoint (an existing function in this file).
static void approx2DOtsuFromSrvHist_only(
	const cv::Mat& Srv_hist,
	int& out_s,
	int& out_t,
	double& whiteRatioPercent,
	double& blackRatioPercent,
	double sigma = 5.0)
{
	// Initialization
	out_s = 0; out_t = 0;
	whiteRatioPercent = 0.0; blackRatioPercent = 100.0;

	if (Srv_hist.empty()) return;

	// Convert the histogram to a CV_64F vector (shape: 256 elements)
	cv::Mat hist64;
	if (Srv_hist.type() == CV_64F) {
		hist64 = Srv_hist.clone();
	}
	else {
		Srv_hist.convertTo(hist64, CV_64F);
	}

	// Shape check: convert 1x256 or 256x1 into a 256-length vector
	cv::Mat histVec;
	if (hist64.rows == 1 && hist64.cols == 256) histVec = hist64.reshape(1, 256);
	else if (hist64.rows == 256 && hist64.cols == 1) histVec = hist64;
	else if (hist64.total() == 256) histVec = hist64.reshape(1, 256);
	else {
		// Unexpected shape
		return;
	}

	// Get the total
	double totalCount = cv::sum(histVec)[0];
	if (totalCount <= 0.0) {
		// All zero
		out_s = out_t = 0;
		whiteRatioPercent = 0.0;
		blackRatioPercent = 100.0;
		return;
	}

	// Create a pseudo joint-histogram (256 x 256)
	cv::Mat jointHist = cv::Mat::zeros(256, 256, CV_64F);

	// Build a Gaussian kernel (only central differences are used, so compute the j weights for each center i)
	const double twoSigma2 = 2.0 * sigma * sigma;
	for (int i = 0; i < 256; ++i) {
		double hi = histVec.at<double>(i, 0);
		if (hi <= 0.0) continue;
		// Weight along the j axis (0..255)
		double sumW = 0.0;
		double wbuf[256];
		for (int j = 0; j < 256; ++j) {
			double d = static_cast<double>(j - i);
			double w = std::exp(-(d * d) / twoSigma2);
			wbuf[j] = w;
			sumW += w;
		}
		// Normalize and add to jointHist
		if (sumW <= 0.0) continue;
		double factor = hi / sumW;
		for (int j = 0; j < 256; ++j) {
			jointHist.at<double>(i, j) += factor * wbuf[j];
		}
	}

	// jointHist is built; use the existing compute2DOtsuFromJoint to find (s,t)
	int best_s = 0, best_t = 0;
	compute2DOtsuFromJoint(jointHist, best_s, best_t);

	// Let C0 be (i <= s && j <= t) and C1 its complement
	double total = cv::sum(jointHist)[0];
	if (total <= 0.0) {
		out_s = best_s; out_t = best_t;
		whiteRatioPercent = 0.0; blackRatioPercent = 100.0;
		return;
	}

	double cntC0 = 0.0, cntC1 = 0.0;
	double sumI_C0 = 0.0, sumI_C1 = 0.0; // Sum of intensity for each class (weighted)
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

	// Mean intensity of each class
	double meanC0 = (cntC0 > 0.0) ? (sumI_C0 / cntC0) : 0.0;
	double meanC1 = (cntC1 > 0.0) ? (sumI_C1 / cntC1) : 0.0;

	// White is the class with the higher mean intensity
	double whiteCount = 0.0;
	if (meanC1 >= meanC0) whiteCount = cntC1;
	else whiteCount = cntC0;

	whiteRatioPercent = 100.0 * (whiteCount / total);
	blackRatioPercent = 100.0 - whiteRatioPercent;

	// Output
	out_s = best_s;
	out_t = best_t;
}



// --- Replace the 2D Otsu part inside drawHistogram with true 2D Otsu ---
// (Replace the section that builds meanImg inside the existing drawHistogram function with this implementation)
int drawHistogram(const cv::Mat& Mst_hist, const cv::Mat& Srv_hist, cv::Mat& histImage, int& Deff_median_OUT, Mat& img_Mst, Mat& img_Srv)
{
	int histSize = 256;
	int hist_w = 768; // Histogram width
	int hist_h = 250; // Histogram base height
	int scale_bar_height = 12;
	int label_height = 40;
	int text_height = 150; // Extended for displaying evaluation metrics

	int total_height = hist_h + scale_bar_height + label_height + text_height;

	// Set background color to white
	histImage = cv::Mat(total_height, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));
	// Normalization (scale the vertical axis based on a fixed value)
	//cv::Mat normMstHist, normSrvHist;
	//cv::normalize(Mst_hist, normMstHist, 0, hist_h, cv::NORM_MINMAX);
	//cv::normalize(Srv_hist, normSrvHist, 0, hist_h, cv::NORM_MINMAX);


	// Draw the two histograms on the same vertical scale (scaled by the common max)
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

	// Draw Srv_hist in black (originally white)
	for (int i = 1; i < histSize; i++) {
		cv::line(histImage,
			cv::Point(bin_w * (i - 1), hist_h - cvRound(normSrvHist.at<float>(i - 1))),
			cv::Point(bin_w * i, hist_h - cvRound(normSrvHist.at<float>(i))),
			cv::Scalar(0, 0, 0), 2, 8, 0); // Black
	}

	// Draw Mst_hist in red (unchanged)
	for (int i = 1; i < histSize; i++) {
		cv::line(histImage,
			cv::Point(bin_w * (i - 1), hist_h - cvRound(normMstHist.at<float>(i - 1))),
			cv::Point(bin_w * i, hist_h - cvRound(normMstHist.at<float>(i))),
			cv::Scalar(0, 0, 255), 2, 8, 0); // Red
	}

	// Draw grayscale bar (unchanged)
	for (int i = 0; i < hist_w; i++) {
		int v = cvRound((double)i / hist_w * 255.0);
		cv::line(histImage,
			cv::Point(i, hist_h),
			cv::Point(i, hist_h + scale_bar_height - 1),
			cv::Scalar(v, v, v), 1);
	}

	// Tick marks and labels (color set to black)
	int scale_y = hist_h + scale_bar_height;
	for (int i = 0; i <= histSize; i += 10) {
		int x = bin_w * i;
		cv::line(histImage, cv::Point(x, hist_h + scale_bar_height), cv::Point(x, scale_y + 6), cv::Scalar(0, 0, 0), 1);
		if (i == 0 || i == 128 || i == 255) {
			cv::putText(histImage, std::to_string(i), cv::Point(x - 12, scale_y + 24),
				cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
		}
	}

	// Axis label (color set to black)
	cv::putText(histImage, "Intensity (Grayscale 0-255)", cv::Point(425, total_height - text_height - 25),
		cv::FONT_HERSHEY_SIMPLEX, 0.73, cv::Scalar(0, 0, 0), 2);

	// --- Compute image entropy ---
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

	// --- Compute spatial frequency ---
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

	// --- Compute entropy and spatial frequency from the original images ---
	double entropy_Mst = 0.0, entropy_Srv = 0.0, sf_Mst = 0.0, sf_Srv = 0.0;
	if (!img_Mst.empty()) {
		entropy_Mst = calcEntropy(img_Mst);
		sf_Mst = calcSpatialFrequency(img_Mst);
	}
	if (!img_Srv.empty()) {
		entropy_Srv = calcEntropy(img_Srv);
		sf_Srv = calcSpatialFrequency(img_Srv);
	}

	// --- Draw numeric values below the histogram ---
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(3);

	int out_s = 0, out_t = 0;
	double whiteRatioPercent;
	double blackRatioPercent;

	approx2DOtsuFromSrvHist_only(Srv_hist, out_s, out_t, whiteRatioPercent, blackRatioPercent, 5.0);
	// Display (2D Otsu)
	std::ostringstream otsuStream, otsuPercStream;
	otsuStream << "Otsu2D(approx) s=" << out_s << " t=" << out_t;
	otsuPercStream << " W:" << std::fixed << std::setprecision(2) << whiteRatioPercent << "% B:" << blackRatioPercent << "%";
	cv::putText(histImage, otsuStream.str() + otsuPercStream.str(), cv::Point(10, total_height - text_height + 25 * 0),
		cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 0, 0), 2);

	// --- Continued: draw mean, median, and standard deviation (reuse the original code as-is) ---
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

	// Deff statistics
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

	// Draw the Mst_hist side
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

	int Mst_x = 270; // Center-aligned position (adjust as needed)
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

	// Draw the Deff side (center-aligned)
	std::ostringstream DeffMeanStream, DeffMedianStream, DeffStddevStream;
	DeffMeanStream << std::fixed << std::setprecision(1) << Deff_mean;
	DeffMedianStream << std::fixed << std::setprecision(1) << Deff_median;
	DeffStddevStream << std::fixed << std::setprecision(1) << Deff_std;
	int deff_x = 530; // Center-aligned position (adjust as needed)
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
	cv::line(histImage, cv::Point(x_median_inp, 0), cv::Point(x_median_inp, hist_h), cv::Scalar(0, 0, 255), 2); // Green (2D)
	// Display the label to the upper-right of the line (position adjusted to fit within the histogram top)
	{
		std::string label2 = "median_inp";
		int label2_x = std::min(x_median_inp + 6, hist_w - 1 - 80); // Prevent overflow at the right edge (accounting for text width)
		int label2_y = std::max(12, 40);
		cv::putText(histImage, label2, cv::Point(label2_x, label2_y),
			cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
	}

	int x_median_out = cvRound((double)Srv_median * bin_w);
	cv::line(histImage, cv::Point(x_median_out, 0), cv::Point(x_median_out, hist_h), cv::Scalar(0, 0, 0), 2); // Green (2D)
	// Display the label to the upper-right of the line (position adjusted to fit within the histogram top)
	{
		std::string label = "median_out";
		int label_x = std::min(x_median_out + 6, hist_w - 1 - 60); // Prevent overflow at the right edge (accounting for text width)
		int label_y = std::max(12, 18);
		cv::putText(histImage, label, cv::Point(label_x, label_y),
			cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
	}

	return total_height;
}

#include <shlobj.h>
std::string getDesktopWKCGHFolder() {
	char desktopPath[MAX_PATH];
	// Get the desktop path
	if (SUCCEEDED(SHGetFolderPathA(NULL, CSIDL_DESKTOPDIRECTORY, NULL, 0, desktopPath))) {
		std::string wkCghPath = std::string(desktopPath) + "\\Kinoform_Optimization\\";
		// Create the folder if it doesn't exist
		DWORD attrib = GetFileAttributesA(wkCghPath.c_str());
		if (attrib == INVALID_FILE_ATTRIBUTES || !(attrib & FILE_ATTRIBUTE_DIRECTORY)) {
			CreateDirectoryA(wkCghPath.c_str(), NULL);
		}
		return wkCghPath;
	}
	return "";
}

string MITImg_Read(string FILE_DTOP, string FileName, Mat& grayImage, int curSize, int BLK)
{
	string FName_END = "NULL";
	// 1. Read the file
	// By specifying IMREAD_ANYCOLOR | IMREAD_ANYDEPTH, the floating-point (32-bit/16-bit)
	// data specific to EXR can be read while preserving it.
	std::string filename = FILE_DTOP + "data\\" +  FileName + ".exr";//test
#if MODE_CUT == 0
	std::cout << "Name of the loaded file: " << filename << std::endl;
#endif //MODE_CUT
	// To support paths with multi-byte (Japanese) characters, read the file via std::ifstream + imdecode
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

	// Check whether loading succeeded
	if (image.empty()) {
		std::cerr << "Error: The file could not be loaded." << std::endl;
		return "-1";
	}
	// 2. Convert to a bitmap (8-bit)
	// EXR is normally 32-bit floating point, so it needs to be converted to 8-bit for display or general saving.
	cv::Mat display_image;

	// A simple conversion tends to cause blown-out highlights or crushed shadows,
	// so it's common to first fit brightness values into the 0.0-1.0 range and then multiply by 255.

	// Scaling (example that sets the max value to 255)
	normalize(image, image, 0.0, 1.0, NORM_MINMAX);// Normalize to 0.0-1.0
	image.convertTo(display_image, CV_8UC3, 255);// Convert to 0-255
	// --- Grayscale conversion process starts here ---

	// 3. Convert from color (BGR) to grayscale
	cv::Mat gray_image;
	cv::cvtColor(display_image, gray_image, cv::COLOR_BGR2GRAY);

	// Compose gray_image on a 384x384 black background, aligned to the top-left corner
	cv::Mat comb_image = cv::Mat::zeros(cv::Size(curSize, curSize), CV_8U);
	const int copy_w0 = std::min(gray_image.cols, curSize);
	const int copy_h0 = std::min(gray_image.rows, curSize);
	gray_image(cv::Rect(0, 0, copy_w0, copy_h0)).copyTo(comb_image(cv::Rect(0, 0, copy_w0, copy_h0)));

	const int dst_x = curSize - copy_w0; // Right-aligned
	const int dst_y = 0;              // Top-aligned

	gray_image(cv::Rect(0, 0, copy_w0, copy_h0))
		.copyTo(comb_image(cv::Rect(dst_x, dst_y, copy_w0, copy_h0)));
	//BLK = 1;
	int border = BLK; // Number of pixels to trim
	cv::Mat COPYtarget_mat = comb_image(cv::Rect(border, border, comb_image.cols - 2 * border, comb_image.rows - 2 * border)).clone();
	cv::copyMakeBorder(COPYtarget_mat, grayImage, BLK, BLK, BLK, BLK, cv::BORDER_CONSTANT, cv::Scalar(255));
	//int big_width = target_mat.cols > target_mat.rows ? target_mat.cols : target_mat.rows;//Which is longer, width or height?
	//double ratio = ((double)M_MAT / (double)big_width);//Ratio
	//cv::resize(target_mat, grayImage, cv::Size(), ratio, ratio, cv::INTER_NEAREST);//Resize

	return  FName_END;
}

// Function to swap FFT quadrants (equivalent to Python's np.fft.fftshift)
void fftShift(Mat& magI) {
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-left
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-right

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
	// 1. Load images (grayscale, treated as CV_32F)
	//Mat imgRef = imread("reference.png", IMREAD_GRAYSCALE);
	//Mat imgTest = imread("test.png", IMREAD_GRAYSCALE);

	if (imgRef.empty() || imgTest.empty()) {
		cout << "The image could not be loaded." << endl;
		return -1;
	}


	// Prepare HVS weights
	cv::Mat weights = generateHVSWeight(cv::Size(8, 8));
	// Calculate PSNR-HVS-M
	double psnrHVSM;//
	psnrHVSM = calculatePSNRHVSM(imgRef, imgTest, weights);

	Mat ref32, test32;
	imgRef.convertTo(ref32, CV_32F);
	imgTest.convertTo(test32, CV_32F);

	// 2. Compute the difference image
	Mat diff = test32 - ref32;

	// 3. Apply a window function (Hanning Window)
	// Suppresses cross-shaped noise caused by discontinuities at the image edges
	Mat hann;
	createHanningWindow(hann, diff.size(), CV_32F);
	Mat diffWindowed = diff.mul(hann);// When applying the window function

	// 4. Zero padding (extend to the optimal DFT size)
	// To increase frequency resolution, it's also possible to intentionally set a larger size (e.g. a power of 2)
	int m = 2048;// getOptimalDFTSize(diff.rows);
	int n = 2048;// getOptimalDFTSize(diff.cols);
	Mat padded;
	copyMakeBorder(diffWindowed, padded, 0, m - diff.rows, 0, n - diff.cols, BORDER_CONSTANT, Scalar::all(0));

	// 5. Perform FFT (expand into the complex plane)
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);

	// 6. Compute the magnitude
	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magI = planes[0];

	// 7. Visualize using log scale
	magI += Scalar::all(1);
	log(magI, magI);

	// 8. Swap quadrants
	fftShift(magI);

	// 9. Normalize and display (scale to 0.0-1.0)
	normalize(magI, magI, 0, 1, NORM_MINMAX);
#if  DISP_CUT == 0
	normalize(diffWindowed, diffWindowed, 0, 1, NORM_MINMAX);
	//imshow("Input: Difference (Windowed)", diffWindowed);// Scaled for display
	//imshow("Spectrum: Error Analysis", magI);
#endif //  DISP_CUT == 0

	// Write PSNR-HVS-M onto magI (convert to 8-bit / BGR for drawing text)
	cv::Mat magI_u8;
	magI.convertTo(magI_u8, CV_8U, 255.0);

	// magI_u8: gradations 170 and below become black (0), 171 and above pass through unchanged
	cv::Mat magI_u8_masked;
	cv::threshold(magI_u8, magI_u8_masked, 170, 255, cv::THRESH_TOZERO);

	// --- Filter complexI with a frequency mask, perform inverse FFT, and convert to an image ---
// Prerequisites:
//  - complexI : complex spectrum after dft (CV_32FC2), size = padded.size()
//  - magI_u8_masked : mask (CV_8U) for the display image after fftShift(magI), size = padded.size()
// Purpose:
//  - Obtain the spatial image from an inverse FFT that keeps only the frequencies within the mask region

// 1) Convert magI_u8_masked -> mask (0/1 float)
	cv::Mat freqMask_u8;
	cv::compare(magI_u8_masked, 0, freqMask_u8, cv::CMP_GT); // 255 or 0

	cv::Mat freqMaskShifted_f;
	freqMask_u8.convertTo(freqMaskShifted_f, CV_32F, 1.0 / 255.0); // 1.0 or 0.0

	// 2) Restore the fftShift-ed mask back to the "original DFT layout" (equivalent to ifftshift)
	cv::Mat freqMask_f = freqMaskShifted_f.clone();
	fftShift(freqMask_f); // fftShift is self-inverse, so calling it again restores the original

	// 3) Apply the mask to the complex spectrum (multiply into both channels)
	std::vector<cv::Mat> cplxPlanes(2);
	cv::split(complexI, cplxPlanes);
	cplxPlanes[0] = cplxPlanes[0].mul(freqMask_f);
	cplxPlanes[1] = cplxPlanes[1].mul(freqMask_f);

	cv::Mat complexMasked;
	cv::merge(cplxPlanes, complexMasked);

	// 4) Inverse DFT (to the spatial domain)
	cv::Mat inv;
	cv::idft(complexMasked, inv, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

	// 5) Crop padded back to the original diff size
	cv::Mat invCropped = inv(cv::Rect(0, 0, diff.cols, diff.rows)).clone();

	// 6) Normalize for display and convert to 8-bit (if needed)
	cv::Mat invVis;
	cv::normalize(invCropped, invVis, 0, 255, cv::NORM_MINMAX);
	invVis.convertTo(invVis, CV_8U);

	cv::resize(invVis, invVis, cv::Size(256, 256), 0.0, 0.0, cv::INTER_NEAREST);

	//cv::imshow("iFFT (Masked Spectrum)", invVis);
	string Wstr = std::to_string(NsLoop + 1);
	string FName_0 = FILE_DTOP + "output\\" + Wstr + "_iFFT (Masked Spectrum).bmp";
	cv::imwrite(FName_0, invVis);

	// Resize magI_u8_masked to 256x256
	cv::Mat magI_u8_256;
	cv::resize(magI_u8_masked, magI_u8_256, cv::Size(256, 256), 0.0, 0.0, cv::INTER_NEAREST);

	//imshow("MaskDiff", magI_u8_256);//For display
	string Wstri = std::to_string(NsLoop + 1);
	string FName_1 = FILE_DTOP + "output\\" + Wstri + "_MASK (Masked Spectrum).bmp";
	cv::imwrite(FName_1, magI_u8_256);

	string Ws = std::to_string(NsLoop + 1);
	string FName = FILE_DTOP + "output\\" + Ws + "(Original) Difference Image FFT.bmp";
	cv::imwrite(FName, magI_u8);

	// Overlay imgRef onto the top-left corner of magI_u8 (clip so the size doesn't overflow)
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

	// Background (for visibility)
	const int x = 10;
	const int y = n - 30;
	const int p = 4;
	cv::putText(magI_bgr, psnrText.str(), cv::Point(x, y),
		cv::FONT_HERSHEY_SIMPLEX, p, cv::Scalar(0, 255, 255), 11, cv::LINE_AA);
	// Foreground
	cv::putText(magI_bgr, psnrText.str(), cv::Point(x, y),
		cv::FONT_HERSHEY_SIMPLEX, p, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);

	// From here, use the BGR side for display
	//cv::imshow("Spectrum: Error Analysis", magI_bgr);

	string Wst = std::to_string(NsLoop + 1);
	string FName_ = FILE_DTOP + "output\\" + Wst + "_Difference Image FFT.bmp";
	cv::imwrite(FName_, magI_bgr);

	return 0;
}

#define NloopMIT 4000//Number of MIT processing loop iterations

#define Image0to7 8 //Additional loop count for reference images

static double Hdata[7][NloopMIT + Image0to7 + 1];//Initialization (for storing history data)
struct EvalResults { double psnr=0.0, ssim=0.0, de=0.0; Mat reconImg; };

// ============================================================
// (A) Revised evaluateHologram
//     - Metric computation only (aggregation block removed)
//     - Hdata write -> critical(eval)
//     - Console output -> critical(console)
// ============================================================
EvalResults evaluateHologram(
	Mat& COPYtarget_mat_M_256,
	Mat& COPYtarget_mat_S_256,
	int NsLoop,
	double Mde_physics,
	string FILE_DTOP,
	int WKKK,
	string inputCom)
{
	EvalResults res;

	// Weight matrices are computed thread-locally (no sharing)
	cv::Mat weights = generateHVSWeight(cv::Size(8, 8));

	// ===== Compute each metric thread-locally (no critical section needed) =====
	double w = calcPSNR(COPYtarget_mat_M_256, COPYtarget_mat_S_256);

	double psnrHVS = calculatePSNRHVS(COPYtarget_mat_M_256, COPYtarget_mat_S_256, weights);
	double w_hvs = std::isinf(psnrHVS) ? 0.0 : psnrHVS;

	double psnrHVSM = calculatePSNRHVSM(COPYtarget_mat_M_256, COPYtarget_mat_S_256, weights);
	double w_hvsm = std::isinf(psnrHVSM) ? 0.0 : psnrHVSM;

	double ssim_standard = getStandardSSIM(COPYtarget_mat_M_256, COPYtarget_mat_S_256);
	cv::Scalar ssim = getSSIM(COPYtarget_mat_M_256 / 255.0, COPYtarget_mat_S_256 / 255.0);

	// Correct DE to 100% when MSE=0
	double Mde_out = (w_hvsm == 0.0) ? 100.0 : Mde_physics;

	// ===== Write to Hdata: protected by critical(eval) =====
#pragma omp critical(eval)
	{
		Hdata[0][NsLoop + 1] = w;
		Hdata[1][NsLoop + 1] = w_hvs;
		Hdata[2][NsLoop + 1] = w_hvsm;
		Hdata[3][NsLoop + 1] = ssim[0];
		Hdata[6][NsLoop + 1] = ssim_standard;
		Hdata[4][NsLoop + 1] = Mde_out;
		Hdata[5][NsLoop + 1] = Mde_out;
	}

	// ===== Console output: protected by critical(console) =====
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
		std::cout << " DE " << Mde_out << "%" << std::endl;
		SetConsoleColor(FOREGROUND_RED | FOREGROUND_INTENSITY);
#endif
	}

	res.de = Mde_out;
	res.psnr = w;
	res.ssim = ssim[0];
	return res;
}


// ============================================================
// (B) Average aggregation of 8 photos
//     Call from the main thread after the NsLoop parallel loop ends
//     (previously the NsLoop==7 block)
//     - static variable -> changed to a local variable
//     - cin.get() -> removed (to prevent deadlock)
// ============================================================
void aggregatePhotoResults()
{
	// Removed static, made local variables (safe since this is a single call after thread completion)
	double avg[7] = {};
	int    count[7] = {};
	int    ZEROcount[7] = {};

	const int startIdx = 1;  // NsLoop+1: NsLoop=0 -> index 1
	const int endIdx = Image0to7;    // NsLoop+1: NsLoop=7 -> index 8

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

	// cin.get() removed (would cause a deadlock in a parallel environment)
}


// ============================================================
// (C) Average aggregation of MIT images + imwrite
//     Call from the main thread after the NsLoop parallel loop ends
//     (previously the NsLoop==NloopMIT-1+Image0to7 block)
//     - static variables -> changed to local variables / std::vector
// ============================================================
void aggregateMITResults(int WKKK, const string& FILE_DTOP, const string& inputCom)
{
	// NloopMIT and Image0to7 use the global macros directly
	const int startIdx = Image0to7 + 1;
	const int endIdx = NloopMIT + Image0to7;
	const int sampleCount = endIdx - startIdx + 1;
	// Removed static, made local variables
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
		<< " SSIM,DE:" << count[3] << "," << count[4] << " \n";

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
	//if (avg[4] == 100) std::cout << "   DE  100 % ";
	//else               std::cout << "   DE  " << avg[4] << " % ";
	std::cout << "   DE  " << avg[4] << " % ";
	std::cout << "Average" << std::endl;

	// --- Hdata[0] single histogram ---
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
int histogramMAIN(Mat& COPYtarget_mat_M_256, Mat& COPYtarget_mat_S_256, Mat& intensity_roi, Mat& img_Mst, Mat& img_Srv, int curSize, Mat& histImage)// Histogram calculation and display
{
	//// Histogram calculation and display
	// Histogram calculation for img_Mst
	cv::Mat Mst_hist;
	int Mst_histSize = 256;// Number of bins
	float Mst_range[] = { 0, 256 };// Range
	const float* Mst_histRange = { Mst_range };
	cv::calcHist(&COPYtarget_mat_M_256, 1, 0, cv::Mat(), Mst_hist, 1, &Mst_histSize, &Mst_histRange, true, false);
	// Histogram calculation for img_Srv
	cv::Mat Srv_hist;
	int Srv_histSize = 256;// Number of bins
	float Srv_range[] = { 0, 256 };// Range
	const float* Srv_histRange = { Srv_range };


	// M: reference, S: target
	double mstMin = 0.0, mstMax = 0.0;
	double srvMin = 0.0, srvMax = 0.0;

	cv::minMaxLoc(COPYtarget_mat_M_256, &mstMin, &mstMax);
	cv::minMaxLoc(intensity_roi, &srvMin, &srvMax);

	// Scaling to align the target-side max with the reference-side max
	if (srvMax > 0.0) {
		double scale = mstMax / srvMax;
		intensity_roi.convertTo(intensity_roi, intensity_roi.type(), scale);
		intensity_roi.convertTo(COPYtarget_mat_S_256, CV_8U);
	}
	cv::calcHist(&COPYtarget_mat_S_256, 1, 0, cv::Mat(), Srv_hist, 1, &Srv_histSize, &Srv_histRange, true, false);
	intensity_roi.convertTo(img_Srv, CV_8U);//Reconstructed-image drawing data

	int Deff_median_OUT = 0;
	int hist_w = 768;// Width of the histogram display image
	int total_height = 0;// Height of the histogram display image
	total_height = drawHistogram(Mst_hist, Srv_hist, histImage, Deff_median_OUT, img_Mst, img_Srv);// Draw the histogram
#if MODE_CUT == 0
#if DISP_CUT == 0
	cv::resizeWindow("Histogram", hist_w, total_height);// Resize the window
	cv::imshow("Histogram", histImage);// Show the histogram
	cv::waitKey(1);
#endif // DISP_CUT				
#endif // MODE_CUT
	return Deff_median_OUT;
}


Mat Img_IFFT(Mat & WkP, int curSize)
{
	Mat re = Mat_<double>(curSize, curSize);// Real part
	Mat im = Mat_<double>(curSize, curSize);// Imaginary part
	Mat complexImage = cv::Mat::zeros(cv::Size(curSize, curSize), CV_64FC2); // 2-channel double image
	Mat planes[] = { re.clone(), im.clone() };// Split into 2 channels for complex representation

/***************************************************************************************/
	// --- Convert from amplitude type to phase-only type ---
	// 1. Convert WkP (assumed to be scaled to 0.0-1.0) into a 0-2*pi phase (radians)
	Mat phi = WkP * 2.0 * CV_PI;
	// 2. Amplitude is all 1.0 (light is not absorbed, 100% transmitted/reflected)
	Mat mag = Mat::ones(curSize, curSize, CV_64F);
	// 3. Convert from polar coordinates (mag, phi) to Cartesian coordinates (real part planes[0], imaginary part planes[1])
	cv::polarToCart(mag, phi, planes[0], planes[1]);

	// Perform IFFT for display
	cv::merge(planes, 2, complexImage);
	cv::dft(complexImage, complexImage, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);// Perform IFFT with scaling
	cv::split(complexImage, planes);

	cv::Mat intensity_roi = Mat_<double>(curSize, curSize);
	intensity_roi = cv::Mat::zeros(intensity_roi.size(), intensity_roi.type());
	cv::magnitude(planes[0], planes[1], intensity_roi);// magnitude = sqrt(real^2 + imag^2)
	/*Since this is the square-root variable of sqrt(X^2+Y^2), it always takes a value greater than zero*/
	/***************************************************************************************/
	// Compute the clipping value
	cv::Mat CLIP_intensity_roi = Mat_<UCHAR>(curSize, curSize);// = intensity_roi.clone();
	intensity_roi.convertTo(CLIP_intensity_roi, CV_8U, 255);
	// Histogram calculation
	cv::Mat CLIP_hist;
	int CLIP_histSize = 256; // Number of bins
	float CLIP_range[] = { 0, 256 }; // Range
	const float* CLIP_histRange = { CLIP_range };
	cv::calcHist(&CLIP_intensity_roi, 1, 0, cv::Mat(), CLIP_hist, 1, &CLIP_histSize, &CLIP_histRange, true, false);

	// Manually compute the cumulative distribution (1D histogram version)
	cv::Mat cumulative_hist(CLIP_hist.rows, CLIP_hist.cols, CV_64F);

	double running_sum = 0.0;
	if (CLIP_hist.rows == 1) {
		// Case of 1 x N (row vector)
		for (int i = 0; i < CLIP_hist.cols; ++i) {
			double v = static_cast<double>(CLIP_hist.at<float>(0, i));
			running_sum += v;
			cumulative_hist.at<double>(0, i) = running_sum;
		}
	}
	else if (CLIP_hist.cols == 1) {
		// Case of N x 1 (column vector)
		for (int i = 0; i < CLIP_hist.rows; ++i) {
			double v = static_cast<double>(CLIP_hist.at<float>(i, 0));
			running_sum += v;
			cumulative_hist.at<double>(i, 0) = running_sum;
		}
	}
	else {
		// Unexpected shape (print a message for safety)
		std::cerr << "CLIP_hist must be 1D (1xN or Nx1)." << std::endl;
	}

	// Compute the top-10% threshold
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
	// --- Generate a mask from CLIP_intensity_roi(CV_8U) -> fill the masked range of intensity_roi(CV_64F) with the minimum value ---
	CV_Assert(CLIP_intensity_roi.type() == CV_8U);
	CV_Assert(intensity_roi.type() == CV_64F);
	CV_Assert(CLIP_intensity_roi.size() == intensity_roi.size());

	// 1) Create the mask (e.g., target pixels exceeding clip_value)
	cv::Mat mask; // CV_8U(0/255)
	cv::compare(CLIP_intensity_roi, clip_value, mask, cv::CMP_GT);

	// 2) Get the minimum value within the masked range (intensity_roi remains CV_64F)
	double minValMasked = 0.0;
	double maxValMasked = 0.0;
	cv::minMaxLoc(intensity_roi, &minValMasked, &maxValMasked, nullptr, nullptr, mask);

	// 3) Fill the masked range with the minimum value (still CV_64F)
	if (cv::countNonZero(mask) > 0) {
		cv::Mat fillMat(intensity_roi.size(), CV_64F, cv::Scalar(minValMasked));
		fillMat.copyTo(intensity_roi, mask);
	}

	// For compatibility (equivalent to the original used_threshold)
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
	cv::waitKey(1);// After a wait time of 1.0 msec, the interference fringe display appears.
	cv::imshow("Master Image", COPYtarget_mat_M_256);
	cv::imshow("Target Image", COPYtarget_mat_S_256);//Display the target
	cv::waitKey(1);// After a wait time of 1.0 msec, the interference fringe display appears.
	cv::imshow("CGH", img_CGH);
	cv::waitKey(1);// After a wait time of 1.0 msec, the interference fringe display appears.
#endif //DISP_CUT
#if MODE_CUT == 0
#if DISP_CUT == 0
	cv::normalize(img_all, img_all, 0, 255, cv::NORM_MINMAX);
	cv::imshow("Overall Output Image", img_all);
	cv::waitKey(1);// After a wait time of 1.0 msec, the interference fringe display appears.
	//cv::imwrite(FName_all, img_all);
	//std::cout << "Enter any key to continue...img_ALL End" << std::endl;
	//std::cin.get(); // Wait for key input
#endif // DISP_CUT
				// Compute the difference image between COPYtarget_mat_M_256,BLK and COPYtarget_mat_S_256,BLK
	cv::Mat diff_img;
	COPYtarget_mat_M_256.convertTo(WkT, CV_64F);
	COPYtarget_mat_S_256.convertTo(WkS, CV_64F);
	cv::absdiff(WkT, WkS, diff_img);

	// Normalize the difference image for visualization
	cv::Mat diff_img_norm;
	cv::normalize(diff_img, diff_img_norm, 0, 255, cv::NORM_MINMAX);
	diff_img_norm.convertTo(diff_img_norm, CV_8U);

	// Display the difference image
#if DISP_CUT == 0
	cv::imshow("Difference Image", diff_img_norm);
	cv::waitKey(1); // Wait for display
#endif // DISP_CUT				
	if (NsLoop < Image0to7)// Perform the difference calculation for loop iterations 0-7
	{
		Diffmain(COPYtarget_mat_M_256, COPYtarget_mat_S_256, FILE_DTOP, NsLoop);//Display the difference calculation
	}

#endif // MODE_CUT

#if MODE_CUT == 0
	if (inputCom.find("save") != std::string::npos) {// Executed only when "save" is included
		if (NsLoop < Image0to7)// Save during loop iterations 0-7
		{
			// Perform the save process
			//cv::normalize(img_all, img_all, 0, 255, cv::NORM_MINMAX);
			cv::imwrite(FName_all, img_all);
			cv::imwrite(FName_Src, COPYtarget_mat_S_256);
			cv::imwrite(FName_Mst, COPYtarget_mat_M_256);
			std::cout << "The image has been exported to a file." << FName_Srv << endl;
			cv::imwrite(FName_cgh, img_CGH);
			cv::imwrite(FName_his, histImage);


			// Output to a CSV file
			/*Do not remove the subsequent processing*/
			img_Srv.convertTo(WkP, CV_64F);
			string filenames = FILE_DTOP + "output\\" + std::string("Srv.csv");
			writeMatToCSV(WkP, filenames);
			//cout << "Exported to CSV file " << filenames << endl;

			// Output to a CSV file
			/*Do not remove the subsequent processing*/
			img_Mst.convertTo(WkP, CV_64F);
			filenames = FILE_DTOP + "output\\" + std::string("Mst.csv");
			writeMatToCSV(WkP, filenames);
			//cout << "Exported to CSV file " << filenames << endl;
		}
	}
	if (inputCom.find("step") != std::string::npos) // When "step" is included
	{
		std::cout << "    [step: The stop operation is skipped due to parallelization.]\n";
		//std::cout << "    Enter any other key to continue...Stop >> ";// << std::endl;
		//std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Clear the buffer
		//std::cin.get(); // Wait for any key input
		//std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Clear the buffer
	}
#endif // MODE_CUT
	return;
}
/**
 * @brief Calculates the diffraction efficiency (DE) of a phase-only hologram using a signal ROI mask.
 * @param inputObjectImage The reference input image (CV_64F) used to create the ROI mask.
 * @param holoForDE The quantized phase hologram pattern (CV_64F, 0.0-1.0).
 * @param curSize Matrix dimension (e.g., 256 or 384).
 * @return Diffraction efficiency (DE) in percentage (%).
 */
double calculateDiffractionEfficiencyWithMask(
	const cv::Mat& inputObjectImage,
	const cv::Mat& holoForDE,
	int curSize)
{
	cv::Mat holoPlanesDE[2];
	cv::Mat holoComplexForDE;
	cv::Mat reconstructedComplexForDE;
	cv::Mat mask_8u_de;
	cv::Mat mask_64F_de;
	double total_intensity_reconstructed_D = 0.0;
	double phaseModulationDepth = 2.0 * CV_PI;
	// --- Denominator calculation ---
	double total_input_energy = 0.0;

	// --- 2. Create the mask image ---
	// 1. Create a mask image (non-zero becomes 255, zero stays 0)
	mask_8u_de = (inputObjectImage != 0.0);
	int borderSize = 1; // Mask border size
	mask_8u_de = mask_8u_de(cv::Rect(borderSize, borderSize, mask_8u_de.cols - 2 * borderSize, mask_8u_de.rows - 2 * borderSize)).clone();
	cv::copyMakeBorder(mask_8u_de, mask_8u_de, borderSize, borderSize, borderSize, borderSize, cv::BORDER_CONSTANT, cv::Scalar(255));
	// 2. Apply the mask to the target image
	mask_8u_de.convertTo(mask_64F_de, CV_64F, 1.0 / 255.0);

#if MODE_CUT == 0
#if DISP_CUT == 0
#pragma omp critical(gui)
	{
		cv::Mat img_ROI_de;
		mask_8u_de.convertTo(img_ROI_de, CV_8U, 255);
		cv::imshow("MASK", img_ROI_de);
	}
#endif // DISP_CUT
#endif // MODE_CUT

	// 1. Map the interference fringe pattern WkP to the phase distribution phi(u,v)
	// Adjust the scale to match the device's maximum phase modulation depth (e.g. 2.19*pi)
	cv::Mat phiDE = holoForDE.clone();
	cv::normalize(phiDE, phiDE, 0, phaseModulationDepth, cv::NORM_MINMAX);

	cv::Mat magDE = cv::Mat::ones(holoForDE.size(), CV_64F);// Amplitude reflectance a = 1.0
	cv::Mat reDE, imDE;// Real and imaginary parts
	// Compute the complex exponential exp(+j * phi)
	cv::polarToCart(magDE, phiDE, reDE, imDE);// Convert from polar to Cartesian coordinates
// Compute the complex exponential exp(-j * phi)
/*etc
for (int y = 0; y < curSize; y++) {
	for (int x = 0; x < curSize; x++) {
		double p = phiDE.at<double>(y, x);
		holoPlanesDE[0].at<double>(y, x) = cos(p);  // Real part
		holoPlanesDE[1].at<double>(y, x) = -sin(p); // Imaginary part (matched to the conjugate image direction)
	}
}
*/
	holoPlanesDE[0] = reDE;// Real part
	holoPlanesDE[1] = imDE;// Imaginary part
	cv::merge(holoPlanesDE, 2, holoComplexForDE);

	// 3. Perform physical IFFT (apply scale to preserve energy)
	cv::dft(holoComplexForDE, reconstructedComplexForDE, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);

	// 4. Compute the diffraction efficiency (DE) (using raw data before brightness correction)
	// --- Below: strict diffraction efficiency computation based on the physical model (Parseval's theorem) ---
	// 4-1. Compute intensity (squared amplitude) for each pixel

	// --- Numerator: compute the total intensity of the "target region D" of the reconstructed image ---
	Mat re = Mat_<double>(curSize, curSize);// Real part
	Mat im = Mat_<double>(curSize, curSize);// Imaginary part
	Mat DEchannels[] = { re.clone(), im.clone() };// Split into 2 channels for complex representation

	cv::split(reconstructedComplexForDE, DEchannels);
	cv::Mat total_intensity_mat;
	cv::magnitude(DEchannels[0], DEchannels[1], total_intensity_mat);
	cv::multiply(total_intensity_mat, total_intensity_mat, total_intensity_mat);
	// 4-2. Numerator: total energy within the signal region (ROI)
	cv::Mat masked_intensity;
	cv::multiply(total_intensity_mat, mask_64F_de, masked_intensity);
	// 4-3. Denominator: total energy of the entire reconstructed image (equal to the total SLM reflected light by Parseval's theorem)
	total_intensity_reconstructed_D = cv::sum(masked_intensity)[0];

	// Fix: total energy of the incident light (assumed plane wave)
	// Assume light with intensity 1.0 is incident over an area of curSize * curSize
	//reference_energy = static_cast<double>(inputObjectImage.rows * inputObjectImage.cols);
	total_input_energy = 1.0;// reference_energy;
	// 4-4. Compute the ratio (mathematically never exceeds 100%)
	return (total_intensity_reconstructed_D / total_input_energy) * 100.0;
}

double REminVal, REmaxVal;
double IMminVal, IMmaxVal;
std::string inputCom;

int main(int argc, char* argv[])
{
	int WKKK = 0;

	std::string folderPath = getDesktopWKCGHFolder();
	// folderPath + "\\filename" can be used for saving files, etc.

	// Written near the beginning of the main function
	_putenv_s("OPENCV_IO_ENABLE_OPENEXR", "1");

	std::string FILE_DTOP;
	std::string computerName = getComputerName();
		FILE_DTOP = folderPath;
		std::cout << "This computer is " << computerName << "(" << FILE_DTOP << ")" << std::endl;

#if DISP_CUT == 0
	namedWindow("Input Image", WINDOW_AUTOSIZE);// Create the window
	cv::moveWindow("Input Image", 10, 100);// Specify the window position (x-coordinate, y-coordinate)
	cv::resizeWindow("Input Image", 260, 260);// Specify the window size (width, height)
	namedWindow("CGH", WINDOW_AUTOSIZE);// Create the window
	cv::moveWindow("CGH", 10 + 300 * 1, 100);// Specify the window position (x-coordinate, y-coordinate)
	cv::resizeWindow("CGH", 260, 260);// Specify the window size (width, height)

	namedWindow("Master Image", WINDOW_AUTOSIZE);// Create the window
	cv::moveWindow("Master Image", 10, 150 + 270 + 200);// Specify the window position (x-coordinate, y-coordinate)
	cv::resizeWindow("Master Image", 260, 130);// Specify the window size (width, height)
	namedWindow("Target Image", WINDOW_AUTOSIZE);// Create the window
	cv::moveWindow("Target Image", 10, 150 + 270);// Specify the window position (x-coordinate, y-coordinate)
	cv::resizeWindow("Target Image", 260, 130);// Specify the window size (width, height)
#endif //DISP_CUT
#if MODE_CUT == 0
#if DISP_CUT == 0
	namedWindow("Overall Output Image", WINDOW_AUTOSIZE);// Create the window
	cv::moveWindow("Overall Output Image", 10 + 300, 150 + 270);// Specify the window position (x-coordinate, y-coordinate)
	cv::resizeWindow("Overall Output Image", 260, 260);// Specify the window size (width, height)
	namedWindow("Histogram", WINDOW_AUTOSIZE);
	cv::moveWindow("Histogram", 10 + 300 * 2, 100);// 150 + 270);
	namedWindow("Difference Image", WINDOW_AUTOSIZE);
	cv::moveWindow("Difference Image", 10, 150 + 270 + 400);
	namedWindow("MASK", WINDOW_AUTOSIZE);
	cv::moveWindow("MASK", 10 + 300, 150 + 270 + 400);
#endif //DISP_CUT
#endif //MODE_CUT

	DWORD elapsedall = 0;

	srand((unsigned)time(NULL)); /*Seed the random number generator*/


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
		////	FName_his;//File name for image saving
		// Save current state
		std::ios::fmtflags curret_flag = std::cout.flags();

		//Pad numbers like 123 with 5 leading zeros to make 8 digits
		std::ostringstream Wss;
		Wss << std::setw(3) << std::setfill('0') << WKKK;// << "\n";
		std::string Ws(Wss.str());
		std::cout << Ws;

		WKKK++;
		//std::cout << "WKKK=" << WKKK << "  \\ \n";
	Retry:

		std::cout << "Enter save(save images),step(pause each time),run(execute: normalize, round, median correction, GS method: with dummy region)>> ";// << std::endl;
		std::cin >> inputCom; // Get the input
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Clear the buffer

		if (inputCom.find("run") == std::string::npos) goto Retry;// Re-enter input if 'run' is not included

		DWORD start = timeGetTime();

		////const std::string csvPath = FILE_DTOP  +"output\\"+ "alpha_sweep.csv";
		////std::ofstream csv(csvPath, std::ios::out | std::ios::trunc);
		////if (!csv.is_open()) {
		////	std::cerr << "[CSV] open failed: " << csvPath << std::endl;
		////	// Abort the sweep here (or return / continue)
		////	continue;
		////}


// Moved outside the loop: static int -> std::atomic<int>
		static std::atomic<int> GS_logCounter(0);

		// Parallelize the NsLoop
#pragma omp parallel for schedule(dynamic)
		for (int NsLoop = 0; NsLoop <= NloopMIT - 1 + Image0to7; NsLoop++)
		{
			int BLK = 1;
			int curSize = M_MAT;// Fixed at 256x256
			if (NsLoop >= Image0to7) curSize = 384;// Fixed at 384x384

			// Define all FName variables as local variables within the loop (to eliminate thread contention)
			std::string FName, FName_Mst, FName_all,
				FName_Srv_, FName_Srv, FName_Src,
				FName_cgh, FName_his, FName_END;

			// The following Mat variables remain as in the original code (already local variables)
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

			if (NsLoop < Image0to7)// File names for image output during loop iterations 0-7
			{
				str = std::to_string(NsLoop);
				// Assign the FName variables to local variables
				FName = FILE_DTOP + "output\\" + Ws + "_photo " + str + "_" + inputCom + ".bmp";
				FName_Mst = FILE_DTOP + "output\\" + Ws + "_Mst_photo " + str + "_" + inputCom + ".bmp";
				FName_all = FILE_DTOP + "output\\" + Ws + "_ALL_photo " + str + "_" + inputCom + ".bmp";
				FName_Srv_ = FILE_DTOP + "output\\" + Ws + "_Srv_photo " + str + "_" + inputCom + "_";
				FName_Src = FILE_DTOP + "output\\" + Ws + "_Src_photo " + str + "_" + inputCom + ".bmp";
				FName_cgh = FILE_DTOP + "output\\" + Ws + "_CGH_photo " + str + "_" + inputCom + ".bmp";
				FName_his = FILE_DTOP + "output\\" + Ws + "_His_photo " + str + "_" + inputCom + ".bmp";
				// SetConsoleColor/cout are protected with critical
#pragma omp critical(console)
				{
					SetConsoleColor(FOREGROUND_GREEN | FOREGROUND_INTENSITY);
					std::cout << FName;
					SetConsoleColor(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
				}
			}
			if (NsLoop < Image0to7) {// File names for image output during loop iterations 0-7
				FName_END = Img_Read(FILE_DTOP, NsLoop, grayImage, target_mat, curSize, BLK);
				FName_Srv = FName_Srv_ + FileNameBMP[NsLoop];
#pragma omp critical(console)
				{
					{ cout << "\nFName_END=" << FName_END << std::endl; }
				}
			}
			if (NsLoop >= Image0to7)// File names for image output for loop iterations 8 and beyond
			{
				curSize = 384;
				std::ostringstream Wss;
				Wss << std::setw(4) << std::setfill('0') << (NsLoop - Image0to7);
				std::string Ws_loc(Wss.str()); // Avoid name collision with the outer Ws
#if MODE_CUT == 0
				// [Step4]
#pragma omp critical(console)
				{ std::cout << Ws_loc; }
#endif
				FName_END = MITImg_Read(FILE_DTOP, Ws_loc, grayImage, curSize, 1);
			}

#if DISP_CUT == 0
			// imshow/waitKey are protected with critical
#pragma omp critical(gui)
			{
				cv::waitKey(1);
				cv::imshow("Input Image", grayImage);
			}
#endif

			grayImage.convertTo(inputImage, CV_64F, 1.0 / 255.0);// Convert to CV_64F and scale to 0.0-1.0
			WkP = cv::Mat::zeros(cv::Size(curSize, curSize), CV_64F);// Initialize WkP to zeros
			re = cv::Mat::zeros(cv::Size(curSize, curSize), CV_64F);// Real part
			im = cv::Mat::zeros(cv::Size(curSize, curSize), CV_64F);// Imaginary part

			Img_FFT(NsLoop, inputImage, re, im, complexImage, planes, Trans, curSize);

#if MODE_CUT == 0
#if FILE_OUT == 1
			cv::magnitude(re, im, WkP);
			string filenames = FILE_DTOP + "output\\" + std::string("A_sinpuku.csv");
			// [Step4]
#pragma omp critical(console)
			{ cout << "Exported to CSV file " << filenames << endl; }
#endif
#endif

			// === GS method ===
			// Generates a phase-only hologram (POH) using the Gerchberg-Saxton iterative method.
			// The object plane (image) and hologram plane (frequency domain) are propagated back and forth via FFT/IFFT,
			// while imposing amplitude constraints on each plane to converge the phase distribution.
			//   - Object plane   : force the amplitude of the signal region to the target amplitude A_target
			//   - Hologram plane : force the amplitude to 1 (phase-only = POH constraint)
			// The lower part of the image is treated as a dummy region (for noise escape); by letting the amplitude vary freely there,
			// this scheme greatly improves the reconstruction quality (PSNR) of the signal region.
			int num_gs_iterations = 119;                    // Number of GS iterations (default: signal 50% / dummy 50%)
			cv::Mat A_target;
			cv::sqrt(inputImage, A_target);                 // Target amplitude = sqrt(intensity). Converts the intensity image to amplitude
			cv::Mat current_phase = Trans.clone();          // Initial phase: start from the random phase Trans
			cv::Mat current_mag = A_target.clone();         // Initial amplitude: the target amplitude itself
			cv::Mat gs_complex, gs_holo, gs_mag, gs_phase;  // Working variables (complex field, hologram, amplitude, phase)
			cv::Mat gs_planes[2];                           // Real/imaginary planes of the complex field
			int signalRows = curSize / 2;                   // Number of rows in the signal region (top half). The rest is the dummy region (signal 50% / dummy 50%) (best: average PSNR approx. 47.8dB)

			// Switch the signal/dummy ratio and number of iterations based on the command string.
			// DUMMYxx = signal region occupies xx% of the whole (the rest is the dummy region).
			// The number of iterations is set higher as the dummy region narrows, since convergence is slower (determined experimentally).
			if (inputCom.find("DUMMY75") != std::string::npos) {
				num_gs_iterations = 470;            // Signal 75% / dummy 25% (slow convergence -> 470 iterations)
				signalRows = curSize * 3 / 4;
			}
			else if (inputCom.find("DUMMY60") != std::string::npos) {
				num_gs_iterations = 251;            // Signal 60% / dummy 40%
				signalRows = curSize * 6 / 10;
			}

			// Log to confirm the GS settings (output only once every 50 calls to avoid excessive verbosity)
			// Retrieve the counter via atomic
			int myLogCount = GS_logCounter.fetch_add(1);
			const bool doLog = ((myLogCount % 50) == 0);
			if (doLog) {
				// [Step4]
#pragma omp critical(console)
				{
					std::cout << "[GS] num_gs_iterations = " << num_gs_iterations
						<< " signalRows=" << signalRows
						<< " dummyRows=" << (curSize - signalRows)
						<< " dummyRatio=" << (100.0 * (curSize - signalRows) / curSize) << "%"
						<< std::endl;
				}
			}

			// --- GS iteration loop ---
			for (int iter = 0; iter < num_gs_iterations; iter++) {
				// [Step1] Compose the complex field of the object plane (amplitude x exp(i*phase)) and propagate to the hologram plane via FFT
				cv::polarToCart(current_mag, current_phase, gs_planes[0], gs_planes[1]);
				cv::merge(gs_planes, 2, gs_complex);
				cv::dft(gs_complex, gs_holo, cv::DFT_COMPLEX_OUTPUT);
				// [Step2] Extract the phase gs_phase at the hologram plane
				cv::split(gs_holo, gs_planes);
				cv::cartToPolar(gs_planes[0], gs_planes[1], gs_mag, gs_phase);
				// [Step3] POH constraint: force the amplitude of the hologram plane to 1 (retain only the phase information)
				cv::Mat mag_ones = cv::Mat::ones(curSize, curSize, CV_64F);
				cv::polarToCart(mag_ones, gs_phase, gs_planes[0], gs_planes[1]);
				cv::merge(gs_planes, 2, gs_holo);
				// [Step4] Propagate back to the object plane via IDFT (normalized by 1/N^2 with DFT_SCALE), obtaining the reconstructed amplitude gs_mag and phase
				//         The phase current_phase is carried forward directly to the next iteration
				cv::dft(gs_holo, gs_complex, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);
				cv::split(gs_complex, gs_planes);
				cv::cartToPolar(gs_planes[0], gs_planes[1], gs_mag, current_phase);

				// [Step5] Compute the energy-matching scale.
				// Compare the energy (sum of amplitude^2) of the target amplitude and the reconstructed amplitude within the signal region,
				// and use scale = sqrt(energy_A/energy_gs) to match the amplitude level of the dummy region to that of the signal side.
				// Note: applying an extra multiplier (e.g., BOOST>1) to scale is strictly forbidden, as it causes divergence via positive feedback.
				double energy_A = 0.0, energy_gs = 0.0;
				// Removed #pragma omp parallel for reduction (to avoid double parallelism with the outer NsLoop)
				for (int y = 0; y < signalRows; y++) {
					const double* pA = A_target.ptr<double>(y);
					const double* pG = gs_mag.ptr<double>(y);
					for (int x = 0; x < curSize; x++) {
						energy_A += pA[x] * pA[x];
						energy_gs += pG[x] * pG[x];
					}
				}
				double scale = (energy_gs > 0.0) ? std::sqrt(energy_A / energy_gs) : 1.0;

				// [Step6] Amplitude constraint of the object plane (update current_mag for the next iteration)
				//   Signal region (y < signalRows): force to the target amplitude A_target (faithfully reproduce the image)
				//   Dummy region (y >= signalRows): allow the reconstructed amplitude as-is (only matched via scale)
				//                               -> "noise escape" that lets the error energy flow into this region
				// Removed #pragma omp parallel for (to avoid double parallelism with the outer NsLoop)
				for (int y = 0; y < curSize; y++) {
					for (int x = 0; x < curSize; x++) {
						if (y < signalRows)
							current_mag.at<double>(y, x) = A_target.at<double>(y, x);
						else
							current_mag.at<double>(y, x) = gs_mag.at<double>(y, x) * scale;
					}
				}
			}

			// [Step7] Wrap the final hologram phase gs_phase into [0, 2*pi),
			//         normalize it to [0,1], and store it in the phase hologram WkP (unchanged from the original code)
			for (int y = 0; y < curSize; y++) {
				for (int x = 0; x < curSize; x++) {
					double p = gs_phase.at<double>(y, x);
					p = std::fmod(p, 2.0 * CV_PI);          // Take the remainder modulo 2*pi
					if (p < 0.0) p += 2.0 * CV_PI;          // For negative values, add +2*pi to fit within [0, 2*pi)
					WkP.at<double>(y, x) = p / (2.0 * CV_PI); // Normalize to 0-1 (phase/2*pi)
				}
			}
			// === End of GS method ===
			// From here: quantize WkP to bitDepth to generate img_CGH (hologram for display/DE evaluation)

			double scalingFactor = 1.0;// Scaling factor for diffraction efficiency evaluation
			double bitDepth = 8.0;// Bit depth for quantization (8-bit or 16-bit)
			normalize(WkP, WkP, 0.0, 1.0, NORM_MINMAX);// Normalize to 0-1
			double numLevels = std::pow(2.0, bitDepth) - 1.0f;// Number of quantization levels (e.g., 255 for 8-bit, 65535 for 16-bit)
			WkP.convertTo(WkP, CV_64F, numLevels);// Scale to 0-255 (8-bit) or 0-65535 (16-bit)
			WkP += 0.5;// Add 0.5 for rounding
			if (bitDepth <= 8.0)// If bit depth is 8 or less, convert to CV_8U; otherwise, convert to CV_16U
				WkP.convertTo(img_CGH, CV_8U);
			else
				WkP.convertTo(img_CGH, CV_16U);
			scalingFactor = numLevels;// Update scaling factor

			cv::Mat holoForDE;
			img_CGH.convertTo(holoForDE, CV_64F);
			if (scalingFactor != 1.0 && scalingFactor != 0.0)
				holoForDE /= scalingFactor;// Scale back to 0-1 for diffraction efficiency evaluation
			// Strict diffraction efficiency based on the physical model (Parseval's theorem)
			double Mde_physics = calculateDiffractionEfficiencyWithMask(inputImage, holoForDE, curSize);
			
			cv::Mat intensity_roi = Mat_<double>(curSize, curSize);
			intensity_roi = Img_IFFT(holoForDE, curSize);
			cv::multiply(intensity_roi, intensity_roi, intensity_roi);
			cv::normalize(intensity_roi, intensity_roi, 0.0, 1.0, cv::NORM_MINMAX);
			intensity_roi.convertTo(img_all, CV_8U, 255);

			int border = BLK;// The border size for cropping the signal region (1 pixel)
			Mat COPYtarget_mat_M_BLK(cv::Size(curSize - BLK * 2, (curSize / 2) - BLK * 2), CV_8U);
			Mat COPYtarget_mat_M_256(cv::Size(curSize, (curSize / 2)), CV_8U);
			grayImage.rowRange(0, (curSize / 2)).copyTo(img_Mst);
			COPYtarget_mat_M_BLK = img_Mst(cv::Rect(border, border, img_Mst.cols - 2 * border, img_Mst.rows - 2 * border)).clone();
			cv::copyMakeBorder(COPYtarget_mat_M_BLK, COPYtarget_mat_M_256, BLK, BLK, BLK, BLK, cv::BORDER_CONSTANT, cv::Scalar(0));

			if (NsLoop >= Image0to7)
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

			// [Step4]
#pragma omp critical(console)
			{
				std::cout << "Deff_median(" << Deff_median_OUT << ") ";
			} //<< std::endl; }

			// evaluateHologram is called as-is (it uses critical internally)
			evaluateHologram(COPYtarget_mat_M_256, COPYtarget_mat_S_256,
				NsLoop, Mde_physics,FILE_DTOP, WKKK, inputCom);

			// Disp_Save_img (may include imshow) is protected with critical
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
		// ===== Aggregate on a single thread after the parallel loop ends =====

        // (B) Average aggregation of 8 photos (previously the NsLoop==7 block)
		aggregatePhotoResults();// Aggregate the results of 8 photos

		// (C) Average aggregation of MIT images + imwrite (previously the trailing block)
		aggregateMITResults(WKKK, FILE_DTOP, inputCom);// Aggregate the results of MIT images

		DWORD elapsed = (timeGetTime() - start);
		std::cout << "t=" << elapsed << "ms.(" << (elapsed / 1000.0) << "sec. " << (elapsed / 60000.0) << "min.) / " << (NloopMIT + Image0to7) << " = " << (elapsed / 1000.0) / (NloopMIT + Image0to7) << "sec[" << 1.0 / ((elapsed / 1000.0) / (NloopMIT + Image0to7)) << "fps]\n";
		////csv.close();
		std::cout << "Enter any key to exit..." << std::endl;
		std::cin.get(); // Wait for key input

		// Destroy the windows
		// Release COM
		CoUninitialize();
		return -1;
	}
}