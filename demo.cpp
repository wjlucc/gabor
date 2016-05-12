#include "cvgabor.h"

int main()
{
	double Sigma = 2 * PI;
	double F = sqrt(2.0);
	CvGabor gabor(PI / 4, 3, Sigma, F);

	//获得实部并显示它
	Mat kernel(gabor.get_mask_width(), gabor.get_mask_width(), CV_8UC1);
	gabor.get_image(CV_GABOR_REAL, kernel);
	imshow("Kernel", kernel);
	cout << kernel.rows << endl;
	cout << kernel.cols << endl;
	//cvWaitKey(0);
	Mat img = imread("D:\\data\\fruit\\other\\9_2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Original Image", img);
	//cvWaitKey(0);

	//获取载入图像的gabor滤波响应的实部并且显示
	Mat reimg(img.rows, img.cols, CV_32FC1);
	gabor.conv_img(img, reimg, CV_GABOR_REAL);
	imshow("After Image", reimg);
	cvWaitKey(0);

}
