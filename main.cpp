#include <opencv2/opencv.hpp>
#include <opencv2/shape/hist_cost.hpp>
#include <opencv2/shape/shape_distance.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat in_X = imread("X.png");
    Mat img_X;
    cvtColor(in_X, img_X, COLOR_BGR2GRAY);

    Mat in_a = imread("a.png");
    Mat img_a;
    cvtColor(in_a, img_a, COLOR_BGR2GRAY);

    vector<vector<Point> > con1;
    vector<Vec4i> hierarchy1;
    findContours(img_X, con1, hierarchy1, RETR_CCOMP, CHAIN_APPROX_TC89_KCOS);

    vector<vector<Point> > con2;
    vector<Vec4i> hierarchy2;
    findContours(img_a, con2, hierarchy2, RETR_CCOMP, CHAIN_APPROX_TC89_KCOS);


    auto hd = createHausdorffDistanceExtractor();
    auto sd = createShapeContextDistanceExtractor();

    cout << hd->computeDistance(con1[0], con2[0]) << endl;
    cout << sd->computeDistance(con1[0], con2[0]) << endl;

    cout << "working" << endl;
    return 0;
}

