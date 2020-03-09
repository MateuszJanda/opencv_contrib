#include <opencv2/opencv.hpp>
#include <opencv2/shape/hist_cost.hpp>
#include <opencv2/shape/shape_distance.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    Mat img_X = imread("X.png", COLOR_BGR2GRAY);
    Mat img_a = imread("a.png", COLOR_BGR2GRAY);
    Mat img_M = imread("M.png", COLOR_BGR2GRAY);

    vector<vector<Point> > con_X;
    vector<Vec4i> hierarchy1;
    findContours(img_X, con_X, hierarchy1, RETR_CCOMP, CHAIN_APPROX_TC89_KCOS);

    vector<vector<Point> > con_M;
    vector<Vec4i> hierarchy2;
    findContours(img_M, con_M, hierarchy2, RETR_CCOMP, CHAIN_APPROX_TC89_KCOS);

    vector<vector<Point> > con_a;
    vector<Vec4i> hierarchy3;
    findContours(img_a, con_a, hierarchy3, RETR_CCOMP, CHAIN_APPROX_TC89_KCOS);

    auto hd = createHausdorffDistanceExtractor();
    auto sd = createShapeContextDistanceExtractor();

    cout << "X <-> M" << endl;
    cout << hd->computeDistance(con_X[0], con_M[0]) << endl;
    cout << sd->computeDistance(con_X[0], con_M[0]) << endl;

    cout << "X <-> a" << endl;
    cout << hd->computeDistance(con_X[0], con_a[0]) << endl;
    cout << sd->computeDistance(con_X[0], con_a[0]) << endl;

    return 0;
}

