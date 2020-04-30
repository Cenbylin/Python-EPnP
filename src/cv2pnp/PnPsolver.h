
#ifndef PNPSOLVER_H
#define PNPSOLVER_H

#include <opencv2/core.hpp>
#include <vector>

using namespace std;

/*
PnPsolver 仅用于Tracking::Relocalization 函数中
原本计划迁移本类中所有 opencv 1.0 的函数到 opencv 3.0
但是因为 cv::SVDecomp 与 cvSVD 行为有些许不同(排除法，还没有看opencv源代码差别)，迁移过去后重定位失败
然后观察到 PnPsolver 与 cv::solvePnPRansac 有许多相似之处，然后尝试修改，没想到居然可行
原始代码看起来对相机位姿的重复多次优化，可以得到更好的位姿，此修改，做了类似的迭代求解，目前测试正常。
*/

class PnPsolver {
public:
    PnPsolver(const Frame &F, const vector<MapPoint *> &vpMapPointMatches);

    ~PnPsolver();

    cv::Mat iterate(int iterationsCount, bool &bNoMore, vector<bool> &vbInliers, int &nInliers);

private:
    // cam
    double fx, fy, cx, cy;

    // 2D Points
    vector<cv::Point2f> mvP2D;

    // 3D Points
    vector<cv::Point3f> mvP3Dw;

    // 当前已迭代次数
    int mnCurrentIterationsCount;
    // 旋转向量
    cv::Mat mrvec;
    // 位置向量
    cv::Mat mtvec;

    // 最大迭代次数
    int mMaxIterationsCount = 300;
    // 重投影误差，重投影误差小于该值的点对的编号会加入inliers
    // 该值太小，会引起重跟踪变得困难，太大，误差也会变大
    float mReprojectionError = 3.f;
    float mConfidence = 0.99;
    // 至少有这个数量的点对的重投影误差小于 reprojectionError，否则返回空矩阵
    int mMinInliers = 50;

};

#endif //PNPSOLVER_H