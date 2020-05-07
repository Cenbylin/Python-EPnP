#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include <cmath>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include "DUtils/Random.h"
#include "CV2PnPsolver.h"
CV2PnPsolver::CV2PnPsolver(vector<float> vLevelSigma2, float mfx, float mfy, float mcx, float mcy,
                     vector<cv::KeyPoint> &vpKp, vector<tuple<int, float, float, float>> mtPointMatches) :
        mnCurrentIterationsCount(0) {
    // 预先分配空间
    mvP2D.reserve(vpKp.size());
    mvP3Dw.reserve(vpKp.size());

    // vector[tuple(idx,x,y,z)]
    for (auto &tKpMp:mtPointMatches) {
        int i = get<0>(tKpMp);
        mvP3Dw.push_back(cv::Point3f(get<1>(tKpMp), get<2>(tKpMp), get<3>(tKpMp)));
        const cv::KeyPoint *kp = &vpKp[i];//得到2维特征点, 将KeyPoint类型变为Point2f
//        mvP2D 和 mvP3Dw 是一一对应的匹配点
        mvP2D.push_back(kp->pt);//存放到mvP2D容器
    }


    // Set camera calibration parameters
    fx = mfx;
    fy = mfy;
    cx = mcx;
    cy = mcy;

    mMaxIterationsCount = 300;
    mReprojectionError = 3.f;
    mConfidence = 0.99;
    mMinInliers = 50;
}

CV2PnPsolver::~CV2PnPsolver() {
}

tuple<cv::Mat, bool, vector<bool>, int> CV2PnPsolver::iterate(int iterationsCount) {
    // 这三个是要返回的
    bool bNoMore = false;
    vector<bool> vbInliers;
    int nInliers = 0;

    if (iterationsCount < 1) {
        cerr << "Error：iterationsCount must be greater than 0, but now is " << iterationsCount
             << "; location PnPsolver::solve" << endl;
        iterationsCount = 1;
    }
    if (mnCurrentIterationsCount >= mMaxIterationsCount) {
        bNoMore = true;
        return make_tuple(cv::Mat(), bNoMore, vbInliers, nInliers);
    }

    float cammat[9] = {fx, 0, cx, 0, fy, cy, 0, 0, 0};
    cv::Mat camMat(3, 3, CV_32F, (void *) cammat);
    cv::Mat M = cv::Mat::eye(4, 4, CV_32F);
    cv::Mat rmat;
    cv::Mat tmat;
    cv::Mat inliers;// 误差小于指定值的点对编号

    //cv::solvePnPRansac(mvP3Dw, mvP2D, camMat, cv::noArray(), rvec, tvec, false, mRansacMaxIts, mRansacEpsilon, mRansacProb, inliers, cv::SOLVEPNP_EPNP);
    cv::solvePnPRansac(mvP3Dw, mvP2D, camMat, cv::noArray(), mrvec, mtvec, mnCurrentIterationsCount > 0,
                       iterationsCount, mReprojectionError, mConfidence, inliers, cv::SOLVEPNP_EPNP);

    cv::Rodrigues(mrvec, rmat);
    rmat.copyTo(M.rowRange(0, 3).colRange(0, 3));
    tmat = mtvec.reshape(1, 1);
    tmat.copyTo(M.row(3).colRange(0, 3));
    nInliers = inliers.rows;

    mnCurrentIterationsCount += iterationsCount;
    if (mnCurrentIterationsCount >= mMaxIterationsCount)
        bNoMore = true;

    //if (nInliers < mRansacMinInliers)
    if (nInliers < mMinInliers) {
        return make_tuple(cv::Mat(), bNoMore, vbInliers, nInliers);
    }

    vbInliers.resize(mvP3Dw.size(), false);
    for (int i = 0; i < inliers.rows; ++i) {
        vbInliers[((int *) inliers.data)[i]] = true;
    }
    return make_tuple(M, bNoMore, vbInliers, nInliers);
}

