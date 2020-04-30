
#include <iostream>
#include <cmath>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include "DUtils/Random.h"

PnPsolver::PnPsolver(const Frame &F, const vector<MapPoint *> &vpMapPointMatches) :
        mnCurrentIterationsCount(0) {
    // 预先分配空间
    mvP2D.reserve(F.mvpMapPoints.size());
    mvP3Dw.reserve(F.mvpMapPoints.size());

    for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++) {
        MapPoint *pMP = vpMapPointMatches[i];

        if (pMP) {
            if (!pMP->isBad()) {
                const cv::KeyPoint &kp = F.mvKeysUn[i];
                mvP2D.push_back(kp.pt);

                cv::Mat Pos = pMP->GetWorldPos();
                mvP3Dw.push_back(cv::Point3f(Pos.at<float>(0), Pos.at<float>(1), Pos.at<float>(2)));
            }
        }
    }

    // Set camera calibration parameters
    // 得到当前帧的相机内部参数
    fx = F.fx;
    fy = F.fy;
    cx = F.cx;
    cy = F.cy;

    mMaxIterationsCount = 300;
    mReprojectionError = 3.f;
    mConfidence = 0.99;
    mMinInliers = 50;
}

PnPsolver::~PnPsolver() {
}

cv::Mat PnPsolver::iterate(int iterationsCount, bool &bNoMore, vector<bool> &vbInliers, int &nInliers) {
    bNoMore = false;
    vbInliers.clear();
    nInliers = 0;

    if (iterationsCount < 1) {
        cerr << "Error：iterationsCount must be greater than 0, but now is " << iterationsCount
             << "; location PnPsolver::solve" << endl;
        iterationsCount = 1;
    }
    if (mnCurrentIterationsCount >= mMaxIterationsCount) {
        bNoMore = true;
        return cv::Mat();
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
        return cv::Mat();
    }

    vbInliers.resize(mvP3Dw.size(), false);
    for (int i = 0; i < inliers.rows; ++i) {
        vbInliers[((int *) inliers.data)[i]] = true;
    }
    return M;
}

