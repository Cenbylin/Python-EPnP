/**
* This file is part of ORB-SLAM2.
* This file is a modified version of EPnP <http://cvlab.epfl.ch/EPnP/index.php>
*/

#ifndef PNPSOLVER_H
#define PNPSOLVER_H

#include <opencv2/core/core.hpp>
#include <pybind11/stl.h>

using namespace std;


class PnPKeyPoint {
public:
    float angle;
    cv::Point2f pt;
    int class_id;
    int octave;
    float response;
    float size;

    PnPKeyPoint(float angle, tuple<float, float> tpt, float response, float size = 31, int class_id = -1, int octave = 0) :
            angle(angle), response(response), size(size), class_id(class_id), octave(octave) {
        pt.x = get<0>(tpt);
        pt.y = get<1>(tpt);
    };

    PnPKeyPoint(){
        // for test
    }
    cv::Mat test(){
        return cv::Mat();
    }

    ~ PnPKeyPoint() {

    };
};

class PnPsolver {
public:
    PnPsolver(vector<float> vLevelSigma2, float fx, float fy, float cx, float cy,
              vector<PnPKeyPoint *> vpKp, map<int, tuple<float, float, float>> vtMapPointMatches);

    ~PnPsolver();

    void SetRansacParameters(double probability = 0.99, int minInliers = 8, int maxIterations = 300, int minSet = 4,
                             float epsilon = 0.4,
                             float th2 = 5.991);

    tuple<cv::Mat, bool, vector<bool>, int> find();

    tuple<cv::Mat, bool, vector<bool>, int> iterate(int nIterations);

private:

    void CheckInliers();

    bool Refine();

    // Functions from the original EPnP code
    void set_maximum_number_of_correspondences(const int n);

    void reset_correspondences(void);

    void add_correspondence(const double X, const double Y, const double Z,
                            const double u, const double v);

    double compute_pose(double R[3][3], double T[3]);

    void relative_error(double &rot_err, double &transl_err,
                        const double Rtrue[3][3], const double ttrue[3],
                        const double Rest[3][3], const double test[3]);

    void print_pose(const double R[3][3], const double t[3]);

    double reprojection_error(const double R[3][3], const double t[3]);

    void choose_control_points(void);

    void compute_barycentric_coordinates(void);

    void fill_M(cv::Mat *M, const int row, const double *alphas, const double u, const double v);

    void compute_ccs(const double *betas, const double *ut);

    void compute_pcs(void);

    void solve_for_sign(void);

    void find_betas_approx_1(const cv::Mat *L_6x10, const cv::Mat *Rho, double *betas);

    void find_betas_approx_2(const cv::Mat *L_6x10, const cv::Mat *Rho, double *betas);

    void find_betas_approx_3(const cv::Mat *L_6x10, const cv::Mat *Rho, double *betas);

    void qr_solve(cv::Mat *A, cv::Mat *b, cv::Mat *X);

    double dot(const double *v1, const double *v2);

    double dist2(const double *p1, const double *p2);

    void compute_rho(double *rho);

    void compute_L_6x10(const double *ut, double *l_6x10);

    void gauss_newton(const cv::Mat *L_6x10, const cv::Mat *Rho, double current_betas[4]);

    void compute_A_and_b_gauss_newton(const double *l_6x10, const double *rho,
                                      double cb[4], cv::Mat *A, cv::Mat *b);

    double compute_R_and_t(const double *ut, const double *betas,
                           double R[3][3], double t[3]);

    void estimate_R_and_t(double R[3][3], double t[3]);

    void copy_R_and_t(const double R_dst[3][3], const double t_dst[3],
                      double R_src[3][3], double t_src[3]);

    void mat_to_quat(const double R[3][3], double q[4]);


    double uc, vc, fu, fv;

    double *pws, *us, *alphas, *pcs;
    int maximum_number_of_correspondences;
    int number_of_correspondences;

    double cws[4][3], ccs[4][3];
    double cws_determinant;

//    vector<tuple<int, int, int>> mvtMapPointMatches;
    int nKpNum;

    // 2D Points
    vector<cv::Point2f> mvP2D;
    vector<float> mvSigma2;

    // 3D Points
    vector<cv::Point3f> mvP3Dw;

    // Index in Frame
    vector<size_t> mvKeyPointIndices;

    // Current Estimation
    double mRi[3][3];
    double mti[3];
    cv::Mat mTcwi;
    vector<bool> mvbInliersi;
    int mnInliersi;

    // Current Ransac State
    int mnIterations;
    vector<bool> mvbBestInliers;
    int mnBestInliers;
    cv::Mat mBestTcw;

    // Refined
    cv::Mat mRefinedTcw;
    vector<bool> mvbRefinedInliers;
    int mnRefinedInliers;

    // Number of Correspondences
    int N;

    // Indices for random selection [0 .. N-1]
    vector<size_t> mvAllIndices;

    // RANSAC probability
    double mRansacProb;

    // RANSAC min inliers
    int mRansacMinInliers;

    // RANSAC max iterations
    int mRansacMaxIts;

    // RANSAC expected inliers/total ratio
    float mRansacEpsilon;

    // RANSAC Threshold inlier/outlier. Max error e = dist(P1,T_12*P2)^2
    float mRansacTh;

    // RANSAC Minimun Set used at each iteration
    int mRansacMinSet;

    // Max square error associated with scale level. Max error = th*th*sigma(level)*sigma(level)
    vector<float> mvMaxError;

};

#endif //PNPSOLVER_H