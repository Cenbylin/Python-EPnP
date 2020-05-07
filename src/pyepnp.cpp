// Wrapper for most external modules
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <exception>
// Opencv includes
#include <opencv2/opencv.hpp>
// type catser for Numpy <=> cv:Mat
#include "ndarray_converter.h"
#include "opencv_type_converter.h"

#include "PnPsolver.h"
#include "CV2PnPsolver.h"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(PyEPnP, m) {
    NDArrayConverter::init_numpy();

    m.doc() = R"pbdoc(
        Pybind11 for EPnP
    )pbdoc";

    // Class

    py::class_<PnPsolver>(m, "PnPsolver")
            .def(py::init<vector<float>, float, float, float, float, vector<cv::KeyPoint> &, vector<tuple<int, float, float, float>>>(),
                 py::arg("vLevelSigma2"),py::arg("fx"),py::arg("fy"),py::arg("cx"),py::arg("cy"),py::arg("vpKp"),py::arg("vtMapPointMatches"))
            .def("SetRansacParameters", &PnPsolver::SetRansacParameters,
                    py::arg("probability"), py::arg("minInliers"), py::arg("maxIterations"),
                    py::arg("minSet"), py::arg("epsilon"), py::arg("th2"))
            .def("iterate", &PnPsolver::iterate, py::arg("nIterations"),  py::return_value_policy::take_ownership)
            .def("find", &PnPsolver::find , py::return_value_policy::take_ownership)

            ;

    py::class_<CV2PnPsolver>(m, "CV2PnPsolver")
            .def(py::init<vector<float>, float, float, float, float, vector<cv::KeyPoint> &, vector<tuple<int, float, float, float>>>(),
                 py::arg("vLevelSigma2"),py::arg("fx"),py::arg("fy"),py::arg("cx"),py::arg("cy"),py::arg("vpKp"),py::arg("vtMapPointMatches"))
            .def("iterate", &CV2PnPsolver::iterate, py::arg("nIterations"),  py::return_value_policy::take_ownership)
            ;



#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}