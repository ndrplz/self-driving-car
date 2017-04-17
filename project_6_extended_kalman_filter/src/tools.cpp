#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */

	size_t num_estimations = estimations.size();
	size_t num_ground_truths = ground_truth.size();

	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	// sanity check of input validity
	if (estimations.size() != ground_truth.size() || estimations.size() < 1) {
		std::cout << "Cannot compute RMSE metric. Invalid input size.";
		return rmse;
	}

	// accumulate residuals
	for (size_t i = 0; i < estimations.size(); ++i) {
		VectorXd residual = estimations[i] - ground_truth[i];
		residual = residual.array() * residual.array();
		rmse += residual;
	}

	// compute mean
	rmse /= estimations.size();

	// compute squared root
	rmse = rmse.array().sqrt();

	return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
	Eigen::MatrixXd temp = MatrixXd(2, 2);
	return temp;
}
