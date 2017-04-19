#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

	size_t num_estimations = estimations.size();
	size_t num_ground_truths = ground_truth.size();

	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	// sanity check of input validity
	if (estimations.size() != ground_truth.size() || estimations.size() < 1) {
		std::cout << "Cannot compute RMSE metric. Invalid input size." << std::endl;
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

	MatrixXd Hj(3, 4);

	// Unroll state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);
	
	// Pre-compute some term which recur in the Jacobian
	float c1 = px * px + py * py;
	float c2 = sqrt(c1);
	float c3 = c1 * c2;

	// Sanity check to avoid division by zero
	if (std::abs(c1) < 0.0001) {
		std::cout << "Error in CalculateJacobian. Division by zero." << std::endl;
		return Hj;
	}

	// Actually compute Jacobian matrix
	Hj << (px / c2),				(py / c2),					0,			0,
		-(py / c1),					(px / c1),					0,			0,
		py * (vx*py - vy*px) / c3,	px * (vy*px - vx*py) / c3,	px / c2,	py / c2;

	return Hj;

}
