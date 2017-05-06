#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
	MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
	x_ = x_in;
	P_ = P_in;
	F_ = F_in;
	H_ = H_in;
	R_ = R_in;
	Q_ = Q_in;
}

void KalmanFilter::Predict() {

	// Predict the state
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::UpdateRoutine(const VectorXd& y) {

	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();

	// Compute Kalman gain
	MatrixXd K = P_ * Ht * Si;

	// Update estimate
	x_ = x_ + K * y;
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z) {

	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;

	UpdateRoutine(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

	// Recover explicitly status information
	float px = x_(0);
	float py = x_(1);
	float vx = x_(2);
	float vy = x_(3);

	// Map predicted state into measurement space
	float rho = sqrt(px * px + py * py);

	// Sanity checks to avoid division by zero
	if (std::abs(rho) < 0.0001) rho = 0.0001;
	if (std::abs(px)  < 0.0001) px = 0.0001;

	float phi = atan2(py, px);
	float rho_dot = (px * vx + py * vy) / rho;

	//// Normalize angle
	//while (phi > M_PI) phi -= 2 * M_PI;
	//while (phi < M_PI) phi += 2 * M_PI;

	VectorXd z_pred(3);
	z_pred << rho, phi, rho_dot;

	VectorXd y = z - z_pred;

	UpdateRoutine(y);

}
