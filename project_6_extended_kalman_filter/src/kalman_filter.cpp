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
	P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {

	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd S = H_ * P_ * H_.transpose() + R_;
	MatrixXd K = P_ * H_.transpose() * S.inverse();	

	// Update estimate
	x_ = x_ + K * y;
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

	// Recover explicitly status information
	float px = x_(0);
	float py = x_(1);
	float vx = x_(2);
	float vy = x_(3);

	// Map predicted state into measurement space
	VectorXd z_pred(3);
	float rho_pred = sqrt(px * px + py * py);
	float phi_pred = atan(py / px);
	float rhodot_pred = (px * vx + py * vy) / rho_pred;

	// Sanity check to avoid division by zero
	if (std::abs(rho_pred) < 0.0001)
		rho_pred = 0.0001; 

	// Normalize angle
	while (phi_pred > M_PI) phi_pred -= 2 * M_PI;
	while (phi_pred < M_PI) phi_pred += 2 * M_PI;

	z_pred << rho_pred, phi_pred, rhodot_pred;

	VectorXd y = z - z_pred;
	MatrixXd S = H_ * P_ * H_.transpose() + R_;
	MatrixXd K = P_ * H_.transpose() * S.inverse();

	// Update estimate
	x_ = x_ + K * y;
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}
