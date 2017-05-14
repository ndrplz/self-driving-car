#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>


using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  // If this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // If this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Initial state vector
  x_ = VectorXd(5);

  // Initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Number of sigma points
  n_sigma_points_ = 2 * n_aug_ + 1;

  // Predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, n_sigma_points_);

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Weights of sigma points
  weights_ = VectorXd(n_sigma_points_);
  weights_.segment(1, 2 * n_aug_).fill(0.5 / (n_aug_ + lambda_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  // Initialize Normalized Innovation Squared (NIS) value for both sensors
  NIS_laser_ = 0.;
  NIS_radar_ = 0.;

  // Sensor's measurement size
  n_z_radar_ = 3; // radar -> rho, phi, rho_dot
  n_z_lidar_ = 2; // lidar -> px, py

  // Measurement covariance matrices
  R_lidar_ = MatrixXd(n_z_lidar_, n_z_lidar_);
  R_radar_ = MatrixXd(n_z_radar_, n_z_radar_);
}


UKF::~UKF() {}


/**
 * @param {MeasurementPackage} measurement_pack The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {

	/*****************************************************************************
	*  Initialization
	****************************************************************************/
	if (!is_initialized_) {

		if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {

			// Extract values from measurement
			float rho = measurement_pack.raw_measurements_(0);
			float phi = measurement_pack.raw_measurements_(1);
			float rho_dot = measurement_pack.raw_measurements_(2);
			
			// Convert from polar to cartesian coordinates
			float px = rho * cos(phi);
			float py = rho * sin(phi);

			// Initialize state
			x_ << px, py, rho_dot, 0.0, 0.0;
		}
		else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {

			// Extract values from measurement
			float px = measurement_pack.raw_measurements_(0);
			float py = measurement_pack.raw_measurements_(1);
			
			// Initialize state
			x_ << px, py, 0.0, 0.0, 0.0;
		}

		// Initialize state covariance matrix
		P_ = MatrixXd::Identity(n_x_, n_x_);

		// Update last measurement
		previous_timestamp_ = measurement_pack.timestamp_;

		// Done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}

	/*****************************************************************************
	*  Prediction
	****************************************************************************/

	// Compute elapsed time from last measurement
	float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;

	// Update last measurement
	previous_timestamp_ = measurement_pack.timestamp_;

	Prediction(dt);

	/*****************************************************************************
	*  Update
	****************************************************************************/

	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
		// Radar updates 
		UpdateRadar(measurement_pack);
	else
		// Laser updates
		UpdateLidar(measurement_pack);
	

}


MatrixXd UKF::ComputeSigmaPoints(double dt) {

	// Augmented mean state
	VectorXd x_aug = VectorXd(n_aug_);
	x_aug.head(5) = x_;
	x_aug(5) = 0;
	x_aug(6) = 0;

	// Augmented state covariance
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
	P_aug.fill(0.0);
	P_aug.topLeftCorner(5, 5) = P_;
	P_aug(5, 5) = std_a_ * std_a_;
	P_aug(6, 6) = std_yawdd_ * std_yawdd_;

	// Compute sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma_points_);
	Xsig_aug.col(0) = x_aug;
	MatrixXd L = P_aug.llt().matrixL(); // square root of P
	for (int i = 0; i < n_aug_; i++) {
		Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
		Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
	}

	// Actual computation of sigma points
	for (int i = 0; i < n_sigma_points_; i++) {

		// Auxiliary variables for readability
		double p_x		= Xsig_aug(0, i);
		double p_y		= Xsig_aug(1, i);
		double v		= Xsig_aug(2, i);
		double yaw		= Xsig_aug(3, i);
		double yawd		= Xsig_aug(4, i);
		double nu_a		= Xsig_aug(5, i);
		double nu_yawdd	= Xsig_aug(6, i);

		// Sanity check
		if (fabs(p_x) < 0.001 && fabs(p_y) < 0.001) {
			p_x = 0.1;
			p_y = 0.1;
		}

		// Predicted state values
		double px_p, py_p;
		if (fabs(yawd) > 0.001) {
			px_p = p_x + v / yawd * (sin(yaw + yawd * dt) - sin(yaw));
			py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * dt));
		}
		else {
			px_p = p_x + v * dt * cos(yaw);
			py_p = p_y + v * dt * sin(yaw);
		}

		double v_p = v;
		double yaw_p = yaw + yawd * dt;
		double yawd_p = yawd;

		// Handle noise
		px_p = px_p + 0.5 * nu_a * dt * dt * cos(yaw);
		py_p = py_p + 0.5 * nu_a * dt * dt * sin(yaw);
		v_p = v_p + nu_a * dt;
		yaw_p = yaw_p + 0.5 * nu_yawdd * dt * dt;
		yawd_p = yawd_p + nu_yawdd * dt;

		// Fill current column of Xsig_pred matrix with sigma point just computed
		Xsig_pred_(0, i) = px_p;
		Xsig_pred_(1, i) = py_p;
		Xsig_pred_(2, i) = v_p;
		Xsig_pred_(3, i) = yaw_p;
		Xsig_pred_(4, i) = yawd_p;
	}

	return Xsig_pred_;
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} dt the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double dt) {

	// Compute predicted sigma points
	MatrixXd Xsig_pred = ComputeSigmaPoints(dt);

	// Predicted state mean (5 x 1) (vectorised form: x = Xsig_pred * weights_ could also work)
	VectorXd x = VectorXd(n_x_);
	x.fill(0.0);
	for (int i = 0; i < n_sigma_points_; i++) 
		x = x + weights_(i) * Xsig_pred.col(i);
	
	// Predicted state covariance matrix (5 x 5)
	MatrixXd P = MatrixXd(n_x_, n_x_);
	P.fill(0.0);
	for (int i = 0; i < n_sigma_points_; i++) {

		VectorXd x_diff = Xsig_pred.col(i) - x;

		// Normalize angle
		normalize_between_minus_PI_and_PI(x_diff(3));

		P = P + weights_(i) * x_diff * x_diff.transpose();
	}

	// Update state vector and covariance matrix
	x_ = x;
	P_ = P;
}


/**
* Updates the state and the state covariance matrix using a laser measurement.
* @param {MeasurementPackage} measurement_pack
*/
void UKF::UpdateLidar(MeasurementPackage measurement_pack) {

	/*****************************************************************************
	*  Prediction
	****************************************************************************/
	
	// Project sigma points onto measurement space
	MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z_lidar_, n_sigma_points_); 

	// Predicted measurement mean
	VectorXd z_pred = VectorXd(n_z_lidar_);
	z_pred.fill(0.0);
	for (int i = 0; i < n_sigma_points_; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	// Predicted measurement covariance matrix
	MatrixXd S = MatrixXd(n_z_lidar_, n_z_lidar_);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {

		VectorXd z_diff = Zsig.col(i) - z_pred;

		// Normalize angle in range [-PI, PI]
		while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
		while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	// Handle measurement noise 
	R_lidar_ << std_laspx_ * std_laspx_, 0, 0, std_laspy_ * std_laspy_;
	S = S + R_lidar_;

	/*****************************************************************************
	*  Update
	****************************************************************************/

	// Parse laser measurement
	VectorXd z = VectorXd(n_z_lidar_);
	z << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1];

	// Ccompute cross correlation matrix
	MatrixXd Tc = MatrixXd(n_x_, n_z_lidar_); 	// 5 x 2
	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {

		// Residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		// Normalize angle
		normalize_between_minus_PI_and_PI(z_diff(1));

		// State difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		// Normalize angle
		normalize_between_minus_PI_and_PI(x_diff(3));

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	// Compute Kalman gain;
	MatrixXd K = Tc * S.inverse();

	// Residual
	VectorXd z_diff = z - z_pred;

	// Normalize angle
	normalize_between_minus_PI_and_PI(z_diff(1));

	// Update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();

	// Compute NIS for laser sensor
	NIS_laser_ = (measurement_pack.raw_measurements_ - z_pred).transpose() * S.inverse() *
		(measurement_pack.raw_measurements_ - z_pred);

}

/**
* Updates the state and the state covariance matrix using a radar measurement.
* @param {MeasurementPackage} measurement_pack
*/
void UKF::UpdateRadar(MeasurementPackage measurement_pack) {

	/*****************************************************************************
	*  Prediction
	****************************************************************************/

	// Project sigma points onto measurement space
	MatrixXd Zsig = MatrixXd(n_z_radar_, 2 * n_aug_ + 1);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {

		// extract values for better readability
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);

		double v1 = cos(yaw) * v;
		double v2 = sin(yaw) * v;

		// Measurement model

		Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);

		if (fabs(p_y) > 0.001 && fabs(p_x) > 0.001)
			Zsig(1, i) = atan2(p_y, p_x);
		else
			Zsig(1, i) = 0.0;

		if (fabs(sqrt(p_x * p_x + p_y * p_y)) > 0.001)
			Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);
		else
			Zsig(2, i) = 0.0;
		
	}

	// Predicted measurement mean
	VectorXd z_pred = VectorXd(n_z_radar_);
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	// Predicted measurement covariance matrix
	MatrixXd S = MatrixXd(n_z_radar_, n_z_radar_);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {

		VectorXd z_diff = Zsig.col(i) - z_pred;

		// Normalize angle
		normalize_between_minus_PI_and_PI(z_diff(1));

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	// Handle measurement noise
	R_radar_ << std_radr_ * std_radr_, 0, 0,
		0, std_radphi_ * std_radphi_, 0,
		0, 0, std_radrd_ * std_radrd_;
	S = S + R_radar_;

	/*****************************************************************************
	*  Update
	****************************************************************************/

	// Parse radar measurement
	VectorXd z = VectorXd(n_z_radar_);
	z << measurement_pack.raw_measurements_[0],
		measurement_pack.raw_measurements_[1],
		measurement_pack.raw_measurements_[2];

	// Compute cross correlation matrix
	MatrixXd Tc = MatrixXd(n_x_, n_z_radar_); 	// 5 x 3
	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  // iterate over sigma points

		// Residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		// Normalize angle
		normalize_between_minus_PI_and_PI(z_diff(1));

		// State difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		// Normalize angle
		normalize_between_minus_PI_and_PI(x_diff(3));

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	// Compute Kalman gain
	MatrixXd K = Tc * S.inverse();

	// Residual
	VectorXd z_diff = z - z_pred;

	// Normalize angle
	normalize_between_minus_PI_and_PI(z_diff(1));

	// Update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();

	// Compute NIS for radar sensor
	NIS_radar_ = (measurement_pack.raw_measurements_ - z_pred).transpose() * S.inverse() *
		(measurement_pack.raw_measurements_ - z_pred);
}


/**
* Helper function to normalize angles in range [-PI, PI]
* @param phi angle to be normalized
*/
void UKF::normalize_between_minus_PI_and_PI(double& phi)
{
	while (phi > M_PI) phi -= 2. * M_PI;
	while (phi < -M_PI) phi += 2. * M_PI;
}
