#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/* Constructor. */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // Initialize measurement covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
			  0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ <<	0.09,	0,		0,
				0,		0.0009, 0,
				0,		0,		0.09;

  // Lidar - measurement matrix
  H_laser_ = MatrixXd(2, 4);
  H_laser_	<<	1, 0, 0, 0,
				0, 1, 0, 0;

  // Radar - jacobian matrix
  Hj_ = MatrixXd(3, 4);
  Hj_		<<	0, 0, 0, 0,
				0, 0, 0, 0,
				0, 0, 0, 0;

  // Initialize state covariance matrix P
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_	<<	1,	0,	0,	 0,
				0,	1,	0,	 0,
				0,	0, 1000, 0,
				0,	0, 0,	1000;

  // Initial transition matrix F_
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ <<	1, 0, 1, 0,
				0, 1, 0, 1,
				0, 0, 1, 0,
				0, 0, 0, 1;

  // Initialize process noise covariance matrix
  ekf_.Q_ = MatrixXd(4, 4); 
  ekf_.Q_ <<	0, 0, 0, 0,
				0, 0, 0, 0,
				0, 0, 0, 0,
				0, 0, 0, 0;

  // Initialize ekf state
  ekf_.x_ = VectorXd(4);
  ekf_.x_ << 1, 1, 1, 1;

  noise_ax = 9;
  noise_ay = 9;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {

		// Extract values from measurement
		float rho		= measurement_pack.raw_measurements_(0);
		float phi		= measurement_pack.raw_measurements_(1);
		float rho_dot	= measurement_pack.raw_measurements_(2);

		// Convert from polar to cartesian coordinates
		float px = rho * cos(phi);
		float py = rho * sin(phi);
		float vx = rho_dot * cos(phi);
		float vy = rho_dot * sin(phi);

		// Initialize state
		ekf_.x_ << px, py, vx, vy;

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {

		// Extract values from measurement
		float px = measurement_pack.raw_measurements_(0);
		float py = measurement_pack.raw_measurements_(1);

		// Initialize state
		ekf_.x_ << px, py, 0.0, 0.0;
    }

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

  // Update state transition matrix F (according to elapsed time dt)
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // Compute process noise covariance matrix
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  ekf_.Q_	<<	dt_4 / 4 * noise_ax,	0,						dt_3 / 2 * noise_ax,	0,
				0,						dt_4 / 4 * noise_ay,	0,						dt_3 / 2 * noise_ay,
				dt_3 / 2 * noise_ax,	0,						dt_2 * noise_ax,		0,
				0,						dt_3 / 2 * noise_ay,	0,						dt_2 * noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
	  ekf_.R_ = R_radar_;
	  ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
	  ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
	  ekf_.R_ = R_laser_;
	  ekf_.H_ = H_laser_;
	  ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
