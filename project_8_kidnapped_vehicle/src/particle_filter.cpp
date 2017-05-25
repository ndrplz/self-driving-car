/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double gps_x, double gps_y, double theta, double sigma_pos[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.

	// Set the number of particles 
	num_particles = 100;

	// Creates normal (Gaussian) distribution for x, y and theta
	default_random_engine gen;
	normal_distribution<double> dist_x(gps_x, sigma_pos[0]);
	normal_distribution<double> dist_y(gps_y, sigma_pos[1]);
	normal_distribution<double> dist_theta(theta, sigma_pos[2]);

	for (size_t i = 0; i < num_particles; ++i) {
		
		// Instantiate a new particle
		Particle p;
		p.id		= int(i);
		p.weight	= 1.0;
		p.x			= dist_x(gen);
		p.y			= dist_y(gen);
		p.theta		= dist_theta(gen);

		// Add the particle to the particle filter set
		particles.push_back(p);
	}
}

void ParticleFilter::prediction(double delta_t, double sigma_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.

	for (size_t i = 0; i < num_particles; ++i) {
		
		// Gather old values
		double theta_old = particles[i].theta;
		double x_old = particles[i].x;
		double y_old = particles[i].y;

		// Apply equations of motion model
		double theta_pred = theta_old + yaw_rate * delta_t;
		double x_pred = x_old + velocity / yaw_rate * (sin(theta_pred) - sin(theta_old));
		double y_pred = y_old + velocity / yaw_rate * (cos(theta_old) - cos(theta_pred));

		// Initialize distributions for adding noise
		default_random_engine gen;
		normal_distribution<double> dist_x(x_pred, sigma_pos[0]);
		normal_distribution<double> dist_y(y_pred, sigma_pos[1]);
		normal_distribution<double> dist_theta(theta_pred, sigma_pos[2]);

		// Update particle with noisy prediction
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find sigma_pos::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
