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
	//   p_x, p_y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.

	// Set the number of particles 
	num_particles = 100;

	// Creates normal (Gaussian) distribution for p_x, p_y and theta
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

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double sigma_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.

	for (size_t i = 0; i < num_particles; ++i) {
		
		// Gather old values
		double x_old	 = particles[i].x;
		double y_old	 = particles[i].y;
		double theta_old = particles[i].theta;

		double theta_pred, x_pred, y_pred;

		if (abs(yaw_rate) > 1e-5) {
			// Apply equations of motion model (turning)
			theta_pred = theta_old + yaw_rate * delta_t;
			x_pred	   = x_old + velocity / yaw_rate * (sin(theta_pred) - sin(theta_old));
			y_pred	   = y_old + velocity / yaw_rate * (cos(theta_old) - cos(theta_pred));
		} else {
			// Apply equations of motion model (going straight)
			theta_pred = theta_old;
			x_pred	   = x_old + velocity * delta_t * cos(theta_old);
			y_pred	   = y_old + velocity * delta_t * sin(theta_old);
		}

		// Initialize normal distributions centered on predicted values
		default_random_engine gen;
		normal_distribution<double> dist_x(x_pred, sigma_pos[0]);
		normal_distribution<double> dist_y(y_pred, sigma_pos[1]);
		normal_distribution<double> dist_theta(theta_pred, sigma_pos[2]);

		// Update particle with noisy prediction
		particles[i].x	   = dist_x(gen);
		particles[i].y	   = dist_y(gen);
		particles[i].theta = dist_theta(gen);

	}

}

// Finds which observations correspond to which landmarks by using a nearest - neighbors data association
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

	for (auto& obs : observations) {
		double min_dist = numeric_limits<double>::max();

		for (const auto& pred_obs : predicted) { 
			double d = dist(obs.x, obs.y, pred_obs.x, pred_obs.y);
			if (d < min_dist) {
				obs.id	 = pred_obs.id;
				min_dist = d;
			}
		}
	}

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
	//   for the fact that the map's p_y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	// Gather std values for readability
	double std_x = std_landmark[0];
	double std_y = std_landmark[1];

	// Iterate over all particles
	for (size_t i = 0; i < num_particles; ++i) {

		// Gather current particle values 
		double p_x	 = particles[i].x;
		double p_y	 = particles[i].y;
		double theta = particles[i].theta;
		
		// List all landmarks within sensor range
		vector<LandmarkObs> predicted_landmarks;

		for (const auto& map_landmark : map_landmarks.landmark_list) {
			int l_id   = map_landmark.id_i;
			double l_x = (double) map_landmark.x_f;
			double l_y = (double) map_landmark.y_f;

			double d = dist(p_x, p_y, l_x, l_y);
			if (d < sensor_range) {
				LandmarkObs l_pred;
				l_pred.id = l_id;
				l_pred.x = l_x;
				l_pred.y = l_y;
				predicted_landmarks.push_back(l_pred);
			}
		}

		// List all observations in map coordinates
		vector<LandmarkObs> observed_landmarks_map_ref;
		for (size_t j = 0; j < observations.size(); ++j) { 

			// Convert observation from particle(vehicle) to map coordinate system
			LandmarkObs rototranslated_obs;
			rototranslated_obs.x = cos(theta) * p_x - sin(theta) * p_y + observations[j].x;
			rototranslated_obs.y = sin(theta) * p_x + cos(theta) * p_y + observations[j].y;

			observed_landmarks_map_ref.push_back(rototranslated_obs); 
		}

		// Find which observations correspond to which landmarks (associate ids)
		dataAssociation(predicted_landmarks, observed_landmarks_map_ref);

		// Compute the likelihood for each particle, that is the probablity of obtaining
		// current observations being in state (particle_x, particle_y, particle_theta)
		double particle_likelihood = 1.0;

		double mu_x, mu_y;
		for (const auto& obs : observed_landmarks_map_ref) {

			// Multivariate gaussian is centered on corresponding landmark on map
			for (const auto& land: predicted_landmarks) {
				if (obs.id == land.id) {
					mu_x = land.x;
					mu_y = land.y;
					break;
				}
			}
			double obs_prob = (exp(-pow(obs.x - mu_x, 2)) / (2 * M_PI * std_x))	* (exp(-pow(obs.y - mu_y, 2)) / (2 * M_PI * std_y));

			particle_likelihood *= obs_prob + numeric_limits<double>::epsilon();
		}

		particles[i].weight = particle_likelihood;

	} // end loop for each particle

	// Compute weight normalization factor
	double norm_factor = 0.0;
	for (const auto& particle : particles)
		norm_factor += particle.weight;

	// Normalize weights s.t. they sum to one
	for (auto& particle : particles)
		particle.weight /= norm_factor;
}

// Resample particles with replacement with probability proportional to their weight. 
void ParticleFilter::resample() {

	vector<double> particle_weights;
	for (const auto& particle : particles)
		particle_weights.push_back(particle.weight);

	default_random_engine gen;
	discrete_distribution<int> weighted_distribution(particle_weights.begin(), particle_weights.end());

	vector<Particle> resampled_particles;
	for (size_t i = 0; i < num_particles; ++i) {
		int k = weighted_distribution(gen);
		resampled_particles.push_back(particles[k]);
	}
	
	particles = resampled_particles;

	// Reset weights for all particles
	for (auto& particle : particles)
		particle.weight = 1.0;


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
