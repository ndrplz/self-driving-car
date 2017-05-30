#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;


// Create only once the default random engine
static default_random_engine gen;


// Particle filter initialization.
// Set number of particles and initialize them to first position based on GPS estimate.
void ParticleFilter::init(double gps_x, double gps_y, double theta, double sigma_pos[]) {

    // Set the number of particles
    num_particles = 100;

    // Creates normal (Gaussian) distribution for p_x, p_y and p_theta
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


// Move each particle according to bicycle motion model (taking noise into account)
void ParticleFilter::prediction(double delta_t, double sigma_pos[], double velocity, double yaw_rate) {

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
        normal_distribution<double> dist_x(x_pred, sigma_pos[0]);
        normal_distribution<double> dist_y(y_pred, sigma_pos[1]);
        normal_distribution<double> dist_theta(theta_pred, sigma_pos[2]);

        // Update particle with noisy prediction
        particles[i].x	   = dist_x(gen);
        particles[i].y	   = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}


// Finds which observations correspond to which landmarks by using a nearest-neighbor data association
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


// Update the weight of each particle taking into account current measurements.
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {

    // Gather std values for readability
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];

    // Iterate over all particles
    for (size_t i = 0; i < num_particles; ++i) {

        // Gather current particle values
        double p_x	   = particles[i].x;
        double p_y	   = particles[i].y;
        double p_theta = particles[i].theta;

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
            rototranslated_obs.x = cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y + p_x;
            rototranslated_obs.y = sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y + p_y;

            observed_landmarks_map_ref.push_back(rototranslated_obs);
        }

        // Find which observations correspond to which landmarks (associate ids)
        dataAssociation(predicted_landmarks, observed_landmarks_map_ref);

        // Compute the likelihood for each particle, that is the probablity of obtaining
        // current observations being in state (particle_x, particle_y, particle_theta)
        double particle_likelihood = 1.0;

        double mu_x, mu_y;
        for (const auto& obs : observed_landmarks_map_ref) {

            // Find corresponding landmark on map for centering gaussian distribution
            for (const auto& land: predicted_landmarks)
                if (obs.id == land.id) {
                    mu_x = land.x;
                    mu_y = land.y;
                    break;
                }

            double norm_factor = 2 * M_PI * std_x * std_y;
            double prob = exp( -( pow(obs.x - mu_x, 2) / (2 * std_x * std_x) + pow(obs.y - mu_y, 2) / (2 * std_y * std_y) ) );

            particle_likelihood *= prob / norm_factor;
        }

        particles[i].weight = particle_likelihood;

    } // end loop for each particle

    // Compute weight normalization factor
    double norm_factor = 0.0;
    for (const auto& particle : particles)
        norm_factor += particle.weight;

    // Normalize weights s.t. they sum to one
    for (auto& particle : particles)
        particle.weight /= (norm_factor + numeric_limits<double>::epsilon());
}


// Resample particles with replacement with probability proportional to their weight.
void ParticleFilter::resample() {

    vector<double> particle_weights;
    for (const auto& particle : particles)
        particle_weights.push_back(particle.weight);

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


Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}


string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}


string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}


string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
