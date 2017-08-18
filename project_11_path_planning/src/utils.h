#ifndef UTILS_H
#define UTILS_H

#include <math.h>
#include <limits>
#include <vector>
#include "json.hpp"

const double pi = M_PI;

const double lane_width		= 4.0;		// width of a lane					(m)
const double safety_margin	= 20.0;		// distance to keep from other cars	(m)
const double max_safe_speed	= 49.5;		// max reference speed in the limit	(mph)


struct Vehicle {
	double d;
	double vx, vy;
	double speed;
	double s;
	Vehicle(nlohmann::json sensor_fusion) {
		this->vx = sensor_fusion[3];
		this->vy = sensor_fusion[4];
		this->s  = sensor_fusion[5];
		this->d  = sensor_fusion[6];
		this->speed = sqrt(vx * vx + vy * vy);
	}
};

// For converting back and forth between radians and degrees.
double deg2rad(double x) { return x * pi / 180; }
double rad2deg(double x) { return x * 180 / pi; }

// Calculate the Euclidea Distance between two points
double distance(double x1, double y1, double x2, double y2) {
	return sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
}

// Check if a vehicle is is a certain lane
bool is_in_lane(double d, int lane) {
	return (d > lane_width * lane) && (d < lane_width * lane + lane_width);
}

#endif