#ifndef UTILS_H
#define UTILS_H

#include <math.h>
#include <limits>
#include <vector>

const double pi = M_PI;

// For converting back and forth between radians and degrees.
double deg2rad(double x) { return x * pi / 180; }
double rad2deg(double x) { return x * 180 / pi; }

// Calculate the Euclidea Distance between two points
double distance(double x1, double y1, double x2, double y2) {
	return sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
}

#endif