#include "utils.h"


// Choose in the map of highway waypoints the one closest to the car
int get_closest_waypoint(double x, double y, std::vector<double> maps_x, std::vector<double> maps_y) {

	double closest_len = std::numeric_limits<int>::max();
	int closest_waypoint = 0;

	for (int i = 0; i < maps_x.size(); i++) {

		double map_x = maps_x[i];
		double map_y = maps_y[i];

		double dist = distance(x, y, map_x, map_y);
		if (dist < closest_len) {
			closest_len = dist;
			closest_waypoint = i;
		}
	}
	return closest_waypoint;
}


// Choose in the map of highway waypoints the closest before the car (that is the next).
// The actual closest waypoint could be behind the car.
int get_next_waypoint(double x, double y, double theta, std::vector<double> maps_x, std::vector<double> maps_y) {

	int closest_waypoint = get_closest_waypoint(x, y, maps_x, maps_y);

	double map_x = maps_x[closest_waypoint];
	double map_y = maps_y[closest_waypoint];

	double heading = atan2(map_y - y, map_x - x);

	double angle = abs(theta - heading);

	if (angle > pi / 4)
		closest_waypoint++;

	return closest_waypoint;
}