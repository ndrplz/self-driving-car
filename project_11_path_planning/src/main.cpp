#include <fstream>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"
#include "utils.h"
#include "coords_transform.h"


using namespace std;
using json = nlohmann::json;


// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}


int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }
	
  // Start on lane 1 
  int lane = 1;

  // Reference velocity (mph)
  double ref_vel = 0.0; 

  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy,&ref_vel, &lane](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
        	// Main car's localization Data
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = j[1]["yaw"];
          	double car_speed = j[1]["speed"];

          	// Previous path data given to the Planner
          	auto previous_path_x = j[1]["previous_path_x"];
          	auto previous_path_y = j[1]["previous_path_y"];
          	// Previous path's end s and d values 
          	double end_path_s = j[1]["end_path_s"];
          	double end_path_d = j[1]["end_path_d"];

          	// Sensor Fusion Data, a list of all other cars on the same side of the road.
          	auto sensor_fusion = j[1]["sensor_fusion"];
			
			int prev_size = previous_path_x.size();

			if (prev_size > 0)
				car_s = end_path_s;

			bool too_close = false;

			for (size_t i = 0; i < sensor_fusion.size(); ++i) {

				// Check if the `i_th` car is on our lane
				double d = sensor_fusion[i][6];
				if (d < (2 + 4 * lane + 2) && d >(2 + 4 * lane - 2)) {
					double vx = sensor_fusion[i][3];
					double vy = sensor_fusion[i][4];
					double check_speed = sqrt(vx * vx + vy * vy);  // speed module
					double check_car_s = sensor_fusion[i][5];

					// use previous points to project s value outward
					check_car_s += (double)prev_size * 0.02 * check_speed;

					if (check_car_s > car_s && (check_car_s - car_s) < 30) {
						// add the logic here, for example either decrease speed or prepare for lane change
						too_close = true;
						if (lane > 0)
							lane = 0;
					}
				}
			}

			if (too_close)
				ref_vel -= 0.224; // around 5 m/s^2
			else if (ref_vel < 49.5)
				ref_vel += 0.224;

			// List of widely spaced (x, y) waypoints. These will be later interpolated with a spline,
			// filling it with more points which control speed.
			vector<double> pts_x;
			vector<double> pts_y;

			// Reference x, y, yaw state 
			double ref_x = car_x;
			double ref_y = car_y;
			double ref_yaw = deg2rad(car_yaw);

			// If previous size is almost empty, use the car as a starting reference
			if (prev_size < 2) {
				double prev_car_x = car_x - cos(car_yaw);
				double prev_car_y = car_y - sin(car_yaw);

				pts_x.push_back(prev_car_x); pts_x.push_back(car_x);
				pts_y.push_back(prev_car_y); pts_y.push_back(car_y);
			}
			// Use the previous path's end points as starting reference
			else {
				ref_x = previous_path_x[prev_size - 1];
				ref_y = previous_path_y[prev_size - 1];

				double ref_x_prev = previous_path_x[prev_size - 2];
				double ref_y_prev = previous_path_y[prev_size - 2];
				ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

				pts_x.push_back(ref_x_prev); pts_x.push_back(ref_x);
				pts_y.push_back(ref_y_prev); pts_y.push_back(ref_y);
			}

			// In Frenet coordinates, add evenly 30m spaced points ahead of the starting reference
			vector<double> next_wp0 = frenet_to_cartesian(car_s + 30, (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
			vector<double> next_wp1 = frenet_to_cartesian(car_s + 60, (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
			vector<double> next_wp2 = frenet_to_cartesian(car_s + 90, (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);

			pts_x.push_back(next_wp0[0]); pts_x.push_back(next_wp1[0]); pts_x.push_back(next_wp2[0]);
			pts_y.push_back(next_wp0[1]); pts_y.push_back(next_wp1[1]); pts_y.push_back(next_wp2[1]);

			// Rototranslate into car's reference system to make the math easier
			for (size_t i = 0; i < pts_x.size(); ++i) {
				double shift_x = pts_x[i] - ref_x;
				double shift_y = pts_y[i] - ref_y;
				pts_x[i] = shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw);
				pts_y[i] = shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw);
			}

			// Create a spline
			tk::spline s;
			s.set_points(pts_x, pts_y);
			
			// Define he actual (x, y) points will be used for the planner
			vector<double> next_x_vals;
			vector<double> next_y_vals;

			// Start with all previous points from last time
			for (size_t i = 0; i < previous_path_x.size(); ++i) {
				next_x_vals.push_back(previous_path_x[i]);
				next_y_vals.push_back(previous_path_y[i]);
			}

			// Calculate how to break up spline points to travel at reference velocity
			double target_x = 30.0;
			double target_y = s(target_y);
			double target_dist = sqrt(target_x * target_x + target_y * target_y);

			double x_add_on = 0.0;

			for (size_t i = 1; i <= 50 - previous_path_x.size(); ++i) {

				double N = target_dist / (0.02 * ref_vel / 2.24);
				double x_point = x_add_on + target_x / N;
				double y_point = s(x_point);

				x_add_on = x_point;

				double x_ref = x_point;
				double y_ref = y_point;

				// Rotate back into previous coordinate system
				x_point = x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw);
				y_point = x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw);

				x_point += ref_x;
				y_point += ref_y;

				next_x_vals.push_back(x_point);
				next_y_vals.push_back(y_point);
			}

          	json msgJson;
          	msgJson["next_x"] = next_x_vals;
          	msgJson["next_y"] = next_y_vals;

          	auto msg = "42[\"control\","+ msgJson.dump()+"]";

          	//this_thread::sleep_for(chrono::milliseconds(1000));
          	ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
