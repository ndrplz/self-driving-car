#include <math.h>
#include <uWS/uWS.h>
#include <thread>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }

double deg2rad(double x) { return x * pi() / 180; }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
    auto found_null = s.find("null");
    auto b1 = s.find_first_of("[");
    auto b2 = s.rfind("}]");
    if (found_null != string::npos) {
        return "";
    } else if (b1 != string::npos && b2 != string::npos) {
        return s.substr(b1, b2 - b1 + 2);
    }
    return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
    double result = 0.0;
    for (int i = 0; i < coeffs.size(); i++) {
        result += coeffs[i] * pow(x, i);
    }
    return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
    assert(xvals.size() == yvals.size());
    assert(order >= 1 && order <= xvals.size() - 1);
    Eigen::MatrixXd A(xvals.size(), order + 1);

    for (int i = 0; i < xvals.size(); i++) {
        A(i, 0) = 1.0;
    }

    for (int j = 0; j < xvals.size(); j++) {
        for (int i = 0; i < order; i++) {
            A(j, i + 1) = A(j, i) * xvals(j);
        }
    }

    auto Q = A.householderQr();
    auto result = Q.solve(yvals);
    return result;
}

int main() {
    uWS::Hub h;

    // MPC is initialized here!
    MPC mpc;

    h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                       uWS::OpCode opCode) {
        // "42" at the start of the message means there's a websocket message event.
        // The 4 signifies a websocket message
        // The 2 signifies a websocket event
        string sdata = string(data).substr(0, length);
        cout << sdata << endl;
        if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
            string s = hasData(sdata);
            if (s != "") {
                auto j = json::parse(s);
                string event = j[0].get<string>();
                if (event == "telemetry") {
                    // j[1] is the data JSON object
                    vector<double> ptsx = j[1]["ptsx"];
                    vector<double> ptsy = j[1]["ptsy"];
                    double px = j[1]["x"];
                    double py = j[1]["y"];
                    double psi = j[1]["psi"];
                    double v = j[1]["speed"];
                    double delta = j[1]["steering_angle"];
                    double a = j[1]["throttle"];

                    // Rotate and shift such that new reference system is centered on the origin @ 0 degrees
                    for (size_t i = 0; i < ptsx.size(); ++i) {

                        double shift_x = ptsx[i] - px;
                        double shift_y = ptsy[i] - py;

                        ptsx[i] = shift_x * cos(-psi) - shift_y * sin(-psi);
                        ptsy[i] = shift_x * sin(-psi) + shift_y * cos(-psi);
                    }

                    // Convert to Eigen::VectorXd
                    double *ptrx = &ptsx[0];
                    Eigen::Map<Eigen::VectorXd> ptsx_transform(ptrx, 6);

                    double *ptry = &ptsy[0];
                    Eigen::Map<Eigen::VectorXd> ptsy_transform(ptry, 6);

                    // Fit coefficients of third order polynomial
                    auto coeffs = polyfit(ptsx_transform, ptsy_transform, 3);

                    double cte = polyeval(coeffs, 0);

                    // before reference system change: double epsi = psi - atan(coeffs[1] + 2*px*coeffs[2] + 3*coeffs[3] * pow(px,2));
                    //double epsi = psi - atan(coeffs[1] + 2*px*coeffs[2] + 3*coeffs[3] * pow(px,2));
                    double epsi = -atan(coeffs[1]);

                    // Latency for predicting time at actuation
                    const double dt = 0.1;

                    const double Lf = 2.67;

                    // Predict future state (take latency into account)
                    // x, y and psi are all zero in the new reference system
                    double pred_px        = 0.0 + v * dt;               // psi is zero, cos(0) = 1, can leave out
                    const double pred_py  = 0.0;                        // sin(0) = 0, y stays as 0 (y + v * 0 * dt)
                    double pred_psi       = 0.0 + v * -delta / Lf * dt;
                    double pred_v         = v + a * dt;
                    double pred_cte       = cte + v * sin(epsi) * dt;
                    double pred_epsi      = epsi + v * -delta / Lf * dt;

                    // Feed in the predicted state values
                    Eigen::VectorXd state(6);
                    state << pred_px, pred_py, pred_psi, pred_v, pred_cte, pred_epsi;

                    auto vars = mpc.Solve(state, coeffs);

                    // Display the waypoints / reference line
                    vector<double> next_x_vals;
                    vector<double> next_y_vals;

                    double poly_inc = 2.5; // step on x
                    int num_points = 25;    // how many point "in the future" to be plotted
                    for (int i = 1; i < num_points; ++i) {
                        double future_x = poly_inc * i;
                        double future_y = polyeval(coeffs, future_x);
                        next_x_vals.push_back(future_x);
                        next_y_vals.push_back(future_y);
                    }

                    // Normalize steering angle range [-deg2rad(25), deg2rad(25] -> [-1, 1].

                    const double angle_norm_factor = deg2rad(25) * Lf;
                    double steer_value = vars[0] / angle_norm_factor;
                    double throttle_value = vars[1];

                    //Display the MPC predicted trajectory
                    vector<double> mpc_x_vals;
                    vector<double> mpc_y_vals;

                    for (size_t i = 2; i < vars.size(); ++i) {
                        if (i % 2 == 0) mpc_x_vals.push_back(vars[i]);
                        else            mpc_y_vals.push_back(vars[i]);
                    }

                    // Compose message for simulator client
                    json msgJson;
                    msgJson["steering_angle"]   = steer_value;
                    msgJson["throttle"]         = throttle_value;
                    msgJson["mpc_x"]            = mpc_x_vals;
                    msgJson["mpc_y"]            = mpc_y_vals;
                    msgJson["next_x"]           = next_x_vals;
                    msgJson["next_y"]           = next_y_vals;

                    auto msg = "42[\"steer\"," + msgJson.dump() + "]";
                    std::cout << msg << std::endl;

                    // Latency
                    // The purpose is to mimic real driving conditions where the car does actuate
                    // the commands instantly. Feel free to play around with this value but should
                    // be to drive around the track with 100ms latency.
                    // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE SUBMITTING.
                    this_thread::sleep_for(chrono::milliseconds(100));
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
    // program doesn't compile :-(
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
