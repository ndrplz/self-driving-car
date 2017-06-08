#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {
    error_current_    = 0.0;
    error_derivative_ = 0.0;
    error_total_      = 0.0;
}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    Kp_ = Kp;
    Ki_ = Ki;
    Kd_ = Kd;
}

void PID::UpdateError(double cte) {
    error_derivative_ = cte - error_current_;
    error_total_     += cte;
    error_current_    = cte;
}

double PID::ErrorIntegral() {
    return error_total_;
}

double PID::ErrorDerivative() {
    return error_derivative_;
}

