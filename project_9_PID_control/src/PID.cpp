#include "PID.h"


using namespace std;


PID::PID() {
    error_proportional_ = 0.0;
    error_integral_     = 0.0;
    error_derivative_   = 0.0;
}


PID::~PID() {}


void PID::Init(double Kp, double Ki, double Kd) {
    Kp_ = Kp;
    Ki_ = Ki;
    Kd_ = Kd;
}


void PID::UpdateError(double cte) {
    error_integral_     += cte;
    error_derivative_    = cte - error_proportional_;
    error_proportional_  = cte;
}


double PID::TotalError() {
    return -(Kp_ * error_proportional_ + Ki_ * error_integral_ + Kd_ * error_derivative_);
}

