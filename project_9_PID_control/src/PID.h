#ifndef PID_H
#define PID_H

class PID {
public:
    /*
    * Errors
    */
    double error_current_;
    double error_total_;
    double error_derivative_;

    /*
    * Coefficients
    */
    double Kp_;
    double Ki_;
    double Kd_;

    /*
    * Constructor
    */
    PID();

    /*
    * Destructor.
    */
    virtual ~PID();

    /*
    * Initialize PID.
    */
    void Init(double Kp, double Ki, double Kd);

    /*
    * Update the PID error variables given cross track error.
    */
    void UpdateError(double cte);

    /*
    * Calculate the total PID error.
    */
    double ErrorIntegral();

    /*
    * Calculate the error derivative
    */
    double ErrorDerivative();


};

#endif /* PID_H */
