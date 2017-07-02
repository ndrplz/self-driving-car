#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"


using CppAD::AD;

size_t N = 10;  // how many timesteps into the future
double dt = .1; // duration

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// References values that will be included in objective function
double ref_cte  = 0;
double ref_epsi = 0;
double ref_v    = 100;

// Indexes on the 1D vector (for readability)
size_t x_start      = 0;
size_t y_start      = x_start       + N;
size_t psi_start    = y_start       + N;
size_t v_start      = psi_start     + N;
size_t cte_start    = v_start       + N;
size_t epsi_start   = cte_start     + N;
size_t delta_start  = epsi_start    + N;
size_t a_start      = delta_start   + N - 1;


class FG_eval {
public:
    // Fitted polynomial coefficients
    Eigen::VectorXd coeffs;

    FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

    typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

    void operator()(ADvector &fg, const ADvector &vars) {
        // `fg` a vector of the cost constraints
        // `vars` is a vector of variable values (state & actuators)

        fg[0] = 0;

        // Objective term 1: Keep close to reference values
        for (size_t i = 0; i < N; ++i) {
            fg[0] += 2000 * CppAD::pow(vars[cte_start  + i] - ref_cte,  2);
            fg[0] += 2000 * CppAD::pow(vars[epsi_start + i] - ref_epsi, 2);
            fg[0] += 100  * CppAD::pow(vars[v_start    + i] - ref_v,    2);
        }

        // Objective term 2:  Avoid to actuate, as much as possible
        for (size_t i = 0; i < N - 1; ++i) {
            fg[0] += 5 * CppAD::pow(vars[delta_start + i], 2);
            fg[0] += 5 * CppAD::pow(vars[a_start     + i], 2);
        }

        // Objective term 3:  Enforce actuators smoothness in change
        for (size_t i = 0; i < N - 2; ++i) {
            fg[0] += 200 * CppAD::pow(vars[delta_start + i + 1] - vars[delta_start + i], 2);
            fg[0] += 10  * CppAD::pow(vars[a_start     + i + 1] - vars[a_start     + i], 2);
        }

        // Initial constraints
        fg[1 + x_start]     = vars[x_start];
        fg[1 + y_start]     = vars[y_start];
        fg[1 + psi_start]   = vars[psi_start];
        fg[1 + v_start]     = vars[v_start];
        fg[1 + cte_start]   = vars[cte_start];
        fg[1 + epsi_start]  = vars[epsi_start];

        for (size_t i = 0; i < N - 1; ++i) {

            // Values at time (t)
            AD<double> x_0      = vars[x_start    + i];
            AD<double> y_0      = vars[y_start    + i];
            AD<double> psi_0    = vars[psi_start  + i];
            AD<double> v_0      = vars[v_start    + i];
            AD<double> cte_0    = vars[cte_start  + i];
            AD<double> epsi_0   = vars[epsi_start + i];

            // Values at time (t+1)
            AD<double> x_1      = vars[x_start    + i + 1];
            AD<double> y_1      = vars[y_start    + i + 1];
            AD<double> psi_1    = vars[psi_start  + i + 1];
            AD<double> v_1      = vars[v_start    + i + 1];
            AD<double> cte_1    = vars[cte_start  + i + 1];
            AD<double> epsi_1   = vars[epsi_start + i + 1];

            AD<double> delta_0  = vars[delta_start + i];
            AD<double> a_0      = vars[a_start     + i];

            AD<double> f_0 = coeffs[0] + \
                             coeffs[1] * x_0 + \
                             coeffs[2] * x_0 * x_0 + \
                             coeffs[3] * x_0 * x_0 * x_0;

            AD<double> psides_0 = CppAD::atan(coeffs[1] + 2 * coeffs[2] * x_0 + 3 * coeffs[3] * x_0 * x_0);

            // Setup other model constraints
            fg[2 + x_start + i]     = x_1    - (x_0 + v_0 * CppAD::cos(psi_0) * dt);
            fg[2 + y_start + i]     = y_1    - (y_0 + v_0 * CppAD::sin(psi_0) * dt);
            fg[2 + psi_start + i]   = psi_1  - (psi_0 - v_0 * delta_0 / Lf * dt);
            fg[2 + v_start + i]     = v_1    - (v_0 + a_0 * dt);
            fg[2 + cte_start + i]   = cte_1  - (f_0 - y_0 + (v_0 + CppAD::sin(epsi_0) * dt));
            fg[2 + epsi_start + i]  = epsi_1 - (psi_0 - psides_0 - v_0 * delta_0 / Lf * dt);
        }
    }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}

MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {

    bool ok = true;
    typedef CPPAD_TESTVECTOR(double) Dvector;

    // Explicitly gather state values for readability
    double x    = state[0];
    double y    = state[1];
    double psi  = state[2];
    double v    = state[3];
    double cte  = state[4];
    double epsi = state[5];

    // Set the number of model variables (includes both states and inputs).
    size_t n_vars = N * 6 + (N - 1) * 2;

    // Set the number of constraints
    size_t n_constraints = N * 6;

    // Initial all of independent variables to zero.
    Dvector vars(n_vars);
    for (size_t i = 0; i < n_vars; i++) {
        vars[i] = 0;
    }

    Dvector vars_lowerbound(n_vars);
    Dvector vars_upperbound(n_vars);

    // Set limits for non-actuators (avoid numerical issues during optimization)
    for (size_t i = 0; i < delta_start; ++i) {
        vars_lowerbound[i] = numeric_limits<float>::min();
        vars_upperbound[i] = numeric_limits<float>::max();
    }

    // Set upper and lower constraints for steering
    double max_degree   = 25;
    double max_radians  = max_degree * M_PI / 180;
    for (size_t i = delta_start; i < a_start; ++i) {
        vars_lowerbound[i] = - max_radians * Lf;
        vars_upperbound[i] = + max_radians * Lf;
    }

    // Set upper and lower constraints for acceleration
    double max_acceleration_value  = 1.0;
    for (size_t i = a_start; i < n_vars; ++i) {
        vars_lowerbound[i] = - max_acceleration_value;
        vars_upperbound[i] = + max_acceleration_value;
    }

    // Initialize to zero lower and upper limits for the constraints
    Dvector constraints_lowerbound(n_constraints);
    Dvector constraints_upperbound(n_constraints);
    for (size_t i = 0; i < n_constraints; ++i) {
        constraints_lowerbound[i] = 0;
        constraints_upperbound[i] = 0;
    }

    // Force the solver to start from current state in optimization space
    constraints_lowerbound[x_start] = x;        constraints_upperbound[x_start] = x;
    constraints_lowerbound[y_start] = y;        constraints_upperbound[y_start] = y;
    constraints_lowerbound[psi_start] = psi;    constraints_upperbound[psi_start] = psi;
    constraints_lowerbound[v_start] = v;        constraints_upperbound[v_start] = v;
    constraints_lowerbound[cte_start] = cte;    constraints_upperbound[cte_start] = cte;
    constraints_lowerbound[epsi_start] = epsi;  constraints_upperbound[epsi_start] = epsi;

    // object that computes objective and constraints
    FG_eval fg_eval(coeffs);

    //
    // NOTE: You don't have to worry about these options
    //
    // options for IPOPT solver
    std::string options;
    // Uncomment this if you'd like more print information
    options += "Integer print_level  0\n";
    // NOTE: Setting sparse to true allows the solver to take advantage
    // of sparse routines, this makes the computation MUCH FASTER. If you
    // can uncomment 1 of these and see if it makes a difference or not but
    // if you uncomment both the computation time should go up in orders of
    // magnitude.
    options += "Sparse  true        forward\n";
    options += "Sparse  true        reverse\n";
    // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
    // Change this as you see fit.
    options += "Numeric max_cpu_time          0.5\n";

    // place to return solution
    CppAD::ipopt::solve_result<Dvector> solution;

    // solve the problem
    CppAD::ipopt::solve<Dvector, FG_eval>(
            options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
            constraints_upperbound, fg_eval, solution);

    // Check some of the solution values
    ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

    // Cost
    auto cost = solution.obj_value;
    std::cout << "Cost " << cost << std::endl;

    // Return the first actuator values. The variables can be accessed with `solution.x[i]`.
    vector<double> result;
    result.push_back(solution.x[delta_start]);
    result.push_back(solution.x[a_start]);

    // Add "future" solutions (where MPC is going)
    for (size_t i = 0; i < N - 1; ++i) {
        result.push_back(solution.x[x_start + i + 1]);
        result.push_back(solution.x[y_start + i + 1]);
    }

    return result;
}
