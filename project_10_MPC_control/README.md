# CarND-Controls-MPC
Self-Driving Car Engineer Nanodegree Program

---

## Project Description

### Overview

**Model predictive control (MPC)** is an advanced method of process control which relies on dynamic models of the process.
Differently from previously implemented [PID controller](https://github.com/ndrplz/self-driving-car/tree/master/project_9_PID_control), MPC controller has the ability to anticipate future events and can take control actions accordingly. Indeed, future time steps are taking into account while optimizing current time slot.

The MPC controller framework consists in four main components:
 - **Trajectory** taken in consideration during optimization. This is parametrized by a number of time steps ***N*** spaced out by a time ***dt***. Clearly, the number of variables optimized is directly proportional to *N*, so this must be considered in case there are computational constraints.
 
 - **Vehicle Model**, which is the set of equations that describes system behavior and updates across time steps. In our case, we used a simplified kinematic model (so called *bycicle model*) described by a state of six parameters:
   - **x** car position (*x-axis*)
   - **y** car position (*y-axis*)
   - **psi** car's heading direction
   - **v** car's velocity
   - **cte** cross-track error
   - **epsi** orientation error
   
   Vehicle model update equations are implemented at lines 117-123 in [MPC.cpp](https://github.com/ndrplz/self-driving-car/blob/master/project_10_MPC_control/src/MPC.cpp).
   
 - **Contraints** necessary to model contrants in actuators' respose. For instance, a vehicle will never be able to steer 90 deegrees in a single time step. In this project we set these constraints as follows:
   - **steering**: bounded in range [-25°, 25°]
   - **acceleration**: bounded in range [-1, 1] from full brake to full throttle
   
 - **Cost Function** on whose optimization is based the whole control process. Usually cost function is made of the sum of different terms. Besides the main terms that depends on reference values (*e.g.* cross-track or heading error), other regularization terms are present to enforce the smoothness in the controller response (*e.g.* avoid abrupt steering).
 
   In this project the cost function is implemented at lines 54-79 in [MPC.cpp](https://github.com/ndrplz/self-driving-car/blob/master/project_10_MPC_control/src/MPC.cpp).
   
### Tuning Trajectory Parameters

Both ***N*** and ***dt*** are fundamental parameters in the optimization process. In particular, ***T = N * dt*** constitutes the *prediction horizon* considered during optimization. These values have to be tuned keeping in mind a couple of things:
  - large *dt* result in less frequent actuations, which in turn could result in the difficulty in following a continuous reference trajectory (so called *discretization error*) 
  - despite the fact that having a large *T* could benefit the control process, consider that predicting too far in the future does not make sense in real-world scenarios.
  - large *T* and small *dt* lead to large *N*. As mentioned above, the number of variables optimized is directly proportional to *N*, so will lead to an higher computational cost.

In the current project I empirically set (by visually inspecting the vehicle's behavior in the simulator) these parameters to be ***N=10*** and ***dt=0.1***, for a total of ***T=1s*** in the future. 

### Changing Reference System

Simulator provides coordinates in global reference system. In order to ease later computation, these are converted into car's own reference system at lines 94-102 in [main.cpp](https://github.com/ndrplz/self-driving-car/blob/master/project_10_MPC_control/src/main.cpp).

### Dealing with Latency

To mimic real driving conditions where the car does actuate the commands instantly, a *100ms* latency delay has been introduced before sending the data message to the simulator (line 185 in [main.cpp](https://github.com/ndrplz/self-driving-car/blob/master/project_10_MPC_control/src/main.cpp)). In order to deal with latency, state is predicted one time step ahead before feeding it to the solver (lines 125-132 in [main.cpp](https://github.com/ndrplz/self-driving-car/blob/master/project_10_MPC_control/src/main.cpp)).

---

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Fortran Compiler
  * Mac: `brew install gcc` (might not be required)
  * Linux: `sudo apt-get install gfortran`. Additionall you have also have to install gcc and g++, `sudo apt-get install gcc g++`. Look in [this Dockerfile](https://github.com/udacity/CarND-MPC-Quizzes/blob/master/Dockerfile) for more info.
* [Ipopt](https://projects.coin-or.org/Ipopt)
  * Mac: `brew install ipopt`
  * Linux
    * You will need a version of Ipopt 3.12.1 or higher. The version available through `apt-get` is 3.11.x. If you can get that version to work great but if not there's a script `install_ipopt.sh` that will install Ipopt. You just need to download the source from the Ipopt [releases page](https://www.coin-or.org/download/source/Ipopt/) or the [Github releases](https://github.com/coin-or/Ipopt/releases) page.
    * Then call `install_ipopt.sh` with the source directory as the first argument, ex: `bash install_ipopt.sh Ipopt-3.12.1`. 
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [CppAD](https://www.coin-or.org/CppAD/)
  * Mac: `brew install cppad`
  * Linux `sudo apt-get install cppad` or equivalent.
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./mpc`.

## Tips

1. It's recommended to test the MPC on basic examples to see if your implementation behaves as desired. One possible example
is the vehicle starting offset of a straight line (reference). If the MPC implementation is correct, after some number of timesteps
(not too many) it should find and track the reference line.
2. The `lake_track_waypoints.csv` file has the waypoints of the lake track. You could use this to fit polynomials and points and see of how well your model tracks curve. NOTE: This file might be not completely in sync with the simulator so your solution should NOT depend on it.
3. For visualization this C++ [matplotlib wrapper](https://github.com/lava/matplotlib-cpp) could be helpful.

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!

More information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/f1820894-8322-4bb3-81aa-b26b3c6dcbaf/lessons/b1ff3be0-c904-438e-aad3-2b5379f0e0c3/concepts/1a2255a0-e23c-44cf-8d41-39b8a3c8264a)
for instructions and the project rubric.

