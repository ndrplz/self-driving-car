# CarND-Controls-PID
This projects implements a PID controller for keeping the car on track by appropriately adjusting the steering angle.

## Overview
<table style="width:100%">
  <tr>
    <th>
      <p align="center">
       <img src="./img/control_none.gif" alt="Overview" width="80%">
      </p>
    </th>
    <th>
      <p align="center">
       <img src="./img/control_p.gif" alt="Overview" width="80%">
      </p>
    </th>
    <th>
      <p align="center">
       <img src="./img/control_pid.gif" alt="Overview" width="80%">
      </p>
    </th>
  </tr>
  </table>
  
#### What's a PID controller

A **proportional–integral–derivative controller** (PID controller) is one of the most common control loop feedback mechanisms. A PID controller continuously calculates an **error function** (which in our case is the distance from the center of the lane) and applies a correction based on proportional (P), integral (I), and derivative (D) terms.

#### Choosing PID Parameters
The behavior of a PID controller depends on three main parameters, namely the **proportional**, **integral** and **derivative gain**. Each one of these three parameters controls the strenght of the respective controller's response. In particular:
1. **Proportional gain** regulates how large the change in the output will be for a given change in the error. If the proportional gain is too high, the system can become unstable (see *p controller* gif above).
2. **Integral gain** contributes in proportion to both the magnitude of the error and the duration of the error. In this way controller is able to eliminate the residual steady-state error that occurs with a pure proportional controller (*i.e.* a purely proportional controller operates only when error is non-zero) and is able to deal with systematic biases.
3. **Derivative gain** decides how much the error's rate of change is taken into account when computing the response. In other words, if the desired setpoint is getting closer (= error is decreasing) the response must be smoothed in order not to overshoot the target. Derivative component benefits the system's stability and settling time.

In the current project, parameters have been manually tuned by qualitatively inspecting the driving behaviour in the simulator in response to parameter's changes. Parameter's validation could also be easily performed automatically in a simulator in which headless mode was available.

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
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!

More information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/f1820894-8322-4bb3-81aa-b26b3c6dcbaf/lessons/e8235395-22dd-4b87-88e0-d108c5e5bbf4/concepts/6a4d8d42-6a04-4aa6-b284-1697c0fd6562)
for instructions and the project rubric.


