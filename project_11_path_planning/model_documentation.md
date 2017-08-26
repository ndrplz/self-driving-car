# Model Documentation

This file contains the documentation for the **Udacity Path-Planning Project**.

## Task Description

The goal in this project is to **build a path planner that is able to create smooth, safe trajectories** for the car to follow. The highway track has other vehicles, all going different speeds, but approximately obeying the 50 MPH speed limit.

The car transmits its location, along with its sensor fusion data, which estimates the location of all the vehicles on the same side of the road.

## What is the Path

A **trajectory** is constituted by as set of (x, y) points. Every 20 ms the car moves to the next point on the list.
In this framework, the car moves from point to point perfectly.

Since the car always moves to the next waypoint after 20ms, the **velocity** of the vehicle is given by how much future waypoints are spaced.
In other words, the larger the space between waypoints, the faster the car will be.

## Rubric

Here's a bullet list of rubric points along with an explanation of how constraints are satisfied.

- **The car drives according to the speed limit**

    A [reference velocity variable](https://github.com/ndrplz/self-driving-car/blob/7b0d9e057931667a04422649fee6d017b3ef7475/project_11_path_planning/src/main.cpp#L74) is used to keep track of current speed of the vehicle. Also, [another constant variable](https://github.com/ndrplz/self-driving-car/blob/7b0d9e057931667a04422649fee6d017b3ef7475/project_11_path_planning/src/utils.h#L13) is used to make sure the car never exeed the maximum speed of 50 mph.

- **Max acceleration and jerk are not exceeded.**
    
    The rubric sets as hard constraint the fact that car does not exceed a total acceleration of 10 m/s^2 and a jerk of 10 m/s^3. This is obtained by setting the initial speed of vehicle to zero, and [gradually accelerating](https://github.com/ndrplz/self-driving-car/blob/7b0d9e057931667a04422649fee6d017b3ef7475/project_11_path_planning/src/main.cpp#L181-L182) until the vehicle has reached the maximum speed allowed (either by the speed limit or by the current road situation).

- **Car does not have collisions.**

    Under the reasonable assumption that other cars drive reasonably well, the main risk to hit another car is *rear-ending*. This is avoided using [sensor fusion data to make sure that no other car is dangerously close](https://github.com/ndrplz/self-driving-car/blob/7b0d9e057931667a04422649fee6d017b3ef7475/project_11_path_planning/src/main.cpp#L133-L134) in our same lane. If this is the case, the car [gracefully decelerates](https://github.com/ndrplz/self-driving-car/blob/7b0d9e057931667a04422649fee6d017b3ef7475/project_11_path_planning/src/main.cpp#L179-L180) until the situation gets safer.
    
    Another possibly dangerous situation is the one of *lane changing*. This will be examined in the next point.
    
- **The car stays in its lane, except for the time between changing lanes.**
    
    One of the first things that we would demand from a self-driving car is being able to drive inside a certain lane. In our framework, we know that each lane is 4m width, and that there are three lanes for each direction of travel.
    
    For the task of lane keeping we'll rely on *Frenet coordinates*, which measure longitudinal (s) and lateral (d) motion along the road and are much easier to deal with w.r.t. euclidean ones. By the way, the code for converting between Eunclidean and Frenet coordinates is in [`coords_transform.h`](https://github.com/ndrplz/self-driving-car/blob/master/project_11_path_planning/src/coords_transform.h).
    
    In order for the car to keep the lane, [three future waypoints are generated equally spaced 30m, 60m and 90m before the car](https://github.com/ndrplz/self-driving-car/blob/50adb2c54ac2e0d5c0878f4c3c73894275ab20c3/project_11_path_planning/src/main.cpp#L215-L218) respectively. These are quite far apart one from the other, so in order to obtain a smooth trajectory a [spline](https://github.com/ndrplz/self-driving-car/blob/611428a30cc28b958dd38a2b3a837897b2a0c0b7/project_11_path_planning/src/main.cpp#L232-L233) is used to interpolate intermediate locations.
    
    Notice that these waypoints are created [using the `lane` variable](https://github.com/ndrplz/self-driving-car/blob/50adb2c54ac2e0d5c0878f4c3c73894275ab20c3/project_11_path_planning/src/main.cpp#L215-L218) to make sure that the `d` coordinate is just in the middle of the lane the car is currently on. In case a lane change is needed, the `lane` variable will already contain the coordinate of the future lane (see below). 

- **The car is able to change lanes**
    
    When it makes sense to do, the vehicle will try to change lane. This will happens if a slower moving car obstructs traffic lane. A simple **FSM** (Finite State Machine) able to deal with safe lane changing is implemented at [these lines](https://github.com/ndrplz/self-driving-car/blob/611428a30cc28b958dd38a2b3a837897b2a0c0b7/project_11_path_planning/src/main.cpp#L119-L182) of `main.cpp`.
    
    Basically, our car uses sensor fusion data to collect information about nearby vehicles. Vehicles which travel on our same lane are of [particular interest](https://github.com/ndrplz/self-driving-car/blob/611428a30cc28b958dd38a2b3a837897b2a0c0b7/project_11_path_planning/src/main.cpp#L133-L139), in particular when they get dangerously close. In this latter case, the vehicle attempts a lane change maneuver. 
    
    The car has now entered the state `prepare_for_lane_change`. However, in order to safely change lane, the car has to check if at least another lane is sufficiently clear of traffic for allowing a safe lane change. This controls are implemented [here](https://github.com/ndrplz/self-driving-car/blob/611428a30cc28b958dd38a2b3a837897b2a0c0b7/project_11_path_planning/src/main.cpp#L143-L170). If at least one lane free is found, the flag [`ready_for_lane_change` is set](https://github.com/ndrplz/self-driving-car/blob/611428a30cc28b958dd38a2b3a837897b2a0c0b7/project_11_path_planning/src/main.cpp#L166-L167) to signal that a safe lane changing is now possible.
    
    The actual lane change happen by changing the `lane` variable to the new value. This takes place [here](https://github.com/ndrplz/self-driving-car/blob/611428a30cc28b958dd38a2b3a837897b2a0c0b7/project_11_path_planning/src/main.cpp#L172-L176). As we saw before, the `lane` variable is used for computing the next waypoints. This implies that the future trajectory will drive the car in the new lane.
