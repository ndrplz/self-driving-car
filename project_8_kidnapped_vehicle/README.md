# Overview
This repository contains all the code needed to complete the final project for the Localization course in Udacity's Self-Driving Car Nanodegree.

#### Submission
All you will submit is your completed version of `particle_filter.cpp`, which is located in the `src` directory. You should probably do a `git pull` before submitting to verify that your project passes the most up-to-date version of the grading code (there are some parameters in `src/main.cpp` which govern the requirements on accuracy and run time.)

## Project Introduction
Your robot has been kidnapped and transported to a new location! Luckily it has a map of this location, a (noisy) GPS estimate of its initial location, and lots of (noisy) sensor and control data.

In this project you will implement a 2 dimensional particle filter in C++. Your particle filter will be given a map and some initial localization information (analogous to what a GPS would provide). At each time step your filter will also get observation and control data. 

## Running the Code
Once you have this repository on your machine, `cd` into the repository's root directory and run the following commands from the command line:

```
> ./clean.sh
> ./build.sh
> ./run.sh
```

> **NOTE**
> If you get any `command not found` problems, you will have to install 
> the associated dependencies (for example, 
> [cmake](https://cmake.org/install/))

If everything worked you should see something like the following output:

Time step: 2444
Cumulative mean weighted error: x .1 y .1 yaw .02
Runtime (sec): 38.187226
Success! Your particle filter passed!

```
Otherwise you might get
.
.
.
Time step: 100
Cumulative mean weighted error: x 39.8926 y 9.60949 yaw 0.198841
Your x error, 39.8926 is larger than the maximum allowable error, 1
```

Your job is to build out the methods in `particle_filter.cpp` until the last line of output says:

```
Success! Your particle filter passed!
```

# Implementing the Particle Filter
The directory structure of this repository is as follows:

```
root
|   build.sh
|   clean.sh
|   CMakeLists.txt
|   README.md
|   run.sh
|
|___data
|   |   control_data.txt
|   |   gt_data.txt
|   |   map_data.txt
|   |
|   |___observation
|       |   observations_000001.txt
|       |   ... 
|       |   observations_002444.txt
|   
|___src
    |   helper_functions.h
    |   main.cpp
    |   map.h
    |   particle_filter.cpp
    |   particle_filter.h
```

The only file you should modify is `particle_filter.cpp` in the `src` directory. The file contains the scaffolding of a `ParticleFilter` class and some associated methods. Read through the code, the comments, and the header file `particle_filter.h` to get a sense for what this code is expected to do.

If you are interested, take a look at `src/main.cpp` as well. This file contains the code that will actually be running your particle filter and calling the associated methods.

## Inputs to the Particle Filter
You can find the inputs to the particle filter in the `data` directory. 

#### The Map*
`map_data.txt` includes the position of landmarks (in meters) on an arbitrary Cartesian coordinate system. Each row has three columns
1. x position
2. y position
3. landmark id

> * Map data provided by 3D Mapping Solutions GmbH.


#### Control Data
`control_data.txt` contains rows of control data. Each row corresponds to the control data for the corresponding time step. The two columns represent
1. vehicle speed (in meters per second)
2. vehicle yaw rate (in radians per second)

#### Observation Data
The `observation` directory includes around 2000 files. Each file is numbered according to the timestep in which that observation takes place. 

These files contain observation data for all "observable" landmarks. Here observable means the landmark is sufficiently close to the vehicle. Each row in these files corresponds to a single landmark. The two columns represent:
1. x distance to the landmark in meters (right is positive) RELATIVE TO THE VEHICLE. 
2. y distance to the landmark in meters (forward is positive) RELATIVE TO THE VEHICLE.

> **NOTE**
> The vehicle's coordinate system is NOT the map coordinate system. Your 
> code will have to handle this transformation.

## Success Criteria
If your particle filter passes the current grading code (you can make sure you have the current version at any time by doing a `git pull`), then you should pass! 

The two things the grading code is looking for are:

1. **Accuracy**: your particle filter should localize vehicle position and yaw to within the values specified in the parameters `max_translation_error` (maximum allowed error in x or y) and `max_yaw_error` in `src/main.cpp`.
2. **Performance**: your particle filter should complete execution within the time specified by `max_runtime` in `src/main.cpp`.



