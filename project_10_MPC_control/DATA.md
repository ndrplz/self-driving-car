# Websocket Data

This document describes the JSON object send back from the simulator command server.

Fields:

* `ptsx` (Array<float>) - The global x positions of the waypoints.
* `ptsy` (Array<float>) - The global y positions of the waypoints. This corresponds to the z coordinate in Unity
since y is the up-down direction.
* `psi` (float) - The orientation of the vehicle in **radians** converted from the Unity format to the standard format expected in most mathemetical functions (more details below).
* `psi_unity` (float) - The orientation of the vehicle in **radians**. This is an orientation commonly used in [navigation](https://en.wikipedia.org/wiki/Polar_coordinate_system#Position_and_navigation).
* `x` (float) - The global x position of the vehicle.
* `y` (float) - The global y position of the vehicle.
* `steering_angle` (float) - The current steering angle in **radians**.
* `throttle` (float) - The current throttle value [-1, 1].
* `speed` (float) - The current velocity in **mph**.


### `psi` and `psi_unity` representations

`psi`

```
//            90
//
//  180                   0/360
//
//            270
```


`psi_unity`

```
//            0/360
//
//  270                   90
//
//            180
```

