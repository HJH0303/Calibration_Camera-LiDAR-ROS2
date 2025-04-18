# Camera-3D-LiDAR-extrinsic-calibration-ROS2
The camera and 3D LiDAR extrinic prameter calibration projects with ROS2 humble package.

## Getting Started
* You have to know your [camera intrinic parameters](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
*  You will need to publish 3D LiDAR points and camera images to ROS2.
```
sensor_msgs/msg/Image
sensor_msgs/msg/PointCloud2
```
### Docker usage
Go to the this [docker_usage.md](https://github.com/HJH0303/Camera-3D-LiDAR-extrinsic-calibration-ROS2/blob/main/dcoker_usage.md)

## Installing
```
git clone https://github.com/HJH0303/Camera-3D-LiDAR-extrinsic-calibration-ROS2.git
cd Camera-3D-LiDAR-extrinsic-calibration-ROS2
mv ./calib_pkg ~/your_ws/src
cd ~/your_ws
colcon build
```

## Running calibration node
```
ros2 run calib_pkg calib_extrinics_param.py
```
## Usage
https://github.com/user-attachments/assets/8b300b63-a796-4731-baf8-84469ef2b8ea

## Results
https://github.com/user-attachments/assets/a5c554dc-e2f8-44b4-8676-763501389c32

![image](https://github.com/user-attachments/assets/8f22a4f0-e05f-463f-b5b1-24755e291e33)

## References
[1] Zhou, Lipu, Zimo Li, and Michael Kaess. “Automatic Extrinsic Calibration of a Camera and a 3D LiDAR Using Line and Plane Correspondences.” In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 5562–69. Madrid: IEEE, 2018. https://doi.org/10.1109/IROS.2018.8593660.

[2] [Matlab LiDAR Toolbox](https://ww2.mathworks.cn/help/lidar/ug/lidar-and-camera-calibration.html)

