# This Docker is for the Jetson(jetpack 6.0.0) users

## Getting Started
You can download docker images at here.
```
docker pull eveeeee/calib_cam_lidar
```
You can use docker_run.sh files for building the image.
```
./docker_run.sh image_name container_name
```
There is an rosbag folder to play the camera and 3d LiDAR data.
```
cd ~/root
ros2 bag play velo_zed2.bag
```
