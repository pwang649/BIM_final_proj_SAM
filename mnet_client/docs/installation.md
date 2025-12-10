# Installation

This page provides detailed instructions to install the ROS 2 version of the mnet-client.

mnet-client is distributed as a ROS 1/2 package via source code.



## Prerequisites

* Ubuntu ([https://ubuntu.com/](https://ubuntu.com/)) and ROS 2 ([https://www.ros.org/](https://www.ros.org/))

* FFmpeg ([https://ffmpeg.org/](https://ffmpeg.org/)) with [x264 encoder](https://trac.ffmpeg.org/wiki/Encode/H.264) for video processing

  ```shell
  sudo apt install ffmpeg libavcodec-dev libavformat-dev libswscale-dev libx264-dev
  ```

* OpenCV ([https://opencv.org/](https://opencv.org/)) with FFmpeg compatible

  ```shell
  pip install opencv-python # or sudo apt install python3-opencv
  ```

  **Notice**: if you installed OpenCV before FFmpeg was available on your system, then OpenCV was built without FFmpeg support. To make OpenCV actually use FFmpeg, you’ll need to **reinstall or rebuild OpenCV after FFmpeg is installed**.

* Connect ROS 2 with OpenCV

  ```shell
  sudo apt install ros-<distro>-cv-bridge \
                   ros-<distro>-vision-opencv \
                   ros-<distro>-image-transport \
                   ros-<distro>-compressed-image-transport \
                   ros-<distro>-image-common
  ```

* [Requests](https://requests.readthedocs.io/en/latest/) for HTTP requests

  ```shell
  pip install requests
  ```

* [Pydantic](https://docs.pydantic.dev/latest/) (>=2.0) for message transfer

  ```
  pip install pydantic>=2.0
  ```

* [PyBullet](https://pybullet.org/wordpress/) (3.2.7 recommended, other versions could also work) for scene rendering

  ```
  pip install pybullet
  ```

* [pupil-apriltags](https://pypi.org/project/pupil-apriltags/) (Python bindings for the [apriltags3](https://april.eecs.umich.edu/software/apriltag) library)

  ```shell
  pip install pupil-apriltags
  ```

  

## Install via Source Code

Enter your ROS 2 workspace:

```sh
cd ros2_ws/src
```

Clone the repository in your ROS 2 Workspace

```
git clone --branch ros_2 https://github.com/ManipulationNet/mnet_client.git --recursive
```

Compile the ROS 2 workspace

```
cd ..
colcon build
```



## Update your Client

Enter your mnet-client package and do:

```
git pull
git submodule update --init --recursive 
```

