# Installation
Go to `sam-3d-objects`, `sam3`, `realworld/xArm-Python-SDK`, and `mnet_client` and follow the README installation procedure respectively.

```commandline
pip install pyrealsense2 pybullet
```
There might be several other depend packages you might install

# Running

Specify the `self.api_key` field in `demo.py` if you plan to use GPT-4o API.

And simply run:
```commandline
python demo.py
```

Make sure the xArm and the Realsense camera are connected and the robot IP is set correctly in `realworld/robot_config.py`.

# Contributions
Peter: Worked on real world deployment and SAM3D and GPT4o API system integration. Setup the language-to-SAM3-to-SAM3D pipeline. Integrate the pipeline into the robotics system. Implement the occlusion handling concept. Deploy system on hardware. Conduct real world deployment and tests. Work on final deliverables.

Nick: Worked on TRELLIS (previous deprecated version of the project) code + dataset + fine-tuning. Collaboratively setup the language-to-SAM3-to-SAM3D pipeline. Setup mnet_client evaluation and baselines. Setup real world deployment and tests. Work on final deliverables.