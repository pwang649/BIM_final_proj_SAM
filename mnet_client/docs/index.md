# Welcome to ManipulationNet

For more details about ManipulationNet, please visit [manipulation-net.org](https://manipulation-net.org/). For any questions, please contact <u>support@manipulation-net.org</u>.



## Overview

**ManipulationNet delivers standardized task setups:** Each task setup is defined by a physical object set and a task protocol that governs its use. ManipulationNet designs, fabricates, and distributes task setups to registered teams worldwide. Available tasks are listed [here](https://manipulation-net.org/index.html#tasks).

**ManipulationNet evaluates authentic task performance:** ManipulationNet decouples performance collection from result verification. Manipulation performance is collected decentrally via the **<u>mnet-client</u>** with time and location flexibility, while final verification occurs centrally by the organizing committee for comparable results.



This document introduces how to properly use the **<u>mnet-client</u>** to participate in each hosted benchmark task under ManipulationNet.



## What is mnet-client?

The mnet-client is a **middle layer** between the **robotic system** and the **mnet-server** to support distributed manipulation benchmarking on standardized task setups. The robotic system communicates with the mnet-client through ROS services and topics. In general, the mnet-client is responsible for: 

1. **collect** authentic manipulation performance on standardized task setups and upload it to the server for comparable research;

2. **deliver** task instructions from the server to the robotic system in real-time. This could involve language/visual prompts, task-specific instructions, and more;

3. **report** task execution and human intervention logs from the robotic system to the server in real-time to better describe the manipulation performance.

   

This document introduces the ROS 2 version of mnet-client; for the ROS 1 version, please refer to [here](https://mnet-client.readthedocs.io/ros_1/).

```{toctree}
:maxdepth: 2
:caption: Contents

installation.md
general.md
local_test.md
connection_test.md
submission.md

```
