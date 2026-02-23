# 🤖 Team Vulcans — ABU Robocon 2026: *Kung Fu Quest*

[![ROS 2 Humble](https://img.shields.io/badge/ROS%202-Humble-blue)](https://docs.ros.org/en/humble/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Viraj281105/Team-Vulcans-Robocon-2026/blob/main/LICENSE)
![MATLAB](https://img.shields.io/badge/MATLAB-73.7%25-orange)
![Python](https://img.shields.io/badge/Python-25.9%25-blue)
![Shell](https://img.shields.io/badge/Shell-0.4%25-lightgrey)
![GitHub stars](https://img.shields.io/github/stars/Viraj281105/Team-Vulcans-Robocon-2026?style=social)

Welcome to the official source code repository for **Team Vulcans'** entry into the **ABU Robocon 2026** competition, *"Kung Fu Quest"*, hosted in **Hong Kong**.

This repository contains all the software, simulation models, and AI components for our two robots. Our project focuses on developing a **robust and collaborative robotic system** with a strong emphasis on:

- Autonomous navigation
- Real-time computer vision for object recognition
- Intelligent strategic decision-making

---

## 🔧 Tech Stack

Our system is built on an industry-standard robotics stack for reliability and performance.

| Category               | Technology                               |
| ---------------------- | ---------------------------------------- |
| **Operating System**   | Ubuntu 22.04 LTS *"Jammy Jellyfish"*     |
| **Robotics Framework** | ROS 2 Humble Hawksbill (LTS)             |
| **Simulation**         | Gazebo 11                                |
| **Primary Language**   | Python 3.10                              |
| **AI / Vision**        | PyTorch, Ultralytics YOLOv8, OpenCV      |
| **Build System**       | colcon                                   |
| **Version Control**    | Git & GitHub                             |

---

## 📁 Repository Structure

```
Team-Vulcans-Robocon-2026/
├── src/                        # ROS 2 packages — robot nodes, navigation, vision, strategy
├── teams/                      # Team-specific docs, research, design notes (non-ROS)
│   ├── team-1_navigation/      # Sensor datasheets, path planning research
│   ├── team-2_vision/          # Dataset plans, model research
│   ├── team-3_strategy-ui/     # State machine diagrams, UI sketches
│   └── team-4_integration/     # System architecture diagrams, setup notes
├── .gitignore
├── LICENSE                     # MIT License
└── README.md
```

> **Note:** Non-code content such as research notes, meeting minutes, and design documents belongs in the `teams/` directory — not inside `src/`.

---

## 🚀 Getting Started

### 1️⃣ Prerequisites

Before starting, ensure that you've completed the **dual-boot installation** following our team's setup guide. This ensures the correct OS, ROS 2 version, and dependencies are installed.

- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill
- Gazebo 11
- Python 3.10+
- colcon build tool

---

### 2️⃣ Cloning the Repository

> ⚠️ GitHub no longer supports password authentication over HTTPS.
> You must use the **SSH URL** linked to your configured SSH key.

```bash
# Navigate to your ROS 2 workspace
cd ~/ros2_ws

# Clone via SSH
git clone git@github.com:Viraj281105/Team-Vulcans-Robocon-2026.git

# Move into the repo
cd Team-Vulcans-Robocon-2026
```

---

### 3️⃣ Building the Workspace

```bash
# From the workspace root
cd ~/ros2_ws

# Install dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build all packages
colcon build

# Source the workspace
source install/setup.bash
```

---

### 4️⃣ Running the Simulation

```bash
# Launch Gazebo simulation
ros2 launch <package_name> simulation.launch.py

# Run the navigation stack
ros2 launch <package_name> navigation.launch.py

# Run vision pipeline
ros2 run <package_name> vision_node
```

---

## 👥 Team

| Sub-Team           | Focus Area                              |
| ------------------ | --------------------------------------- |
| **Team 1**         | Navigation & Localisation               |
| **Team 2**         | Computer Vision (YOLOv8)                |
| **Team 3**         | Strategy, State Machines & UI           |
| **Team 4**         | Integration & System Architecture       |

**Contributors:** 2 active contributors — [Viraj281105](https://github.com/Viraj281105) + team member

---

## 🏆 Competition

- **Event**: ABU Robocon 2026 — *"Kung Fu Quest"*
- **Venue**: Hong Kong
- **Theme**: Dual-robot collaborative Kung Fu challenge

---

## ⚖️ License

MIT License — see [LICENSE](LICENSE) file for details.

---

**Status**: 🚧 Active Development
