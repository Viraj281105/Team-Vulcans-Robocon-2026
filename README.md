# 🤖 Team Vulcans — ABU Robocon 2026: *Kung Fu Quest*

Welcome to the official source code repository for **Team Vulcans’** entry into the **ABU Robocon 2026** competition, *“Kung Fu Quest”*, hosted in **Hong Kong**.

This repository contains all the software, simulation models, and AI components for our two robots.  
Our project focuses on developing a **robust and collaborative robotic system** with a strong emphasis on:
- Autonomous navigation  
- Real-time computer vision for object recognition  
- Intelligent strategic decision-making  

---

## 🔧 Tech Stack

Our system is built on an industry-standard robotics stack for reliability and performance.

| Category | Technology |
|-----------|-------------|
| **Operating System** | Ubuntu 22.04 LTS *"Jammy Jellyfish"* |
| **Robotics Framework** | ROS 2 Humble Hawksbill (LTS) |
| **Simulation** | Gazebo 11 |
| **Primary Language** | Python 3.10 |
| **AI / Vision** | PyTorch, Ultralytics YOLOv8, OpenCV |
| **Build System** | colcon |
| **Version Control** | Git & GitHub |

---

## 🚀 Getting Started

### 1️⃣ Prerequisites

Before starting, ensure that you’ve completed the **dual-boot installation** following our team’s setup guide.  
This ensures the correct OS, ROS 2 version, and dependencies are installed.

---

### 2️⃣ Cloning the Repository (Cleanly!)

> ⚠️ GitHub no longer supports password authentication over HTTPS.  
> You must use the **SSH URL** linked to your configured SSH key.

```bash
# Navigate to your workspace
cd ~/ros2_ws

# Clone via SSH
git clone git@github.com:Viraj281105/Team-Vulcans-Robocon-2026.git
