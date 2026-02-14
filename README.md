# Industrial-Stone-Measurement
A computer vision–based 3D measurement and analysis system designed to estimate natural stone dimensions and provide a foundation for industrial cutting optimization with reduced material waste.

## Project Motivation
In industrial stone processing, inefficient cutting strategies lead to high material waste (fire) and increased production costs.  
This project aims to develop a **measurement-driven system** that accurately analyzes stone dimensions and supports future **cutting strategy recommendations**.

The current implementation focuses on **3D measurement and analysis**.  
Automated optimal cutting recommendations are part of the **future development roadmap**.

## Current Capabilities
- ZED stereo camera integration for depth-aware data acquisition
- 3D point cloud generation and preprocessing
- Calibration, noise filtering, ground plane removal, and stone segmentation
- Oriented Bounding Box (OBB)–based dimension estimation
- Volume and weight estimation based on material density
- 3D visualization and mesh export (OBJ / PLY)
- Industrial-oriented measurement and efficiency reporting

## System Overview
ZED Stereo Camera → Point Cloud Acquisition → Filtering & Segmentation → 3D Stone Model → Dimension & Volume Estimation → Industrial Analysis


## Technologies Used
- Python
- Zed SDK (pyzed.sl)
- OpenCV
- Open3D
- NumPy

## Project Status
- ✔ Functional prototype
- ✔ Real-time ZED camera integration
- ✔ Reliable dimension estimation under controlled conditions
- ⏳ Cutting optimization logic under development

This project is an **engineering prototype**, not a finished industrial automation system.
This repository represents the **measurement and analysis foundation** of a larger industrial optimization vision.

## Known Limitations
- Cutting recommendations are not yet fully automated
- Optimization logic is currently rule-based
- Measurement accuracy depends on calibration quality and environment

## Future Work
- Automated optimal cutting strategy generation
- Orientation-aware and kerf-aware cutting optimization
- Advanced point cloud refinement for higher accuracy
- AI-based decision and optimization models
- Industrial production line integration
