# GPR Signal Denoising with Wave-U-Net

## Overview
This project applies the **Wave-U-Net** architecture, originally designed for audio denoising, to remove noise from **Ground Penetrating Radar (GPR) signals**.


## Please cite:
H. Namdari, M. Moradikia, S. Zekavat, R. Askari, O. Mangoubi and D. Petkie, "Advancing Precision Agriculture: Machine Learning-Enhanced GPR Analysis for Root-Zone Soil Moisture Assessment in Mega Farms," in IEEE Transactions on AgriFood Electronics, doi: 10.1109/TAFE.2024.3455238.
keywords: {Feature extraction;Mathematical models;Data models;Soil moisture;Synthetic data;Moisture;Radar;Permittivity;Training;Soil measurements;Feature extraction;gprMAX simulations;ground penetration radar (GPR);machine learning (ML);root zone soil moisture estimation (SME)},


## Features
- Uses **Wave-U-Net** with **skip connections** for improved signal recovery.
- Trains on **synthetic or real GPR data**.
- Outputs **denoised GPR signals**.
- Saves the trained model for future inference.

## Installation
Clone this repository and install the dependencies:
```bash
git clone https://github.com/YOUR_USERNAME/GPR-Denoising-WaveUNet.git
cd GPR-Denoising-WaveUNet
pip install -r requirements.txt


This project is an experimental implementation. The model may require further tuning to generalize well to real-world GPR data. Users are encouraged to fine-tune hyperparameters and test with diverse datasets.
