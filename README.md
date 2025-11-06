# ACCESS_DENIED_SJBIT
Repository  for AXIOM hackathon @ SJBIT

Project Overview
This project is a mobile application designed to help smallholder farmers rapidly diagnose crop diseases in the field using augmented reality (AR) overlays and real-time pattern recognition. The app operates with minimal labeled training data, leveraging transfer learning and lightweight models for on-device inference. Farmers can simply scan crops using their phone camera, receive immediate diagnosis, and get practical treatment tips—localized for usability in English and Kannada.

Features
AR overlays: Highlights affected crop regions (leaf, fruit, etc.) in real-time via the phone camera.

Minimal input AI: Uses few-shot learning/transfer learning for disease recognition with low data requirements.

Immediate diagnosis & treatment tips: Concise information delivered instantly after scanning.

Language support: Treatment instructions and diagnoses are available in English and Kannada.

Offline operation: All major functions run without internet connectivity.

README and documentation: Clear instructions for dataset usage, model training/conversion, app setup, and offline running.

Architecture
Layer	Tech Used	Description
UI/Frontend	Flutter / ReactNative	Camera interface, AR overlays, language toggle
AR Engine	ARCore/Unity	Augmented reality feedback via overlays
ML Model	TensorFlow Lite	Transfer-learned image model for disease detection
Data	Public datasets	PlantVillage, PlantDoc, etc.; with local samples for tuning
Dataset Sources
PlantVillage: Kaggle link

PlantDoc: PlantDoc dataset

Local Fine-Tune Samples: Collected during hackathon as per requirement.

Quick Start & Setup
Clone the repository

Install dependencies

Flutter/React Native packages

TensorFlow Lite runtime

Model training/conversion

Use Python/TensorFlow notebook to fine-tune MobileNet/EfficientNet-Lite model on crop disease images (see model_training/).

Convert model to .tflite format for mobile use.

App build/run

Deploy mobile app to device/emulator.

Test with sample images from dataset, then live camera for AR overlays.

Language toggle

Check accessibility by toggling the treatment tips between English and Kannada.

Offline check

Ensure app responds without internet connectivity.

Usage
Open the app and point your device camera at the crop you wish to scan.

The app will highlight affected regions via AR overlays and provide an immediate probable diagnosis.

Treatment information will be shown in your selected language.

No network or external service needed after install.

Folder Structure
text
project-root/
├── app/                   # Source code for mobile application
├── model_training/        # Scripts/notebooks to train and convert ML models
├── datasets/              # Links/sample images for transfer learning
├── translations/          # Resources for Kannada/English tips
├── README.md              # Project documentation
└── screenshots/           # Sample app screens/demos
License
This project is developed for SJBIT Bangalore Hackathon 2025. Open-source for educational and non-commercial purposes.

Contributors
Your Team Name + Team Members

References
PlantVillage/Kaggle

TensorFlow Lite documentation

ARCore/Unity documentation
