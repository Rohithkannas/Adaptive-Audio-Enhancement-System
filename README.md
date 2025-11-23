<h1 align="center">Adaptive Audio Enhancement System</h1>

<p align="center">
  <img src="https://img.shields.io/badge/MATLAB-R2024a-orange" />
  <img src="https://img.shields.io/badge/Status-Completed-green" />
  <img src="https://img.shields.io/badge/Architecture-Hybrid_(DSP_+_ML)-blueviolet" />
</p>

<p align="center">
  <img src="[Cover_Image.png](https://github.com/Rohithkannas/Adaptive-Audio-Enhancement-System/blob/f5b950792ca2ebdc9709a8649c48357a6555698d/MatlabprojectImage.png)" alt="Project Cover" width="100%">
</p>

## 📖 About this Project
This project addresses the critical challenge of background noise in modern real-time communication platforms (VoIP, Zoom, Microsoft Teams). As remote work becomes the norm, the demand for clear, noise-free audio has skyrocketed.

Traditional AI noise suppression can be computationally heavy and introduce lag. This project proposes a **Hybrid Solution** that combines the mathematical precision and speed of **Digital Signal Processing (DSP)** with the intelligent monitoring capabilities of **Deep Learning**. The result is a system that cleans audio in real-time (<2.5ms latency) while using AI to verify and benchmark the output quality.

## ⚙️ Technical Architecture
The system operates on a block-processing basis with a **Dual-Stage Architecture**:

### 1. The Engine (DSP Layer)
* **Sign-LMS Adaptive Filter:** The core component. It models and subtracts noise mathematically.
    * *Why this?* It has $O(N)$ complexity, making it ultra-fast and suitable for embedded hardware.
* **VAD (Voice Activity Detection):** Uses energy thresholds to prevent the filter from distorting silence.

### 2. The Monitor (AI Layer)
* **Deep Learning (CNN):** A Convolutional Neural Network analyzes spectrograms of the output audio to classify it as "Clean" or "Noisy" with **92.5% accuracy**.
* **Machine Learning (SVM/RF):** Classical models (Random Forest) provide a secondary validation layer based on MFCC features.

## 📊 System Output
![Dashboard Results](https://github.com/Rohithkannas/Adaptive-Audio-Enhancement-System/blob/ca85e3989f0390d7ac7b76dfff228f93856f65cf/Dashboard_results.png)
*Figure 1: Signal processing results (Top) and ML-based Quality Classification Matrix (Bottom).*

## ⚙️ Technical Architecture
The system operates on a block-processing basis with three distinct stages:

### 1. Signal Processing Layer (The Engine)
* **VAD:** Energy-based Voice Activity Detection to isolate speech segments.
* **Adaptive Filter:** **Sign-LMS algorithm** for computational efficiency ($O(N)$) and fast convergence.

### 2. Feature Engineering Layer
* **Spectral Analysis:** Extraction of Spectral Centroid, Rolloff, and Flux.
* **Audio Features:** MFCC (Mel-frequency cepstral coefficients), Zero Crossing Rate (ZCR), and Pitch.

### 3. Machine Learning Layer (The Monitor)
* **Deep Learning:** A **Convolutional Neural Network (CNN)** processes spectrogram images to visually classify audio segments (Clean vs. Noisy).
* **Classical ML:** **Random Forest** and **SVM** classifiers provide redundant quality assessment based on extracted feature vectors.

## 📈 Performance Metrics
| Parameter | Result | Notes |
| :--- | :--- | :--- |
| **Noise Attenuation** | **+10-15 dB** | Effective suppression via LMS Filter |
| **Algorithm Latency** | **~2.5 ms** | Real-time capable (<20ms standard) |
| **CNN Accuracy** | **92.5%** | High confidence in spectrogram classification |
| **ML Accuracy (RF)** | **91.5%** | Robust feature-based quality validation |

## 🚀 Usage Instructions
1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR-USERNAME/Adaptive-Audio-Enhancement-System.git](https://github.com/YOUR-USERNAME/Adaptive-Audio-Enhancement-System.git)
    ```
2.  Open `Smart_Meeting_Assistant.m` in MATLAB.
3.  **Requirements:** Signal Processing Toolbox, Statistics and Machine Learning Toolbox, Deep Learning Toolbox.
4.  Run the simulation to generate synthetic audio, apply the filter, and view the ML classification results.

## 🏁 Conclusion
The **Adaptive Audio Enhancement System** demonstrates that a hybrid architecture is the optimal solution for real-time audio processing. By offloading the heavy filtering task to the efficient Sign-LMS algorithm and reserving AI models for high-level quality monitoring, the system achieves a latency of **~2.5ms**—well within the industry standard of 20ms—while maintaining high spectral fidelity. This architecture offers a scalable path for deploying noise cancellation on low-power devices.

## 📬 Contact
If you have any questions or would like to discuss this project further, please feel free to connect:

* **LinkedIn:** [Your Name](https://www.linkedin.com/in/rohith4510/)
* **Email:** [your.email@example.com](rohithkanna.ss@gmail.com)
