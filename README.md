# Fingerprint Blood Group Detector

A desktop application powered by a CNN (AlexNet) model to detect blood groups from fingerprint images. It supports training on custom datasets, single-image predictions, and live detection via a connected ESP32/Arduino fingerprint sensor over serial communication.

---

## 🚀 Features

- **Model Training**: Train a custom CNN model on categorized fingerprint datasets directly from the GUI.
- **Single Prediction**: Select any fingerprint image to instantly predict the blood group with confidence percentages and probability charts.
- **ESP32/Arduino Integration**: Live capture and transmission of fingerprint data from an external sensor with feedback displayed on the ESP32.
- **Modern UI**: A responsive, dark-themed dashboard built with Tkinter.

---

## 📂 Repository Structure

```text
Finger_blood_det/
├── esp32/
│   └── esp32.ino              # Arduino/ESP32 firmware code
├── .gitignore                 # Standard Python & large weights ignores
├── environment.yml            # Conda environment dependency list
├── instruction.md             # Detailed installation and user guide
├── main.py                    # Main GUI application entry point
├── model_blood_group.keras    # Trained CNN weights (locally stored)
└── set_env.bat                # Windows setup script to automate Conda setup
```

---

## 🔧 Getting Started

1. **Setup the Environment**: Run [set_env.bat](file:///c:/Users/naim/Documents/projects/Finger_blood_det/set_env.bat) once to install Miniconda and configure the required libraries.
2. **Launch the App**:
   ```bash
   conda activate fingerprint
   python main.py
   ```
3. **Firmware Deployment**: Upload [esp32.ino](file:///c:/Users/naim/Documents/projects/Finger_blood_det/esp32/esp32.ino) to your microcontroller using the Arduino IDE.

For deep-dive usage instructions, troubleshooting, dataset formatting, and training tips, see the [instruction.md](file:///c:/Users/naim/Documents/projects/Finger_blood_det/instruction.md) guide.

---

## 📝 License

This project is open-source and available under the MIT License.
