# Fingerprint Blood Group Detector — User Guide

> [!WARNING]
> **Disclaimer**: This is a dummy project created for demonstration and educational purposes. It is **not** scientifically possible to determine a person's blood group from fingerprint images. This application is purely a proof-of-concept simulation and must not be used for any real-world medical or diagnostic purposes.

A desktop application that detects blood group from fingerprint images using a CNN (AlexNet) model, with optional live detection via ESP32/Arduino over serial.

---

## Requirements

- Windows 10 or 11
- Internet connection (first-time setup only)
- `set_env.bat` and `environment.yml` (provided alongside this guide)

---

## 1. Environment Setup

Run the setup script **once** before first use.

1. Double-click `set_env.bat`
2. If prompted by Windows Defender, click **More info → Run anyway**
3. The script will:
   - Install Miniconda (via `winget`)
   - Create a conda environment named `fingerprint`
   - Install all required packages from `environment.yml`
4. Wait for the message:

```
=====================================
Environment setup complete!
=====================================
```

> If `winget` is not available, install it from the Microsoft Store (search **App Installer**) and re-run the bat file.

---

## 2. Launching the App

After setup, launch the app each time with:

```bat
conda activate fingerprint
python main.py
```

Or create a shortcut that runs:

```bat
cmd /k "conda activate fingerprint && python main.py"
```

---

## 3. Dataset Structure

Organize your fingerprint images exactly like this before training:

```
dataset_blood_group/
├── A+/
│   ├── img001.bmp
│   └── ...
├── A-/
├── B+/
├── B-/
├── AB+/
├── AB-/
├── O+/
└── O-/
```

Supported image formats: `.bmp`, `.jpg`, `.jpeg`, `.png`

---

## 4. Tab ① — Train

Use this tab to train a new model on your dataset.

1. Click **SELECT DATASET FOLDER** and choose your `dataset_blood_group/` folder
2. Confirm the detected classes appear below the folder path
3. Adjust settings if needed:

| Setting    | Default | Range    |
|------------|---------|----------|
| Epochs     | 20      | 1 – 100  |
| Batch Size | 32      | 8 – 128  |
| Image Size | 256     | 64 – 512 |

4. Click **▶ START TRAINING**
5. Watch progress in the log and progress bar
6. When done, the model saves automatically as `model_blood_group.keras` next to `main.py`

> Higher image size and more epochs improve accuracy but take longer to train.

---

## 5. Tab ② — Predict

Use this tab to predict blood group from a single fingerprint image.

1. The model auto-loads on startup if `model_blood_group.keras` exists
2. If not loaded, click **AUTO-LOAD** or **LOAD MODEL FILE** to pick a `.keras` / `.h5` file
3. Click **🔍 SELECT IMAGE & PREDICT**
4. Choose a fingerprint image — the result appears instantly with:
   - Predicted blood group (large display)
   - Confidence percentage
   - Probability bar chart for all 8 classes

---

## 6. Tab ④ — Arduino / ESP32

Use this tab for live detection with a connected ESP32 fingerprint sensor.

### Connecting

1. Plug in the ESP32 via USB
2. Click **🔄 REFRESH** to scan available serial ports
3. Select the correct **PORT** (e.g., `COM3`) and set **BAUD** to `115200`
4. Click **🔌 CONNECT**
5. The app sends `READY` automatically — the ESP32 will capture and transmit an image

### Flow

```
PC sends READY  →  ESP32 captures fingerprint & sends image
ESP32 sends image  →  App predicts blood group
App sends RESULT:<group>:<confidence>  →  ESP32 displays result
ESP32 sends ACK  →  App sends READY again
```

### Manual Control

- **📡 SEND READY (manual)** — trigger a new capture manually
- **🗑 CLEAR LOG** — clear the serial log

### Troubleshooting Serial

| Problem | Fix |
|---|---|
| No ports listed | Click REFRESH; check USB cable |
| Connection fails | Verify baud rate matches ESP32 firmware |
| No ACK received | App retries 3 times, then continues automatically |

---

## 7. Tab ③ — About

Contains architecture details, preprocessing notes, and a usage summary for quick reference inside the app.

---

## Notes

- The model uses `rescale=1./255` normalisation. Do **not** swap in ImageNet preprocessing — it will cause the classifier to collapse to 1–2 classes.
- The trained model file (`model_blood_group.keras`) is saved next to `main.py` and is reused across sessions automatically.
- To retrain from scratch, simply click START TRAINING again — the file will be overwritten.
