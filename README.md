# Workplace Safety PPE Detection using YOLOv8

This project provides a real-time object detection system to monitor whether people in a workplace environment are wearing essential Personal Protective Equipment (PPE), specifically safety helmets and vests. The system is built using the YOLOv8 model.

---

## Key Features

* **Real-Time Detection:** Capable of processing video streams from files or a live webcam to identify PPE in real-time.
* **High Accuracy:** Trained on a large dataset of over 23,000 images, achieving an mAP50 of 73.5% overall and over 86% for key classes like helmets and vests.
* **State-of-the-Art Model:** Utilizes YOLOv8, a powerful and efficient object detection architecture.
* **Focused Detection:** While trained on 17 classes, the inference script is configured to specifically highlight helmets and vests for workplace safety monitoring.

---

## Technology Stack

* **Python 3.8+**
* **PyTorch**
* **Ultralytics YOLOv8**
* **OpenCV**
* **Kaggle Notebooks** (for training)

---

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

* Python 3.8 or newer
* Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/hafizqaim/Workspace-Safety-Detection-using-YOLOv8.git](https://github.com/hafizqaim/Workspace-Safety-Detection-using-YOLOv8.git)
    cd Workspace-Safety-Detection-using-YOLOv8
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1.  **Download the Trained Model:**
    The trained model file (`best.pt`) is required to run the inference. Download it from the **[Releases](https://github.com/hafizqaim/Workspace-Safety-Detection-using-YOLOv8/releases)** page of this repository.

2.  **Place the Model:**
    Place the downloaded `best.pt` file in the root directory of the project.

3.  **Run the Inference Script:**
    The `inference.py` script is configured to run on your webcam by default.
    ```bash
    python inference.py
    ```
    * To use a video file instead, open `inference.py` and modify the script to point to your video file.

---

## Model Performance

The model was trained for 10 epochs on the "PPE Detection v3" dataset from Roboflow.

| Class               | Precision | Recall | mAP50 | mAP50-95 |
| :------------------ | :-------- | :----- | :---- | :------- |
| **Overall** | 0.72      | 0.715  | 0.735 | 0.456    |
| **`head_helmet`** | 0.784     | 0.824  | 0.866 | 0.584    |
| **`vest`** | 0.841     | 0.897  | 0.935 | 0.705    |

<img width="1440" height="810" alt="Screenshot 2025-07-16 at 11 53 44 AM" src="https://github.com/user-attachments/assets/d01fadda-a590-4525-acbc-7006c007cd19" />
<img width="1314" height="771" alt="Screenshot 2025-07-16 at 11 55 09 AM" src="https://github.com/user-attachments/assets/0fe27deb-48b2-4dae-bfc2-6fec0673bfa1" />

