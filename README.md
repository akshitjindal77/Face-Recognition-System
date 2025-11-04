# Face Recognition Tool (InsightFace + Streamlit)

A **multi-user face recognition system** built with **InsightFace**, **OpenCV**, and **Streamlit**.  
It lets you register faces, build an embedding database, train an optional classifier (SVM), and recognize users in real time — all locally.

---

## Features
**Register new users** via webcam  
**Build & save face embeddings** using InsightFace (ArcFace 512-D)  
**Optional SVM training** for higher accuracy and probability-based “Unknown” detection  
**Live recognition** (real-time face ID from webcam)  
**Streamlit web interface** + CLI support  
**CPU / GPU toggle**  
**Automatic alignment & high-accuracy embeddings**

---

## Project Structure

face-recognition-tool/
├─ ui/
│ └─ app.py # Streamlit UI
├─ main.py # Command-line version
├─ data/
│ ├─ users/ # User images (ignored by Git)
│ ├─ insightface_db.pkl # Generated DB (ignored)
│ └─ .gitkeep
├─ requirements.txt
├─ .gitignore
└─ README.md

---

## Installation

### 1️. Clone the repo

git clone https://github.com/<akshitjindal77>/face-recognition-tool.git
cd face-recognition-tool
### 2️. Create a virtual environment
Using conda (recommended)
conda create -n faceid python=3.10 -y
conda activate faceid
Or using venv
python -m venv .venv
.\.venv\Scripts\activate     # Windows
source .venv/bin/activate    # macOS/Linux
### 3️. Install dependencies
pip install -r requirements.txt
 - If you have an NVIDIA GPU, replace onnxruntime with onnxruntime-gpu for faster inference:

pip uninstall onnxruntime -y
pip install onnxruntime-gpu
## How to Run
### Option 1 — CLI (Command Line)
 1. Register a user
        python main.py register --name Akshit --cpu
 2.  Build database
        python main.py build-db --cpu
 3.  Train SVM (optional, improves accuracy)
        python main.py train-svm
 4. Start live recognition
        python main.py recognize --cpu
 - Use --cpu for CPU mode or omit it to use GPU (if available).
 - Adjust unknown threshold with --thresh, e.g. --thresh 0.65.

### Option 2 — Streamlit Web UI

streamlit run ui/app.py
This opens the interface in your browser with three tabs:

 - Register → capture user faces
 - Build DB → compute embeddings for all users
 - Recognize → real-time recognition feed

#### Sidebar options:
1. Camera index (if multiple webcams)
2. CPU/GPU toggle
3. Threshold control for “Unknown”
4. Rebuild DB, list users, and more

#### Data Storage
All data is stored locally under the data/ folder:

File/Folder	Purpose
data/users/	Stores captured face images per user
data/insightface_db.pkl	Saved embeddings (automatically generated)
data/svm_insightface.pkl	Trained classifier (if trained)

These files are ignored by Git (.gitignore) to keep your repo clean and privacy-safe.

## Tech Stack
| Component                  | Library                                                                   |
| -------------------------- | ------------------------------------------------------------------------- |
| Face Detection & Embedding | [InsightFace](https://github.com/deepinsight/insightface) (ArcFace 512-D) |
| Classifier (optional)      | Scikit-learn LinearSVC + Calibrated Probabilities                         |
| Video / Image I/O          | OpenCV                                                                    |
| Web App                    | Streamlit                                                                 |
| Math / Utils               | NumPy, SciPy, Joblib                                                      |


## Troubleshooting
1. "Microsoft Visual C++ 14.0 required"
Install Microsoft Build Tools →
Select “Desktop development with C++” workload → Restart → Reinstall InsightFace.

2. "No users found / DB empty"
Make sure you registered users before building the database.
If still empty, check the Data directory path shown at the top of the Streamlit UI.

3. "Recognition too strict / too loose"
Adjust Unknown threshold:
Higher = fewer false positives (more Unknowns)
Lower = fewer Unknowns but may misclassify

## License
MIT License © 2025 Akshit Jindal
You may freely use, modify, and distribute this project for non-commercial or educational purposes.


## Credits
Developed by Akshit Jindal
Powered by InsightFace, OpenCV, and Streamlit