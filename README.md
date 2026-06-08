# 🎬 Demo

![Demo](assets/gif.gif)

# 🚀 Face Watchlist System

A modular face recognition and tracking pipeline that processes video input, detects and identifies faces using deep learning embeddings, and generates structured visual and analytical outputs.

Built with **InsightFace, OpenCV, and Streamlit**, this system bridges computer vision inference with an interactive analytics interface, enabling both programmatic execution and real-time visual exploration.

The pipeline converts raw video into:
- Frame-level face detections
- Identity-matched bounding boxes
- Structured JSON metadata
- Interactive playback and analysis dashboard

---

## 🎯 What This Project Demonstrates

This project focuses on building a **production-style computer vision pipeline**, covering:

- Face detection and recognition using deep embeddings (InsightFace)
- Cosine similarity-based identity matching
- Video-to-frame transformation pipelines
- Structured metadata generation for downstream analytics
- Interactive visualization via Streamlit
- Separation of backend pipeline and frontend UI logic

---

## ⚙️ Why It Matters

Unlike basic face detection demos, this system is designed with a **real-world architecture mindset**, separating:

- 🧠 Core inference logic (offline pipeline)
- 📊 Structured outputs (JSON + frames)
- 🖥️ User-facing visualization layer (Streamlit app)

This makes it suitable for:
- scalable batch processing
- reproducible experiments
- interactive demonstrations

---

## 🧪 Ethical Design

This system is designed for **controlled and ethical use cases only**.  
All test inputs use **public VIP figures** to avoid privacy violations.

It is intended for:
- research
- educational use
- controlled environment simulations

---

## 🧠 Key Capabilities

- 🎥 Video ingestion and frame extraction
- 🧑 Face detection using InsightFace (`buffalo_l`)
- 🧠 Face recognition via cosine similarity embeddings
- 📊 JSON-based structured metadata output
- 🖼 Frame-level annotation and storage
- 📈 Streamlit dashboard with:
  - Detection table
  - Identity filtering
  - Timeline scrubber
  - Adjustable playback speed
  - Adjustable frame size

---

## ⚙️ Architecture Overview

- `main.py` → Core pipeline (no UI)
- `streamlit_app.py` → Interactive interface
- `oop_face_detector.py` → Face detection + recognition logic
- `output/` → Generated frames + JSON metadata
- `data/` → Input video + known faces

---

## 📌 Use Cases

- Research in facial recognition systems
- Media/video analysis pipelines
- Security simulation environments
- Computer vision experimentation
- Identity-based video indexing

---

## 📊 Output Format

The system generates:

- Annotated frame images (`output/saved_frames/`)
- JSON metadata (`output/json_files/`)

Each detection includes:
- Identity label
- Confidence score
- Bounding box coordinates
- Timestamp (seconds)

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.

It must not be used for:
- unauthorized surveillance
- processing non-consensual biometric data
- any unlawful application of facial recognition

Users are responsible for ensuring compliance with applicable laws and regulations.

---

## 👤 Author

**y-prog**



# ⚙️ How to Run This Project

This project can be executed in two ways:
1. 🧠 Backend pipeline (no UI) via `main.py`
2. 🖥️ Interactive Streamlit application

It is recommended to run this project on **Linux or WSL (Windows Subsystem for Linux)** due to better compatibility with OpenCV and InsightFace dependencies.

---

## 📁 Project Structure

```text
face_watchlist_system/
├── data/
│   ├── known_faces/        # VIP face dataset
│   └── videos/             # input videos
├── output/
│   ├── json_files/         # generated metadata
│   └── saved_frames/       # annotated frames
├── src/
│   ├── oop_face_detector.py
│   ├── main.py
│   └── streamlit_app.py
├── README.md
```

### OPTION 1 — Run Backend Pipeline (No UI)
Step 1 — Clone repository
```
git clone https://github.com/y-prog/face_watchlist_system.git
cd face_watchlist_system
```
Step 2 — Create virtual environment

```python3 -m venv .venv
source .venv/bin/activate
```
Step 3 — Install dependencies
```
pip install -r requirements.txt
```

If requirements.txt is missing:

```
pip install streamlit opencv-python insightface numpy pandas pillow onnxruntime
```
Step 4 — Run pipeline

```
python src/main.py
```
📌 Outputs:

Annotated frames → output/saved_frames/
JSON metadata → output/json_files/

## OPTION 2 — Run Streamlit App (Interactive UI)
Step 1 — Start Streamlit
```
streamlit run src/streamlit_app.py
```
Step 2 — Use the app
-Upload MP4 video
1Upload known face images (VIP dataset)
-Set frame interval + similarity threshold
-Click Start Processing
Explore:
📊 detection table
🧑 identity filtering
🎞 frame playback
⏱ timeline scrubber
🎚 speed control
📏 frame size control
🧠 SYSTEM NOTES

Recommended OS: Linux / WSL
Model: InsightFace (buffalo_l)
Runs on CPU (GPU optional)
Uses VIP/public figures only for safe testing
Fully reproducible pipeline (video → frames → metadata)
⚠️ RECOMMENDATION

✔ Use Linux / WSL for best compatibility
✔ Keep input videos short for faster processing
✔ Use only public/VIP images for testing
