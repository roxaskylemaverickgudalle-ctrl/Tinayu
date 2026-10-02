\# Tinayu



> AI-assisted personal color analysis using computer vision, facial landmarks, color science, and recommendation algorithms.



Tinayu is a web-based personal color analysis project that analyzes a user's photo to extract visual color information and generate personalized clothing, makeup, and accent color recommendations.



The project combines computer vision with a deterministic recommendation engine and a natural-language explanation layer.



\## ✨ Features



\- Face detection using MediaPipe

\- 478-point facial landmark detection

\- Facial skin-region extraction

\- Hair-region extraction

\- Iris-region extraction

\- RGB, HSV, and LAB color analysis

\- LAB-based lighting normalization

\- Skin tone and color characteristic interpretation

\- Seasonal color heuristics

\- Clothing color recommendations

\- Makeup color recommendations

\- Accent color recommendations

\- Image quality and detection confidence indicators

\- FastAPI backend

\- Next.js frontend

\- Webcam capture

\- Image upload and clipboard paste

\- Responsive web interface



\## 🧠 Architecture



```text

User Image

&#x20;   │

&#x20;   ▼

Next.js Web Interface

&#x20;   │

&#x20;   ▼

FastAPI API

&#x20;   │

&#x20;   ▼

MediaPipe Face Detection

&#x20;   │

&#x20;   ▼

Facial Landmarks

&#x20;   │

&#x20;   ├── Skin Region

&#x20;   ├── Hair Region

&#x20;   └── Eye Region

&#x20;   │

&#x20;   ▼

OpenCV Color Analysis

&#x20;   │

&#x20;   ├── RGB

&#x20;   ├── HSV

&#x20;   └── LAB

&#x20;   │

&#x20;   ▼

Lighting Normalization

&#x20;   │

&#x20;   ▼

Color Profile

&#x20;   │

&#x20;   ▼

Recommendation Engine

&#x20;   │

&#x20;   ├── Clothing

&#x20;   ├── Makeup

&#x20;   └── Accents

&#x20;   │

&#x20;   ▼

Structured Analysis Result

