# Interactive Learning Platform API

Backend API for an interactive educational platform focused on:

- Image Processing
- Computer Vision
- Computer Graphics

## Technologies

- Python
- FastAPI

## Running locally

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Install backend dependencies and run API:

```bash
cd backend
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

Run frontend (no build step required):

```bash
cd ..
python3 -m http.server 5500
```

Open:

- Frontend lab: http://127.0.0.1:5500/frontend/
- API docs: http://127.0.0.1:8000/docs

## Implemented endpoints

- GET /api/v1/health
- POST /api/v1/image-processing/grayscale
- POST /api/v1/image-processing/blur/gaussian
- POST /api/v1/image-processing/edges/sobel
- POST /api/v1/image-processing/edges/canny

Global image constraints for all Image Processing endpoints:

- minimum size: 50x50 pixels
- maximum size: 200x200 pixels

Gaussian blur parameters:

- kernel_size: odd integer greater than 1
- sigma: float greater than or equal to 0

Sobel parameters:

- kernel_size: one of 1, 3, 5, 7

Canny parameters:

- threshold1: float between 0 and 255
- threshold2: float between 0 and 255
- rule: threshold1 must be lower than threshold2

## Intial Structure

```bash
backend/
├── app/
│   ├── __init__.py
│   │
│   ├── main.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── health.py
│   │
│   └── core/
│       └── __init__.py
│
├── tests/
│   └── __init__.py
│
├── requirements.txt
├── .gitignore
└── README.md
```