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

## Implemented endpoints

- GET /api/v1/health
- POST /api/v1/image-processing/grayscale
- POST /api/v1/image-processing/blur/gaussian

Gaussian blur parameters:

- kernel_size: odd integer greater than 1
- sigma: float greater than or equal to 0

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