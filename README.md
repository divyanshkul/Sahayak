# Sahayak AI 🎓

**Empowering teachers in multi-grade classrooms through AI**

[![Python 3.8+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Vertex%20AI-blue.svg)](https://cloud.google.com/vertex-ai)

## Overview

Sahayak AI is a comprehensive AI-powered platform designed specifically for multi-grade classrooms in under-resourced Indian schools. It provides four specialized AI agents to assist teachers in planning, teaching, assessment, and administration.

### 🎯 Problem Statement

- **30%+** of India's government schools operate in Multi-Grade Multi-Level (MGML) model
- **9%+** are single-teacher schools
- Teacher shortages, lack of training, and absence of localized materials make effective teaching nearly impossible

### 💡 Solution

An AI-powered multilingual mobile app that supports single teachers handling 3-5 grades simultaneously with:

- NCERT curriculum alignment
- Regional language support
- Automated administrative tasks
- Interactive content generation

## Four AI Agents

### 1. शिक्षक मित्र (Shikshak Mitra) - Teaching Companion

- Weekly lesson scheduling and material preparation
- Smart blackboard layouts for multi-grade teaching
- Hyper-local content generation with cultural adaptation
- Manim animation creation for mathematical concepts
- Textbook photo scanning and digitization

### 2. परीक्षा गुरु (Pariksha Guru) - Exam/Quiz Agent

- Personalized quiz generation based on weak concepts
- NCERT-aligned exam paper creation
- OMR/OCR automated answer checking
- Multi-level difficulty adaptation
- Real-time performance analytics

### 3. प्रबंधन साथी (Prabandhan Saathi) - Admin Partner

- Photo-based face recognition attendance
- Automatic parent messaging (below 60% attendance)
- Performance dashboards and analytics
- Teaching improvement suggestions

### 4. सहायता चैट (Sahayata Chat) - Help Assistant

- Student curiosity question handling
- News-to-lesson content conversion
- Real-world connection generation
- Cultural context integration
- Personal teaching help chatbot

## 🏗️ Architecture

### Technology Stack

**Frontend:**

- Flutter & Dart for cross-platform mobile app

**Backend:**

- Python with FastAPI
- Google Cloud Platform (Firestore, PostgreSQL, Object Store)
- Vertex AI for ML deployment

**AI/ML:**

- Gemini API (Large Language Model)
- Google Agent Development Kit
- PaliGemma (Vision-Language Model)
- Google ML Kit (Face Recognition)
- Manim (Mathematical Animation Engine)

**Data:**

- RAG System with NCERT knowledge retrieval
- Firestore Vector Database
- PostgreSQL for structured data

## 🚀 Getting Started

### Prerequisites

- Python 3.12.9
- Google Cloud Platform account
- Google AI API key
- Required system dependencies for face recognition

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/divyanshkul/Sahayak.git
cd Sahayak
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

```bash
# Create .env file
cp .env.example .env

# Configure your settings
GOOGLE_API_KEY=your_google_api_key
GCP_BUCKET_NAME=your_bucket_name
GCP_CREDENTIALS_PATH=path_to_service_account.json
```

4. **Set up Google Cloud credentials**

```bash
# Place your service account JSON file in the secrets/ directory
mkdir secrets/
# Copy your service account file to secrets/
```

5. **Run the application**

```bash
python main.py
```

The API will be available at `http://localhost:4000`

### API Documentation

Once running, visit:

- Swagger UI: `http://localhost:4000/docs`
- ReDoc: `http://localhost:4000/redoc`

## API Endpoints

### Health Check

```http
GET /api/v1/health/
```

### Shikshak Mitra (Teaching Companion)

```http
POST /api/v1/shikshak-mitra/generation-questions
Content-Type: application/json

{
  "question": "Generate questions related to triangles for class 5"
}
```

```http
POST /api/v1/shikshak-mitra/generate-animation
Content-Type: application/json

{
  "prompt": "Create an animation showing how angles in a triangle add up to 180 degrees"
}
```

### Prabandhan Saathi (Admin Partner)

```http
POST /api/v1/prabhandhak/attendance/upload-photo
Content-Type: multipart/form-data

photo: [image file]
class_id: "class_123"
```

## 🎯 Key Features

### 🔍 Face Recognition Attendance

- Upload class photos for automatic attendance marking
- Advanced face recognition using trained models
- Generates attendance reports with visual feedback
- Automatic parent notifications for low attendance

### 🎬 Mathematical Animation Generation

- AI-powered Manim animation creation
- Mathematical concept visualization
- Automatic video generation and cloud storage
- Educational content localization

### 📝 Question Generation with RAG

- NCERT curriculum-aligned question generation
- Context-aware retrieval from vector database
- In-context learning with SQL-backed question bank
- Multi-language support

### 🌐 Multilingual Support

- Hindi, Marathi, Kannada, Punjabi, and English
- Cultural context adaptation
- Regional example generation

## Features Demo

### Question Generation

The system generates contextually relevant questions based on NCERT curriculum:

<div align="center">
  <img src="https://github.com/user-attachments/assets/0f4c120b-d2ad-4bdf-b4ba-9e18f0df835e" alt="Swipe Left Action" width="250" style="margin: 10px;">
  <img src="https://github.com/user-attachments/assets/d6b68c57-f502-4120-82bd-016a151a0390" alt="Swipe Right Action" width="250" style="margin: 10px;">
  <img src="https://github.com/user-attachments/assets/d51a3c5e-0567-4bd4-9bc1-a232714a39c3" alt="Quiz Interface" width="250" style="margin: 10px;">
</div>

### Attendance Processing

Upload a class photo and get detailed attendance analysis:

<div align="center">
 <img src="https://github.com/user-attachments/assets/3c1e659b-dce5-49f5-a43f-d2fb8827a5ef" alt="Take Attendance" width="300" style="margin: 10px;">
 <img src="https://github.com/user-attachments/assets/348d5da1-a9cc-4714-a510-81f94a95f488" alt="Attendance Results" width="300" style="margin: 10px;">
</div>


## 🔧 Configuration

### Environment Variables

| Variable               | Description               | Required |
| ---------------------- | ------------------------- | -------- |
| `GOOGLE_API_KEY`       | Google AI API key         | Yes      |
| `GCP_BUCKET_NAME`      | Cloud Storage bucket      | Yes      |
| `GCP_CREDENTIALS_PATH` | Service account JSON path | Yes      |
| `MANIM_SERVER_PATH`    | Path to Manim MCP server  | No       |
| `PYTHON_ENV_PATH`      | Python executable path    | No       |

### Face Recognition Setup

1. Create a `train/` directory
2. Add student photos named as `student_name.jpg`
3. The system will automatically load and train face encodings

## Google Cloud Agentic AI Day Hackathon

- Team: "The Fast and Fourier"
- Problem Statement: Empowering teachers in multi-grade classrooms

**Made with ❤️ for teachers and students in multi-grade classrooms across India**
