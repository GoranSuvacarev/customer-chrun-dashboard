# 🎯 Customer Churn Prediction Dashboard

A full-stack machine learning application for predicting customer churn in the telecommunications industry. Built with FastAPI, React, and MongoDB, featuring an interactive dashboard for analyzing churn patterns and identifying at-risk customers.

---

## 📋 Table of Contents

- Features
- Tech Stack
- Prerequisites
- Installation
- Usage Guide
- API Documentation
- Project Structure
- Future Improvements

---

## ✨ Features

### 🎨 Frontend Features
- **Interactive Dashboard** - Real-time metrics displaying total customers, churn rate, at-risk customers, and model accuracy
- **Customer Management** - Browse, search, and filter customers with pagination support
- **Detailed Customer Profiles** - View comprehensive customer information with churn risk visualization
- **Analytics & Visualizations** - Interactive charts showing:
  - Risk distribution (pie chart)
  - Top features driving churn (bar chart)
  - Churn rate by contract type (bar chart)
- **CSV Upload Interface** - Drag-and-drop file upload with validation
- **ML Pipeline Controls** - Train models and generate predictions directly from the UI
- **Responsive Design** - Mobile-friendly interface using Tailwind CSS

### 🔧 Backend Features
- **RESTful API** - FastAPI-powered backend with automatic documentation
- **Machine Learning Pipeline** - Random Forest classifier with 77% accuracy
- **Data Preprocessing** - Automated feature engineering and data transformation
- **Batch Predictions** - Process thousands of customers efficiently
- **MongoDB Integration** - Scalable NoSQL database for customer data
- **Analytics Endpoints** - Aggregation queries for insights and reporting
- **Error Handling** - Comprehensive error messages and validation

### 🤖 ML Capabilities
- **Churn Probability Scoring** - Predict likelihood of customer churn (0-100%)
- **Risk Level Classification** - Categorize customers as Low, Medium, or High risk
- **Feature Importance Analysis** - Identify key factors driving churn
- **Top Risk Factors** - Personalized explanations for each customer's churn risk
- **Model Metadata Tracking** - Store and retrieve model performance metrics

---

## 🛠️ Tech Stack

### Frontend
- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **React Router** - Client-side routing
- **Tailwind CSS** - Utility-first styling
- **Recharts** - Data visualization library
- **Axios** - HTTP client
- **Lucide React** - Icon library

### Backend
- **FastAPI** - Modern Python web framework
- **Python 3.9+** - Programming language
- **Motor** - Async MongoDB driver
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning library
- **Joblib** - Model serialization
- **Pydantic** - Data validation

### Database & Tools
- **MongoDB** - NoSQL database
- **Uvicorn** - ASGI server
- **Node.js & npm** - JavaScript runtime and package manager

---

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.9 or higher** - [Download Python](https://www.python.org/downloads/)
- **Node.js 18 or higher** - [Download Node.js](https://nodejs.org/)
- **MongoDB** - [Install MongoDB](https://www.mongodb.com/try/download/community)
- **Git** - [Download Git](https://git-scm.com/downloads)

### Verify Installation
```bash
# Check Python version
python --version  # Should be 3.9+

# Check Node.js version
node --version    # Should be 18+

# Check MongoDB installation
mongod --version

# Check Git
git --version
```

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/customer-churn-dashboard.git
cd customer-churn-dashboard
```

### 2. Backend Setup

#### Create Virtual Environment

**Windows:**
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
```

#### Install Python Dependencies
```bash
pip install --break-system-packages -r requirements.txt
```

#### Configure Environment Variables

Create a `.env` file in the `backend/` directory:
```env
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017
DATABASE_NAME=churn_dashboard

# Application Settings
ENVIRONMENT=development
```

#### Start MongoDB

**Windows:**
```bash
# Start MongoDB service
net start MongoDB
```

**Mac:**
```bash
brew services start mongodb-community
```

**Linux:**
```bash
sudo systemctl start mongod
```

#### Run Backend Server
```bash
uvicorn app.main:app --reload
```

Backend will be available at: **http://localhost:8000**

API Documentation: **http://localhost:8000/docs**

### 3. Frontend Setup

Open a **new terminal window** and navigate to the frontend directory:
```bash
cd frontend
```

#### Install Node Dependencies
```bash
npm install
```

#### Start Development Server
```bash
npm run dev
```

Frontend will be available at: **http://localhost:5173**

---

## 📖 Usage Guide

### Step 1: Upload Customer Data

1. Navigate to **Upload** page in the dashboard
2. Click **"Choose File"** or drag-and-drop your CSV file
3. Click **"Upload CSV"** button
4. Wait for confirmation message showing records created/updated

**Required CSV Columns:**
- customerID, gender, SeniorCitizen, Partner, Dependents
- tenure, PhoneService, MultipleLines, InternetService
- OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport
- StreamingTV, StreamingMovies, Contract, PaperlessBilling
- PaymentMethod, MonthlyCharges, TotalCharges, Churn

**Sample Dataset:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Step 2: Train the ML Model

1. On the **Upload** page, scroll to **"Step 2: Train Model"**
2. Click **"Train Model"** button
3. Wait 30-60 seconds for training to complete
4. View model metrics:
   - Accuracy: ~77%
   - ROC-AUC: ~0.84
   - Precision, Recall, F1-Score

### Step 3: Generate Predictions

1. Scroll to **"Step 3: Generate Predictions"**
2. Click **"Predict All Customers"**
3. Confirm the action (processes all customers in database)
4. Wait for completion (processes ~18 customers/second (6 minutes total for the whole dataset))
5. View prediction summary

### Step 4: Explore the Dashboard

#### Dashboard Page
- View key metrics: total customers, churn rate, at-risk count
- See model accuracy
- Quick action buttons to navigate

#### Customers Page
- Browse all customers with pagination
- Search by customer ID
- Filter by risk level (High/Medium/Low)
- Click any row to view detailed customer profile

#### Customer Detail Page
- View comprehensive customer information
- See churn probability gauge (0-100%)
- Review top 5 risk factors with explanations
- Understand personal, account, service, and financial details

#### Analytics Page
- **Risk Distribution Chart** - Pie chart showing Low/Medium/High risk breakdown
- **Feature Importance Chart** - Bar chart of top factors driving churn
- **Churn by Contract Chart** - Bar chart comparing contract types

---

## 📚 API Documentation

### Base URL
```
http://localhost:8000/api
```

### Endpoints

#### Customers
- `GET /customers` - Get paginated customer list
- `GET /customers/{id}` - Get single customer by MongoDB ID
- `POST /upload/csv` - Upload customer data from CSV

#### Machine Learning
- `POST /ml/train` - Train churn prediction model
- `GET /ml/model-info` - Get current model metadata
- `POST /ml/predict/{customer_id}` - Predict single customer
- `POST /ml/predict-all` - Batch predict all customers

#### Analytics
- `GET /analytics/overview` - Dashboard statistics
- `GET /analytics/risk-distribution` - Risk level counts
- `GET /analytics/feature-importance` - Top model features
- `GET /analytics/churn-by-contract` - Churn rates by contract type
- `GET /analytics/churn-by-tenure` - Churn rates by tenure brackets

**Full API Documentation:** Visit `http://localhost:8000/docs` after starting the backend

---

## 📁 Project Structure
```
customer-churn-dashboard/
│
├── backend/                          # FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI application entry point
│   │   ├── database.py               # MongoDB connection
│   │   ├── models.py                 # Pydantic data models
│   │   │
│   │   ├── routes/                   # API endpoints
│   │   │   ├── __init__.py
│   │   │   ├── customers.py          # Customer CRUD operations
│   │   │   ├── upload.py             # CSV upload handling
│   │   │   ├── ml.py                 # ML training & prediction
│   │   │   └── analytics.py          # Analytics & aggregations
│   │   │
│   │   └── services/                 # Business logic
│   │       ├── __init__.py
│   │       └── ml_service.py         # ML preprocessing & training
│   │
│   ├── models/                       # Saved ML models
│   │   └── churn_model_*.pkl
│   │
│   ├── requirements.txt              # Python dependencies
│   └── .env                          # Environment variables
│
├── frontend/                         # React frontend
│   ├── src/
│   │   ├── components/               # Reusable UI components
│   │   │   ├── layout/
│   │   │   └── common/
│   │   │
│   │   ├── pages/                    # Page components
│   │   │   ├── Dashboard.jsx         # Main dashboard
│   │   │   ├── CustomerList.jsx      # Customer table
│   │   │   ├── CustomerDetail.jsx    # Customer profile
│   │   │   ├── Upload.jsx            # Upload & ML workflow
│   │   │   └── Analytics.jsx         # Charts & insights
│   │   │
│   │   ├── services/
│   │   │   └── api.js                # API client (Axios)
│   │   │
│   │   ├── App.jsx                   # Main app component
│   │   ├── index.css                 # Tailwind imports
│   │   └── main.jsx                  # React entry point
│   │
│   ├── package.json                  # Node dependencies
│   ├── tailwind.config.js            # Tailwind configuration
│   └── vite.config.js                # Vite configuration
│
├── data/                             # CSV datasets (gitignored)
│   └── telco_churn.csv
│
└── README.md                         # This file
```

---

## 🚀 Future Improvements

### Features
- [ ] **Authentication & Authorization** - User login and role-based access
- [ ] **Real-time Predictions** - WebSocket support for live updates
- [ ] **Email Notifications** - Alert when high-risk customers are detected
- [ ] **Export Reports** - PDF/Excel export of analytics
- [ ] **Customer Segmentation** - K-means clustering for customer groups
- [ ] **A/B Testing** - Test different retention strategies
- [ ] **Automated Retraining** - Schedule model retraining on new data

### Technical Enhancements
- [ ] **Docker Containerization** - Easy deployment with Docker Compose
- [ ] **CI/CD Pipeline** - Automated testing and deployment
- [ ] **Unit Tests** - Comprehensive test coverage
- [ ] **API Rate Limiting** - Prevent abuse
- [ ] **Caching** - Redis for improved performance
- [ ] **Logging & Monitoring** - ELK stack or similar
- [ ] **Model Versioning** - MLflow for experiment tracking

### UI/UX Improvements
- [ ] **Dark Mode** - Theme toggle
- [ ] **Advanced Filters** - Multi-criteria customer filtering
- [ ] **Bulk Actions** - Select and act on multiple customers
- [ ] **Data Visualization** - More interactive charts (D3.js)
- [ ] **Progressive Web App** - Offline support
- [ ] **Accessibility** - WCAG 2.1 AA compliance

---