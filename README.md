# 🧠 Explainable AI Model Debugger  
## 🎓 DEPI Graduation Project – Phase 1

---

## 🚀 Project Vision

This project is part of a larger Explainable AI & Model Debugging Platform aimed at transforming machine learning systems from black-box models into transparent, interpretable, and debuggable AI systems.

Phase 1 delivers the core data intelligence layer along with initial UI and reporting capabilities, forming the foundation of a full-scale AI system.

---

## 🧱 System Architecture Overview

┌───────────────────────────┐
│       User Interface      │
│   (HTML / CSS Reports)   │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│      EDA Pipeline Core     │
│     (EDAPipeline Engine)  │
└─────────────┬─────────────┘
              │
 ┌────────────┼────────────┐
 │            │            │
 ▼            ▼            ▼
DataLoader  DataValidator  DataPreprocessor
                              │
                              ▼
                       OutlierHandler
                              │
                              ▼
                       ReportGenerator

---

## 🔄 End-to-End Data Flow

[ Raw Dataset ]
        │
        ▼
[ DataLoader ]
        │
        ▼
[ DataValidator ]
        │
        ▼
[ DataPreprocessor ]
        │
        ▼
[ OutlierHandler ]
        │
        ▼
[ ReportGenerator ]
        │
        ▼
[ HTML Report + UI Display ]

---

## 📌 Phase 1 Scope

✔ Data ingestion and loading  
✔ Data validation and quality checks  
✔ Data preprocessing pipeline  
✔ Outlier detection and handling  
✔ Automated report generation  
✔ Pipeline orchestration  
✔ Basic UI (HTML/CSS)  
✔ Exported HTML reports  

---

## 🧩 Core Components Breakdown

### Data Layer
- DataLoader → Reads and structures datasets  
- DataValidator → Ensures data quality  

### Processing Layer
- DataPreprocessor → Cleans and prepares data  
- OutlierHandler → Handles anomalies  

### Output Layer
- ReportGenerator → Produces insights  
- HTML Reports → Visual output  

### Control Layer
- EDAPipeline → Orchestrates full workflow  

---

## 🌐 Frontend & Reporting

┌────────────────────────────┐
│        index.html          │
│   (User Interaction UI)    │
└─────────────┬──────────────┘
              │
┌─────────────▼──────────────┐
│        style.css           │
│   (UI Styling Layer)       │
└─────────────┬──────────────┘
              │
┌─────────────▼──────────────┐
│    final_report.html       │
│ (Generated Analysis View)  │
└────────────────────────────┘

---

## 📁 Project Structure

Graduation-Project-DEPI-Round_4/
│
├── Phase_1/
│   ├── data_loader.py
│   ├── data_validator.py
│   ├── data_preprocessor.py
│   ├── outlier_handler.py
│   ├── report_generator.py
│   ├── eda_pipeline.py
│   ├── main.py
│   └── template.html
│
├── templates/
│   └── index.html
│
├── static/
│   └── css/
│       └── style.css
│
├── reports/
│   └── final_report.html
│
└── README.md

---

## 🛠️ Technologies Used

Python  
Pandas  
NumPy  
OOP (Object-Oriented Programming)  
Statistical Analysis  
HTML / CSS  

---

## 📌 Current Status

✔ Phase 1 Completed  
✔ Fully Functional EDA Pipeline  
✔ Clean Modular Architecture  
✔ Data Validation & Preprocessing  
✔ Outlier Detection Engine  
✔ Automated Reporting System  
✔ Basic UI + Styled Interface  
✔ Exportable HTML Reports  

---

## 👥 Project Team

- Mostafa Fathalla  
- Mostafa Gamal Fouda  
- Mariam Gaber  
- Tasneem Radwan  
- Samuel Adel  
- Abdelhamid Ibrahim  

---

## 📈 Future Work

[ Phase 2 ]
→ Explainable AI (SHAP / LIME)

[ Phase 3 ]
→ Machine Learning Models + Debugging

[ Phase 4 ]
→ Streamlit Dashboard + Visualization

[ Phase 5 ]
→ Full Deployment (API + Cloud)

---

## ⚡ Key Insight

Phase 1 establishes the system’s backbone by combining:

✔ Data Engineering  
✔ Data Analysis  
✔ Pipeline Automation  
✔ Basic User Interface  

This is the first step toward building a full Explainable AI system capable of transforming complex ML models into transparent and trustworthy systems.