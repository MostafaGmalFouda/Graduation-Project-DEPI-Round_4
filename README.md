# Graduation-Project-DEPI-Round_4# 
## 🧠 Explainable AI Model Debugger  
## 🎓 DEPI Graduation Project – Phase 1

---

## 🚀 Project Vision

This project is part of a larger Explainable AI & Model Debugging Platform aimed at transforming machine learning systems from black-box models into transparent, interpretable, and debuggable AI systems.

Phase 1 focuses on building the core data processing and analytical pipeline architecture, which serves as the foundation for future explainability, visualization, and AI debugging capabilities.

---

## 📌 Phase 1 Scope

This phase implements a complete EDA + Data Processing Pipeline System using a modular OOP architecture.

It includes:
- Data ingestion and loading  
- Data validation and quality control  
- Data preprocessing pipeline  
- Outlier detection and handling  
- Automated reporting system  
- End-to-end pipeline orchestration  

---

## 🏗️ System Architecture (Phase 1)

DataLoader  
↓  
DataValidator  
↓  
DataPreprocessor  
↓  
OutlierHandler  
↓  
ReportGenerator  
↓  
EDAPipeline (Orchestrator)  

Each module is designed to be independent, reusable, and scalable for future AI explainability integration.

---

## 🧩 Core Components

### DataLoader
- Load CSV files  
- Load Excel files  
- Return structured dataset  

### DataValidator
- Detect missing values  
- Validate data types  
- Identify duplicates  
- Generate data quality reports  

### DataPreprocessor
- Handle missing values using strategies  
- Convert data types  
- Remove duplicates  
- Initial outlier handling  
- Return cleaned dataset  

### OutlierHandler
- IQR-based detection  
- Z-score detection  
- Remove outliers  
- Cap extreme values  

### ReportGenerator
- Descriptive statistics  
- Correlation matrix  
- Automated reports  

### EDAPipeline (Orchestrator)
- Controls full workflow execution  
- Connects all modules  
- Ensures smooth data flow from raw data to final output  

---

## 🔄 Data Flow

Raw Dataset → DataLoader → DataValidator → DataPreprocessor → OutlierHandler → ReportGenerator → Final Clean Data + Report  

---

## 🛠️ Technologies Used

Python  
Pandas  
NumPy  
Object-Oriented Programming (OOP)  
Data Analysis Techniques  
Statistical Methods  

---

## 📁 Project Structure

/Phase_1  
- data_loader.py  
- data_validator.py  
- data_preprocessor.py  
- outlier_handler.py  
- report_generator.py  
- eda_pipeline.py  
- main.py  

---

## 📌 Current Status

✔ Phase 1 Completed  
✔ Full EDA Pipeline Implemented  
✔ Modular OOP Architecture  
✔ Data Validation System  
✔ Preprocessing Engine  
✔ Outlier Detection System  
✔ Reporting Module  
✔ Pipeline Orchestrator  

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

- Explainable AI integration (SHAP / LIME)  
- Machine Learning model layer  
- Bias detection module  
- Streamlit dashboard  
- REST API (Flask / FastAPI)  
- Real-time AI debugging system  

---

## ⚡ Key Insight

This project establishes the foundation of an intelligent AI debugging system that will evolve into a full Explainable AI platform capable of:
- Understanding model behavior  
- Explaining predictions  
- Detecting bias  
- Debugging ML pipelines  