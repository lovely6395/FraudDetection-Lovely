# Fraud Detection System — IEEE-CIS Dataset

## Project Overview
End-to-end fraud detection system using LightGBM, SHAP explainability, 
and a live Streamlit dashboard. Built as a capstone project for the 
AI & Data Analytics internship.

## Setup
pip install -r requirements.txt

## Run the Dashboard
cd dashboard
streamlit run app.py

## Live Dashboard URL
[Add your Streamlit Cloud URL here after deployment]

## Project Structure
- analysis.ipynb — main notebook with all 8 tasks
- dashboard/app.py — Streamlit dashboard
- charts/ — all saved visualizations
- data/ — train_transaction.csv, train_identity.csv

## Key Results
- Best Model: LightGBM (Tuned)
- PR-AUC: [add your value]
- ROC-AUC: [add your value]
- Fraud Capture Rate (Critical Risk tier): ~85%