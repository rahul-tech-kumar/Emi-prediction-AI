# 💰 EMI Prediction AI

An intelligent **Streamlit web app** that predicts a user's **EMI eligibility**, identifies **High-Risk applicants**, and estimates their **maximum affordable EMI** using **Machine Learning** models trained on real-world financial data.

---

## 🚀 Overview

**EMIPredict AI** helps financial institutions and individuals evaluate EMI loan eligibility.  
It uses a trained **classification model** to determine the risk category:
- ❌ **Not Eligible**  
- 🚨 **High Risk**  
- ✅ **Eligible**

and a **regression model** to estimate the **maximum monthly EMI** the applicant can afford.

---

## 🧠 Features

✅ EMI eligibility classification (Eligible / High Risk / Not Eligible)  
✅ Confidence scores for each class  
✅ EMI amount prediction (regression model)  
✅ Interactive data visualization using Plotly and Seaborn  
✅ Admin & Model Monitoring panel  
✅ Cached model loading for fast performance  
✅ Deployed seamlessly on **Streamlit Cloud**

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **Backend / Models** | Python, scikit-learn, joblib |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Model Tracking** | MLflow |
| **Deployment** | Streamlit Cloud |
| **Data Handling** | Pandas, NumPy |

---

## 🗂️ Project Structure

Emi-prediction-AI/
│
├── app.py # Main Streamlit app
├── emi_prediction_dataset.csv # Training dataset (required)
├── models/
│ ├── bestmodel_emi_classifier.pkl # Classification model
│ └── bestmodel_emi_regressor.pkl # Regression model
│
├── requirements.txt # All dependencies
├── README.md # Project documentation
└── screenshots/ # (Optional) App screenshots

## Install Dependencies

- pip install -r requirements.txt

## Run Locally
- streamlit run app.py
- Then open the link shown in the terminal (usually http://localhost:8501)


## 🧾 Requirements
streamlit
pandas
numpy
matplotlib
seaborn
plotly
mlflow
joblib
scikit-learn


## 📈 MLflow Tracking
- mlflow ui --port 5006
- Then open → http://localhost:5006

## 🌐 Deployment on Streamlit Cloud
- Push your repository to GitHub

- Go to Streamlit Cloud

- Connect the repo and deploy

- Make sure these files exist in GitHub:

- app.py

- requirements.txt

- models/bestmodel_emi_classifier.pkl

- models/bestmodel_emi_regressor.pkl

- emi_prediction_dataset.csv


## 🖼️ Screenshots(ML Flow UI)
<img width="1920" height="1080" alt="Screenshot (135)" src="https://github.com/user-attachments/assets/f16f63b5-38e0-4c57-9b35-89d0c490a1c6" />
<img width="1920" height="1080" alt="Screenshot (136)" src="https://github.com/user-attachments/assets/86f03b3f-711a-4f05-9549-f234c88e6870" />

# comparision of Models
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/9602253f-50a7-44c2-beb6-cc7056783ee9" />

<img width="1920" height="1080" alt="Screenshot (137)" src="https://github.com/user-attachments/assets/08560598-0e91-4957-872e-da46fdf7a721" />


## 🖼️ Screenshots(Sreamlit)
<img width="1920" height="1080" alt="Screenshot (139)" src="https://github.com/user-attachments/assets/bab49583-eda6-4fd4-a2fb-78432b5cd72a" />
<img width="1920" height="1080" alt="Screenshot (140)" src="https://github.com/user-attachments/assets/9f723820-b522-4834-baad-e0e1c454ec78" />




## 🧑‍💻 Author

- Rahul Kumar
- rahulkumar11062003@gmail.com







