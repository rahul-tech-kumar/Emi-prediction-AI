import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from mlflow.tracking import MlflowClient
import plotly.express as px

st.set_page_config(page_title="EMIPredict AI", layout="wide")
st.sidebar.title("📘 EMIPredict AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Data Explorer", "Model Monitor", "Admin"])


# Load models (only once)
@st.cache_resource
def load_models():
    classifier = joblib.load("models/bestmodel_emi_classifier.pkl")
    regressor = joblib.load("models/bestmodel_emi_regressor.pkl")
    return classifier, regressor

classifier, regressor = load_models()


# Load dataset to get feature structure
df = pd.read_csv("emi_prediction_dataset.csv", low_memory=False)
feature_cols = [
    'age','monthly_salary','years_of_employment','monthly_rent','family_size','dependents',
    'school_fees','college_fees','travel_expenses','groceries_utilities','other_monthly_expenses',
    'existing_loans','current_emi_amount','credit_score','bank_balance','emergency_fund',
    'requested_amount','requested_tenure','expense_income_ratio','total_expense',
    'saving_ratio','emi_ratio','financial_stability_index','dependents_ratio','affordability_ratio',
    'gender_0','gender_1','marital_status_Married','marital_status_Single',
    'education_Graduate','education_High School','education_Post Graduate','education_Professional',
    'employment_type_Government','employment_type_Private','employment_type_Self-employed',
    'company_type_Large Indian','company_type_MNC','company_type_Mid-size',
    'company_type_Small','company_type_Startup','house_type_Family','house_type_Own',
    'house_type_Rented','emi_scenario_E-commerce Shopping EMI','emi_scenario_Education EMI',
    'emi_scenario_Home Appliances EMI','emi_scenario_Personal Loan EMI','emi_scenario_Vehicle EMI'
]

# feature_columns = df.drop(columns=["emi_eligibility", "max_monthly_emi"]).columns.tolist()

import streamlit as st
import pandas as pd




# --- User Input Function ---
def user_input_features():
    st.sidebar.header("📋 Enter Applicant Details")

    # --- Numeric Inputs (only those used in training) ---
    age = st.sidebar.number_input("Age", 18, 70, 30)
    monthly_salary = st.sidebar.number_input("Monthly Salary (₹)", 5000, 500000, 50000, 1000)
    years_of_employment = st.sidebar.number_input("Years of Employment", 0, 50, 3)
    credit_score = st.sidebar.number_input("Credit Score", 300, 900, 700)
    bank_balance = st.sidebar.number_input("Bank Balance (₹)", 0, 5000000, 100000)
    emergency_fund = st.sidebar.number_input("Emergency Fund (₹)", 0, 5000000, 50000)
    requested_amount = st.sidebar.number_input("Requested Loan Amount (₹)", 10000, 5000000, 500000)
    requested_tenure = st.sidebar.number_input("Requested Tenure (Months)", 6, 240, 60)
    emi_ratio = st.sidebar.slider("EMI Ratio", 0.0, 1.0, 0.2)
    affordability_ratio = st.sidebar.slider("Affordability Ratio", 0.0, 1.0, 0.7)
    dependents_ratio = st.sidebar.slider("Dependents Ratio", 0.0, 1.0, 0.2)
    saving_ratio = st.sidebar.slider("Saving Ratio", 0.0, 1.0, 0.3)
    expense_income_ratio = st.sidebar.slider("Expense/Income Ratio", 0.0, 1.0, 0.5)

    # --- Categorical Inputs ---
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married"])
    education = st.sidebar.selectbox("Education", ["Graduate", "High School", "Post Graduate", "Professional"])
    employment_type = st.sidebar.selectbox("Employment Type", ["Government", "Private", "Self-employed"])
    company_type = st.sidebar.selectbox("Company Type", ["MNC", "Startup", "Large Indian", "Mid-size", "Small"])
    house_type = st.sidebar.selectbox("House Type", ["Own", "Rented", "Family"])
    emi_scenario = st.sidebar.selectbox("EMI Scenario", [
        "E-commerce Shopping EMI", "Education EMI", "Home Appliances EMI", "Personal Loan EMI", "Vehicle EMI"
    ])

    # --- Define all columns used in training ---
    feature_cols = [
        'age','monthly_salary','years_of_employment','credit_score','bank_balance','emergency_fund',
        'requested_amount','requested_tenure','emi_ratio','affordability_ratio','dependents_ratio',
        'saving_ratio','expense_income_ratio','gender_0','gender_1','marital_status_Married',
        'marital_status_Single','education_Graduate','education_High School','education_Post Graduate',
        'education_Professional','employment_type_Government','employment_type_Private',
        'employment_type_Self-employed','company_type_Large Indian','company_type_MNC',
        'company_type_Mid-size','company_type_Small','company_type_Startup','house_type_Family',
        'house_type_Own','house_type_Rented','emi_scenario_E-commerce Shopping EMI',
        'emi_scenario_Education EMI','emi_scenario_Home Appliances EMI',
        'emi_scenario_Personal Loan EMI','emi_scenario_Vehicle EMI'
    ]

    # --- Initialize all columns with 0 ---
    input_dict = {col: 0 for col in feature_cols}

    # --- Fill numeric fields ---
    input_dict.update({
        "age": age,
        "monthly_salary": monthly_salary,
        "years_of_employment": years_of_employment,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
        "emi_ratio": emi_ratio,
        "affordability_ratio": affordability_ratio,
        "dependents_ratio": dependents_ratio,
        "saving_ratio": saving_ratio,
        "expense_income_ratio": expense_income_ratio,
    })

    # --- One-hot Encoding ---
    input_dict[f"gender_{1 if gender.lower()=='male' else 0}"] = 1
    input_dict[f"marital_status_{marital_status}"] = 1
    input_dict[f"education_{education}"] = 1
    input_dict[f"employment_type_{employment_type}"] = 1
    input_dict[f"company_type_{company_type}"] = 1
    input_dict[f"house_type_{house_type}"] = 1
    input_dict[f"emi_scenario_{emi_scenario}"] = 1

    return pd.DataFrame([input_dict])



# --- Page: Prediction ---
if page == "Prediction":
    st.title("🔮 EMI Prediction")
    import streamlit as st

    # Stylish header with emojis and icons
    st.markdown("""
        <style>
            .title-container {
                background-color: #f0f2f6;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .title-text {
                font-size: 26px;
                font-weight: 700;
                color: #2c3e50;
                text-align: center;
            }
            .subtitle-text {
                font-size: 16px;
                color: #4a4a4a;
                text-align: center;
                margin-top: 8px;
            }
            .divider {
                border-bottom: 2px solid #3498db;
                width: 80%;
                margin: 15px auto;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="title-container">
        <div class="title-text">💰 EMI Eligibility & Loan Insights Dashboard</div>
        <div class="divider"></div>
        <div class="subtitle-text">
            Predict your <b>EMI Eligibility</b> and estimate the <b>Maximum EMI Amount</b> 
            based on your salary, expenses, and financial stability indicators.
            <br><br>
            🔍 Enter your details below to receive instant, data-driven predictions powered by 
            <b>Machine Learning</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Info box (optional)
    st.info("✨ Tip: Adjust your salary, dependents, and loan preferences to see how your eligibility changes dynamically!")

    input_df = user_input_features()

    # if st.button("Predict Eligibility"):
    #     pred = classifier.predict(input_df)
    #     st.success(f"Eligibility: {'✅ Eligible' if pred[0] else '❌ Not Eligible'}")
    #     st.balloons()
    if st.button("Predict Eligibility"):
        pred_proba = classifier.predict_proba(input_df)[0]
        pred_class = np.argmax(pred_proba)

        class_labels = {0: "❌ Not Eligible", 1: "🚨 High Risk", 2: "✅ Eligible"}
        risk_colors = {0: "red", 1: "orange", 2: "green"}

        st.markdown(f"### Prediction Result: <span style='color:{risk_colors[pred_class]}'>{class_labels[pred_class]}</span>", unsafe_allow_html=True)
        st.progress(float(pred_proba[pred_class]))

        st.info(f"""
        **Model Confidence:**  
        - Not Eligible: {pred_proba[0]:.2f}  
        - High Risk: {pred_proba[1]:.2f}  
        - Eligible: {pred_proba[2]:.2f}
        """)

        if pred_class == 0:
            st.error("💸 Applicant is not eligible due to low financial stability or insufficient income.")
        elif pred_class == 1:
            st.warning("⚠️ Applicant is in the High Risk category — EMI approval unlikely without additional documents.")
        else:
            st.success("✅ Applicant is financially strong and likely eligible for EMI approval.")
            st.balloons()
            
        st.subheader("📊 Risk Probability Distribution")
        st.bar_chart({
            "Not Eligible": [pred_proba[0]],
            "High Risk": [pred_proba[1]],
            "Eligible": [pred_proba[2]]
        })



    if st.button("Predict Maximum EMI"):
        emi_pred = regressor.predict(input_df)
        st.success(f"Predicted Maximum EMI: ₹{emi_pred[0]:,.2f}")
        st.balloons()





if page == "Home":
    st.title("🏠 EMI Prediction Home")
    st.markdown(
    """
    <div style='
        background-color: #D6EAF8;
        padding: 15px;
        border-radius: 10px;
        color: #154360;
        font-weight: bold;
        font-size:16px;
    '>
        This application predicts EMI eligibility and max monthly EMI based on user financial data.
    </div>
    """,
    unsafe_allow_html=True
    )


    df = pd.read_csv("emi_prediction_dataset.csv")
    st.markdown("---")
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Summary Statistics")
    st.write(df.describe())
    
elif page=="Data Explorer":
    st.title("📊 Data Visualization")
    df = pd.read_csv("emi_prediction_dataset.csv")

    st.subheader("🏛️ Employment Type")
    employment = st.selectbox("Select Employment Type", df['employment_type'].unique())
    filtered_df = df[df['employment_type'] == employment]

    st.dataframe(filtered_df.head())
    
    st.subheader("📈 Visualization")
    # Countplot using Seaborn + Matplotlib
    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(x="employment_type", data=df, palette="viridis", ax=ax)
    ax.set_title("Employment Type Distribution")
    st.pyplot(fig)
    
    st.markdown("---")
    # Existing Loans vs EMI Eligibility
    st.subheader("🧮 Existing Loans vs EMI Eligibility")
    
    fig,ax=plt.subplots(figsize=(6,4))
    sns.countplot(x='existing_loans', hue='emi_eligibility', data=df)
    ax.set_title("Existing Loans")
    st.pyplot(fig)
    
    
elif page=="Model Monitor":
    
    
    st.title("📊 Model Performance Monitor")

    client = MlflowClient()

    # ------------------ CLASSIFICATION SECTION -------------------------------------
    st.subheader("🎯 Classification Model Performance")

    exp_cls = client.get_experiment_by_name("EMI_Classification_Model")

    if exp_cls is not None:
        runs_cls = client.search_runs(exp_cls.experiment_id)

        if runs_cls:
            df_cls = pd.DataFrame([{
                "Model": r.data.tags.get("mlflow.runName"),
                "Run ID": r.info.run_id,
                "Accuracy": r.data.metrics.get("accuracy"),
                "F1 Score": r.data.metrics.get("f1_score"),
                "MSE": r.data.metrics.get("MSE"),
                "MAE": r.data.metrics.get("MAE")
            } for r in runs_cls])

            st.dataframe(df_cls)
            
            
            st.subheader("📈 F1 Score vs Models")
            
            fig = px.line(df_cls, y="F1 Score", x="Model", title="F1 Score Across Models")
            st.plotly_chart(fig)
            
            st.subheader("📈 F1 Score vs Accuracy")

            # Plot F1 vs Accuracy
            fig_cls = px.bar(
                df_cls,
                x="Model",
                y="F1 Score",
                color="Accuracy",
                title="Classification Models: F1 Score vs Accuracy",
                text_auto=".3f"
            )
            st.plotly_chart(fig_cls, use_container_width=True)
        else:
            st.warning("No classification runs found.")
    else:
        st.error("Classification experiment not found.")


    # ------------------ REGRESSION SECTION -----------------------------
    st.markdown("---")
    st.subheader("🎯 Regression Model Performance")

    exp_reg = client.get_experiment_by_name("EMI_Regression_Model")

    if exp_reg is not None:
        runs_reg = client.search_runs(exp_reg.experiment_id)

        if runs_reg:
            df_reg = pd.DataFrame([{
                "Model": r.data.tags.get("mlflow.runName"),
                "Run ID": r.info.run_id,
                "R² Score": r.data.metrics.get("R2_score"),
                "RMSE": r.data.metrics.get("RMSE"),
                "MAE": r.data.metrics.get("MAE"),
                "MSE": r.data.metrics.get("MSE")
            } for r in runs_reg])

            st.dataframe(df_reg)
            
            st.subheader("📈 R² Score vs Models")
            
            fig = px.line(df_reg, y="R² Score", x="Model", title="R² Score Across Models")
            st.plotly_chart(fig)
            
            st.subheader('📈 R² Score vs RMSE ')

            # Plot R² comparison
            fig_reg = px.bar(
                df_reg,
                x="Model",
                y="R² Score",
                color="RMSE",
                title="Regression Models: R² Score vs RMSE",
                text_auto=".3f"
            )
            st.plotly_chart(fig_reg, use_container_width=True)
        else:
            st.warning("No regression runs found.")
    else:
        st.error("Regression experiment not found.")
        
        
elif page=="Admin":
    st.title("🛠️ Admin Dashboard")
        
        # ====== HEADER ======
    st.markdown("""
        <style>
            .admin-title {
                font-size: 28px;
                font-weight: 700;
                color: #1f3b73;
                text-align: center;
                margin-bottom: 10px;
            }
            .divider {
                border-bottom: 2px solid #3498db;
                width: 70%;
                margin: 10px auto 25px auto;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="admin-title">🧑‍💼 Admin Dashboard — EMI Eligibility Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ====== LOAD DATA ======
    st.sidebar.header("📂 Data Controls")
    uploaded_file = st.sidebar.file_uploader("Upload dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.success(f"✅ Data Loaded Successfully! {df.shape[0]} records found.")
        
        # Optional filters
        st.sidebar.header("🔍 Filters")
        eligible_filter = st.sidebar.multiselect("Filter by EMI Eligibility", options=df['emi_eligibility'].unique())
        gender_filter = st.sidebar.multiselect("Filter by Gender", options=df['gender'].unique())
        
        if eligible_filter:
            df = df[df['emi_eligibility'].isin(eligible_filter)]
        if gender_filter:
            df = df[df['gender'].isin(gender_filter)]

        # ====== METRICS SECTION ======
        st.markdown("### 📈 Key Insights")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Applicants", len(df))
        col2.metric("Eligible Count", (df['emi_eligibility'] == 'Eligible').sum())
        col3.metric("High Risk Count", (df['emi_eligibility'] == 'High_Risk').sum())

        # ====== DATA TABLE ======
        st.markdown("### 📋 Applicant Overview")
        st.dataframe(df.head(10), use_container_width=True)

        # ====== VISUALIZATIONS ======
        st.markdown("### 📊 Data Visualizations")

        tab1, tab2, tab3 = st.tabs(["💰 Salary & Expenses", "📉 Credit & EMI", "🧮 Financial Ratios"])

        with tab1:
            fig1 = px.histogram(df, x='monthly_salary', color='emi_eligibility', nbins=30, title="Salary Distribution by Eligibility")
            st.plotly_chart(fig1, use_container_width=True)
            
            if 'expense_income_ratio' in df.columns:
                fig2 = px.scatter(df, x='monthly_salary', y='expense_income_ratio', color='emi_eligibility',
                                title="Expense-Income Ratio vs Salary")
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            if 'credit_score' in df.columns:
                fig3 = px.box(df, x='emi_eligibility', y='credit_score', color='emi_eligibility',
                            title="Credit Score Distribution by Eligibility")
                st.plotly_chart(fig3, use_container_width=True)

            if 'emi_ratio' in df.columns:
                fig4 = px.violin(df, x='emi_eligibility', y='emi_ratio', color='emi_eligibility',
                                title="EMI Ratio by Eligibility")
                st.plotly_chart(fig4, use_container_width=True)

        with tab3:
            if 'saving_ratio' in df.columns and 'affordability_ratio' in df.columns:
                fig5 = px.scatter(df, x='saving_ratio', y='affordability_ratio', color='emi_eligibility',
                                size='monthly_salary', hover_data=['credit_score'],
                                title="Savings vs Affordability (Colored by Eligibility)")
                st.plotly_chart(fig5, use_container_width=True)

    else:
        st.warning("⚠️ Please upload a dataset to view the admin dashboard.")
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Admin Panel Info")
    st.sidebar.info(
        """
        Manage loan records, view user predictions,  
        and monitor model performance here.
        """
    )
    st.sidebar.caption("👨‍💼 Admin Access | © 2025 Rahul Kumar")






# ---------------------------Sidebar footer-------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 💼 About This App")
st.sidebar.info(
    """
    **EMI Prediction System**  
    Developed by *Rahul Kumar*  
    Version: 1.0.0  
    Powered by Streamlit & XGBoost
    """
)
st.sidebar.caption("© 2025 All rights reserved.")
