import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Prediction App", layout="wide")

st.title("Churn Prediction Model Comparison")
st.markdown("""
This application allows you to upload a customer data CSV file, 
compare various machine learning models for churn prediction, 
and make predictions on new customer data.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview:")
    st.write(data.head())

    # Data Preprocessing
    data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(data[['Gender', 'Geography']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Gender', 'Geography']))
    data = pd.concat([data.drop(['Gender', 'Geography'], axis=1), encoded_df], axis=1)

    x = data.drop('Exited', axis=1)
    y = data['Exited']
    
    # Splitting the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    # Feature scaling and handling class imbalance with SMOTE
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    smote = SMOTE(random_state=0)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Decision Tree": tree.DecisionTreeClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric='logloss'),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "LightGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0)
    }

    accuracy_results = {}

    for name, model in models.items():
        with st.spinner(f"Training {name}..."):
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)
            accuracy_results[name] = accuracy

    st.success("Training complete!")
    
    # Display Model Comparison Results
    st.subheader("Model Comparison")
    results_df = pd.DataFrame.from_dict(accuracy_results, orient='index', columns=['Accuracy'])
    results_df.sort_values('Accuracy', ascending=False, inplace=True)

    st.write(results_df.style.highlight_max(axis=0))

    best_model_name = results_df.index[0]
    best_model = models[best_model_name]

    st.subheader(f"Best Model: {best_model_name}")

    if best_model_name in ["Random Forest", "XGBoost", "Gradient Boosting", "LightGBM", "CatBoost"]:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(x_test)
        original_feature_names = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                                  'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                                  'Gender_Female', 'Gender_Male',
                                  'Geography_France', 'Geography_Germany', 'Geography_Spain']
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, x_test, feature_names=original_feature_names, plot_type="bar", show=False)
        st.subheader("SHAP Summary Plot")
        st.pyplot(fig)

# Prediction Section
st.subheader("Simulate Churn Prediction")
st.markdown("""
Adjust the fields below to see how they affect the probability of customer churn.
""")

# Create input fields for user interaction
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", value=600.0, format="%.2f", min_value=300.0, max_value=850.0)
    age = st.number_input("Age", value=35, format="%d", min_value=18, max_value=100)
    tenure = st.number_input("Tenure", value=5, format="%d", min_value=0, max_value=10)

with col2:
    balance = st.number_input("Balance", value=50000.0, format="%.2f", min_value=0.0)
    num_of_products = st.number_input("Number of Products", value=1, format="%d", min_value=1, max_value=5)
    has_credit_card = st.selectbox("Has Credit Card?", options=[0, 1])

col3, col4 = st.columns(2)

with col3:
    is_active_member = st.selectbox("Is Active Member?", options=[0, 1])
    estimated_salary = st.number_input("Estimated Salary", value=100000.0, format="%.2f", min_value=0.0)

with col4:
    geography = st.selectbox("Select Geography", options=['France', 'Spain', 'Germany'])
    gender = st.selectbox("Select Gender", options=['Female', 'Male'])

# Prepare input data for prediction
input_data = {
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_credit_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary,
    "Geography": geography,
    "Gender": gender
}

input_df = pd.DataFrame([input_data])
encoded_input_features = encoder.transform(input_df[['Gender', 'Geography']])
encoded_input_df = pd.DataFrame(
    encoded_input_features,
    columns=encoder.get_feature_names_out(['Gender', 'Geography'])
)
input_df = pd.concat([input_df.drop(['Gender', 'Geography'], axis=1), encoded_input_df], axis=1)

expected_feature_order = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Gender_Female', 'Gender_Male',
    'Geography_France', 'Geography_Germany', 'Geography_Spain'
]
input_df = input_df[expected_feature_order]
input_scaled = scaler.transform(input_df)

# Make predictions using the best model
if best_model_name in ["Random Forest", "XGBoost", "Gradient Boosting", "LightGBM", "CatBoost"]:
    prediction_class = best_model.predict(input_scaled)
    probability_class_1 = best_model.predict_proba(input_scaled)[0][1]
else:
    prediction_class = (best_model.predict(input_scaled) > 0.5).astype(int)
    probability_class_1 = best_model.predict(input_scaled)[0][0]

# Display the results
if prediction_class[0] == 1:
    st.error(f"The customer is likely to churn. Probability: {probability_class_1:.2f}")
else:
    st.success(f"The customer is likely to stay. Probability: {1 - probability_class_1:.2f}")

# Displaying the probability as a metric
st.metric(label="Churn Probability", value=f"{probability_class_1:.2%}")