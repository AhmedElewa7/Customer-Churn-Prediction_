import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Prediction App", layout="centered")
st.title("Customer Churn Prediction App")
st.image("Project Photo.jfif")
st.write("Enter customer details to predict churn probability.")

model = joblib.load("churn model.pkl")

def predict_churn(data): # data is a dataframe with a single row 
    return model.predict_proba(data)[:, 1][0]

with st.sidebar:
    credit_score = st.slider("Credit Score", 350, 850, 600)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 18, 92, 35)
    tenure = st.slider("Tenure", 0, 10, 5)
    balance = st.number_input("Balance", value=0.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.radio("Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.radio("Is Active Member?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    estimated_salary = st.number_input("Estimated Salary", value=0.0, format="%.2f")

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary
    }])

    prob = predict_churn(input_df)

    if prob > 0.4:
        st.error(f"⚠ High churn risk: {prob:.2%}")
    else:
        st.success(f"✅ Low churn risk: {prob:.2%}")


# نجيب أسماء الأعمدة بعد preprocessing
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# نجيب الأهميات من Random Forest
importances = model.named_steps["classifier"].feature_importances_

fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)
fi_df["feature"] = (
    fi_df["feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
    .str.replace("bin__", "", regex=False))
top_5_features = fi_df.head(5)
st.subheader("Top 5 Important Features Affecting Churn Prediction")
st.dataframe(top_5_features)
st.bar_chart(top_5_features.set_index("feature")["importance"])

