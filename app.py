import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load('Mental_Health_Model.pkl')

model = load_model()


top_countries = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']


st.set_page_config(page_title="Student Mental Health Predictor", page_icon="🧠", layout="centered")
st.title("🧠 Student Mental Health ML Predictor")
st.markdown("Enter the student's details below to predict their mental health score.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    country = st.text_input("Country", value="India")
    academic_level = st.selectbox("Academic Level", ['Undergraduate', 'Graduate', 'High School'])
    most_used_platform = st.selectbox("Most Used Platform", 
        ['Facebook', 'LinkedIn', 'Instagram', 'Snapchat', 'Twitter', 'YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat'])
    purpose_of_use = st.selectbox("Purpose of Use", ['Networking', 'Education', 'Entertainment', 'News'])

with col2:
    avg_daily_usage_hours = st.number_input("Avg Daily Usage Hours", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
    daily_unlocks = st.number_input("Daily Unlocks", min_value=0, value=10)
    study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
    physical_activity_hours = st.number_input("Physical Activity Hours", min_value=0.0, max_value=24.0, value=1.0, step=0.5)
    sleep_hours_per_night = st.number_input("Sleep Hours Per Night", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    stress_level = st.selectbox("Stress Level", ['Low', 'Medium', 'High', 'Very High'])


if st.button("Predict Mental Health Score", type="primary"):
    country_group = country if country in top_countries else "Other"
    
    input_row = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Country': country,
        'Academic_Level': academic_level,
        'Most_Used_Platform': most_used_platform,
        'Purpose_Of_Use': purpose_of_use,
        'Avg_Daily_Usage_Hours': avg_daily_usage_hours,
        'Daily_Unlocks': daily_unlocks,
        'Study_Hours': study_hours,
        'Physical_Activity_Hours': physical_activity_hours,
        'Sleep_Hours_Per_Night': sleep_hours_per_night,
        'Stress_Level': stress_level,
        'Grouped_country': country_group
    }])
    
    try:
        prediction = model.predict(input_row)[0]
        st.success(f"### 🎯 Predicted Mental Health Score: {round(float(prediction), 2)}")
    except Exception as e:
        st.error(f"Prediction failed. Please check your inputs. Error: {e}")