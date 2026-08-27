import os
import sys


os.system(f"{sys.executable} -m pip install --upgrade pip")
os.system(f"{sys.executable} -m pip install joblib pandas scikit-learn==1.6.1")

import streamlit as st
import pandas as pd
import joblib



st.set_page_config(page_title="Mental Health Signal", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load('Mental_Health_Model.pkl')

model = load_model()



top_countries = ['India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']
platforms = ['Facebook', 'LinkedIn', 'Instagram', 'Snapchat', 'Twitter', 'YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat']
purposes = ['Networking', 'Education', 'Entertainment', 'News']
academic_levels = ['High School', 'Undergraduate', 'Graduate']
stress_levels = ['Low', 'Medium', 'High', 'Very High']


st.title(" Mental Health Signal")
st.markdown("A quick read on how habits, screen time, and stress are trending — modeled from your daily rhythm, not a diagnosis.")
st.markdown("---")

# Form
with st.form("prediction_form"):
    st.subheader("1. Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=20, step=1)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        country = st.text_input("Country (e.g., India, USA)", value="India").strip()

    st.subheader("2. Academic & Digital Habits")
    col1, col2 = st.columns(2)
    with col1:
        academic_level = st.selectbox("Academic Level", academic_levels)
        most_used_platform = st.selectbox("Most Used Platform", platforms)
    with col2:
        purpose_of_use = st.selectbox("Primary Purpose", purposes)
        avg_daily_usage_hours = st.number_input("Avg. Daily Screen Time (hours)", min_value=0.0, max_value=24.0, value=4.5, step=0.1)
    
    daily_unlocks = st.number_input("Daily Phone Unlocks", min_value=0, value=50, step=1)

    st.subheader("3. Lifestyle & Stress")
    col1, col2, col3 = st.columns(3)
    with col1:
        study_hours = st.number_input("Study Hours / Day", min_value=0.0, max_value=24.0, value=6.0, step=0.1)
    with col2:
        physical_activity_hours = st.number_input("Physical Activity / Day (hours)", min_value=0.0, max_value=24.0, value=1.0, step=0.1)
    with col3:
        sleep_hours_per_night = st.number_input("Sleep / Night (hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.1)
    
    stress_level = st.selectbox("Perceived Stress Level", stress_levels)

    submit_button = st.form_submit_button("Read My Signal", type="primary", use_container_width=True)


if submit_button:
    with st.spinner("Analyzing your data..."):
        #
        grouped_country = country if country in top_countries else 'Other'

       
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Grouped_Country': grouped_country,
            'Academic_Level': academic_level,
            'Most_Used_Platform': most_used_platform,
            'Purpose_Of_Use': purpose_of_use,
            'Avg_Daily_Usage_Hours': avg_daily_usage_hours,
            'Daily_Unlocks': daily_unlocks,
            'Study_Hours': study_hours,
            'Physical_Activity_Hours': physical_activity_hours,
            'Sleep_Hours_Per_Night': sleep_hours_per_night,
            'Stress_Level': stress_level
        }])

        try:
            
            prediction = model.predict(input_data)[0]
            score = float(round(prediction, 2))
            
            st.markdown("---")
            st.success("✅ Prediction Complete!")
            

            st.markdown(f"### Your Mental Health Signal Score: `{score} / 10`")
            
            if score < 4:
                st.warning("**Signal: Strained**\n\nYour responses suggest elevated strain right now. Small shifts in sleep or screen time can go a long way.")
            elif score < 7:
                st.info("**Signal: Balanced**\n\nYour rhythm looks fairly steady, with some room to recover and reset.")
            else:
                st.success("**Signal: Strong**\n\nYour habits point to a well-supported, resilient baseline. Keep it up!")
                
        except Exception as e:
            st.error(f"⚠️ An error occurred during prediction: {e}")
            st.write("Error!!'")


st.markdown("---")
st.caption("Built for informational purposes only — this is not a clinical assessment. If you're struggling, please talk to someone you trust.")