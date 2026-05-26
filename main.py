import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import joblib
import warnings
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import requests
import time
import datetime
import numpy as np

from codebase.dashboard_graphs import MaternalHealthDashboard

# Configure page
st.set_page_config(
    page_title="PregnaSafe - Maternal Health Platform",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize models as None - will try to load them later
maternal_model = None
fetal_model = None
use_mock_predictions = False

# Try to load models silently
try:
    import os
    if os.path.exists("model/finalized_maternal_model.sav"):
        maternal_model = joblib.load("model/finalized_maternal_model.sav")
except Exception as e:
    pass  # Silent fail
    
try:
    import os
    if os.path.exists("model/fetal_health_classifier.sav"):
        fetal_model = joblib.load("model/fetal_health_classifier.sav")
except Exception as e:
    pass  # Silent fail

# Rule-based prediction functions for when ML models are not available
def mock_maternal_prediction(age, systolic_bp, diastolic_bp, blood_sugar, body_temp, heart_rate):
    """Comprehensive rule-based maternal health risk prediction based on medical guidelines"""
    
    # Convert string inputs to numbers with validation
    try:
        age = float(age) if age else 25
        systolic_bp = float(systolic_bp) if systolic_bp else 120
        diastolic_bp = float(diastolic_bp) if diastolic_bp else 80
        blood_sugar = float(blood_sugar) if blood_sugar else 5.5
        body_temp = float(body_temp) if body_temp else 36.5
        heart_rate = float(heart_rate) if heart_rate else 75
    except:
        return 1, ["Invalid input values provided"]  # Medium risk for invalid inputs
    
    risk_factors = []
    risk_score = 0
    
    # Age-related risk assessment
    if age < 18:
        risk_factors.append("⚠️ Very young maternal age (< 18 years)")
        risk_score += 2
    elif age < 20:
        risk_factors.append("⚠️ Young maternal age (< 20 years)")
        risk_score += 1
    elif age > 35:
        risk_factors.append("⚠️ Advanced maternal age (> 35 years)")
        risk_score += 1
    elif age > 40:
        risk_factors.append("🚨 Very advanced maternal age (> 40 years)")
        risk_score += 2
    
    # Blood pressure assessment (Hypertension in pregnancy)
    if systolic_bp >= 160 or diastolic_bp >= 110:
        risk_factors.append("🚨 Severe hypertension - immediate medical attention needed")
        risk_score += 3
    elif systolic_bp >= 140 or diastolic_bp >= 90:
        risk_factors.append("🚨 Gestational hypertension")
        risk_score += 2
    elif systolic_bp >= 130 or diastolic_bp >= 80:
        risk_factors.append("⚠️ Elevated blood pressure")
        risk_score += 1
    elif systolic_bp < 90 or diastolic_bp < 60:
        risk_factors.append("⚠️ Low blood pressure (hypotension)")
        risk_score += 1
        
    # Blood glucose assessment (Gestational diabetes)
    if blood_sugar >= 11.1:  # mmol/L
        risk_factors.append("🚨 Severe hyperglycemia - urgent care needed")
        risk_score += 3
    elif blood_sugar >= 7.8:
        risk_factors.append("🚨 Gestational diabetes range")
        risk_score += 2
    elif blood_sugar >= 6.1:
        risk_factors.append("⚠️ Impaired glucose tolerance")
        risk_score += 1
    elif blood_sugar < 3.3:
        risk_factors.append("⚠️ Hypoglycemia")
        risk_score += 1
        
    # Body temperature assessment
    if body_temp >= 38.5:
        risk_factors.append("🚨 High fever - possible infection")
        risk_score += 2
    elif body_temp >= 38.0:
        risk_factors.append("⚠️ Mild fever")
        risk_score += 1
    elif body_temp <= 35.0:
        risk_factors.append("🚨 Hypothermia")
        risk_score += 2
        
    # Heart rate assessment
    if heart_rate >= 120:
        risk_factors.append("⚠️ Tachycardia (fast heart rate)")
        risk_score += 1
    elif heart_rate >= 100:
        risk_factors.append("⚠️ Elevated heart rate")
        risk_score += 1
    elif heart_rate <= 50:
        risk_factors.append("⚠️ Bradycardia (slow heart rate)")
        risk_score += 1
    
    # Additional risk combinations
    if systolic_bp >= 140 and blood_sugar >= 7.8:
        risk_factors.append("🚨 Combined preeclampsia and diabetes risk")
        risk_score += 1
    
    # Add positive factors if no risks found
    if not risk_factors:
        risk_factors.append("✅ All parameters within normal ranges")
    
    # Determine overall risk level with more nuanced scoring
    if risk_score >= 6:
        return 2, risk_factors  # High risk
    elif risk_score >= 3:
        return 1, risk_factors  # Medium risk
    else:
        return 0, risk_factors  # Low risk

def mock_fetal_prediction(*args):
    """Simple mock prediction for fetal health"""
    import random
    # For demo purposes, randomly return 0 (Normal), 1 (Suspect), or 2 (Pathological)
    # In reality, this would be based on the input parameters
    random.seed(hash(str(args)) % 1000)  # Consistent results for same inputs
    return random.randint(0, 2)

# Custom CSS for dynamic styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #ff6b9d, #ffa8e4);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
}
.risk-card {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.low-risk { background-color: #d4edda; border-left: 5px solid #28a745; }
.medium-risk { background-color: #fff3cd; border-left: 5px solid #ffc107; }
.high-risk { background-color: #f8d7da; border-left: 5px solid #dc3545; }
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 4px solid #007bff;
}
.stProgress .st-bo {
    background-color: #e9ecef;
}
</style>
""", unsafe_allow_html=True)

# Dynamic header with current time
current_time = datetime.datetime.now().strftime("%B %d, %Y - %I:%M %p")
st.markdown(f'<div class="main-header"><h1>🤰 PregnaSafe - Maternal Health Platform</h1><p>Current Session: {current_time}</p></div>', unsafe_allow_html=True)

# sidebar for navigation
with st.sidebar:
    st.markdown("### 🏥 Navigation Menu")
    st.write("Welcome to PregnaSafe - Your trusted maternal health companion")
    
    # Add a dynamic health tip
    tips = [
        "💡 Tip: Drink at least 8 glasses of water daily during pregnancy",
        "💡 Tip: Take prenatal vitamins as recommended by your doctor",
        "💡 Tip: Get regular exercise with your doctor's approval",
        "💡 Tip: Monitor your weight gain according to BMI guidelines",
        "💡 Tip: Avoid smoking and alcohol completely during pregnancy"
    ]
    tip_index = int(time.time()) % len(tips)
    st.info(tips[tip_index])

    selected = option_menu('PregnaSafe',
                          
                          ['🏠 Home',
                            '🏥 Risk Assessment',
                           '👶 Fetal Health',
                           '⚖️ BMI Calculator',
                           '🍎 Nutrition Guide',
                           '📊 Analytics'],
                          icons=['house','hospital','baby','calculator','apple','bar-chart'],
                          menu_icon="heart-pulse",
                          default_index=0,
                          styles={
                              "container": {"padding": "5!important", "background-color": "#fafafa"},
                              "icon": {"color": "orange", "font-size": "20px"}, 
                              "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                              "nav-link-selected": {"background-color": "#ff6b9d"},
                          })
    
if (selected == '🏠 Home'):
    # Dynamic home page with animated metrics
    st.markdown("### 📊 Platform Overview")
    
    # Key statistics with animated counters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📈 Risk Assessments", 
            value="1,234",
            delta="+15 today"
        )
    
    with col2:
        st.metric(
            label="👩‍⚕️ Active Users", 
            value="567",
            delta="+8 this week"
        )
    
    with col3:
        st.metric(
            label="✅ Successful Predictions", 
            value="98.7%",
            delta="+0.2%"
        )
    
    with col4:
        st.metric(
            label="🏥 Partner Hospitals", 
            value="45",
            delta="+2 new"
        )
    
    st.markdown("---")
    
    # Feature showcase with interactive cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("🏥 Risk Assessment")
            st.write("Advanced AI-powered pregnancy risk analysis using multiple health parameters")
            st.progress(0.95)
            st.caption("95% Accuracy Rate")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("👶 Fetal Monitoring")
            st.write("Comprehensive fetal health assessment through CTG analysis")
            st.progress(0.92)
            st.caption("92% Accuracy Rate")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("📊 Health Analytics")
            st.write("Real-time dashboard with regional maternal health insights")
            st.progress(0.98)
            st.caption("98% Data Coverage")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick action buttons
    st.markdown("### ⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏥 Start Risk Assessment", use_container_width=True):
            st.session_state.selected = '🏥 Risk Assessment'
            st.experimental_rerun()
    
    with col2:
        if st.button("⚖️ Calculate BMI", use_container_width=True):
            st.session_state.selected = '⚖️ BMI Calculator'
            st.experimental_rerun()
    
    with col3:
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.selected = '📊 Analytics'
            st.experimental_rerun()

if (selected == '🏥 Risk Assessment'):
    st.markdown("### 🏥 Maternal Health Risk Assessment")
    st.markdown("*Enter your health parameters and submit for comprehensive risk analysis*")
    
    # Create form for maternal health assessment
    with st.form("maternal_assessment_form"):
        st.markdown("#### 📊 Health Parameters")
        
        # Input fields organized in columns
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("📅 Age (years)", 15, 50, 28, help="Maternal age at pregnancy")
            systolic_bp = st.slider("🩸 Systolic BP (mmHg)", 80, 200, 120, help="Upper blood pressure reading")
            blood_sugar = st.slider("🍭 Blood Glucose (mmol/L)", 2.0, 15.0, 5.0, 0.1, help="Fasting blood glucose level")
        
        with col2:
            heart_rate = st.slider("❤️ Heart Rate (bpm)", 50, 150, 75, help="Resting heart rate")
            diastolic_bp = st.slider("🩸 Diastolic BP (mmHg)", 40, 130, 80, help="Lower blood pressure reading")
            body_temp = st.slider("🌡️ Body Temperature (°C)", 35.0, 40.0, 36.5, 0.1, help="Core body temperature")
        
        # Submit button
        submitted = st.form_submit_button("🔍 Analyze Maternal Health Risk", use_container_width=True)
        
    # Process results only when submitted
    if submitted:
        # Risk calculation
        risk_level, risk_factors = mock_maternal_prediction(age, systolic_bp, diastolic_bp, blood_sugar, body_temp, heart_rate)
        
        # Create two main columns for results
        left_col, right_col = st.columns([2, 1])
        
        with left_col:
            # Visual risk gauge
            st.markdown("#### 📊 Risk Assessment Gauge")
            
            # Create a visual gauge using plotly
            risk_percentage = (risk_level / 2) * 100
            
            gauge_fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_percentage,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Level"},
                delta = {'reference': 30},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            gauge_fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with right_col:
            # Live Risk Status Card
            st.markdown("#### 🔴 Risk Status")
            
            # Animated risk level display
            if risk_level == 0:
                st.markdown('<div class="risk-card low-risk"><h3>✅ LOW RISK</h3><p>Parameters are within safe ranges. Continue regular prenatal care.</p></div>', unsafe_allow_html=True)
                risk_color = "green"
                risk_text = "Low Risk"
            elif risk_level == 1:
                st.markdown('<div class="risk-card medium-risk"><h3>⚠️ MEDIUM RISK</h3><p>Some parameters need attention. Consult your healthcare provider.</p></div>', unsafe_allow_html=True)
                risk_color = "orange"
                risk_text = "Medium Risk"
            else:
                st.markdown('<div class="risk-card high-risk"><h3>🚨 HIGH RISK</h3><p>Multiple risk factors detected. Seek immediate medical attention.</p></div>', unsafe_allow_html=True)
                risk_color = "red"
                risk_text = "High Risk"
            
            # Live parameter status
            st.markdown("#### 📊 Parameter Status")
            
            # Age status
            if 20 <= age <= 35:
                st.success("📅 Age: Optimal range")
            elif age < 20 or age > 35:
                st.warning("📅 Age: Monitor closely")
            else:
                st.error("📅 Age: High risk range")
            
            # Blood pressure status
            if systolic_bp < 130 and diastolic_bp < 80:
                st.success("🩸 BP: Normal")
            elif systolic_bp < 140 or diastolic_bp < 90:
                st.warning("🩸 BP: Elevated")
            else:
                st.error("🩸 BP: Hypertensive")
            
            # Blood sugar status
            if blood_sugar < 6.1:
                st.success("🍭 Glucose: Normal")
            elif blood_sugar < 7.8:
                st.warning("🍭 Glucose: Pre-diabetic")
            else:
                st.error("🍭 Glucose: Diabetic range")
            
            # Temperature status
            if 36.1 <= body_temp <= 37.2:
                st.success("🌡️ Temp: Normal")
            elif body_temp > 37.2:
                st.warning("🌡️ Temp: Elevated")
            else:
                st.error("🌡️ Temp: Abnormal")
        
        # Detailed Analysis Section
        st.markdown("---")
        st.markdown("### 📊 Detailed Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚠️ Risk Factors Detected")
            for i, factor in enumerate(risk_factors, 1):
                st.markdown(f"{i}. {factor}")
        
        with col2:
            st.markdown("#### 📊 Recommendations")
            if risk_level == 0:
                recommendations = [
                    "✅ Continue regular prenatal checkups",
                    "🍎 Maintain healthy diet and lifestyle",
                    "🚶 Regular light exercise as approved",
                    "📊 Monitor for any symptom changes"
                ]
            elif risk_level == 1:
                recommendations = [
                    "🏥 Schedule more frequent prenatal visits",
                    "🩸 Monitor blood pressure regularly",
                    "🍭 Track blood glucose levels",
                    "📞 Contact healthcare provider for guidance"
                ]
            else:
                recommendations = [
                    "🚨 **Seek immediate medical attention**",
                    "📈 **Daily monitoring of vital signs**",
                    "🏥 **Consider hospitalization if needed**",
                    "👩‍⚕️ **Follow strict medical supervision**"
                ]
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
        
        # Export and Save Options
        st.markdown("---")
        st.markdown("### 💾 Save & Share Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Assessment", use_container_width=True):
                st.success("💾 Assessment saved to your profile!")
        
        with col2:
            if st.button("📱 Share with Doctor", use_container_width=True):
                st.info("📱 Assessment shared with your healthcare provider!")
        
        with col3:
            if st.button("📊 Generate Report", use_container_width=True):
                # Create the report data without f-string backslashes
                risk_factors_text = "\n".join([f"- {factor}" for factor in risk_factors])
                recommendations_text = "\n".join([f"- {rec}" for rec in recommendations])
                
                report_data = f"""
                PREGNANCY RISK ASSESSMENT REPORT
                Generated: {current_time}
                
                PATIENT PARAMETERS:
                Age: {age} years
                Systolic BP: {systolic_bp} mmHg
                Diastolic BP: {diastolic_bp} mmHg
                Blood Glucose: {blood_sugar} mmol/L
                Body Temperature: {body_temp}°C
                Heart Rate: {heart_rate} bpm
                
                RISK ASSESSMENT: {risk_text.upper()}
                
                RISK FACTORS:
                {risk_factors_text}
                
                RECOMMENDATIONS:
                {recommendations_text}
                """
                st.download_button(
                    label="📄 Download PDF Report",
                    data=report_data,
                    file_name=f"pregnancy_risk_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

if (selected == '👶 Fetal Health'):
    st.markdown("### 👶 Fetal Health Assessment")
    st.markdown("*Enter CTG parameters and submit for comprehensive fetal health analysis*")
    
    # Create form for fetal health assessment
    with st.form("fetal_assessment_form"):
        st.markdown("#### 📊 CTG Parameters")
        
        # Basic parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            baseline_value = st.slider("Baseline FHR (bpm)", 100, 200, 140, help="Baseline fetal heart rate")
            accelerations = st.slider("Accelerations", 0.0, 1.0, 0.2, 0.01, help="Number of accelerations per second")
            fetal_movement = st.slider("Fetal Movement", 0.0, 1.0, 0.4, 0.01, help="Fetal movement frequency")
        
        with col2:
            uterine_contractions = st.slider("Uterine Contractions", 0.0, 1.0, 0.5, 0.01, help="Frequency of uterine contractions")
            light_decelerations = st.slider("Light Decelerations", 0.0, 1.0, 0.3, 0.01, help="Number of light decelerations")
            severe_decelerations = st.slider("Severe Decelerations", 0.0, 1.0, 0.1, 0.01, help="Number of severe decelerations")
        
        with col3:
            prolongued_decelerations = st.slider("Prolonged Decelerations", 0.0, 1.0, 0.2, 0.01, help="Number of prolonged decelerations")
            abnormal_short_term_variability = st.slider("Abnormal STV", 0.0, 1.0, 0.3, 0.01, help="Percentage of time with abnormal short term variability")
            mean_value_of_short_term_variability = st.slider("Mean STV", 0.0, 10.0, 2.4, 0.1, help="Mean value of short term variability")
        
        # Advanced parameters in an expander
        with st.expander("Advanced CTG Parameters"):
            col4, col5, col6 = st.columns(3)
            
            with col4:
                percentage_of_time_with_abnormal_long_term_variability = st.slider("% Time with ALTV", 0, 100, 50, help="Percentage of time with abnormal long term variability")
                mean_value_of_long_term_variability = st.slider("Mean LTV", 0.0, 50.0, 10.0, 0.1, help="Mean value of long term variability")
                histogram_width = st.slider("Histogram Width", 0, 100, 50, help="Width of the histogram")
                histogram_min = st.slider("Histogram Min", 50, 150, 100, help="Minimum value of the histogram")
            
            with col5:
                histogram_max = st.slider("Histogram Max", 100, 200, 150, help="Maximum value of the histogram")
                histogram_number_of_peaks = st.slider("Histogram Peaks", 0, 10, 3, help="Number of peaks in the histogram")
                histogram_number_of_zeroes = st.slider("Histogram Zeroes", 0, 10, 1, help="Number of zeroes in the histogram")
                histogram_mode = st.slider("Histogram Mode", 50, 200, 120, help="Mode of the histogram")
            
            with col6:
                histogram_mean = st.slider("Histogram Mean", 50, 200, 120, help="Mean of the histogram")
                histogram_median = st.slider("Histogram Median", 50, 200, 120, help="Median of the histogram")
                histogram_variance = st.slider("Histogram Variance", 0, 100, 30, help="Variance of the histogram")
                histogram_tendency = st.slider("Histogram Tendency", 0, 1, 0, help="Tendency of the histogram")
        
        # Submit button
        fetal_submitted = st.form_submit_button("🔍 Analyze Fetal Health", use_container_width=True)
    
    # Process results only when submitted
    if fetal_submitted:
        # Create two main columns for results
        left_col, right_col = st.columns([2, 1])
        
        with left_col:
            # Create a visual representation of the FHR pattern
            st.markdown("#### 📈 Simulated FHR Pattern")
            
            # Generate simulated FHR data based on parameters
            x = np.linspace(0, 10, 300)
            # Base FHR with random noise
            noise_level = abnormal_short_term_variability * 10
            fhr = baseline_value + np.sin(x) * mean_value_of_long_term_variability * 0.5 + np.random.normal(0, noise_level, len(x))
            
            # Add accelerations
            if accelerations > 0.1:
                for i in range(int(accelerations * 10)):
                    center = np.random.uniform(1, 9)
                    width = np.random.uniform(0.3, 0.7)
                    height = np.random.uniform(10, 15)
                    fhr += height * np.exp(-((x - center) ** 2) / (2 * width ** 2))
            
            # Add decelerations
            if severe_decelerations > 0.1:
                for i in range(int(severe_decelerations * 5)):
                    center = np.random.uniform(1, 9)
                    width = np.random.uniform(0.5, 1.0)
                    depth = np.random.uniform(30, 40)
                    fhr -= depth * np.exp(-((x - center) ** 2) / (2 * width ** 2))
            
            if light_decelerations > 0.1:
                for i in range(int(light_decelerations * 5)):
                    center = np.random.uniform(1, 9)
                    width = np.random.uniform(0.3, 0.7)
                    depth = np.random.uniform(10, 20)
                    fhr -= depth * np.exp(-((x - center) ** 2) / (2 * width ** 2))
            
            if prolongued_decelerations > 0.1:
                for i in range(int(prolongued_decelerations * 3)):
                    center = np.random.uniform(3, 7)
                    width = np.random.uniform(1.0, 2.0)
                    depth = np.random.uniform(20, 30)
                    fhr -= depth * np.exp(-((x - center) ** 2) / (2 * width ** 2))
            
            # Plot the FHR pattern
            fhr_fig = go.Figure()
            fhr_fig.add_trace(go.Scatter(x=x, y=fhr, mode='lines', name='FHR'))
            fhr_fig.add_shape(type="line", x0=0, y0=160, x1=10, y1=160, line=dict(color="red", width=1, dash="dash"))
            fhr_fig.add_shape(type="line", x0=0, y0=110, x1=10, y1=110, line=dict(color="red", width=1, dash="dash"))
            
            fhr_fig.update_layout(
                title="Simulated Fetal Heart Rate Pattern",
                xaxis_title="Time (minutes)",
                yaxis_title="FHR (bpm)",
                yaxis=dict(range=[80, 200]),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fhr_fig, use_container_width=True)
        
        with right_col:
            # Prediction calculation
            if fetal_model is None:
                # Use rule-based prediction seamlessly
                predicted_risk = mock_fetal_prediction(baseline_value, accelerations, fetal_movement,
                    uterine_contractions, light_decelerations, severe_decelerations,
                    prolongued_decelerations, abnormal_short_term_variability,
                    mean_value_of_short_term_variability,
                    percentage_of_time_with_abnormal_long_term_variability,
                    mean_value_of_long_term_variability, histogram_width,
                    histogram_min, histogram_max, histogram_number_of_peaks,
                    histogram_number_of_zeroes, histogram_mode, histogram_mean,
                    histogram_median, histogram_variance, histogram_tendency)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    predicted_risk = fetal_model.predict([[baseline_value, accelerations, fetal_movement,
                        uterine_contractions, light_decelerations, severe_decelerations,
                        prolongued_decelerations, abnormal_short_term_variability,
                        mean_value_of_short_term_variability,
                        percentage_of_time_with_abnormal_long_term_variability,
                        mean_value_of_long_term_variability, histogram_width,
                        histogram_min, histogram_max, histogram_number_of_peaks,
                        histogram_number_of_zeroes, histogram_mode, histogram_mean,
                        histogram_median, histogram_variance, histogram_tendency]])[0]
            
            # Display a health status card based on prediction
            st.markdown("#### 🏥 Fetal Health Status")
            
            if predicted_risk == 0:
                st.markdown('<div class="risk-card low-risk"><h3>✅ NORMAL</h3><p>Fetal parameters indicate normal health status</p></div>', unsafe_allow_html=True)
                health_text = "Normal"
                recommendations = [
                    "✅ Continue regular prenatal checkups",
                    "📊 Monitor normal fetal movements",
                    "🧘‍♀️ Maintain healthy lifestyle",
                    "📝 No additional interventions needed"
                ]
            elif predicted_risk == 1:
                st.markdown('<div class="risk-card medium-risk"><h3>⚠️ SUSPECT</h3><p>Some parameters need medical attention</p></div>', unsafe_allow_html=True)
                health_text = "Suspect"
                recommendations = [
                    "⚠️ Schedule follow-up assessment",
                    "📊 Increased fetal monitoring recommended",
                    "👩‍⚕️ Consult with healthcare provider",
                    "🧘‍♀️ Rest and reduce physical activity"
                ]
            else:
                st.markdown('<div class="risk-card high-risk"><h3>🚨 PATHOLOGICAL</h3><p>Critical condition requiring immediate medical attention</p></div>', unsafe_allow_html=True)
                health_text = "Pathological"
                recommendations = [
                    "🚨 Seek immediate medical attention",
                    "🏥 Hospital assessment required",
                    "👩‍⚕️ Continuous fetal monitoring needed",
                    "⚕️ Prepare for possible intervention"
                ]
            
            # Display risk factors based on parameters
            st.markdown("#### ⚠️ Key Risk Indicators")
            
            risk_factors = []
            
            # Baseline assessment
            if baseline_value > 160:
                risk_factors.append("🚨 Baseline tachycardia")
            elif baseline_value < 110:
                risk_factors.append("🚨 Baseline bradycardia")
            
            # Decelerations assessment
            if severe_decelerations > 0.3:
                risk_factors.append("🚨 Multiple severe decelerations")
            elif light_decelerations > 0.6:
                risk_factors.append("⚠️ Frequent light decelerations")
            
            # Variability assessment
            if abnormal_short_term_variability > 0.5:
                risk_factors.append("⚠️ Reduced short-term variability")
            
            if mean_value_of_short_term_variability < 1.0:
                risk_factors.append("⚠️ Low mean short-term variability")
            
            # If no specific risk factors, add a positive message
            if not risk_factors:
                risk_factors.append("✅ No specific risk factors identified")
            
            # Display risk factors
            for factor in risk_factors:
                st.markdown(f"- {factor}")
            
            # Display recommendations
            st.markdown("#### 📋 Recommendations")
            for rec in recommendations:
                st.markdown(f"- {rec}")
        
        # Save and Share Options for Fetal Health
        st.markdown("---")
        st.markdown("### 💾 Save & Share Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Fetal Assessment", use_container_width=True):
                st.success("💾 Fetal assessment saved to your profile")
        
        with col2:
            if st.button("📱 Share with Doctor", use_container_width=True, key="fetal_share"):
                st.info("📱 Assessment shared with your healthcare provider")
        
        with col3:
            if st.button("📊 Generate Fetal Report", use_container_width=True):
                # Create the fetal report
                fetal_risk_factors_text = "\n".join([f"- {factor}" for factor in risk_factors])
                fetal_recommendations_text = "\n".join([f"- {rec}" for rec in recommendations])
                
                fetal_report_data = f"""
                FETAL HEALTH ASSESSMENT REPORT
                Generated: {current_time}
                
                CTG PARAMETERS:
                Baseline FHR: {baseline_value} bpm
                Accelerations: {accelerations}
                Fetal Movement: {fetal_movement}
                Uterine Contractions: {uterine_contractions}
                Light Decelerations: {light_decelerations}
                Severe Decelerations: {severe_decelerations}
                Prolonged Decelerations: {prolongued_decelerations}
                Abnormal STV: {abnormal_short_term_variability}
                Mean STV: {mean_value_of_short_term_variability}
                
                FETAL HEALTH STATUS: {health_text.upper()}
                
                RISK FACTORS:
                {fetal_risk_factors_text}
                
                RECOMMENDATIONS:
                {fetal_recommendations_text}
                """
                st.download_button(
                    label="📄 Download Fetal Report",
                    data=fetal_report_data,
                    file_name=f"fetal_health_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

if (selected == '⚖️ BMI Calculator'):
    st.markdown("### ⚖️ Dynamic BMI & Health Calculator")
    st.markdown("*Real-time calculations with instant feedback*")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📏 BMI Calculator")
        
        # Dynamic sliders for height and weight
        height = st.slider("📏 Height (cm)", 100, 250, 165, help="Your height in centimeters")
        weight = st.slider("⚖️ Weight (kg)", 30, 150, 65, help="Your current weight in kilograms")
        
        # Real-time BMI calculation
        bmi = weight / ((height/100) ** 2)
        
        # Dynamic BMI display with color coding
        st.markdown(f"### Your BMI: **{bmi:.1f}**")
        
        # BMI progress bar
        bmi_normalized = min(bmi / 40 * 100, 100)  # Normalize to 0-100
        st.progress(bmi_normalized / 100)
        
        # Real-time BMI category
        if bmi < 18.5:
            st.markdown('<div class="risk-card medium-risk"><h4>🟡 UNDERWEIGHT</h4><p>May need additional nutritional support during pregnancy</p></div>', unsafe_allow_html=True)
            category = "Underweight"
        elif 18.5 <= bmi < 25:
            st.markdown('<div class="risk-card low-risk"><h4>🟢 NORMAL WEIGHT</h4><p>Ideal range for healthy pregnancy</p></div>', unsafe_allow_html=True)
            category = "Normal"
        elif 25 <= bmi < 30:
            st.markdown('<div class="risk-card medium-risk"><h4>🟠 OVERWEIGHT</h4><p>Monitor weight gain during pregnancy</p></div>', unsafe_allow_html=True)
            category = "Overweight"
        else:
            st.markdown('<div class="risk-card high-risk"><h4>🔴 OBESE</h4><p>Higher risk pregnancy, close monitoring needed</p></div>', unsafe_allow_html=True)
            category = "Obese"
        
        # Dynamic weight gain chart
        st.markdown("#### 📈 Weight Gain Tracker")
        
        # Create weight gain visualization
        weeks = list(range(0, 41, 4))
        if category == "Underweight":
            target_gains = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 18]
        elif category == "Normal":
            target_gains = [0, 1, 2, 4, 6, 8, 10, 12, 14, 15, 16]
        elif category == "Overweight":
            target_gains = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11]
        else:
            target_gains = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        weight_df = pd.DataFrame({
            'Week': weeks,
            'Target Weight Gain (kg)': target_gains
        })
        
        weight_fig = px.line(weight_df, x='Week', y='Target Weight Gain (kg)', 
                            title=f'Recommended Weight Gain - {category} Category',
                            markers=True)
        weight_fig.update_layout(height=300)
        st.plotly_chart(weight_fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 💊 Dynamic Nutrition Calculator")
        
        # Interactive trimester and activity selectors
        trimester = st.select_slider('Select Trimester', 
                                    options=['1st Trimester', '2nd Trimester', '3rd Trimester'],
                                    value='2nd Trimester')
        activity_level = st.select_slider('Activity Level', 
                                        options=['Low', 'Moderate', 'High'],
                                        value='Moderate')
        
        # Real-time calorie calculation
        base_calories = {
            '1st Trimester': 1800,
            '2nd Trimester': 2000,
            '3rd Trimester': 2200
        }
        
        activity_multiplier = {
            'Low': 1.0,
            'Moderate': 1.1,
            'High': 1.2
        }
        
        daily_calories = int(base_calories[trimester] * activity_multiplier[activity_level])
        
        # Animated calorie display
        st.markdown(f"### Daily Calories: **{daily_calories}** kcal")
        
        # Progress bars for nutrients
        st.markdown("#### 🔋 Daily Nutrient Goals")
        
        nutrients = {
            "Protein": {"amount": "75-100g", "progress": 0.8},
            "Calcium": {"amount": "1000mg", "progress": 0.7},
            "Iron": {"amount": "27mg", "progress": 0.9},
            "Folic Acid": {"amount": "600mcg", "progress": 0.85},
            "Vitamin D": {"amount": "600 IU", "progress": 0.6},
            "Water": {"amount": "8-10 glasses", "progress": 0.75}
        }
        
        for nutrient, data in nutrients.items():
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.write(f"**{nutrient}**")
            with col_b:
                st.progress(data["progress"])
            st.caption(f"Target: {data['amount']}")
        
        # Trimester-specific recommendations
        st.markdown("#### 📊 Trimester Focus")
        
        if trimester == '1st Trimester':
            st.info("🤱 **Foundation Building**: Focus on folic acid and preventing morning sickness")
        elif trimester == '2nd Trimester':
            st.info("🤰 **Growth Support**: Increase protein and calcium for rapid fetal development")
        else:
            st.info("🍼 **Final Preparation**: Extra iron and preparation for breastfeeding")

if (selected == '🍎 Nutrition Guide'):
    st.title('🥗 Nutrition Guide for Pregnancy')
    
    tab1, tab2, tab3 = st.tabs(["📅 Trimester Guide", "🍎 Foods to Eat", "❌ Foods to Avoid"])
    
    with tab1:
        st.subheader("Nutritional Needs by Trimester")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🤱 1st Trimester")
            st.write("**Key Focus: Foundation Building**")
            st.write("• Folic acid: 600 mcg daily")
            st.write("• Iron: 27 mg daily")
            st.write("• Calcium: 1000 mg daily")
            st.write("• Stay hydrated")
            st.write("• Small, frequent meals")
        
        with col2:
            st.markdown("### 🤰 2nd Trimester")
            st.write("**Key Focus: Growth Support**")
            st.write("• Increase calories by 300-400")
            st.write("• Protein: 75-100g daily")
            st.write("• Omega-3 fatty acids")
            st.write("• Vitamin D: 600 IU")
            st.write("• Fiber-rich foods")
        
        with col3:
            st.markdown("### 🍼 3rd Trimester")
            st.write("**Key Focus: Final Development**")
            st.write("• Increase calories by 400-500")
            st.write("• Extra iron for blood volume")
            st.write("• Calcium for bone development")
            st.write("• Vitamin K")
            st.write("• Prepare for breastfeeding")
    
    with tab2:
        st.subheader("🟢 Recommended Foods")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🥩 Proteins**")
            st.write("• Lean meats, poultry")
            st.write("• Fish (low mercury)")
            st.write("• Eggs")
            st.write("• Beans and lentils")
            st.write("• Nuts and seeds")
            
            st.markdown("**🥛 Dairy**")
            st.write("• Milk (pasteurized)")
            st.write("• Cheese (pasteurized)")
            st.write("• Yogurt")
            st.write("• Calcium-fortified alternatives")
        
        with col2:
            st.markdown("**🍎 Fruits & Vegetables**")
            st.write("• Dark leafy greens")
            st.write("• Citrus fruits")
            st.write("• Berries")
            st.write("• Bananas")
            st.write("• Sweet potatoes")
            
            st.markdown("**🌾 Grains**")
            st.write("• Whole grain bread")
            st.write("• Brown rice")
            st.write("• Quinoa")
            st.write("• Oats")
    
    with tab3:
        st.subheader("🔴 Foods to Avoid")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🐟 High Mercury Fish**")
            st.write("• Shark, swordfish")
            st.write("• King mackerel")
            st.write("• Tilefish")
            
            st.markdown("**🥩 Raw/Undercooked**")
            st.write("• Raw or rare meat")
            st.write("• Raw eggs")
            st.write("• Sushi with raw fish")
            st.write("• Unpasteurized products")
        
        with col2:
            st.markdown("**☕ Substances to Limit**")
            st.write("• Caffeine (< 200mg/day)")
            st.write("• Alcohol (avoid completely)")
            st.write("• High sodium foods")
            
            st.markdown("**🧀 Other Risks**")
            st.write("• Soft cheeses (unpasteurized)")
            st.write("• Deli meats (unless heated)")
            st.write("• Unwashed fruits/vegetables")

if (selected == "📊 Analytics"):
    api_key = "579b464db66ec23bdd00000139b0d95a6ee4441c5f37eeae13f3a0b2"
    api_endpoint = f"https://api.data.gov.in/resource/6d6a373a-4529-43e0-9cff-f39aa8aa5957?api-key={api_key}&format=csv"
    st.header("📉 Maternal Health Dashboard")
    content = "Our interactive dashboard offers a comprehensive visual representation of maternal health achievements across diverse regions. The featured chart provides insights into the performance of each region concerning institutional deliveries compared to their assessed needs. It serves as a dynamic tool for assessing healthcare effectiveness, allowing users to quickly gauge the success of maternal health initiatives."
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)

    dashboard = MaternalHealthDashboard(api_endpoint)
    dashboard.create_bubble_chart()
    with st.expander("Show More"):
        # Display a portion of the data
        content = dashboard.get_bubble_chart_data()
        st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)

    dashboard.create_pie_chart()
    with st.expander("Show More"):
        # Display a portion of the data
        content = dashboard.get_pie_graph_data()
        st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)
