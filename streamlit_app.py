#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Loan Approval System v2.0
Created on April 2024 - Optimized Version November 2025
@author: NCSI (HK)
"""

import datarobot as dr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import openai
from typing import Dict, Tuple, Optional
import time

# Page configuration
st.set_page_config(
    page_title="NCS - Automated Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2e86ab 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2e86ab;
    }
    .form-container {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1f4e79 0%, #2e86ab 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Configuration and credentials
@st.cache_data
def load_config():
    """Load configuration settings"""
    return {
        'DATAROBOT_API_TOKEN': st.secrets["DR_API"],
        'DATAROBOT_ENDPOINT': "https://app.datarobot.com/api/v2",
        'DEPLOYMENT_ID': st.secrets["DEPLOYMENT_ID"],
        'OPENAI_KEY': st.secrets["openai"]["openai_key"],
        'OPENAI_ENDPOINT': "https://next-openai-lab.openai.azure.com/",
        'OPENAI_VERSION': "2024-02-15-preview"
    }

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    """Initialize Azure OpenAI client"""
    config = load_config()
    return AzureOpenAI(
        api_key=config['OPENAI_KEY'],
        api_version=config['OPENAI_VERSION'],
        azure_endpoint=config['OPENAI_ENDPOINT']
    )

# Data mapping functions
def get_mappings() -> Tuple[Dict, Dict, Dict]:
    """Get mapping dictionaries for form data"""
    emp_length_map = {"0 - 5 years": 0, "6 - 10 years": 6, "11+ years": 10}
    annual_inc_map = {"Below $30k": 15_000, "$30k - $60k": 40_000, "Above $60k": 100_000}
    feature_name_map = {
        "loan_amnt": "Loan Amount", "term": "Repayment Period",
        "emp_length": "Employment Years", "annual_inc": "Annual Income"
    }
    return emp_length_map, annual_inc_map, feature_name_map

# Header section
def render_header():
    """Render the application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Automated Loan Approval System</h1>
        <p>AI-powered credit assessment with DataRobot and intelligent email generation</p>
    </div>
    """, unsafe_allow_html=True)

# Application form
def render_application_form() -> Optional[Dict]:
    """Render the loan application form"""
    st.markdown("### 📋 Loan Application")
    
    with st.container():
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        
        with st.form("loan_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            
            with col1:
                client = st.text_input("👤 Applicant Name", placeholder="Enter full name")
                loan_amt = st.selectbox("💰 Loan Amount", [5_000, 15_000, 20_000], 
                                      format_func=lambda x: f"${x:,}")
                term = st.selectbox("📅 Repayment Period", [36, 60], 
                                  format_func=lambda x: f"{x} months")
            
            with col2:
                emp_length = st.selectbox("💼 Years of Employment", 
                                        ["0 - 5 years", "6 - 10 years", "11+ years"])
                annual_inc = st.selectbox("💵 Annual Income", 
                                        ["Below $30k", "$30k - $60k", "Above $60k"])
                acknowledgement = st.checkbox("✅ I accept the Terms and Conditions")
            
            submitted = st.form_submit_button("🚀 Submit Application", use_container_width=True)
            
            if submitted:
                if not client.strip():
                    st.error("Please enter applicant name")
                    return None
                if not acknowledgement:
                    st.error("Please accept Terms and Conditions")
                    return None
                
                return {
                    'client': client, 'loan_amt': loan_amt, 'term': term,
                    'emp_length': emp_length, 'annual_inc': annual_inc
                }
        
        st.markdown('</div>', unsafe_allow_html=True)
    return None

# DataRobot prediction
@st.cache_data(ttl=300)
def get_prediction(loan_amt: int, term: int, emp_length: str, annual_inc: str) -> pd.DataFrame:
    """Get prediction from DataRobot with caching"""
    config = load_config()
    emp_map, inc_map, _ = get_mappings()
    
    scoring_data = pd.DataFrame({
        "loan_amnt": [loan_amt],
        "term": [term],
        "emp_length": [emp_map[emp_length]],
        "annual_inc": [inc_map[annual_inc]]
    })
    
    dr.Client(token=config['DATAROBOT_API_TOKEN'], endpoint=config['DATAROBOT_ENDPOINT'])
    job, df = dr.BatchPredictionJob.score_pandas(
        config['DEPLOYMENT_ID'], scoring_data, max_explanations=5
    )
    return df

# Visualization
def create_shap_chart(df: pd.DataFrame) -> go.Figure:
    """Create modern SHAP values visualization"""
    features = []
    impacts = []
    
    for i in range(1, 5):
        feat_col = f"EXPLANATION_{i}_FEATURE_NAME"
        val_col = f"EXPLANATION_{i}_ACTUAL_VALUE"
        impact_col = f"EXPLANATION_{i}_STRENGTH"
        
        if feat_col in df.columns:
            feature_name = df[feat_col].iloc[0]
            feature_value = df[val_col].iloc[0]
            features.append(f"{feature_name}: {feature_value}")
            impacts.append(df[impact_col].iloc[0])
    
    colors = ['#e74c3c' if x < 0 else '#27ae60' for x in impacts]
    
    fig = go.Figure(go.Bar(
        y=features,
        x=impacts,
        orientation='h',
        marker_color=colors,
        text=[f"{x:.3f}" for x in impacts],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Feature Impact on Loan Decision",
        xaxis_title="Impact Score",
        yaxis_title="Features",
        height=400,
        template="plotly_white",
        showlegend=False
    )
    
    return fig

# Results display
def display_results(app_data: Dict, df: pd.DataFrame):
    """Display prediction results with modern UI"""
    risk_score = df["is_bad_1_PREDICTION"].iloc[0]
    
    # Application summary
    st.markdown("### 📊 Application Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Applicant", app_data['client'])
    with col2:
        st.metric("Loan Amount", f"${app_data['loan_amt']:,}")
    with col3:
        st.metric("Term", f"{app_data['term']} months")
    with col4:
        st.metric("Risk Score", f"{risk_score:.1%}", 
                 delta=f"{'High' if risk_score > 0.5 else 'Low'} Risk")
    
    # Risk assessment
    st.markdown("### 🎯 Risk Assessment")
    if risk_score > 0.5:
        st.error(f"⚠️ High risk of default: {risk_score:.1%}")
    else:
        st.success(f"✅ Low risk of default: {risk_score:.1%}")
    
    # SHAP visualization
    st.markdown("### 📈 Feature Impact Analysis")
    fig = create_shap_chart(df)
    st.plotly_chart(fig, use_container_width=True)
    
    return df

# Email generation
def generate_email(client: str, df: pd.DataFrame, decision: str) -> str:
    """Generate email using Azure OpenAI"""
    if decision == "Approve":
        return f"""Dear {client},

We are pleased to inform you that your loan application has been approved! 

Our AI-powered assessment system has evaluated your application favorably based on your financial profile. You can expect to hear from our loan processing team within 2-3 business days to finalize the details.

Thank you for choosing NCS Finance.

Your Sincerely,
NCS(I) Finance HK Limited"""
    
    # For rejection emails
    client_openai = get_openai_client()
    
    explanations = []
    for i in range(1, 4):
        feat_name = df[f"EXPLANATION_{i}_FEATURE_NAME"].iloc[0]
        feat_val = df[f"EXPLANATION_{i}_ACTUAL_VALUE"].iloc[0]
        explanations.append(f"- {feat_name}: {feat_val}")
    
    exp_str = "\n".join(explanations).replace("loan_amnt", "loan amount").replace(
        "emp_length", "employment tenure").replace("term", "loan term").replace(
        "annual_inc", "annual income")
    
    prompt = f"""Generate a professional loan rejection email for {client}. 

Key factors for rejection:
{exp_str}

Requirements:
- Professional and empathetic tone
- Don't mention exact values
- Provide constructive advice for future applications
- Use this format:

Dear {client},
[Content with rejection reasons in bullet points and improvement suggestions]
Your Sincerely,
NCS(I) Finance HK Limited"""
    
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            top_p=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating email: {str(e)}"

# Email section
def render_email_section():
    """Render email generation section"""
    st.markdown("### 📧 Email Generation")
    
    if 'prediction_data' not in st.session_state:
        st.warning("⚠️ Please submit a loan application first to generate emails.")
        return
    
    with st.form("email_form"):
        decision = st.selectbox("Decision", ["Approve", "Reject"], 
                               help="Select the loan decision")
        
        if st.form_submit_button("Generate Email", use_container_width=True):
            with st.spinner("Generating email..."):
                email_content = generate_email(
                    st.session_state.app_data['client'],
                    st.session_state.prediction_data,
                    decision
                )
            
            st.markdown("#### Generated Email")
            st.text_area("Email Content", email_content, height=300)
            
            # Copy to clipboard button
            st.code(email_content, language=None)

# Main application
def main():
    """Main application logic"""
    render_header()
    
    # Application form
    app_data = render_application_form()
    
    if app_data:
        try:
            with st.spinner("Processing application..."):
                df = get_prediction(
                    app_data['loan_amt'], app_data['term'],
                    app_data['emp_length'], app_data['annual_inc']
                )
                
                # Store in session state
                st.session_state.prediction_data = df
                st.session_state.app_data = app_data
                
                # Display results
                display_results(app_data, df)
                
        except Exception as e:
            st.error(f"Error processing application: {str(e)}")
    
    # Email generation section
    render_email_section()
    
    # Footer
    st.markdown("---")
    st.markdown("*Powered by DataRobot ML and Azure OpenAI*")

if __name__ == "__main__":
    main()
