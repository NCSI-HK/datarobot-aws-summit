#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 2024
@author: NCSI (HK)
"""

import datarobot as dr
import pandas as pd
import plotly.express as px
import streamlit as st
import openai


## CREDENTIALS
# DataRobot
DATAROBOT_API_TOKEN = st.secrets["DR_API"]
DATAROBOT_ENDPOINT = "https://app.datarobot.com/api/v2"
DEPLOYMENT_ID = st.secrets["DEPLOYMENT_ID"]

# OpenAI
openai.api_type = "azure"
openai.api_version = "2024-02-15-preview"
openai.api_base = "https://datarobot-oai.openai.azure.com/"
openai.api_key = st.secrets["openai"]["openai_key"]

## CONFIG
st.set_page_config(
    page_title="NCS - Automated Loan Approval with DataRobot",
    layout="wide",
    initial_sidebar_state="auto",
)

@st.cache(allow_output_mutation=True)
def get_df():
    return [None]


## TITLE
title_ttl, title_icon = st.columns([9, 1])

with title_ttl:
    st.header(":blue[Automated Loan Approval System with DataRobot]")
    st.write("Helping you to evaluate credit score and generate email template.")

with title_icon:
    st.write("")
    st.image("./logo_ncs.png")
    # st.image("./logo_datarobot.png")


## APPLICATION FORM
st.subheader("Loan Application Form")

with st.form("loan_form"):
    client = st.text_input("Applicant Name")
    loan_amt = st.selectbox("Loan Amount", [5_000, 15_000, 20_000])
    term = st.selectbox("Repayment Period", [36, 60])
    emp_length = st.selectbox("Years of Employment", ["0 - 5 years", "6 - 10 years", "11+ years"])
    annual_inc = st.selectbox("Annual Income", ["Below 30k", "Between 30k and 60k", "Above 60k"])
    acknowledgement = st.checkbox("Acceptance of Terms and Conditions")
    
    sub_application = st.form_submit_button("Submit")

## DATAROBOT PREDICTION
if sub_application:
    st.markdown(
        f"""
        Your application has been submitted:
        - Applicant Name: `{client}`
        - Loan Amount: `{loan_amt}`
        - Repayment Period: `{term}`
        - Year of Employment: `{emp_length}`
        - Annual Income: `{annual_inc}`
        """
    )
    
    map_emp_length = {
        "0 - 5 years": 0,
        "6 - 10 years": 6,
        "11+ years": 10,
    }
    
    map_annual_inc = {
        "Below 30k": 15_000,
        "Between 30k and 60k": 40_000,
        "Above 60k": 100_000,
    }
    
    scoring_data  = pd.DataFrame({
        "loan_amnt": [loan_amt],
        "term": [term],
        "emp_length": [map_emp_length[emp_length]],
        "annual_inc": [map_annual_inc[annual_inc]]
    })
    
    dr.Client(
        token=DATAROBOT_API_TOKEN,
        endpoint=DATAROBOT_ENDPOINT,
    )
    
    job, df = dr.BatchPredictionJob.score_pandas(
        DEPLOYMENT_ID,
        scoring_data,
        max_explanations=5,
    )
    
    map_feat_name = {
        "loan_amnt": "Loan Amount",
        "term": "Repayment Period",
        "emp_length": "Year of Employment",
        "annual_inc": "Annual Income",
    }
    
    df_sub = df.copy()
    df_sub["ex1_fn"] = df_sub["EXPLANATION_1_FEATURE_NAME"].astype(str) + ": " + df_sub["EXPLANATION_1_ACTUAL_VALUE"].astype(str)
    df_sub["ex2_fn"] = df_sub["EXPLANATION_2_FEATURE_NAME"].astype(str) + ": " + df_sub["EXPLANATION_2_ACTUAL_VALUE"].astype(str)
    df_sub["ex3_fn"] = df_sub["EXPLANATION_3_FEATURE_NAME"].astype(str) + ": " + df_sub["EXPLANATION_3_ACTUAL_VALUE"].astype(str)
    df_sub["ex4_fn"] = df_sub["EXPLANATION_4_FEATURE_NAME"].astype(str) + ": " + df_sub["EXPLANATION_4_ACTUAL_VALUE"].astype(str)
    get_df()[0] = df_sub
    
    st.subheader("Credit Score Explanation")
    st.write("Risk of default is ", df["is_bad_1_PREDICTION"][0], ", and the explanations are")
    
    df_fig = pd.DataFrame({
        "feature": df_sub.filter(regex="ex\d_fn").iloc[-1].to_list(),
        "impact" : df_sub.filter(regex="EXPLANATION_\d_STRENGTH").iloc[-1].to_list()
    })
    
    fig = px.bar(
        df_fig,
        y="feature", x="impact",
        width=1000, height=600,
        orientation="h",
    )
    st.plotly_chart(fig)
else:
    st.write("Fill in the form to proceed the application.")


## EMAIL GENERATION
st.subheader("Email Generation")

with st.form("email_form"):
    approve = st.selectbox(
        "Are you going to approve the loan application?",
        ["Approve", "Reject"],
    )
    
    sub_email = st.form_submit_button("Generate")


## OPENAI GENERATION
if sub_email:
    if approve == "Approve":
        _ = ""
    elif approve == "Reject":
        df_sub = get_df()[0]
        
        exp_str = f"""\
        - {df_sub.loc[0, "EXPLANATION_1_FEATURE_NAME"]}: {df_sub.loc[0, "EXPLANATION_1_ACTUAL_VALUE"]}
        - {df_sub.loc[0, "EXPLANATION_2_FEATURE_NAME"]}: {df_sub.loc[0, "EXPLANATION_2_ACTUAL_VALUE"]}
        - {df_sub.loc[0, "EXPLANATION_3_FEATURE_NAME"]}: {df_sub.loc[0, "EXPLANATION_3_ACTUAL_VALUE"]}
        """.replace("loan_amnt", "loan amount in dollars") \
            .replace("emp_length", "employment tenure in years") \
            .replace("term", "number of months the loan is asked for") \
            .replace("annual_inc", "annual income in dollars")
        
        sys_pmt = f"""\
        You are tasked with generate an email regarding a loan application. \
        Follow the user defined email format as required.
        """.strip().replace("    ", "")
        
        usr_pmt = f"""\
        Below is the top three features that you are going to REJECT the application. \
        You should not mention the exact values of each feature, \
        and need to recommend that is the possible way to increase the chance of loan application in the future.
        
        ```{{top three features}}
        {exp_str}
        ```
        
        You are required to use the below email format.
        
        ```{{email format}}
        Dear {client},
        <content>
        <state the reason of not approving the loan application>
        <the reason should be listed in point form.>
        Your Sincerely,
        Fictional NCS(I) Finance HK Limited
        ```
        """.strip().replace("    ", "")
        
        response = openai.ChatCompletion.create(
            messages = [
                {"role": "system", "content": sys_pmt},
                {"role": "user", "content": usr_pmt}
            ],
            engine="gpt-35-16k",
            temperature=.1,
            top_p=.5,
        )["choices"][0]["message"]["content"]
        
        st.write(response)
