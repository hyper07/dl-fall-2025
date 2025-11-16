import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

from utils import initialize_workspace
from functions.database import SELECTED_COLUMNS

# Initialize workspace path and imports
initialize_workspace()

st.set_page_config(
    layout="wide",
    page_title="Wound Detection Platform",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# Header & footer are rendered by the router (streamlit_app.py) when using st.navigation
# CSS is loaded globally from styles/app.css in the router
# Fallback: load CSS here if page is run directly
try:
    css_path = os.path.join(os.path.dirname(__file__), "styles", "app.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

col_logo1, col_partner, col_logo2 = st.columns([1, 2, 1])
with col_logo1:
    pass  # Logo placeholder - image file not available
# Hero Section
st.markdown("""
<div class="hero-section">
    <h1>Wound Detection Platform</h1>
    <h4>Columbia University SPS × Advanced Medical Imaging</h4>
    <p>AI-Powered Wound Classification & Medical Image Analysis</p>
    <p style="font-size: 0.7rem; color: #95a5a6; margin-top: 0.5rem;">Deep Learning Project Fall 2025</p>

</div>
""", unsafe_allow_html=True)

# Partnership Banner




with col_logo2:
 

    st.markdown("<br>", unsafe_allow_html=True)

# Project Overview
st.markdown("## About the Platform")

st.markdown("""
<div class="info-box">
    <p>
        The <strong>Wound Detection Platform</strong> is an AI-powered solution for automated wound classification and medical image analysis. 
        Built with state-of-the-art convolutional neural networks (CNNs), the platform provides accurate classification of various wound types 
        including abrasions, bruises, burns, cuts, diabetic wounds, lacerations, pressure wounds, surgical wounds, and venous wounds. 
        Our deep learning approach ensures high accuracy in wound identification while providing interpretable results for medical professionals.
    </p>
</div>
""", unsafe_allow_html=True)

# Navigation Section
st.markdown("## Get Started")


col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)

with col_nav1:
    container1 = st.container()
    with container1:
        st.markdown("""
        <div class="nav-card" id="card-training">
            <h4>CNN Model Training</h4>
            <p>Train CNN models for wound classification with real-time progress monitoring</p>
        </div>
        """, unsafe_allow_html=True)
    if st.button("Navigate to Training", key="nav_training_btn", use_container_width=True):
        st.switch_page("pages/1_Training.py")

with col_nav2:
    container2 = st.container()
    with container2:
        st.markdown("""
        <div class="nav-card" id="card-similarity">
            <h4>Similarity Search</h4>
            <p>Find similar wound images using AI-powered vector search and feature extraction</p>
        </div>
        """, unsafe_allow_html=True)
    if st.button("Navigate to Similarity Search", key="nav_similarity_btn", use_container_width=True):
        st.switch_page("pages/2_Similarity_Search.py")

with col_nav3:
    container3 = st.container()
    with container3:
        st.markdown("""
        <div class="nav-card" id="card-analytics">
            <h4>Model Evaluation</h4>
            <p>Analyze trained models performance and view training metrics</p>
        </div>
        """, unsafe_allow_html=True)
    if st.button("Navigate to Evaluation", key="nav_analytics_btn", use_container_width=True):
        st.switch_page("pages/1_Training.py")  # Placeholder

with col_nav4:
    container4 = st.container()
    with container4:
        st.markdown("""
        <div class="nav-card" id="card-data">
            <h4>Data Management</h4>
            <p>Upload datasets, validate images, and manage training data</p>
        </div>
        """, unsafe_allow_html=True)
    if st.button("Navigate to Data Management", key="nav_data_btn", use_container_width=True):
        st.switch_page("pages/1_Training.py")  # Placeholder

st.markdown("<br><br>", unsafe_allow_html=True)

