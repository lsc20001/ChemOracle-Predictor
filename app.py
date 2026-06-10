import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import traceback
import py3Dmol
from stmol import showmol

current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["OMP_NUM_THREADS"] = "1"

from feature_pipeline import extract_all_16_features

st.set_page_config(page_title="Catalyst Oracle Pro", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #2E86C1; padding-bottom: 20px;'>
        Catalyst Activity Oracle (Edge Computing Node)
    </h1>
    <p style='text-align: center; color: gray;'>
        Secure Local Inference | No Geometry Data Leaves Your Device
    </p>
    <hr>
""", unsafe_allow_html=True)

col_m, col_o, col_u = st.columns([1, 1, 2])
with col_m:
    user_metal = st.text_input("Target Metal:", "Mn")
with col_o:
    user_ox = st.number_input("Oxidation State (ox):", value=2, step=1)
with col_u:
    uploaded_file = st.file_uploader("Upload Proprietary XYZ:", type=["xyz"])

st.markdown("<br>", unsafe_allow_html=True)

feature_name_mapping = {
    'ox': 'OX',
    'Debye': 'DP',
    'bo1_2': 'MBO',
    'HOMO-LUMO': 'HLG',
    'ip': 'IP',
    'charges1': 'MQ',
    'B_1': 'B1',
    'B_5': 'B5',
    'P_int2': 'LMP',
    'D_P': 'MP',
    'ar_r': 'MAR',
    'BV': 'BV',
    'EA_Mt': 'MEA',
    'NCA_N': 'NCN',
    'LT': 'TL',
    'NCA_C': 'NCC'
}

if uploaded_file:
    try:
        file_content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        with st.spinner("Executing Local Quantum Physics Pipeline..."):
            
            features = extract_all_16_features(file_content, user_metal, user_ox)
            final_cols = ['ox', 'Debye', 'bo1_2', 'HOMO-LUMO', 'ip', 'charges1', 'B_1', 'B_5', 'P_int2', 'D_P', 'ar_r', 'BV', 'EA_Mt', 'NCA_N', 'LT', 'NCA_C']
            df = pd.DataFrame([features])[final_cols]
            
            TRAIN_MEANS = np.array([2.211765, 3.478055, 0.734077, 0.023137, 10.926487, 0.016713, 4.2422, 8.005451, 43.31487, 65.825882, 241.194118, 0.815912, 8.953459, 2.262353, 8.191765, 1.198824])
            TRAIN_STDS  = np.array([0.645238, 1.086614, 0.148806, 0.018338, 0.509622, 0.365975, 0.495551, 1.14842, 9.15284, 10.168939, 16.116085, 0.0524, 8.350466, 0.831367, 11.881164, 1.387931])
            df_scaled = (df - TRAIN_MEANS) / TRAIN_STDS
            
            model = joblib.load("rf_model.pkl.gz") 
            prediction = model.predict(df)[0]

            st.success("Pipeline Execution Complete. Data Secured.")
            st.markdown("<br>", unsafe_allow_html=True)
            
            top_col1, top_col2 = st.columns([1, 1])
            
            with top_col1:
                st.markdown("<h4 style='text-align: center; color: #555555;'>3D Molecular Structure</h4>", unsafe_allow_html=True)
                view = py3Dmol.view(width=500, height=350)
                view.addModel(file_content, "xyz")
                view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.3}})
                view.setBackgroundColor('#FAFAFA')
                view.zoomTo()
                showmol(view, height=350, width=500)

            with top_col2:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Predicted Activity (Delta G)", 'font': {'size': 20, 'color': '#555555'}},
                    gauge = {
                        'axis': {'range': [None, 80]},
                        'bar': {'color': "#1ABC9C"},
                        'steps': [
                            {'range': [0, 30], 'color': "#FADBD8"},
                            {'range': [30, 50], 'color': "#FCF3CF"},
                            {'range': [50, 80], 'color': "#D5F5E3"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': prediction}
                }))
                fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            ui_cols = [feature_name_mapping.get(col, col) for col in final_cols]

            plot_df = pd.DataFrame({
                'Feature': ui_cols,
                'Deviation (Z-Score)': df_scaled.iloc[0].values
            })
            fig_bar = px.bar(
                plot_df, x='Feature', y='Deviation (Z-Score)',
                title="Feature Deviation from Knowledge Base Average",
                color='Deviation (Z-Score)',
                color_continuous_scale=px.colors.diverging.Tealrose,
                text_auto='.2f'
            )
            fig_bar.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

            with st.expander("View Raw Physical Descriptors Matrix"):
                display_df = df.copy()
                display_df.columns = ui_cols
                st.dataframe(display_df)
            
    except Exception as e:
        error_details = traceback.format_exc()
        st.error("Pipeline Fault:")
        st.code(error_details, language="python")
