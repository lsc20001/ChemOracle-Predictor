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

def auto_align_geometry(xyz_string, target_metal):
    lines = [line.strip() for line in xyz_string.strip().split('\n') if line.strip()]
    if len(lines) < 3:
        raise ValueError("Invalid XYZ file structure.")
        
    num_atoms = lines[0]
    comment = lines[1]
    atom_records = lines[2:]
    
    parsed_atoms = []
    for record in atom_records:
        tokens = record.split()
        if len(tokens) >= 4:
            parsed_atoms.append({
                'element': tokens[0],
                'x': float(tokens[1]),
                'y': float(tokens[2]),
                'z': float(tokens[3]),
                'raw_line': record
            })
            
    metal_atom = None
    metal_index = -1
    for idx, atom in enumerate(parsed_atoms):
        if atom['element'].upper() == target_metal.upper():
            metal_atom = atom
            metal_index = idx
            break
            
    if metal_atom is None:
        raise ValueError(f"Target metal '{target_metal}' not found in the uploaded file.")
        
    pool_without_metal = parsed_atoms[:metal_index] + parsed_atoms[metal_index+1:]
    
    closest_h_atom = None
    closest_h_index = -1
    min_distance = float('inf')
    
    for idx, atom in enumerate(pool_without_metal):
        if atom['element'].upper() == 'H':
            distance = np.sqrt(
                (metal_atom['x'] - atom['x'])**2 +
                (metal_atom['y'] - atom['y'])**2 +
                (metal_atom['z'] - atom['z'])**2
            )
            if distance < min_distance:
                min_distance = distance
                closest_h_atom = atom
                closest_h_index = idx
                
    ordered_records = [metal_atom['raw_line']]
    if closest_h_atom is not None:
        ordered_records.append(closest_h_atom['raw_line'])
        for idx, atom in enumerate(pool_without_metal):
            if idx != closest_h_index:
                ordered_records.append(atom['raw_line'])
    else:
        for atom in pool_without_metal:
            ordered_records.append(atom['raw_line'])
            
    return f"{num_atoms}\n{comment}\n" + "\n".join(ordered_records)

st.set_page_config(page_title="Catalyst Oracle Pro", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #2E86C1; padding-bottom: 20px;'>
        Prediction of Transition Metal-Hydride Dissociation Energies using RF Model
    </h1>
    <hr>
""", unsafe_allow_html=True)

col_m, col_o, col_u = st.columns([1, 1, 2])
with col_m:
    user_metal = st.selectbox("Step 1: Target Metal", options=["Cr", "Mn", "Fe", "Co", "Ni"], index=1)
with col_o:
    user_ox = st.selectbox("Step 2: Oxidation State", options=[1, 2, 3], index=1)
with col_u:
    uploaded_file = st.file_uploader("Step 3: Upload Geometry (XYZ)", type=["xyz"])

st.markdown("<br>", unsafe_allow_html=True)

feature_name_mapping = {
    'ox': 'OX', 'Debye': 'DP', 'bo1_2': 'MBO', 'HOMO-LUMO': 'HLG',
    'ip': 'IP', 'charges1': 'MQ', 'B_1': 'B1', 'B_5': 'B5',
    'P_int2': 'LMP', 'D_P': 'MP', 'ar_r': 'MAR', 'BV': 'BV',
    'EA_Mt': 'MEA', 'NCA_N': 'NCN', 'LT': 'TL', 'NCA_C': 'NCC'
}

if uploaded_file:
    try:
        raw_content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        
        with st.spinner("Executing alignment and pipeline inference..."):
            processed_xyz = auto_align_geometry(raw_content, user_metal)
            
            features = extract_all_16_features(processed_xyz, user_metal, user_ox)
            final_cols = ['ox', 'Debye', 'bo1_2', 'HOMO-LUMO', 'ip', 'charges1', 'B_1', 'B_5', 'P_int2', 'D_P', 'ar_r', 'BV', 'EA_Mt', 'NCA_N', 'LT', 'NCA_C']
            df = pd.DataFrame([features])[final_cols]
            
            TRAIN_MEANS = np.array([2.211765, 3.478055, 0.734077, 0.023137, 10.926487, 0.016713, 4.2422, 8.005451, 43.31487, 65.825882, 241.194118, 0.815912, 8.953459, 2.262353, 8.191765, 1.198824])
            TRAIN_STDS  = np.array([0.645238, 1.086614, 0.148806, 0.018338, 0.509622, 0.365975, 0.495551, 1.14842, 9.15284, 10.168939, 16.116085, 0.0524, 8.350466, 0.831367, 11.881164, 1.387931])
            df_scaled = (df - TRAIN_MEANS) / TRAIN_STDS
            
            model = joblib.load("rf_model.pkl.gz") 
            prediction = model.predict(df)[0]

            st.success("Pipeline Inference Completed Successfully.")
            st.markdown("<br>", unsafe_allow_html=True)
            
            top_col1, top_col2 = st.columns([1, 1])
            
            with top_col1:
                st.markdown("<h4 style='text-align: center; color: #555555;'>Aligned 3D Molecular Structure</h4>", unsafe_allow_html=True)
                view = py3Dmol.view(width=500, height=350)
                view.addModel(processed_xyz, "xyz")
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
        st.error(f"Pipeline Fault: {str(e)}")
        st.code(error_details, language="python")
