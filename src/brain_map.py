import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from pathlib import Path
import json
import streamlit as st

# Simplified but accurate brain regions
BRAIN_REGIONS_3D = {
    "frontal lobe": {"coords": (0, 45, 20), "color": "#00FF00", "full_name": "Frontal Lobe", "size": 25},
    "frontal cortex": {"coords": (0, 45, 20), "color": "#00FF00", "full_name": "Frontal Cortex", "size": 25},
    "temporal lobe": {"coords": (50, 0, -10), "color": "#FF1493", "full_name": "Temporal Lobe", "size": 24},
    "temporal cortex": {"coords": (50, 0, -10), "color": "#FF1493", "full_name": "Temporal Cortex", "size": 24},
    "parietal lobe": {"coords": (0, -20, 50), "color": "#00FFFF", "full_name": "Parietal Lobe", "size": 28},
    "parietal cortex": {"coords": (0, -20, 50), "color": "#00FFFF", "full_name": "Parietal Cortex", "size": 28},
    "occipital lobe": {"coords": (0, -75, 15), "color": "#FFFF00", "full_name": "Occipital Lobe", "size": 25},
    "hippocampus": {"coords": (30, -25, -8), "color": "#FF1493", "full_name": "Hippocampus", "size": 15},
    "amygdala": {"coords": (25, -2, -12), "color": "#DC143C", "full_name": "Amygdala", "size": 10},
    "thalamus": {"coords": (0, -15, 8), "color": "#8A2BE2", "full_name": "Thalamus", "size": 12},
    "cerebellum": {"coords": (0, -55, -20), "color": "#9ACD32", "full_name": "Cerebellum", "size": 25},
    "brainstem": {"coords": (0, -25, -15), "color": "#008000", "full_name": "Brainstem", "size": 18},
}

def create_realistic_brain_surface():
    """Create a realistic brain surface"""
    phi = np.linspace(0, 2*np.pi, 60)
    theta = np.linspace(0, np.pi, 30)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    
    # Brain dimensions
    a, b, c = 65, 80, 50
    
    # Basic brain shape
    r = 1.0 + 0.05 * np.sin(6 * theta_grid) * np.cos(8 * phi_grid)
    
    x = a * r * np.sin(theta_grid) * np.cos(phi_grid)
    y = b * r * np.sin(theta_grid) * np.sin(phi_grid) - 15
    z = c * r * np.cos(theta_grid)
    
    return x, y, z

def create_medical_grade_brain_visualization(detected_regions, findings=None):
    """Create medical-grade brain visualization - FIXED VERSION"""
    
    fig = go.Figure()
    
    # Create brain surface
    x, y, z = create_realistic_brain_surface()
    
    # Add brain surface
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, '#E8E8E8'], [0.5, '#D3D3D3'], [1, '#C0C0C0']],
        showscale=False,
        opacity=0.4,
        name="Brain Surface",
        hoverinfo='name'
    ))
    
    # Add reference regions (dim)
    for region_name, region_info in BRAIN_REGIONS_3D.items():
        x_coord, y_coord, z_coord = region_info["coords"]
        
        # FIXED: Correct tuple unpacking - only 2 values per tuple
        hemispheres = [(-1, "Left"), (1, "Right")]
        
        for x_mult, side_name in hemispheres:  # FIXED: Only unpack 2 values
            # Skip bilateral display for midline structures
            if region_name in ['thalamus', 'brainstem', 'cerebellum'] and x_mult == 1:
                continue
            
            # Use midline for certain structures
            actual_x = 0 if region_name in ['thalamus', 'brainstem', 'cerebellum'] else x_coord * x_mult
            
            fig.add_trace(go.Scatter3d(
                x=[actual_x],
                y=[y_coord],
                z=[z_coord],
                mode='markers',
                marker=dict(size=4, color='lightgray', opacity=0.3),
                name=f"{side_name} {region_info['full_name']}" if actual_x != 0 else region_info['full_name'],
                showlegend=False,
                hovertemplate=f'<b>{side_name} {region_info["full_name"]}</b><br>Status: Normal<extra></extra>'
            ))
    
    # Highlight detected regions
    if detected_regions:
        for region in detected_regions:
            region_lower = region.lower().strip()
            
            # Find matching region
            matched_region = None
            for brain_region, region_info in BRAIN_REGIONS_3D.items():
                if region_lower in brain_region or brain_region in region_lower:
                    matched_region = region_info
                    break
            
            if matched_region:
                x_coord, y_coord, z_coord = matched_region["coords"]
                
                # Color coding based on findings
                color = matched_region["color"]
                size_mult = 1.5
                
                if findings:
                    findings_lower = str(findings).lower()
                    if 'hemorrhage' in findings_lower or 'bleeding' in findings_lower:
                        color = '#FF0000'  # Red
                        size_mult = 2.0
                    elif 'tumor' in findings_lower or 'mass' in findings_lower:
                        color = '#FF4500'  # Orange
                        size_mult = 2.2
                    elif 'infarct' in findings_lower or 'stroke' in findings_lower:
                        color = '#FFD700'  # Gold
                        size_mult = 1.8
                
                # Determine hemispheres
                if 'left' in region_lower:
                    hemispheres_to_show = [(-1, "Left")]
                elif 'right' in region_lower:
                    hemispheres_to_show = [(1, "Right")]
                elif region in ['thalamus', 'brainstem', 'cerebellum']:
                    hemispheres_to_show = [(0, "")]
                else:
                    hemispheres_to_show = [(-1, "Left"), (1, "Right")]
                
                # FIXED: Correct unpacking here too
                for x_mult, side_name in hemispheres_to_show:
                    actual_x = x_coord * x_mult if x_mult != 0 else 0
                    
                    # Main pathology marker
                    fig.add_trace(go.Scatter3d(
                        x=[actual_x],
                        y=[y_coord],
                        z=[z_coord],
                        mode='markers',
                        marker=dict(
                            size=matched_region["size"] * size_mult,
                            color=color,
                            opacity=0.9,
                            line=dict(width=3, color='darkred')
                        ),
                        name=f"AFFECTED: {side_name} {matched_region['full_name']}".strip(),
                        hovertemplate=f'<b>{side_name} {matched_region["full_name"]}</b><br>Status: AFFECTED<br>Finding: {findings or "Abnormality detected"}<extra></extra>'
                    ))
    
    # Layout
    fig.update_layout(
        title="3D Brain Region Analysis",
        scene=dict(
            xaxis=dict(title="Left ← → Right", range=[-80, 80]),
            yaxis=dict(title="Posterior ← → Anterior", range=[-100, 60]),
            zaxis=dict(title="Inferior ← → Superior", range=[-40, 60]),
            bgcolor='white',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=900,
        height=700,
        showlegend=True
    )
    
    return fig

def show_affected_regions(region_list, findings=None, save_path=None):
    """Show affected brain regions with error handling"""
    if not region_list:
        st.info("No brain regions detected in the report.")
        return
    
    try:
        # Create visualization
        fig = create_medical_grade_brain_visualization(region_list, findings)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary
        st.subheader("Detected Regions")
        for i, region in enumerate(region_list, 1):
            st.write(f"{i}. **{region.title()}**")
            
    except Exception as e:
        st.error(f"Brain visualization error: {str(e)}")
        st.info("Showing text summary instead:")
        
        # Fallback text display
        for i, region in enumerate(region_list, 1):
            st.write(f"{i}. **{region.title()}**")
            if findings:
                st.write(f"   Finding: {findings}")

def plot_brain_heatmap(data, title="Brain Region Analysis"):
    """Simple heatmap visualization"""
    if isinstance(data, list):
        from collections import Counter
        region_counts = Counter(data)
    else:
        region_counts = data
    
    if not region_counts:
        st.info("No data to display")
        return
    
    # Create bar chart
    regions = list(region_counts.keys())
    values = list(region_counts.values())
    
    fig = go.Figure([go.Bar(x=regions, y=values)])
    fig.update_layout(
        title=title,
        xaxis_title="Regions",
        yaxis_title="Frequency",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    return fig

# Legacy compatibility
def create_interactive_brain_map(detected_regions, findings=None):
    """Legacy wrapper"""
    return create_medical_grade_brain_visualization(detected_regions, findings)