import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import streamlit as st
import json
from collections import Counter

class RealisticBrainMapper:
    def __init__(self):
        # Anatomically accurate brain region coordinates and colors
        self.brain_regions = {
            "frontal lobe": {
                "coords": [(-30, 40, 20), (30, 40, 20), (-20, 50, 25), (20, 50, 25), (0, 45, 30)],
                "color": "#FF6B6B",
                "area_polygon": self._get_frontal_polygon(),
                "intensity": 0.8
            },
            "frontal cortex": {
                "coords": [(-25, 45, 25), (25, 45, 25), (0, 50, 30)],
                "color": "#FF6B6B", 
                "area_polygon": self._get_frontal_polygon(),
                "intensity": 0.8
            },
            "temporal lobe": {
                "coords": [(-55, -10, -5), (55, -10, -5), (-45, -20, 0), (45, -20, 0), (-50, -30, -10), (50, -30, -10)],
                "color": "#4ECDC4",
                "area_polygon": self._get_temporal_polygon(),
                "intensity": 0.9
            },
            "temporal cortex": {
                "coords": [(-50, -15, 0), (50, -15, 0), (-45, -25, -5), (45, -25, -5)],
                "color": "#4ECDC4",
                "area_polygon": self._get_temporal_polygon(), 
                "intensity": 0.9
            },
            "parietal lobe": {
                "coords": [(-35, -30, 40), (35, -30, 40), (-25, -40, 45), (25, -40, 45), (0, -35, 50)],
                "color": "#45B7D1",
                "area_polygon": self._get_parietal_polygon(),
                "intensity": 0.85
            },
            "parietal cortex": {
                "coords": [(-30, -35, 45), (30, -35, 45), (0, -30, 50)],
                "color": "#45B7D1",
                "area_polygon": self._get_parietal_polygon(),
                "intensity": 0.85
            },
            "occipital lobe": {
                "coords": [(-25, -80, 10), (25, -80, 10), (-15, -90, 5), (15, -90, 5), (0, -85, 15)],
                "color": "#96CEB4",
                "area_polygon": self._get_occipital_polygon(),
                "intensity": 0.7
            },
            "cerebellum": {
                "coords": [(-30, -60, -20), (30, -60, -20), (-20, -70, -25), (20, -70, -25), (0, -65, -15)],
                "color": "#FECA57",
                "area_polygon": self._get_cerebellum_polygon(),
                "intensity": 0.75
            },
            "brainstem": {
                "coords": [(0, -30, -15), (0, -35, -20), (0, -40, -25)],
                "color": "#48CAE4",
                "area_polygon": self._get_brainstem_polygon(),
                "intensity": 0.9
            },
            "hippocampus": {
                "coords": [(-30, -25, -8), (30, -25, -8)],
                "color": "#F093FB",
                "area_polygon": None,
                "intensity": 1.0
            },
            "amygdala": {
                "coords": [(-22, -5, -15), (22, -5, -15)],
                "color": "#F093FB",
                "area_polygon": None,
                "intensity": 1.0
            },
            "thalamus": {
                "coords": [(-10, -15, 5), (10, -15, 5)],
                "color": "#A8E6CF",
                "area_polygon": None,
                "intensity": 0.9
            }
        }
    
    def _get_frontal_polygon(self):
        """Generate frontal lobe area coordinates"""
        return {
            "x": [-35, -30, -20, -10, 0, 10, 20, 30, 35, 30, 20, 10, 0, -10, -20, -30],
            "y": [25, 35, 45, 50, 55, 50, 45, 35, 25, 20, 25, 30, 35, 30, 25, 20]
        }
    
    def _get_temporal_polygon(self):
        """Generate temporal lobe area coordinates"""
        return {
            "x": [-60, -45, -35, -30, -35, -45, -55],
            "y": [5, -5, -15, -25, -35, -30, -10]
        }
    
    def _get_parietal_polygon(self):
        """Generate parietal lobe area coordinates"""
        return {
            "x": [-30, 30, 25, 20, 0, -20, -25],
            "y": [-20, -20, -30, -40, -45, -40, -30]
        }
    
    def _get_occipital_polygon(self):
        """Generate occipital lobe area coordinates"""
        return {
            "x": [-25, 25, 20, 15, 0, -15, -20],
            "y": [-60, -60, -70, -85, -90, -85, -70]
        }
    
    def _get_cerebellum_polygon(self):
        """Generate cerebellum area coordinates"""
        return {
            "x": [-30, 30, 25, 20, 0, -20, -25],
            "y": [-50, -50, -60, -70, -75, -70, -60]
        }
    
    def _get_brainstem_polygon(self):
        """Generate brainstem area coordinates"""
        return {
            "x": [-8, 8, 6, 4, 0, -4, -6],
            "y": [-25, -25, -35, -45, -50, -45, -35]
        }
    
    def create_realistic_brain_outline(self):
        """Create anatomically accurate brain outline"""
        # Create brain outline using parametric equations
        t = np.linspace(0, 2*np.pi, 300)
        
        brain_x = []
        brain_y = []
        
        for angle in t:
            # Base brain shape (modified super-ellipse)
            x = 55 * np.cos(angle) * (1 + 0.2 * np.cos(2*angle))
            y = 70 * np.sin(angle) * (1 + 0.1 * np.sin(3*angle))
            
            # Add frontal prominence
            if -np.pi/4 < angle < np.pi/4:
                y += 15 * np.cos(2*angle) * np.exp(-2*angle**2)
            
            # Add temporal bulges
            if np.pi/4 < angle < 3*np.pi/4:
                x += 15 * np.sin(angle) * np.exp(-(angle - np.pi/2)**2)
            elif -3*np.pi/4 < angle < -np.pi/4:
                x -= 15 * np.sin(angle) * np.exp(-(angle + np.pi/2)**2)
            
            # Add occipital region
            if 3*np.pi/4 < angle < 5*np.pi/4:
                y -= 10 * np.cos(angle - np.pi)
            
            brain_x.append(x)
            brain_y.append(y)
        
        return brain_x, brain_y
    
    def create_brain_visualization(self, detected_regions, findings=None):
        """Create complete brain visualization with detected regions"""
        
        fig = go.Figure()
        
        # Create realistic brain outline
        brain_x, brain_y = self.create_realistic_brain_outline()
        
        # Add brain outline
        fig.add_trace(go.Scatter(
            x=brain_x, y=brain_y,
            mode='lines',
            fill='toself',
            fillcolor='rgba(200, 200, 200, 0.3)',
            line=dict(color="#0c0c0c", width=2),
            name='Brain Tissue',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add anatomical regions (base layer)
        for region_name, region_data in self.brain_regions.items():
            if region_data["area_polygon"]:
                polygon = region_data["area_polygon"]
                fig.add_trace(go.Scatter(
                    x=polygon["x"], y=polygon["y"],
                    mode='lines',
                    fill='toself',
                    fillcolor=f'rgba({self._hex_to_rgb(region_data["color"])}, 0.1)',
                    line=dict(color=region_data["color"], width=1, dash='dot'),
                    name=region_name.title(),
                    showlegend=False,
                    hovertemplate=f'<b>{region_name.title()}</b><br>Status: Normal<extra></extra>'
                ))
        
        # Highlight detected regions
        affected_regions = []
        if detected_regions:
            for region in detected_regions:
                region_lower = region.lower().strip()
                
                # Find matching brain region
                matched_region = None
                for brain_region, region_data in self.brain_regions.items():
                    if region_lower in brain_region or brain_region in region_lower:
                        matched_region = region_data
                        matched_name = brain_region
                        break
                
                if matched_region:
                    affected_regions.append({
                        'name': region.title(),
                        'data': matched_region,
                        'original_name': matched_name
                    })
                    
                    # Highlight area polygon if available
                    if matched_region["area_polygon"]:
                        polygon = matched_region["area_polygon"]
                        fig.add_trace(go.Scatter(
                            x=polygon["x"], y=polygon["y"],
                            mode='lines',
                            fill='toself',
                            fillcolor=f'rgba({self._hex_to_rgb(matched_region["color"])}, {matched_region["intensity"]})',
                            line=dict(color=matched_region["color"], width=3),
                            name=f'AFFECTED: {region.title()}',
                            showlegend=True,
                            hovertemplate=f'<b>{region.title()}</b><br>Status: <b>PATHOLOGICAL</b><br>Intensity: {matched_region["intensity"]:.1f}<extra></extra>'
                        ))
                    
                    # Add activation points
                    coords = matched_region["coords"]
                    for coord in coords:
                        x, y, z = coord
                        
                        # Create pulsing effect with multiple circles
                        for i, radius in enumerate([6, 10, 14]):
                            opacity = 0.8 - i * 0.2
                            fig.add_trace(go.Scatter(
                                x=[x], y=[y],
                                mode='markers',
                                marker=dict(
                                    size=radius,
                                    color=matched_region["color"],
                                    opacity=opacity,
                                    line=dict(width=2, color='darkred') if i == 0 else dict(width=1, color=matched_region["color"])
                                ),
                                showlegend=False,
                                hovertemplate=f'<b>{region.title()}</b><br>Coordinates: ({x}, {y}, {z})<br>Pathology Center<extra></extra>' if i == 0 else None,
                                hoverinfo='skip' if i > 0 else 'all'
                            ))
        
        # Add anatomical labels
        self._add_anatomical_labels(fig)
        
        # Add brain midline
        fig.add_shape(
            type="line",
            x0=0, y0=-90, x1=0, y1=60,
            line=dict(color="gray", width=1, dash="dash"),
        )
        
        # Configure layout
        fig.update_layout(
    title=dict(
        text="Brain Region Analysis - Medical Grade Visualization",
        font=dict(family="Arial", size=18, color="black"),
        x=0.5,
        xanchor="center"
    ),
    width=1000,
    height=800,
    xaxis=dict(
        title=dict(
            text="Left ← → Right (mm)",
            font=dict(family="Arial", size=14, color="black")
        ),
        range=[-80, 80],
        showgrid=True,
        gridcolor='lightgray',
        zeroline=True,
        scaleanchor="y",
        scaleratio=1,
        tickfont=dict(family="Arial", size=12, color="black"),
        linecolor="black",
        mirror=True
    ),
    yaxis=dict(
        title=dict(
            text="Posterior ← → Anterior (mm)",
            font=dict(family="Arial", size=14, color="black")
        ),
        range=[-100, 70],
        showgrid=True,
        gridcolor='lightgray',
        zeroline=True,
        tickfont=dict(family="Arial", size=12, color="black"),
        linecolor="black",
        mirror=True
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family="Arial", size=12, color="black"),  # <-- global fallback
    showlegend=True,
    legend=dict(
        x=1.02,
        y=1,
        font=dict(family="Arial", size=12, color="black"),  # <-- legend labels
        title=dict(text="Pathological Regions", font=dict(color="black", size=13)),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="black",
        borderwidth=1
    )
)

        # If you are adding text annotations for regions:
        fig.add_annotation(
            x=20, y=30,
            text="Hippocampus (Pathology)",
            showarrow=True,
            arrowhead=2,
            font=dict(color="black", size=12)  # <-- forces black label
        )

        return fig, affected_regions
    
    def _add_anatomical_labels(self, fig):
        """Add anatomical region labels with black text for visibility"""
        labels = [
            {"x": 0, "y": 50, "text": "Frontal Lobe", "color": "black"},
            {"x": -50, "y": -15, "text": "Left Temporal", "color": "black"},
            {"x": 50, "y": -15, "text": "Right Temporal", "color": "black"},
            {"x": 0, "y": -35, "text": "Parietal Lobe", "color": "black"},
            {"x": 0, "y": -75, "text": "Occipital Lobe", "color": "black"},
            {"x": 0, "y": -60, "text": "Cerebellum", "color": "black"},
            {"x": 0, "y": -40, "text": "Brainstem", "color": "black"}
        ]
    
        for label in labels:
            fig.add_annotation(
                x=label["x"], y=label["y"],
                text=label["text"],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="black",
                font=dict(size=12, color="black", family="Arial", weight="bold"),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=2
            )
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB string"""
        hex_color = hex_color.lstrip('#')
        return f'{int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}'

# Initialize the brain mapper
brain_mapper = RealisticBrainMapper()

def show_affected_regions(region_list, findings=None, save_path=None):
    """Main function to display brain regions (called by your app)"""
    if not region_list:
        st.info("No brain regions detected in the report.")
        return
    
    try:
        # Create brain visualization
        fig, affected_regions = brain_mapper.create_brain_visualization(region_list, findings)
        
        # Display the visualization
        st.markdown("### Brain Region Map (Realistic Medical Overlay)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Medical summary panel
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Highlighted Atlas Regions:**")
            for region_info in affected_regions:
                color = region_info['data']['color']
                intensity = region_info['data']['intensity']
                st.markdown(f"• <span style='color: {color}; font-weight: bold;'>{region_info['name']}</span> (Intensity: {intensity:.1f})", unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Clinical Assessment:**")
            if findings:
                if isinstance(findings, list):
                    for finding in findings:
                        st.markdown(f"• {finding}")
                else:
                    st.markdown(f"• {findings}")
            else:
                st.markdown("• Abnormalities detected in specified regions")
                st.markdown("• Clinical correlation recommended")
        
        # Intensity legend
        st.markdown("**Pathology Intensity Scale:**")
        intensity_html = """
        <div style='display: flex; align-items: center; margin: 10px 0;'>
            <div style='background: linear-gradient(to right, #ffffff, #ff0000); width: 200px; height: 20px; border: 1px solid #ccc;'></div>
            <span style='margin-left: 10px;'>0.0 (Normal) → 1.0 (Severe)</span>
        </div>
        """
        st.markdown(intensity_html, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Brain visualization error: {str(e)}")
        # Fallback display
        st.write("**Detected Regions:**")
        for region in region_list:
            st.write(f"• {region.title()}")

def plot_brain_heatmap(data, title="Brain Region Frequency Analysis"):
    """Create professional heatmap of brain regions"""
    if isinstance(data, list):
        region_counts = Counter(data)
    elif isinstance(data, dict):
        region_counts = data
    else:
        st.error("Invalid data format for heatmap")
        return
    
    if not region_counts:
        st.info("No data to display")
        return
    
    regions = list(region_counts.keys())
    values = list(region_counts.values())
    
    # Get colors from brain mapper
    colors = []
    for region in regions:
        region_lower = region.lower()
        color = "#0B0B0B"  # Default black if no match
        
        for brain_region, region_data in brain_mapper.brain_regions.items():
            if region_lower in brain_region or brain_region in region_lower:
                color = region_data["color"]
                break
        colors.append(color)
    
    fig = go.Figure([go.Bar(
        x=regions, 
        y=values,
        marker_color=colors,
        text=values,
        textposition='auto',
        textfont=dict(color="black"),  # <-- bar labels black
        hovertemplate='<b>%{x}</b><br>Frequency: %{y}<br>Clinical Significance: High<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Arial", size=18, color="black"),
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title=dict(text="Anatomical Regions", font=dict(color="black", size=14)),
            tickfont=dict(color="black", size=12),
            linecolor="black",
            mirror=True
        ),
        yaxis=dict(
            title=dict(text="Detection Frequency", font=dict(color="black", size=14)),
            tickfont=dict(color="black", size=12),
            linecolor="black",
            mirror=True
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12, color="black"),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    return fig


# Legacy compatibility functions
def create_interactive_brain_map(detected_regions, findings=None):
    """Legacy compatibility wrapper"""
    fig, affected_regions = brain_mapper.create_brain_visualization(detected_regions, findings)
    return fig

def show_professional_brain_mapping(region_list, findings=None):
    """Legacy compatibility wrapper"""
    return show_affected_regions(region_list, findings)