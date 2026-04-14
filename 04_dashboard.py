import streamlit as st 
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# Gruvbox-inspired Palette
BG_COLOR = "#1d2021"      # Deep Dark Grey
GRID_COLOR = "#3c3836"    # Subtle Grid
ACCENT_CYAN = "#8ec07c"   # Aqua (for Pulse)
REAL_COLOR = "#458588"    # Faded Blue
PRED_COLOR = "#fb4934"    # Bright Red
TEXT_COLOR = "#ebdbb2"    # Off-white / Cream

def apply_unified_style(fig, title):
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_COLOR)),
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        margin=dict(l=40, r=20, t=60, b=40)
    )

def prepare_data():
    N_events = 8000
    pe_value = 8.90 # mV/p.e., from 00b_pe_fitting.py
    real_amps = np.load("./data/processed_data_validated.npz")['amplitude'][N_events:]
    real_amps = real_amps / pe_value
    predicted_amps = np.load("./data/predicted_amplitudes_validated_pe.npz")['predicted_amps'].flatten()
    diff = predicted_amps - real_amps
    return real_amps, predicted_amps, diff

@st.cache_resource
def get_lazy_loaders():
    wv_loader = np.load("./data/waveforms_validated.npz", mmap_mode='r')
    proc_loader = np.load("./data/processed_data_validated.npz", mmap_mode='r')
    return wv_loader, proc_loader

def fetch_waveform(event_index):
    wv_loader, proc_loader = get_lazy_loaders()
    waveform = wv_loader['waveforms'][event_index]
    baseline = proc_loader['baseline'][event_index]
    max_sample = proc_loader['STD_position'][event_index]
    return waveform, baseline, max_sample

def add_glowing_line(fig, x, y, color='#458588', name='Signal'):
    """Adds a line with a neon glow effect by layering traces correctly."""
    
    # 1. The Main "Core" Line
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name=name,
        line=dict(color=color, width=2),
        opacity=1.0  # Trace property, not line property
    ))

    # 2. The "Inner Glow"
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        showlegend=False,
        line=dict(color=color, width=8),
        opacity=0.3, # Correct placement
        hoverinfo='skip'
    ))

    # 3. The "Outer Bloom"
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        showlegend=False,
        line=dict(color=color, width=15),
        opacity=0.1, # Correct placement
        hoverinfo='skip'
    ))

def wvfm_build():

    fig = go.Figure()
    
    fig.update_layout(
        title="Dynamic Waveform Display",
        xaxis=dict(range=[0, 800], title="Sample", gridcolor='lightgray'),
        yaxis=dict(range=[0, 10], title="Amplitude (mV)", gridcolor='lightgray'),
        template="plotly_white",
        showlegend=True 
    )
    
    return fig

def wvfm_update(fig, wvfm_index, real_amps, predicted_amps):
    y_data, baseline, max_sample = fetch_waveform(wvfm_index)
    y_corrected = (y_data - baseline)/8.90
    x_data = np.arange(len(y_corrected))

    add_glowing_line(fig, x_data, y_corrected, color='#00ffff', name='Pulse')

    val_real = real_amps[wvfm_index-8000]
    val_pred = predicted_amps[wvfm_index-8000]

    fig.add_trace(go.Scatter(
        x=[max_sample], y=[val_real],
        mode='markers', name='Real Amp',
        marker=dict(color='blue', size=10, symbol='circle-open', line=dict(width=2))
    ))

    fig.add_trace(go.Scatter(
        x=[max_sample], y=[val_pred],
        mode='markers', name='Predicted Amp',
        marker=dict(color='red', size=10, symbol='x', line=dict(width=2))
    ))

    fig.update_layout(title=f"Waveform Display: Event {wvfm_index}")
    fig.update_yaxes(range=[min(y_corrected)-1, max(y_corrected)+1])
    
    return fig  

def sc_build(real_amps, predicted_amps, diff):
    event_indices = np.arange(len(real_amps))
    fig = px.scatter(
        x=real_amps, 
        y=predicted_amps, 
        color=diff,
        # custom_data=event_indices[:,None],
        labels={
            'x': 'Real Amplitudes (p.e.)', 
            'y': 'Predicted Amplitudes (p.e.)',
            'color': 'Difference'
            }, 
        title='Real vs Predicted Amplitudes',
        color_continuous_scale='viridis'
        )
    fig.update_layout(clickmode='event+select', dragmode='zoom',uirevision='0.7')
    fig.update_traces(marker=dict(symbol='square'), unselected=dict(marker=dict(opacity=0.7)))
    return fig

def ehist_build(real, predicted, title="Prediction Error Analysis"):
    """
    Creates a Plotly figure with a Histogram, Pseudo-Voigt fit, and Residuals.
    """
    # 1. Prepare Data
    diff = predicted - real
    
    # Calculate Histogram for fitting
    counts, bin_edges = np.histogram(diff, bins=100, range=(-0.4, 0.4))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # 2. Define and Fit Pseudo-Voigt
    def pseudo_voigt(x, amp, x0, sigma, eta):
        g = np.exp(-np.log(2) * ((x - x0) / sigma)**2)
        l = 1 / (1 + ((x - x0) / sigma)**2)
        return amp * ((1 - eta) * g + eta * l)

    p0_voigt = [np.max(counts), np.mean(diff), np.std(diff)/2, 0.5]
    try:
        p_v, _ = curve_fit(pseudo_voigt, bin_centers, counts, p0=p0_voigt,
                            bounds=([0, -np.inf, 1e-9, 0], [np.inf, np.inf, np.inf, 1]))
    except RuntimeError:
        p_v = p0_voigt # Fallback to initial guess if fit fails

    x_fine = np.linspace(-0.4, 0.4, 1000)
    y_fit = pseudo_voigt(x_fine, *p_v)
    residuals = counts - pseudo_voigt(bin_centers, *p_v)

    # 3. Create Subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25]
    )

    # A. Main Histogram (as a step line)
    fig.add_trace(go.Scatter(
        x=bin_edges, 
        y=np.append(counts, counts[-1]), 
        line_shape='hv', 
        name='Data',
        line=dict(color='#282828', width=2) # Dark grey/black
    ), row=1, col=1)

    # B. Fit Line
    fig.add_trace(go.Scatter(
        x=x_fine, 
        y=y_fit, 
        name=f'Fit (η={p_v[3]:.2f})',
        line=dict(color='#458588', dash='dash', width=2) # Gruvbox Blue
    ), row=1, col=1)

    # C. Residuals
    fig.add_trace(go.Scatter(
        x=bin_centers, 
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(color='#cc241d', size=5), # Gruvbox Red
        error_y=dict(type='data', array=np.sqrt(counts), visible=True, thickness=1, width=0)
    ), row=2, col=1)

    # D. Formatting
    fig.add_hline(y=0, line_dash="dash", line_color="#458588", opacity=0.5, row=2, col=1)
    
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=800,
        height=600,
        hovermode="x unified",
        showlegend=True,
        legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.95),
        margin=dict(l=60, r=20, t=60, b=60)
    )

    fig.update_xaxes(range=[-0.4, 0.4])
    fig.update_yaxes(title_text=f"Counts / {bin_width:.3f} p.e.", row=1, col=1)
    fig.update_yaxes(title_text="Resid.", row=2, col=1)
    fig.update_xaxes(title_text="Prediction Error (Predicted - Real)", row=2, col=1)

    return fig

def amphist_build(real, predicted):
    # 1. Compute histograms manually to create the 'step' look
    # Using the same bins for both ensures a fair comparison
    counts_p, bins_p = np.histogram(predicted, bins=150)
    counts_r, bins_r = np.histogram(real, bins=150)

    # 2. Create Figure
    fig = go.Figure()

    # Add Real Amplitudes (Blue)
    fig.add_trace(go.Scatter(
        x=bins_r, 
        y=np.append(counts_r, counts_r[-1]), # Append last value to close the step
        line_shape='hv', 
        name='Real Amplitudes',
        line=dict(color='#0000FF', width=2),
        fill='tonexty', # Optional: adds a light fill if you want
        opacity=0.7
    ))

    # Add Predicted Amplitudes (Red)
    fig.add_trace(go.Scatter(
        x=bins_p, 
        y=np.append(counts_p, counts_p[-1]), 
        line_shape='hv', 
        name='Predicted Amplitudes',
        line=dict(color='#FF0000', width=2),
        opacity=0.7
    ))

    # 3. Formatting
    fig.update_layout(
        title="Amplitude Distribution Comparison",
        xaxis_title="Amplitude (p.e.)",
        yaxis_title="Number of Events",
        template="plotly_white",
        width=800,
        height=500,
        hovermode="x unified", # Shows both values when hovering
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    return fig

def dashboard():
    st.set_page_config(layout="wide")
    st.title("SiPM Pulse Analysis with CNN")

    real_amps, predicted_amps, diff = prepare_data()

    # Create columns for the top row
    col1, col2 = st.columns(2)

    with col1:
        fig1a = sc_build(real_amps, predicted_amps, diff)
        apply_unified_style(fig1a, "Real vs Predicted Amplitudes")
        event1a = st.plotly_chart(fig1a, selection_mode='points', on_select='rerun', key="scatter_plot")

    with col2:
        fig1b = wvfm_build()
        apply_unified_style(fig1b, "Waveform Display")
    
        selection = event1a.get("selection", {}).get("points", [])
        if selection:
            point_index = selection[0].get("point_index")
            absolute_index = point_index + 8000 
            fig1b = wvfm_update(fig1b, absolute_index, real_amps, predicted_amps)

        st.plotly_chart(fig1b, key="waveform_display")

    fig2 = amphist_build(real_amps, predicted_amps)
    apply_unified_style(fig2, "Amplitude Distribution Comparison")
    st.plotly_chart(fig2, use_container_width=True)
    

    fig3 = ehist_build(real_amps, predicted_amps)
    apply_unified_style(fig3, "Error Distribution with Pseudo-Voigt Fit")
    st.plotly_chart(fig3, use_container_width=True)
    

dashboard()