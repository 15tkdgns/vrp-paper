import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json

# Load data
print("Loading attention analysis data...")
attn_df = pd.read_csv('src/experiments/verification/v63_attention_maps.csv')
with open('src/experiments/verification/v63_results.json', 'r') as f:
    results = json.load(f)

# Extract lag columns
lag_cols = [f'lag_{i}' for i in range(22)]
attn_data = attn_df[lag_cols].values

# Convert Date to datetime
attn_df['Date'] = pd.to_datetime(attn_df['Date'])

# ==========================================
# 1. Interactive Asset Comparison
# ==========================================
print("Creating interactive asset comparison...")

fig1 = go.Figure()

for asset in attn_df['Asset'].unique():
    asset_mask = attn_df['Asset'] == asset
    avg_attn = attn_data[asset_mask].mean(axis=0)
    
    fig1.add_trace(go.Bar(
        name=asset,
        x=list(range(22)),
        y=avg_attn,
        visible=(asset == 'SPY')  # Only SPY visible by default
    ))

# Add dropdown menu
dropdown_buttons = []
for i, asset in enumerate(attn_df['Asset'].unique()):
    visible = [False] * len(attn_df['Asset'].unique())
    visible[i] = True
    dropdown_buttons.append({
        'label': asset,
        'method': 'update',
        'args': [{'visible': visible},
                 {'title': f'{asset} - Average Attention Profile'}]
    })

# Add "All" option
all_visible = [True] * len(attn_df['Asset'].unique())
dropdown_buttons.insert(0, {
    'label': 'All Assets',
    'method': 'update',
    'args': [{'visible': all_visible},
             {'title': 'All Assets - Average Attention Profile'}]
})

fig1.update_layout(
    title='Interactive Asset Attention Comparison',
    xaxis_title='Lag (0=oldest, 21=newest)',
    yaxis_title='Attention Weight',
    updatemenus=[{
        'buttons': dropdown_buttons,
        'direction': 'down',
        'showactive': True,
        'x': 0.17,
        'xanchor': 'left',
        'y': 1.15,
        'yanchor': 'top'
    }],
    hovermode='x unified',
    template='plotly_white',
    height=500
)

# ==========================================
# 2. Time Series Attention Heatmap with Slider
# ==========================================
print("Creating time series heatmap with slider...")

# Use first 500 samples for performance
n_samples = min(500, len(attn_df))
heatmap_data = attn_data[:n_samples]
dates = attn_df['Date'].iloc[:n_samples]
assets = attn_df['Asset'].iloc[:n_samples]

fig2 = go.Figure(data=go.Heatmap(
    z=heatmap_data,
    x=list(range(22)),
    y=dates,
    colorscale='Viridis',
    colorbar=dict(title='Attention<br>Weight'),
    hovertemplate='Date: %{y}<br>Lag: %{x}<br>Weight: %{z:.4f}<extra></extra>'
))

fig2.update_layout(
    title='Attention Weights Over Time (Heatmap)',
    xaxis_title='Lag (0=oldest, 21=newest)',
    yaxis_title='Date',
    template='plotly_white',
    height=600
)

# ==========================================
# 3. Interactive Lag Importance
# ==========================================
print("Creating interactive lag importance...")

overall_avg = attn_data.mean(axis=0)

# Create grouped data
early_vs_recent = pd.DataFrame({
    'Period': ['Early Lags (0-10)', 'Recent Lags (11-21)'],
    'Cumulative Weight': [overall_avg[:11].sum(), overall_avg[11:].sum()],
    'Percentage': [overall_avg[:11].sum() / overall_avg.sum() * 100,
                   overall_avg[11:].sum() / overall_avg.sum() * 100]
})

fig3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Overall Attention Profile', 'Early vs Recent Cumulative'),
    specs=[[{'type': 'scatter'}, {'type': 'bar'}]]
)

# Left: Line plot
fig3.add_trace(
    go.Scatter(
        x=list(range(22)),
        y=overall_avg,
        mode='lines+markers',
        name='Avg Attention',
        line=dict(color='navy', width=2),
        marker=dict(size=8),
        hovertemplate='Lag: %{x}<br>Weight: %{y:.4f}<extra></extra>'
    ),
    row=1, col=1
)

# Add vertical line at split
fig3.add_vline(x=10.5, line_dash="dash", line_color="red", 
               annotation_text="Split", row=1, col=1)

# Right: Bar chart
colors = ['skyblue', 'coral']
fig3.add_trace(
    go.Bar(
        x=early_vs_recent['Period'],
        y=early_vs_recent['Cumulative Weight'],
        marker_color=colors,
        text=[f"{w:.3f}<br>({p:.1f}%)" for w, p in 
              zip(early_vs_recent['Cumulative Weight'], early_vs_recent['Percentage'])],
        textposition='outside',
        hovertemplate='%{x}<br>Weight: %{y:.3f}<extra></extra>'
    ),
    row=1, col=2
)

fig3.update_xaxes(title_text='Lag', row=1, col=1)
fig3.update_xaxes(title_text='', row=1, col=2)
fig3.update_yaxes(title_text='Attention Weight', row=1, col=1)
fig3.update_yaxes(title_text='Cumulative Weight', row=1, col=2)

fig3.update_layout(
    title_text='Lag Importance Analysis',
    showlegend=False,
    template='plotly_white',
    height=400
)

# ==========================================
# 4. Attention by Date Range (Interactive Filter)
# ==========================================
print("Creating date-filtered attention visualization...")

# Calculate monthly average
attn_df['YearMonth'] = attn_df['Date'].dt.to_period('M').astype(str)
monthly_avg = []
for ym in attn_df['YearMonth'].unique():
    mask = attn_df['YearMonth'] == ym
    avg_attn = attn_data[mask].mean(axis=0)
    monthly_avg.append({'YearMonth': ym, 'avg_attn': avg_attn})

monthly_df = pd.DataFrame(monthly_avg)

# Create animation frames
fig4 = px.bar(
    x=list(range(22)),
    y=monthly_df['avg_attn'].iloc[0],
    labels={'x': 'Lag (0=oldest, 21=newest)', 'y': 'Attention Weight'},
    title=f'Monthly Average Attention - {monthly_df["YearMonth"].iloc[0]}'
)

# Add all months as frames
frames = []
for idx, row in monthly_df.iterrows():
    frames.append(go.Frame(
        data=[go.Bar(x=list(range(22)), y=row['avg_attn'])],
        name=row['YearMonth'],
        layout=go.Layout(title_text=f'Monthly Average Attention - {row["YearMonth"]}')
    ))

fig4.frames = frames

# Add play/pause buttons
fig4.update_layout(
    updatemenus=[{
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 500, 'redraw': True}, 
                               'fromcurrent': True}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 
                                  'mode': 'immediate'}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }],
    sliders=[{
        'active': 0,
        'steps': [{'args': [[f.name], {'frame': {'duration': 0, 'redraw': True}, 
                                       'mode': 'immediate'}],
                   'label': f.name,
                   'method': 'animate'} for f in frames],
        'y': 0,
        'len': 0.9,
        'x': 0.1,
        'xanchor': 'left',
        'yanchor': 'top'
    }],
    template='plotly_white',
    height=500
)

# ==========================================
# 5. Combined Dashboard HTML
# ==========================================
print("Creating combined interactive dashboard...")

# Create comprehensive dashboard
from plotly.subplots import make_subplots

# Overview stats
stats_text = f"""
<div style='font-family: Arial; padding: 20px; background-color: #f0f0f0; border-radius: 10px; margin: 20px 0;'>
<h2>Attention Analysis Summary</h2>
<ul>
<li><b>Most Focused Lag:</b> {results['overall_stats']['max_lag']} (weight: {results['overall_stats']['max_weight']:.4f})</li>
<li><b>Early Lags Total:</b> {results['overall_stats']['early_lags_total']:.3f} (47.2%)</li>
<li><b>Recent Lags Total:</b> {results['overall_stats']['recent_lags_total']:.3f} (52.8%)</li>
<li><b>Average Entropy:</b> {results['overall_stats']['avg_entropy']:.3f}</li>
</ul>

<h3>Asset-Specific Focus</h3>
<ul>
"""

for asset, stats in results['asset_stats'].items():
    stats_text += f"<li><b>{asset}:</b> Lag {stats['focus_lag']} (weight: {stats['max_weight']:.4f})</li>\n"

stats_text += """
</ul>
</div>
"""

# Save individual figures
print("Saving individual HTML files...")
fig1.write_html('src/experiments/verification/v63_interactive_asset_comparison.html')
fig2.write_html('src/experiments/verification/v63_interactive_heatmap.html')
fig3.write_html('src/experiments/verification/v63_interactive_lag_importance.html')
fig4.write_html('src/experiments/verification/v63_interactive_monthly.html')

# Create combined HTML
combined_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dual Attention Analysis - Interactive Dashboard</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #ffffff;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 10px;
            background-color: #fafafa;
        }}
        .section h2 {{
            color: #34495e;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
    </style>
</head>
<body>
    <h1>🔍 Dual Attention LSTM - Interactive Analysis Dashboard</h1>
    
    {stats_text}
    
    <div class="section">
        <h2>1. Asset Comparison (Interactive Dropdown)</h2>
        {fig1.to_html(include_plotlyjs='cdn', div_id='fig1')}
    </div>
    
    <div class="section">
        <h2>2. Lag Importance Analysis</h2>
        {fig3.to_html(include_plotlyjs='cdn', div_id='fig3')}
    </div>
    
    <div class="section">
        <h2>3. Time Series Heatmap</h2>
        {fig2.to_html(include_plotlyjs='cdn', div_id='fig2')}
    </div>
    
    <div class="section">
        <h2>4. Monthly Attention Evolution (Animation)</h2>
        {fig4.to_html(include_plotlyjs='cdn', div_id='fig4')}
    </div>
    
    <div style='text-align: center; margin-top: 50px; color: #7f8c8d;'>
        <p>Generated by v63_interactive_dashboard.py</p>
        <p>DualAttentionLSTM Attention Mechanism Analysis</p>
    </div>
</body>
</html>
"""

with open('src/experiments/verification/v63_interactive_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(combined_html)

print("\n" + "="*80)
print("Interactive Dashboard Complete!")
print("="*80)
print("\nGenerated Files:")
print("  - v63_interactive_asset_comparison.html")
print("  - v63_interactive_heatmap.html")
print("  - v63_interactive_lag_importance.html")
print("  - v63_interactive_monthly.html")
print("  - v63_interactive_dashboard.html (Combined Dashboard)")
print("\nOpen the HTML files in a web browser to interact with the visualizations.")
