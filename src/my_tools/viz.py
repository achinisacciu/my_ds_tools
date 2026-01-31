"""
Visualization Module
Imports and configures Matplotlib, Seaborn, and Plotly with the custom design system.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.io as pio

# Import theme variables relative to the package
from .theme import COLOR_PALETTE, PALETTE_BINARY, PALETTE_CATEGORICAL, FONTS

# ==============================================================================
# PLOTLY CONFIGURATION (Auto-applied on import)
# ==============================================================================
# Define custom template based on the design system
pio.templates["my_custom_theme"] = go.layout.Template(
    layout=go.Layout(
        colorway=PALETTE_CATEGORICAL,
        font=dict(
            family=FONTS['family'].split(',')[0],
            color=COLOR_PALETTE['text_primary']
        ),
        title=dict(font=dict(size=FONTS['title_size'])),
        paper_bgcolor=COLOR_PALETTE['background'],
        plot_bgcolor=COLOR_PALETTE['background'],
        xaxis=dict(gridcolor=COLOR_PALETTE['grid_color']),
        yaxis=dict(gridcolor=COLOR_PALETTE['grid_color']),
    )
)

# Set as default
pio.templates.default = "my_custom_theme"

# ==============================================================================
# MATPLOTLIB / SEABORN CONFIGURATION (Auto-applied on import)
# ==============================================================================
# Update global matplotlib params
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [FONTS['family'].split(',')[0]],
    'axes.prop_cycle': plt.cycler(color=PALETTE_CATEGORICAL),
    'axes.titlesize': FONTS['title_size'],
    'axes.grid': True,
    'grid.color': COLOR_PALETTE['grid_color'],
    'figure.facecolor': COLOR_PALETTE['background'],
    'axes.facecolor': COLOR_PALETTE['background'],
    'text.color': COLOR_PALETTE['text_primary'],
    'axes.labelcolor': COLOR_PALETTE['text_primary'],
    'xtick.color': COLOR_PALETTE['text_primary'],
    'ytick.color': COLOR_PALETTE['text_primary']
})

# Update seaborn palette
sns.set_palette(sns.color_palette(PALETTE_CATEGORICAL))

# ==============================================================================
# EXPORT
# ==============================================================================
__all__ = [
    # Visualization Libraries
    'plt', 'ListedColormap', 'sns',
    'px', 'go', 'ff', 'make_subplots', 'pio',
    
    # Theme Variables (Must be listed here to be available via 'from viz import *')
    'COLOR_PALETTE', 'PALETTE_BINARY', 'PALETTE_CATEGORICAL', 'FONTS'
]
