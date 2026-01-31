"""
Design System Module
Defines the core color palettes, fonts, and style configurations used across the library.
"""

# ==============================================================================
# CORE COLOR PALETTE
# ==============================================================================
COLOR_PALETTE = {
    # Backgrounds
    'background': '#ffffff',      # Main white background
    'paper': '#e8e8e6',           # Light grey for cards/sections
    
    # Typography
    'text_primary': '#111111',    # Dark text for readability
    'text_secondary': '#ffffff',  # Light text (e.g., for dark backgrounds)
    
    # Categorical Colors (Safe & Distinct)
    'cat_primary': '#22547C',     # Deep Blue
    'cat_secondary': '#A20F11',   # Deep Red
    'cat_tertiary': '#2B6329',    # Deep Green
    'cat_quaternary': '#733A7C',  # Purple
    'cat_quinary': '#8C4600',     # Brown
    
    # Functional Accents
    'accent_main': '#22547C',     # Main action color (matches primary)
    
    # Structural Elements
    'grid_color': '#dddddd',      # Light grey for gridlines
}

# ==============================================================================
# DERIVED PALETTES
# ==============================================================================

# Binary Palette: High contrast for A/B testing or Yes/No visualizations
# Using Primary Blue vs Secondary Red
PALETTE_BINARY = [
    COLOR_PALETTE['cat_primary'], 
    COLOR_PALETTE['cat_secondary']
]

# Categorical Palette: List of all 5 main categorical colors for plotting
PALETTE_CATEGORICAL = [
    COLOR_PALETTE['cat_primary'],
    COLOR_PALETTE['cat_secondary'],
    COLOR_PALETTE['cat_tertiary'],
    COLOR_PALETTE['cat_quaternary'],
    COLOR_PALETTE['cat_quinary']
]

# ==============================================================================
# TYPOGRAPHY CONFIGURATION
# ==============================================================================
FONTS = {
    'family': 'Arial, sans-serif',
    'title_size': 22,
    'label_size': 14,
    'tick_size': 12
}

# Export these variables so they can be imported elsewhere
__all__ = ['COLOR_PALETTE', 'PALETTE_BINARY', 'PALETTE_CATEGORICAL', 'FONTS']
