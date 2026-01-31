"""
EDA Module
High-level functions for Exploratory Data Analysis (Overview, Missing Values, Reports).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap
from IPython.display import display, Markdown

# Importa il tuo design system interno
from .theme import COLOR_PALETTE, FONTS

# ==============================================================================
# 1. OVERVIEW GENERALE
# ==============================================================================
def eda_overview(df):
    """
    Prints a standard overview of the dataframe: head, info, shape and column types.
    """
    print("Top 5 Rows:")
    display(df.head())

    print("\nData Types Info:")
    df.info()

    print(f"\nDataset Dimensions: {df.shape}")

    # Column separation
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    print(f"\n Categorical Columns ({len(categorical_cols)}):")
    print(f"   {', '.join(categorical_cols)}")

    print(f"\n Numerical Columns ({len(numerical_cols)}):")
    print(f"   {', '.join(numerical_cols)}")
    
    return categorical_cols, numerical_cols

# ==============================================================================
# 2. MISSING VALUES MATRIX (Matplotlib/Seaborn)
# ==============================================================================
def plot_missing_matrix(df, figsize=(15, 7)):
    """
    Plots a binary heatmap showing where missing values are located.
    """
    plt.figure(figsize=figsize)
    
    # Crea colormap binaria dai tuoi colori (Sfondo vs Testo per alto contrasto)
    custom_cmap = ListedColormap([COLOR_PALETTE['background'], COLOR_PALETTE['text_primary']])

    sns.heatmap(
        df.isnull(), 
        cbar=False, 
        cmap=custom_cmap, 
        yticklabels=False
    )

    # Styling
    plt.title(
        "MISSING VALUES MAP", 
        pad=20, 
        fontsize=FONTS['title_size'],
        fontweight='bold', 
        color=COLOR_PALETTE['text_primary']
    )

    plt.xlabel(
        "Variables", 
        fontsize=12, 
        labelpad=10, 
        color=COLOR_PALETTE['text_primary']
    )

    sns.despine(left=True, bottom=True, top=True, right=True)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 3. MISSING VALUES REPORT (Plotly + Pandas Style)
# ==============================================================================
def eda_missing(df):
    """
    Generates a full report on missing values: styled table + interactive bar chart.
    """
    # 1. Calcolo
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })

    # 2. Filtro (Mostra solo chi ha NaN)
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing %', ascending=False)

    if missing_df.empty:
        display(Markdown("#### ✅ No missing values (NaN) found."))
        return

    display(Markdown("#### ⚠️ MISSING VALUES REPORT:"))
    
    # 3. Tabella Pandas Stilizzata
    # Nota: Usiamo il colore secondario (Rosso) per evidenziare i problemi
    display(
        missing_df.style
        .format({'Missing %': '{:.2f}%'})
        .bar(subset=['Missing %'], color=COLOR_PALETTE['cat_secondary'], vmin=0, vmax=100)
    )
    
    # 4. Grafico Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=missing_df.index,
        y=missing_df['Missing %'],
        marker_color=COLOR_PALETTE['cat_secondary'], # Rosso per allarme
        
        # Etichette
        text=missing_df['Missing %'].apply(lambda x: f"{x:.1f}%"),
        textposition='auto',
        textfont=dict(size=14, weight='bold'),
        
        # Hover
        hovertemplate="<b>%{x}</b><br>Missing: %{y:.2f}%<extra></extra>"
    ))

    # Layout Minimalista
    fig.update_layout(
        title=dict(
            text="<b>MISSING VALUES INCIDENCE (%)</b>", 
            font=dict(size=18, color=COLOR_PALETTE['text_primary'])
        ),
        xaxis_title="Variables",
        height=500, 
        paper_bgcolor=COLOR_PALETTE['background'],
        plot_bgcolor=COLOR_PALETTE['background'],
        font=dict(
            family=FONTS['family'].split(',')[0], 
            color=COLOR_PALETTE['text_primary']
        ),
        margin=dict(t=80, b=50),
        
        yaxis=dict(showgrid=True, showticklabels=False, visible=False),
        xaxis=dict(showgrid=False, linecolor=COLOR_PALETTE['text_primary'])
    )

    fig.show()

__all__ = ['eda_overview', 'plot_missing_matrix', 'eda_missing']
