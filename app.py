"""
Understanding Molecular Predictions - interactive dashboard.

Step 3.3: two pages (Dashboard, Dataset) with a sidebar menu. The dashboard
shows key figures and an interactive "measured vs predicted" chart for the
ESOL baseline formula.
Run with:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Settings -----------------------------------------------------------------

# Build the path to the data file relative to THIS file (app.py), so the app
# finds the data no matter which folder the terminal is in.
PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "delaney-processed.csv"

# The original column names are long. We give them short, code-friendly names.
COLUMN_NAMES = {
    "Compound ID": "name",
    "smiles": "smiles",
    "measured log solubility in mols per litre": "logS_measured",
    "ESOL predicted log solubility in mols per litre": "logS_esol",
    "Molecular Weight": "mol_weight",
    "Number of H-Bond Donors": "h_bond_donors",
    "Number of Rings": "n_rings",
    "Number of Rotatable Bonds": "n_rotatable_bonds",
    "Polar Surface Area": "polar_surface_area",
    "Minimum Degree": "min_degree",
}


# --- Data loading -------------------------------------------------------------

# @st.cache_data: Streamlit re-runs this whole script after every click.
# The cache makes it read the CSV only once and reuse the result afterwards.
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """Read the ESOL CSV file, rename its columns and add the ESOL error."""
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_NAMES)
    # Error of the ESOL formula for each molecule: predicted minus measured.
    # Positive = the formula thinks the molecule is MORE soluble than it is.
    df["error_esol"] = df["logS_esol"] - df["logS_measured"]
    return df


# --- Charts -------------------------------------------------------------------

# Diverging colour scale for the error: blue = predicted too low,
# grey = correct, red = predicted too high.
ERROR_COLORSCALE = [[0.0, "#184f95"], [0.5, "#cfcdc8"], [1.0, "#b8302f"]]
ERROR_LIMIT = 3  # errors beyond +/-3 log units get the darkest colour

# Chart size in pixels (wider than tall, to leave room for the colour bar).
CHART_WIDTH = 640
CHART_HEIGHT = 600


def make_parity_plot(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of measured vs predicted log S, one dot per molecule."""
    fig = go.Figure()

    # 1) The diagonal "perfect prediction" line (predicted = measured).
    low = min(df["logS_measured"].min(), df["logS_esol"].min()) - 0.5
    high = max(df["logS_measured"].max(), df["logS_esol"].max()) + 0.5
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            line=dict(color="#8a8984", width=2, dash="dash"),
            name="Perfect prediction",
            hoverinfo="skip",
        )
    )

    # 2) One dot per molecule, coloured by its error.
    fig.add_trace(
        go.Scatter(
            x=df["logS_measured"],
            y=df["logS_esol"],
            mode="markers",
            name="Molecule",
            marker=dict(
                size=8,
                color=df["error_esol"],
                colorscale=ERROR_COLORSCALE,
                cmin=-ERROR_LIMIT,
                cmax=ERROR_LIMIT,
                line=dict(color="white", width=1),  # thin ring separates overlapping dots
                colorbar=dict(title="Error<br>(pred - meas)"),
            ),
            # Extra information shown in the tooltip when hovering a dot.
            customdata=df[["name", "error_esol"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Measured log S: %{x:.2f}<br>"
                "Predicted log S: %{y:.2f}<br>"
                "Error: %{customdata[1]:+.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        # x=0.5 + xanchor="center": centre the title above the chart.
        title=dict(
            text="ESOL formula: predicted vs measured solubility",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Measured log S (mol/L)",
        yaxis_title="Predicted log S (mol/L)",
        # Fixed size: a square plot area plus room for the colour bar.
        width=CHART_WIDTH,
        height=CHART_HEIGHT,
        margin=dict(l=70, r=10, t=60, b=60),  # small margins: the colour bar sits close
        showlegend=False,
        plot_bgcolor="white",
    )
    # Same range on both axes + square aspect, so the diagonal is truly 45 degrees.
    # constrain="domain": shrink the plot area instead of stretching the axis range.
    fig.update_xaxes(
        range=[low, high], dtick=2, constrain="domain",
        gridcolor="#ecebe8", zeroline=False,
    )
    fig.update_yaxes(
        range=[low, high], dtick=2, scaleanchor="x", scaleratio=1, constrain="domain",
        gridcolor="#ecebe8", zeroline=False,
    )
    return fig


# --- Pages --------------------------------------------------------------------

READING_GUIDE = """
**How to read this chart**

- Each dot is one molecule.
- Dots **on the dashed line** are predicted perfectly.
- **Red**: the formula predicts the molecule *more* soluble than it is.
- **Blue**: *less* soluble than it is.
- Hover a dot to see the molecule's name and values.
"""


def dashboard_page() -> None:
    """Overview: key figures, baseline performance and the parity plot."""
    df = load_data(DATA_PATH)

    st.title("Understanding Molecular Predictions")
    st.markdown(
        "Exploring how well molecular structure predicts **aqueous solubility** "
        "(log S, in mol/L) on the ESOL dataset."
    )

    # Three key numbers shown side by side.
    col1, col2, col3 = st.columns(3)
    col1.metric("Molecules", f"{len(df):,}")
    col2.metric("Most soluble (log S)", f"{df['logS_measured'].max():.2f}")
    col3.metric("Least soluble (log S)", f"{df['logS_measured'].min():.2f}")

    # --- Baseline performance ---
    st.subheader("Baseline: the ESOL formula (Delaney, 2004)")

    mae = df["error_esol"].abs().mean()
    rmse = np.sqrt((df["error_esol"] ** 2).mean())

    col1, col2 = st.columns(2)
    col1.metric("MAE (log S)", f"{mae:.2f}", help="Average size of the error.")
    col2.metric("RMSE (log S)", f"{rmse:.2f}", help="Like MAE, but big errors count more.")

    # Chart and reading guide side by side, packed together and vertically centred.
    with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
        st.plotly_chart(make_parity_plot(df), width="content")
        with st.container(width=320):
            st.markdown(READING_GUIDE)


def dataset_page() -> None:
    """The full ESOL table, with readable column names."""
    df = load_data(DATA_PATH)

    st.title("The ESOL dataset")
    st.markdown(
        f"{len(df):,} molecules. Click a column header to sort, "
        "or use the search icon above the table."
    )
    st.dataframe(
        df,
        width="stretch",
        height=600,
        hide_index=True,
        column_order=[
            "name", "logS_measured", "logS_esol", "error_esol", "smiles",
            "mol_weight", "h_bond_donors", "n_rings", "n_rotatable_bonds",
            "polar_surface_area", "min_degree",
        ],
        column_config={
            "name": "Molecule",
            "logS_measured": st.column_config.NumberColumn("Measured log S", format="%.2f"),
            "logS_esol": st.column_config.NumberColumn("ESOL predicted log S", format="%.2f"),
            "error_esol": st.column_config.NumberColumn("ESOL error", format="%+.2f"),
            "smiles": "SMILES",
            "mol_weight": st.column_config.NumberColumn("Molecular weight", format="%.1f"),
            "h_bond_donors": "H-bond donors",
            "n_rings": "Rings",
            "n_rotatable_bonds": "Rotatable bonds",
            "polar_surface_area": st.column_config.NumberColumn("Polar surface area", format="%.1f"),
            "min_degree": "Minimum degree",
        },
    )


# --- App ----------------------------------------------------------------------

st.set_page_config(page_title="Understanding Molecular Predictions", layout="wide")

# Navigation menu in the sidebar: one entry per page.
page = st.navigation([
    st.Page(dashboard_page, title="Dashboard", icon=":material/insights:", default=True),
    st.Page(dataset_page, title="Dataset", icon=":material/table_chart:", url_path="dataset"),
])
page.run()