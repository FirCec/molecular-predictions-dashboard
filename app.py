"""
Understanding Molecular Predictions - interactive dashboard.

Step 3.2: load the ESOL dataset, show key figures and an interactive
"measured vs predicted" chart for the ESOL baseline formula.
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
    """Read the ESOL CSV file and rename its columns."""
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
        title="ESOL formula: predicted vs measured solubility",
        xaxis_title="Measured log S (mol/L)",
        yaxis_title="Predicted log S (mol/L)",
        height=600,
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
        range=[low, high], dtick=2, scaleanchor="x", scaleratio=1,
        gridcolor="#ecebe8", zeroline=False,
    )
    return fig


# --- Page ---------------------------------------------------------------------

st.set_page_config(page_title="Understanding Molecular Predictions", layout="wide")

st.title("Understanding Molecular Predictions")
st.write(
    "Exploring how well molecular structure predicts **aqueous solubility** "
    "(log S, in mol/L) on the ESOL dataset."
)

df = load_data(DATA_PATH)

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

# Chart on the left (wider), reading guide on the right.
chart_col, text_col = st.columns([3, 1])
with chart_col:
    st.plotly_chart(make_parity_plot(df), width="stretch")
with text_col:
    st.markdown(
        """
**How to read this chart**

- Each dot is one molecule.
- Dots **on the dashed line** are predicted perfectly.
- **Red**: the formula predicts the molecule *more* soluble than it is.
- **Blue**: *less* soluble than it is.
- Hover a dot to see the molecule's name and values.
"""
    )

st.subheader("The dataset")
st.dataframe(df, width="stretch", hide_index=True)