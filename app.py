"""
Understanding Molecular Predictions - interactive dashboard.

Step 3.1: load the ESOL dataset and show it as a table.
Run with:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
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
    return df


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

st.subheader("The dataset")
st.dataframe(df, width="stretch", hide_index=True)