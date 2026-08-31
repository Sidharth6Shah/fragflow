"""
FragFlow - Molecule Design Studio
AI-powered molecular generation with GFlowNets
"""
import streamlit as st
import sys
from pathlib import Path
import base64
from io import BytesIO

# Setup
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, QED, AllChem
from rdkit.Contrib.SA_Score import sascorer
from evaluation.sample import MoleculeSampler
import streamlit.components.v1 as components

# Page config
st.set_page_config(
    page_title="FragFlow - Molecule Design Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for design studio vibe
st.markdown("""
<style>
    /* Gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Main container */
    .main {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
    }

    /* Property cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    .metric-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Molecule viewer container */
    .mol-viewer {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
    }

    /* Generate button */
    .stButton > button {
        background: linear-gradient(135deg, #00f0ff 0%, #667eea 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 1rem 3rem;
        border-radius: 50px;
        border: none;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.6);
    }

    /* Header */
    h1 {
        color: white !important;
        text-align: center;
        font-size: 3rem !important;
        margin-bottom: 0 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }

    .subtitle {
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* SMILES display */
    .smiles-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        font-family: monospace;
        color: #333;
        text-align: center;
        font-size: 0.9rem;
    }

    /* Sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
    }

    /* Toggle switch styling */
    .stRadio > label {
        background: rgba(255, 255, 255, 0.95);
        padding: 0.5rem 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generator' not in st.session_state:
    with st.spinner('🧬 Loading FragFlow model...'):
        checkpoint_path = project_root / "checkpoints" / "checkpoint_10000.pt"
        st.session_state.generator = MoleculeSampler(checkpoint_path, vocab_size=200, max_frags=8)
    st.session_state.current_mol = None
    st.session_state.current_smiles = None
    st.session_state.current_reward = None
    st.session_state.current_properties = {}

# Header
st.markdown("<h1>🧪 FragFlow</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Molecule Design Studio</p>", unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Controls")

    # Visualization mode
    st.subheader("Visualization")
    viz_mode = st.radio("Mode", ["2D", "3D"], horizontal=True)

    # Scaffold filter
    st.subheader("Scaffold Filter")
    use_scaffold = st.checkbox("Enable scaffold filter")

    if use_scaffold:
        scaffold_presets = {
            "Benzene Ring": "c1ccccc1",
            "Pyridine": "c1cnccc1",
            "6-Ring": "C1CCCCC1",
            "Carboxylic Acid": "C(=O)O"
        }

        preset = st.selectbox("Preset", ["Custom"] + list(scaffold_presets.keys()))

        if preset == "Custom":
            scaffold_smarts = st.text_input("SMARTS pattern", "c1ccccc1")
        else:
            scaffold_smarts = scaffold_presets[preset]
            st.code(scaffold_smarts)
    else:
        scaffold_smarts = None

    # Property filters
    st.subheader("Property Filters")
    filter_qed = st.slider("Min QED", 0.0, 1.0, 0.0, 0.1)
    filter_sa = st.slider("Max SA Score", 1.0, 10.0, 10.0, 0.5)
    filter_logp_min = st.slider("Min LogP", -5.0, 10.0, -5.0, 0.5)
    filter_logp_max = st.slider("Max LogP", -5.0, 10.0, 10.0, 0.5)

# Generate button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🎲 GENERATE NEW MOLECULE"):
        with st.spinner('Generating molecule...'):
            # Keep generating until we get one that passes filters
            max_attempts = 50
            for attempt in range(max_attempts):
                state, mol, reward = st.session_state.generator.sample_molecule()

                if mol is None:
                    continue

                # Check scaffold filter
                if use_scaffold and scaffold_smarts:
                    pattern = Chem.MolFromSmarts(scaffold_smarts)
                    if pattern and not mol.HasSubstructMatch(pattern):
                        continue

                # Compute properties
                qed = QED.qed(mol)
                sa = sascorer.calculateScore(mol)
                logp = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                tpsa = Descriptors.TPSA(mol)
                rings = Descriptors.RingCount(mol)
                atoms = mol.GetNumHeavyAtoms()

                # Check property filters
                if qed < filter_qed:
                    continue
                if sa > filter_sa:
                    continue
                if logp < filter_logp_min or logp > filter_logp_max:
                    continue

                # Found a valid molecule!
                st.session_state.current_mol = mol
                st.session_state.current_smiles = Chem.MolToSmiles(mol)
                st.session_state.current_reward = reward
                st.session_state.current_properties = {
                    'QED': qed,
                    'SA': sa,
                    'LogP': logp,
                    'MW': mw,
                    'TPSA': tpsa,
                    'Rings': rings,
                    'Atoms': atoms
                }
                break
            else:
                st.error(f"Could not generate molecule matching filters after {max_attempts} attempts. Try relaxing the constraints.")

# Main content area
if st.session_state.current_mol is not None:
    mol = st.session_state.current_mol

    # Molecule viewer
    st.markdown("<div class='mol-viewer'>", unsafe_allow_html=True)

    if viz_mode == "2D":
        # 2D visualization
        img = Draw.MolToImage(mol, size=(600, 600))

        # Convert to base64
        buf = BytesIO()
        img.save(buf, format='PNG')
        img_str = base64.b64encode(buf.getvalue()).decode()

        # Display centered
        st.markdown(
            f"<div style='text-align: center;'><img src='data:image/png;base64,{img_str}' style='max-width: 100%; height: auto;'/></div>",
            unsafe_allow_html=True
        )
    else:
        # 3D visualization
        try:
            mol_3d = Chem.Mol(mol)
            result = AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())

            if result == -1:
                # Fallback to basic embedding
                result = AllChem.EmbedMolecule(mol_3d, randomSeed=42)

            if result != -1:
                AllChem.MMFFOptimizeMolecule(mol_3d)
                mol_block = Chem.MolToMolBlock(mol_3d)

                # py3Dmol viewer
                # Escape backticks in mol_block
                mol_block_escaped = mol_block.replace('`', '\\`')

                viewer_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
                    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
                </head>
                <body>
                    <div id="viewer" style="width: 100%; height: 500px; position: relative; background-color: white;"></div>
                    <script>
                        $(document).ready(function() {{
                            let viewer = $3Dmol.createViewer($("#viewer"), {{backgroundColor: 'white'}});
                            let molBlock = `{mol_block_escaped}`;
                            viewer.addModel(molBlock, "mol");
                            viewer.setStyle({{}}, {{stick: {{radius: 0.15}}, sphere: {{scale: 0.3}}}});
                            viewer.zoomTo();
                            viewer.render();
                            viewer.zoom(1.2, 500);
                        }});
                    </script>
                </body>
                </html>
                """
                components.html(viewer_html, height=520, scrolling=False)
            else:
                st.warning("Could not generate 3D coordinates. Showing 2D instead.")
                img = Draw.MolToImage(mol, size=(600, 600))
                st.image(img, use_container_width=True)
        except Exception as e:
            st.error(f"3D visualization error: {e}")
            img = Draw.MolToImage(mol, size=(600, 600))
            st.image(img, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Property cards
    props = st.session_state.current_properties
    cols = st.columns(5)

    property_data = [
        ("QED", props['QED'], "0-1 scale"),
        ("SA Score", props['SA'], "1-10 scale"),
        ("LogP", props['LogP'], "lipophilicity"),
        ("MW", props['MW'], "Da"),
        ("TPSA", props['TPSA'], "Ų"),
    ]

    for col, (label, value, unit) in zip(cols, property_data):
        with col:
            if label == "QED":
                formatted_value = f"{value:.3f}"
            elif label == "SA Score":
                formatted_value = f"{value:.2f}"
            elif label == "LogP":
                formatted_value = f"{value:.2f}"
            elif label == "MW":
                formatted_value = f"{value:.1f}"
            elif label == "TPSA":
                formatted_value = f"{value:.1f}"

            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{formatted_value}</div>
                <div style='font-size: 0.75rem; color: #999; margin-top: 0.25rem;'>{unit}</div>
            </div>
            """, unsafe_allow_html=True)

    # Additional properties
    st.markdown(f"""
    <div style='text-align: center; color: white; margin-top: 1rem; font-size: 0.9rem;'>
        Heavy Atoms: {props['Atoms']} | Rings: {props['Rings']} | Reward: {st.session_state.current_reward:.3f}
    </div>
    """, unsafe_allow_html=True)

    # SMILES display
    st.markdown(f"""
    <div class='smiles-box'>
        <strong>SMILES:</strong> {st.session_state.current_smiles}
    </div>
    """, unsafe_allow_html=True)

else:
    # Welcome message
    st.markdown("""
    <div style='text-align: center; color: white; padding: 4rem 2rem;'>
        <h2 style='color: white;'>Welcome to FragFlow</h2>
        <p style='font-size: 1.2rem; margin-top: 1rem;'>
            Generate drug-like molecules optimized for QED, synthetic accessibility, and LogP.
        </p>
        <p style='font-size: 1rem; margin-top: 1rem; opacity: 0.8;'>
            Click the button above to generate your first molecule!
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.7); margin-top: 3rem; padding: 1rem; font-size: 0.9rem;'>
    Powered by GFlowNets | Fragment-based molecular generation
</div>
""", unsafe_allow_html=True)
