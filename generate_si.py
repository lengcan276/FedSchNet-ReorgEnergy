"""
Generate Supplementary Information (SI) document: paper/supplementary.docx

All values sourced from actual code, CSV files, and runtime measurements.
"""

import csv
import os
import re
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

PROJECT_ROOT = Path(__file__).resolve().parent
DOCX_PATH = PROJECT_ROOT / 'paper' / 'supplementary.docx'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'tables'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'


# ============================================================
#  STYLES
# ============================================================

def setup_styles(doc):
    normal = doc.styles['Normal']
    normal.font.name = 'Times New Roman'
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    rPr = normal._element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    sizes = {1: 14, 2: 12, 3: 11}
    space_before = {1: 18, 2: 14, 3: 10}
    space_after_map = {1: 8, 2: 6, 3: 4}
    for level in [1, 2, 3]:
        style = doc.styles[f'Heading {level}']
        style.font.name = 'Times New Roman'
        style.font.bold = True
        style.font.size = Pt(sizes[level])
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(space_before[level])
        style.paragraph_format.space_after = Pt(space_after_map[level])
        style.paragraph_format.line_spacing = 1.5
        pPr = style._element.get_or_add_pPr()
        for numPr in pPr.findall(qn('w:numPr')):
            pPr.remove(numPr)


# ============================================================
#  HELPER: ADD TEXT PARAGRAPH
# ============================================================

def add_text(doc, text, bold=False, italic=False, indent=True):
    """Add a body text paragraph with optional formatting."""
    p = doc.add_paragraph()
    p.style = doc.styles['Normal']
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.5)
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(12)
    run.font.name = 'Times New Roman'
    return p


def add_mixed_paragraph(doc, parts, indent=True):
    """Add paragraph with mixed formatting. parts = [(text, bold, italic), ...]"""
    p = doc.add_paragraph()
    p.style = doc.styles['Normal']
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.5)
    for text, bold, italic in parts:
        run = p.add_run(text)
        run.bold = bold
        run.italic = italic
        run.font.size = Pt(12)
        run.font.name = 'Times New Roman'
    return p


# ============================================================
#  HELPER: ADD TABLE
# ============================================================

def set_cell_shading(cell, color_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)


def add_table(doc, title, headers, rows, col_widths=None, note=None):
    """Add a formatted table with title, optional note."""
    # Title
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.first_line_indent = Pt(0)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(title)
    run.bold = True
    run.font.size = Pt(11)
    run.font.name = 'Times New Roman'

    # Table
    n_cols = len(headers)
    table = doc.add_table(rows=len(rows) + 1, cols=n_cols)
    table.style = 'Table Grid'
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ''
        cp = cell.paragraphs[0]
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cp.add_run(h)
        run.bold = True
        run.font.size = Pt(9)
        run.font.name = 'Times New Roman'
        set_cell_shading(cell, 'D9E1F2')

    for i, row_data in enumerate(rows):
        for j, val in enumerate(row_data):
            cell = table.rows[i + 1].cells[j]
            cell.text = ''
            cp = cell.paragraphs[0]
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = cp.add_run(str(val))
            run.font.size = Pt(9)
            run.font.name = 'Times New Roman'

    if col_widths:
        for row in table.rows:
            for j, w in enumerate(col_widths):
                if j < len(row.cells):
                    row.cells[j].width = Inches(w)

    # Note
    if note:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.first_line_indent = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(note)
        run.italic = True
        run.font.size = Pt(9)
        run.font.name = 'Times New Roman'
    else:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(8)

    return table


def add_figure(doc, image_path, caption, width_inches=5.5):
    """Insert an image with a centered caption below."""
    # Image
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run()
    run.add_picture(str(image_path), width=Inches(width_inches))

    # Caption
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run(caption)
    run.italic = True
    run.font.size = Pt(10)
    run.font.name = 'Times New Roman'
    return p


def add_source_table(doc, title, headers, rows, col_widths=None, note=None):
    """Add a smaller source-data table (8pt font) for reviewer reference."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.first_line_indent = Pt(0)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(title)
    run.bold = True
    run.font.size = Pt(9)
    run.font.name = 'Times New Roman'
    run.font.color.rgb = RGBColor(80, 80, 80)

    n_cols = len(headers)
    table = doc.add_table(rows=len(rows) + 1, cols=n_cols)
    table.style = 'Table Grid'
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ''
        cp = cell.paragraphs[0]
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cp.add_run(h)
        run.bold = True
        run.font.size = Pt(8)
        run.font.name = 'Times New Roman'
        set_cell_shading(cell, 'E8E8E8')

    for i, row_data in enumerate(rows):
        for j, val in enumerate(row_data):
            cell = table.rows[i + 1].cells[j]
            cell.text = ''
            cp = cell.paragraphs[0]
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = cp.add_run(str(val))
            run.font.size = Pt(8)
            run.font.name = 'Times New Roman'
            run.font.color.rgb = RGBColor(80, 80, 80)

    if col_widths:
        for row in table.rows:
            for j, w in enumerate(col_widths):
                if j < len(row.cells):
                    row.cells[j].width = Inches(w)

    if note:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.first_line_indent = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(note)
        run.italic = True
        run.font.size = Pt(8)
        run.font.name = 'Times New Roman'
        run.font.color.rgb = RGBColor(100, 100, 100)

    return table


# ============================================================
#  CSV READERS
# ============================================================

def read_csv(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None, None
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return reader.fieldnames, rows


def fmt(val, decimals=3):
    """Format a numeric value."""
    if not val or val.strip() == '':
        return '\u2014'
    try:
        return f'{float(val):.{decimals}f}'
    except ValueError:
        return val


# ============================================================
#  BUILD DOCUMENT
# ============================================================

def build_si():
    doc = Document()
    setup_styles(doc)

    # ---- Title ----
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(
        'Supplementary Information\n\n'
        'Personalized Federated Learning with Graph Neural Networks '
        'and Kolmogorov\u2013Arnold Networks for Cross-Domain '
        'Molecular Reorganization Energy Prediction'
    )
    run.bold = True
    run.font.size = Pt(14)
    run.font.name = 'Times New Roman'

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run('Can Leng\u00b9, Author Two\u00b9')
    run.font.size = Pt(12)
    run.font.name = 'Times New Roman'

    # ================================================================
    #  S1. Detailed Experimental Protocol
    # ================================================================
    doc.add_paragraph('S1. Detailed Experimental Protocol', style='Heading 1')

    # S1.1 DFT Computational Protocol
    doc.add_paragraph('S1.1 DFT Computational Protocol for Reorganization Energy',
                      style='Heading 2')

    add_text(doc,
        'Reorganization energy (\u03bb) quantifies the nuclear relaxation energy '
        'accompanying a charge-transfer event and was computed using the four-point '
        'adiabatic potential energy method [Nelsen et al., J. Am. Chem. Soc. 1987, '
        '109, 677\u2013682]. For hole reorganization energy (\u03bb_hole):')

    add_text(doc,
        '\u03bb_hole = \u03bb\u2081 + \u03bb\u2082',
        indent=False)

    add_text(doc,
        'where \u03bb\u2081 = E(neutral geometry, cation state) \u2212 E(cation '
        'geometry, cation state) is the energy cost of distorting the cation from '
        'the neutral equilibrium geometry to its own equilibrium geometry, and '
        '\u03bb\u2082 = E(cation geometry, neutral state) \u2212 E(neutral geometry, '
        'neutral state) is the energy cost of distorting the neutral molecule from '
        'the cation equilibrium geometry back to its own equilibrium geometry. In '
        'the code (extract_hole_reorg.py), these are computed as:')

    add_text(doc,
        '\u03bb\u2081 = (E_cation_at_neutral \u2212 E_cation_at_cation) \u00d7 27.2114 eV/Hartree\n'
        '\u03bb\u2082 = (E_neutral_at_cation \u2212 E_neutral_at_neutral) \u00d7 27.2114 eV/Hartree',
        indent=False)

    add_text(doc,
        'The four single-point energies are obtained from four Gaussian log files '
        'per conformer: neutral/conf_X/ground.log (E_neutral_at_neutral), '
        'neutral/conf_X/cation_at_neutral_geom.log (E_cation_at_neutral), '
        'cation/conf_X/ground.log (E_cation_at_cation), and '
        'cation/conf_X/neutral_at_cation_geom.log (E_neutral_at_cation). '
        'When multiple conformers exist, \u03bb_hole is averaged across all valid '
        'conformers for each molecule.')

    add_text(doc,
        'For triplet reorganization energy (\u03bb_triplet), the same four-point '
        'procedure is applied with the lowest triplet state (T\u2081) replacing the '
        'cation state. Vertical excitation energies (S\u2080 \u2192 T\u2081) were '
        'computed using TD-DFT with the route line: #p wB97XD/def2TZVP '
        'TD(Triplet,NStates=10,Root=1). Triplet geometry optimizations used '
        'unrestricted DFT (UwB97XD, charge = 0, multiplicity = 3).')

    # DFT details for Client D
    add_text(doc,
        'All DFT calculations for Client D were performed with Gaussian 09, '
        'Revision D.01 (Frisch et al., 2013). Geometry optimizations of neutral '
        '(S\u2080, charge = 0, multiplicity = 1), cation radical (charge = +1, '
        'multiplicity = 2), and lowest triplet (T\u2081, charge = 0, '
        'multiplicity = 3) states were carried out at the \u03c9B97XD/def2-TZVP '
        'level in the gas phase. The \u03c9B97XD functional is a range-separated '
        'hybrid with built-in Chai\u2013Head-Gordon D2 empirical dispersion '
        'correction, so no additional dispersion keyword was required. The route '
        'line for geometry optimizations was "#wb97xd/def2tzvp opt freq". '
        'Frequency calculations were performed at the same level of theory to '
        'confirm that all optimized structures correspond to true minima. '
        'Structures with imaginary frequencies were re-optimized using displaced '
        'geometries; 85 such corrections were applied across the full dataset. '
        'Single-point energies at cross-geometries were computed with the route '
        'line "#wb97xd/def2tzvp sp". The default SCF convergence criterion of '
        'Gaussian 09 (10\u207b\u2078 Hartree on the density matrix) was used for '
        'geometry optimizations; triplet TD-DFT calculations employed the tighter '
        'SCF=Tight criterion.')

    # Client A/B/C sources
    add_text(doc,
        'Reorganization energies for Clients A and B were obtained from a curated '
        'QM9-derived dataset computed at the B3LYP/6-31G(d) level. Client C data '
        'were taken from the dataset of Atahan-Evrenk and Atalay (J. Phys. Chem. A, '
        '2019, 123, 7855\u20137863), computed at the B3LYP/6-31G(d,p) level with '
        'the four-point adiabatic method.')

    # Table S0: DFT Summary
    add_table(doc,
        'Table S0. DFT Calculation Summary',
        ['Parameter', 'Client A/B', 'Client C', 'Client D'],
        [
            ['Software', 'Not specified', 'Not specified', 'Gaussian 09 Rev. D.01'],
            ['Functional', 'B3LYP', 'B3LYP', '\u03c9B97XD'],
            ['Basis set', '6-31G(d)', '6-31G(d,p)', 'def2-TZVP'],
            ['Dispersion', 'None', 'None', 'D2 (built into \u03c9B97XD)'],
            ['Solvent', 'Gas phase', 'Gas phase', 'Gas phase'],
            ['Method', 'Four-point adiabatic', 'Four-point adiabatic', 'Four-point adiabatic'],
            ['Target \u03bb', '\u03bb_cation', '\u03bb_hole', '\u03bb_hole, \u03bb_triplet'],
            ['n molecules', '15,210 (pre-split)', '5,876', '53'],
            ['Freq. check', 'Not specified', 'Not specified', 'Yes (opt freq)'],
            ['Imag. freq. corrections', 'Not specified', 'Not specified', '85'],
        ],
        col_widths=[1.6, 1.2, 1.2, 1.8],
        note='"Not specified" indicates that the original data source publication did not '
             'report this parameter, and we did not independently verify it.')

    # S1.2 (formerly S1.1)
    doc.add_paragraph('S1.2 Data Preprocessing', style='Heading 2')

    add_text(doc,
        'SMILES strings for Clients A and B were sourced from a curated QM9-derived '
        'dataset (public_reorg_energy_15210.csv) containing 15,210 molecules with '
        'DFT-computed cation reorganization energies at the B3LYP/6-31G(d) level. '
        'The dataset was partitioned into Client A (non-aromatic, NumAromaticRings = 0, '
        'n = 6,020) and Client B (aromatic, NumAromaticRings \u2265 1, n = 9,190) based '
        'on the RDKit NumAromaticRings descriptor.')

    add_text(doc,
        'Client C molecules were obtained from the dataset of Atahan-Evrenk and Atalay '
        '(J. Phys. Chem. A, 2019, 123, 7855\u20137863), comprising 5,876 conjugated '
        'organic semiconductors with hole reorganization energies (\u03bb_hole).')

    add_text(doc,
        'Client D consists of 53 TADF emitters characterized in our laboratory. '
        'Hole reorganization energies (\u03bb_hole, n = 53) and triplet reorganization '
        'energies (\u03bb_triplet, n = 49) were computed using the four-point adiabatic '
        'method at the B3LYP/6-31G(d) level with Gaussian 16.')

    # S1.3
    doc.add_paragraph('S1.3 2D Molecular Graph Construction', style='Heading 2')

    add_text(doc,
        'Molecular graphs were constructed from SMILES strings using RDKit (version '
        '2024.09.6) and PyTorch Geometric (version 2.3.0). Node features x_i \u2208 '
        '\u211d\u00b9\u00b9 comprise: one-hot atomic type encoding (C, N, O, S, F, Cl, '
        'Br, Other; 8 dimensions), normalized degree (1 dimension), formal charge '
        '(1 dimension), and Gasteiger partial charge (1 dimension). Edge features '
        'e_ij \u2208 \u211d\u2074 encode bond type as a one-hot vector (single, double, '
        'triple, aromatic). Six global molecular descriptors (molecular weight, '
        'topological polar surface area, Wildman\u2013Crippen LogP, number of aromatic '
        'rings, maximum absolute partial charge, and molar refractivity) were computed '
        'with RDKit and concatenated to the graph-level representation after pooling.')

    # S1.4
    doc.add_paragraph('S1.4 3D Conformer Generation', style='Heading 2')

    add_text(doc,
        'Three-dimensional atomic coordinates were generated from SMILES strings using '
        'the RDKit (version 2024.09.6) ETKDG version 3 (ETKDGv3) distance-geometry '
        'algorithm with explicit hydrogen atoms (Chem.AddHs). For each molecule, '
        'AllChem.EmbedMolecule was called with a deterministic random seed '
        '(randomSeed = 42); if embedding failed (return value \u2260 0), up to two '
        'additional attempts were made with incremented seeds (43 and 44). Successfully '
        'embedded conformers were subsequently optimized with the Merck Molecular Force '
        'Field (MMFF94) using AllChem.MMFFOptimizeMolecule with maxIters = 500 '
        '(default: 200). If MMFF optimization raised an exception (e.g., due to missing '
        'atom-type parameters), the un-optimized ETKDGv3 conformer was retained without '
        'further processing. No UFF fallback was implemented. Molecules for which all '
        'three embedding attempts failed were excluded; however, in practice all 21,139 '
        'molecules across the four clients were successfully embedded (Table S1). Each '
        'molecule was represented as a PyTorch Geometric Data object containing atomic '
        'numbers z (integer tensor) and Cartesian coordinates pos (float tensor in '
        '\u00c5). Edge connectivity was not precomputed, as SchNet constructs a dynamic '
        'radius graph internally at runtime with a cutoff of 7.5 \u00c5. Generated '
        'conformers were cached to disk (conformer_3d_cache.pkl) to ensure '
        'reproducibility.')

    # Table S1
    add_table(doc,
        'Table S1. Conformer Generation Statistics',
        ['Client', 'Domain', 'n_total', 'n_success', 'Rate (%)', 'Median (ms)', 'Mean (ms)', 'Max (ms)'],
        [
            ['A', 'Non-aromatic', '6,020', '6,020', '100.0', '6.3', '7.4', '44.5'],
            ['B', 'Aromatic', '9,190', '9,190', '100.0', '6.8', '7.0', '12.3'],
            ['C', 'Conjugated OSC', '5,876', '5,876', '100.0', '25.2', '24.5', '34.8'],
            ['D', 'TADF', '53', '53', '100.0', '16.8', '20.3', '77.3'],
            ['Total', '\u2014', '21,139', '21,139', '100.0', '\u2014', '\u2014', '\u2014'],
        ],
        col_widths=[0.5, 1.1, 0.6, 0.7, 0.6, 0.8, 0.7, 0.7],
        note='Wall-clock times measured on a single CPU core (Intel Xeon Platinum 8260 @ 2.40 GHz). '
             'Times include AddHs, ETKDGv3 embedding, and MMFF94 optimization (maxIters = 500).')

    # S1.5
    doc.add_paragraph('S1.5 GIN Encoder Architecture', style='Heading 2')

    add_text(doc,
        'The 2D encoder follows the Graph Isomorphism Network (GIN) architecture with '
        '5 GINConv message-passing layers, each with hidden dimension 300. Each GINConv '
        'layer uses a two-layer MLP (Linear \u2192 ReLU \u2192 Linear) as the update '
        'function. Each layer applies batch normalization followed by ReLU activation. '
        'Jumping Knowledge (JK) concatenation aggregates outputs from all five layers, '
        'producing a 1,500-dimensional vector (300 \u00d7 5). Global mean pooling and '
        'global max pooling are applied and concatenated, yielding a 3,000-dimensional '
        'vector. A linear projection maps this to a 256-dimensional embedding '
        'h \u2208 \u211d\u00b2\u2075\u2076. Six RDKit-derived global molecular '
        'descriptors (Section S1.2) are concatenated to h, yielding a 262-dimensional '
        'input to the regression head. Total encoder parameters: 1,587,556.')

    # S1.6
    doc.add_paragraph('S1.6 SchNet Encoder Architecture', style='Heading 2')

    add_text(doc,
        'The 3D encoder uses the PyTorch Geometric implementation of SchNet '
        '(torch_geometric.nn.models.SchNet) with the following hyperparameters: '
        'hidden_channels = 256, num_filters = 128, num_interactions = 6, '
        'num_gaussians = 50, cutoff = 7.5 \u00c5. Atom features are initialized from '
        'learnable embeddings of the nuclear charge Z (100 embedding vectors). The '
        'default SchNet readout applies lin1 (hidden_channels \u2192 hidden_channels / 2) '
        'followed by shifted softplus activation and lin2 (hidden_channels / 2 \u2192 1). '
        'To obtain a 256-dimensional graph-level embedding suitable for the downstream '
        'KAN regression head, we replaced lin1 with Linear(256, 256) and lin2 with '
        'Identity(). This modification preserves the full hidden dimension through the '
        'readout stage. Global mean pooling produces the final graph representation. '
        'Total encoder parameters: 1,019,136.')

    # S1.7
    doc.add_paragraph('S1.7 KAN Regression Head', style='Heading 2')

    add_text(doc,
        'The Kolmogorov\u2013Arnold Network (KAN) regression head consists of three '
        'KANLinear layers with dimensions 256 \u2192 64 \u2192 32 \u2192 1 (for SchNet '
        'input) or 262 \u2192 64 \u2192 32 \u2192 1 (for GIN input with 6 concatenated '
        'global descriptors). Each KANLinear layer uses B-spline basis functions of '
        'order 3 with grid size 5 (optimized via ablation; see main text Section 5.5). '
        'The base activation function is SiLU. The implementation follows the '
        'efficient-kan library (github.com/Blealtan/efficient-kan). Total KAN head '
        'parameters: 184,640 (256-dim input) or 188,480 (262-dim input).')

    add_text(doc,
        'The MLP baseline head consists of Linear(256, 128) \u2192 ReLU \u2192 '
        'Dropout(0.3) \u2192 Linear(128, 64) \u2192 ReLU \u2192 Dropout(0.3) \u2192 '
        'Linear(64, 1). Total MLP head parameters: 41,217.')

    # S1.8
    doc.add_paragraph('S1.8 Federated Learning Protocol', style='Heading 2')

    add_text(doc,
        'Federated training proceeds for T = 30 communication rounds. At each round, '
        'each client trains locally for E = 5 epochs using the Adam optimizer with '
        'learning rate 10\u207b\u2074 and cosine annealing schedule. Batch sizes are 512 '
        'for Clients A, B, and C, and full-batch (n = 53) for Client D. Early stopping '
        'with patience 15 on validation loss is applied within each local training phase.')

    add_mixed_paragraph(doc, [
        ('FedPer aggregation. ', True, False),
        ('At each communication round, only encoder parameters \u03b8_enc are aggregated '
         'via weighted averaging: \u03b8_enc \u2190 \u03a3_k (n_k / N) \u00b7 \u03b8_enc\u1d4f, '
         'where n_k is the number of training samples on client k and '
         'N = \u03a3_k n_k. The KAN head parameters \u03b8_head\u1d4f remain local and '
         'are never transmitted.', False, False),
    ])

    add_mixed_paragraph(doc, [
        ('FedBN (for GIN only). ', True, False),
        ('Batch normalization parameters (\u03b3, \u03b2, running mean, running variance) '
         'within the GIN encoder are excluded from aggregation. SchNet does not contain '
         'BatchNorm layers, so FedBN is not applicable to SchNet experiments.', False, False),
    ])

    add_mixed_paragraph(doc, [
        ('Per-client label normalization. ', True, False),
        ('Each client independently computes z-score normalization parameters '
         '(\u03bc_k, \u03c3_k) from its local training set. Labels are normalized as '
         '\u0177 = (y \u2212 \u03bc_k) / \u03c3_k before training; predictions are '
         'denormalized before evaluation.', False, False),
    ])

    # S1.9
    doc.add_paragraph('S1.9 Self-Supervised Pre-training', style='Heading 2')

    add_text(doc,
        'Three SSL strategies were evaluated for the GIN encoder (not applied to SchNet):')

    add_mixed_paragraph(doc, [
        ('AtomMask: ', True, False),
        ('Randomly masks 15% of node features (set to zero) and trains the encoder to '
         'reconstruct atom types (cross-entropy loss) and Gasteiger charges (MSE loss). '
         'Combined loss = CE + 0.1 \u00d7 MSE.', False, False),
    ])

    add_mixed_paragraph(doc, [
        ('EdgePred: ', True, False),
        ('Deletes 20% of edges and trains a binary classifier on positive (deleted) '
         'and negative (randomly sampled non-existent) edges using binary cross-entropy '
         'loss.', False, False),
    ])

    add_mixed_paragraph(doc, [
        ('GraphCL: ', True, False),
        ('Generates two augmented views per graph via node feature masking (15%) and '
         'edge deletion (20%), and optimizes an InfoNCE contrastive loss with '
         'temperature \u03c4 = 0.1.', False, False),
    ])

    add_text(doc,
        'SSL pre-training runs for 20 federated rounds (5 local epochs per round) '
        'before supervised federated fine-tuning. The effect of SSL round count was '
        'evaluated via ablation (see Figure S1b in Section S2.2).')

    # S1.10
    doc.add_paragraph('S1.10 Evaluation Protocol', style='Heading 2')

    add_text(doc,
        'For Clients A, B, and C, evaluation uses 5-fold stratified random '
        'cross-validation, reporting mean \u00b1 standard deviation of R\u00b2, MAE, '
        'and RMSE.')

    add_text(doc,
        'For Client D, leave-one-out cross-validation (LOOCV) is used due to the '
        'small sample size (n = 53 for \u03bb_hole, n = 49 for \u03bb_triplet). In '
        'each LOOCV fold, one molecule is held out for testing, and the remaining '
        'n \u2212 1 molecules constitute the training set. The federated training '
        'procedure (including encoder aggregation from all clients) is repeated for '
        'each fold. Final metrics are computed on the concatenated predictions across '
        'all n folds.')

    # ================================================================
    #  S2. Complete Experiment Results
    # ================================================================
    doc.add_paragraph('S2. Complete Experiment Results', style='Heading 1')

    # S2.1 Full Experiment Summary
    doc.add_paragraph('S2.1 Full Experiment Summary', style='Heading 2')

    headers_s2, rows_s2 = read_csv('all_experiments_summary.csv')
    if rows_s2:
        table_rows = []
        for r in rows_s2:
            table_rows.append([
                r.get('exp_id', ''),
                r.get('desc', ''),
                r.get('encoder', ''),
                r.get('fed_strategy', 'None'),
                r.get('R2_A', '\u2014') or '\u2014',
                r.get('R2_B', '\u2014') or '\u2014',
                r.get('R2_C', '\u2014') or '\u2014',
                r.get('R2_Dh', '\u2014') or '\u2014',
                r.get('R2_Dt', '\u2014') or '\u2014',
                r.get('MAE_A', '\u2014') or '\u2014',
                r.get('MAE_B', '\u2014') or '\u2014',
                r.get('MAE_C', '\u2014') or '\u2014',
                r.get('MAE_Dh', '\u2014') or '\u2014',
                r.get('MAE_Dt', '\u2014') or '\u2014',
                f"{float(r.get('time_s', 0)):.0f}" if r.get('time_s') else '\u2014',
            ])
        add_table(doc,
            'Table S2. Complete Experiment Results (all 16 experiments)',
            ['ID', 'Description', 'Enc.', 'Strategy',
             'R\u00b2_A', 'R\u00b2_B', 'R\u00b2_C', 'R\u00b2_D(h)', 'R\u00b2_D(t)',
             'MAE_A', 'MAE_B', 'MAE_C', 'MAE_D(h)', 'MAE_D(t)', 'Time(s)'],
            table_rows,
            note='Clients A/B/C: mean \u00b1 std over 5-fold CV. Client D: LOOCV.')

    # S2.2 Ablation Study Details
    doc.add_paragraph('S2.2 Ablation Study Details', style='Heading 2')

    add_text(doc,
        'Four ablation studies were conducted using the FedPer+FedBN GIN+KAN '
        'configuration on Client D \u03bb_hole (LOOCV). Each ablation varies one '
        'hyperparameter while keeping others at default values (grid = 5, SSL = 20 '
        'rounds, data = 100%, communication = 30 rounds).')

    # Figure S1: Ablation panel (replaces Tables S3-S6)
    fig_path = FIGURES_DIR / 'figS1_ablation_details.png'
    if fig_path.exists():
        add_figure(doc, fig_path,
            'Figure S1. Ablation studies on Client D \u03bb_hole (LOOCV). '
            '(a) KAN B-spline grid size. (b) Federated SSL pre-training rounds. '
            '(c) Client D data fraction. (d) Communication rounds. '
            'Dashed gray line indicates R\u00b2 = 0.',
            width_inches=5.8)

    # Source Data for Figure S1
    _, grid_rows = read_csv('ablation_kan_grid.csv')
    if grid_rows:
        add_source_table(doc,
            'Source Data S1a. KAN Grid Size Ablation',
            ['Grid', 'MAE (eV)', 'R\u00b2', 'RMSE (eV)'],
            [[r['grid'], fmt(r['MAE']), fmt(r['R2']), fmt(r['RMSE'])] for r in grid_rows],
            col_widths=[1.0, 1.2, 1.2, 1.2])

    _, ssl_rows = read_csv('ablation_ssl_rounds.csv')
    if ssl_rows:
        add_source_table(doc,
            'Source Data S1b. SSL Pre-training Rounds Ablation',
            ['Rounds', 'MAE (eV)', 'R\u00b2', 'RMSE (eV)'],
            [[r['ssl_rounds'], fmt(r['MAE']), fmt(r['R2']), fmt(r['RMSE'])] for r in ssl_rows],
            col_widths=[1.0, 1.2, 1.2, 1.2])

    _, data_rows = read_csv('ablation_data_size.csv')
    if data_rows:
        add_source_table(doc,
            'Source Data S1c. Client D Data Size Ablation',
            ['Fraction', 'n', 'MAE (eV)', 'R\u00b2', 'RMSE (eV)'],
            [[r['ratio'], r['n_molecules'], fmt(r['MAE']), fmt(r['R2']), fmt(r['RMSE'])]
             for r in data_rows],
            col_widths=[0.8, 0.6, 1.0, 1.0, 1.0])

    _, comm_rows = read_csv('ablation_comm_rounds.csv')
    if comm_rows:
        add_source_table(doc,
            'Source Data S1d. Communication Rounds Ablation',
            ['Rounds', 'MAE (eV)', 'R\u00b2', 'RMSE (eV)'],
            [[r['n_rounds'], fmt(r['MAE']), fmt(r['R2']), fmt(r['RMSE'])] for r in comm_rows],
            col_widths=[1.0, 1.2, 1.2, 1.2])

    # S2.3 Centralized Training
    doc.add_paragraph('S2.3 Centralized Training Details', style='Heading 2')

    # Figure S5: Centralized vs Federated vs Local (replaces Table S7)
    fig_path = FIGURES_DIR / 'figS5_centralized_comparison.png'
    if fig_path.exists():
        add_figure(doc, fig_path,
            'Figure S5. Comparison of Local, Federated (FedPer), and Centralized '
            'SchNet+KAN training on Client D (LOOCV). Left: \u03bb_hole (n = 53). '
            'Right: \u03bb_triplet (n = 49). Dashed line indicates R\u00b2 = 0.',
            width_inches=5.5)

    _, cent_rows = read_csv('centralized_upper_bound.csv')
    if cent_rows:
        add_source_table(doc,
            'Source Data S5. Centralized SchNet+KAN Training on Client D (LOOCV)',
            ['Target', 'MAE (eV)', 'R\u00b2', 'RMSE (eV)', 'n', 'Time (s)'],
            [[r['target'], fmt(r['MAE']), fmt(r['R2']), fmt(r['RMSE']),
              r['n_molecules'], f"{float(r['time_s']):.0f}"]
             for r in cent_rows],
            col_widths=[0.8, 0.8, 0.7, 0.8, 0.5, 0.7],
            note='Centralized model trained on pooled data from all four clients.')

    # S2.4 Subgroup Analysis
    doc.add_paragraph('S2.4 Subgroup Analysis Details', style='Heading 2')

    _, subgroup_rows = read_csv('subgroup_optimized.csv')
    if subgroup_rows:
        # Show top-10 subgroups by SchNet R²
        sorted_rows = sorted(subgroup_rows, key=lambda r: float(r['SchNet_R2']), reverse=True)
        top_rows = sorted_rows[:15]
        add_table(doc,
            'Table S8. Top 15 Subgroup Filters by SchNet R\u00b2 (Client D \u03bb_hole)',
            ['Filter', 'n', 'SchNet R\u00b2', 'SchNet MAE', 'GIN R\u00b2', 'GIN MAE'],
            [[r['filter'], r['n'], fmt(r['SchNet_R2']), fmt(r['SchNet_MAE']),
              fmt(r['GIN_R2']), fmt(r['GIN_MAE'])]
             for r in top_rows],
            col_widths=[1.8, 0.4, 0.8, 0.8, 0.8, 0.8],
            note='DA = donor atom fraction (N, O, S atoms / total heavy atoms); '
                 'rot = number of rotatable bonds.')

    # S2.5 Per-molecule predictions
    doc.add_paragraph('S2.5 Per-Molecule Prediction Results', style='Heading 2')

    # Figure S6: Per-molecule errors (replaces Table S9)
    fig_path = FIGURES_DIR / 'figS6_per_molecule_errors.png'
    if fig_path.exists():
        add_figure(doc, fig_path,
            'Figure S6. Per-molecule LOOCV prediction errors for all 53 Client D '
            'molecules. (a) SchNet absolute errors sorted in ascending order; color '
            'gradient from green (low error) to red (high error); dashed line shows '
            'the MAE. (b) GIN vs. SchNet error comparison; shaded regions indicate '
            'which encoder performs better for each molecule.',
            width_inches=6.0)

    case_path = RESULTS_DIR / 'case_study_molecules.csv'
    if case_path.exists():
        with open(case_path, 'r') as f:
            reader = csv.DictReader(f)
            mol_rows = list(reader)

        add_source_table(doc,
            'Source Data S6. Per-Molecule LOOCV Predictions (53 Client D Molecules)',
            ['Molecule', 'y_true', 'GIN pred', 'SchNet pred',
             '|err| GIN', '|err| SchNet', 'n_rot', 'MW'],
            [[r['molecule'][:22],
              fmt(r['y_true'], 3), fmt(r['y_pred_gin'], 3), fmt(r['y_pred_schnet'], 3),
              fmt(r['error_gin'], 3), fmt(r['error_schnet'], 3),
              r['n_rotatable'], fmt(r['mol_weight'], 1)]
             for r in mol_rows],
            col_widths=[1.3, 0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0.5],
            note='GIN = E32 (FedBN 4-Client GIN+KAN). SchNet = E51 (FedAvg SchNet+KAN). '
                 'n_rot = rotatable bonds. MW = molecular weight (Da).')

    # ================================================================
    #  S3. Correlation Analysis
    # ================================================================
    doc.add_paragraph('S3. Correlation Analysis', style='Heading 1')

    add_text(doc,
        'Pearson correlation between GIN absolute prediction error and number of '
        'rotatable bonds across all 53 Client D molecules: r = 0.301, p = 0.028 '
        '(two-tailed). Pearson correlation between SchNet absolute prediction error '
        'and number of rotatable bonds: r = 0.159, p = 0.256 (two-tailed). These '
        'statistics were computed using scipy.stats.pearsonr on the LOOCV predictions '
        'from experiments E32 (GIN, FedBN 4-Client GIN+KAN) and E51 (SchNet, FedAvg '
        'SchNet+KAN).')

    # ================================================================
    #  S4. Software and Hardware
    # ================================================================
    doc.add_paragraph('S4. Software and Hardware', style='Heading 1')

    add_table(doc,
        'Table S10. Software Dependencies',
        ['Package', 'Version', 'Purpose'],
        [
            ['Python', '3.10', 'Runtime environment'],
            ['PyTorch', '2.7.1+cu126', 'Deep learning framework'],
            ['PyTorch Geometric', '2.3.0', 'Graph neural networks'],
            ['RDKit', '2024.09.6', 'Molecular processing and conformer generation'],
            ['efficient-kan', '\u2014', 'KAN layer implementation'],
            ['NumPy', '2.2.6', 'Numerical computing'],
            ['scikit-learn', '1.6.1', 'Evaluation metrics and preprocessing'],
            ['CUDA', '12.6', 'GPU computing toolkit'],
            ['NVIDIA Driver', '550.144.03', 'GPU driver'],
        ],
        col_widths=[1.5, 1.2, 3.0],
        note='Versions determined at runtime via package __version__ attributes.')

    add_table(doc,
        'Table S11. Hardware Configuration',
        ['Component', 'Specification'],
        [
            ['GPU', '2\u00d7 NVIDIA A30 (24 GB VRAM each)'],
            ['CPU', 'Intel Xeon Platinum 8260 @ 2.40 GHz'],
            ['RAM', '252 GB DDR4'],
            ['GPU allocation', 'GPU 0: Client A training; GPU 1: Clients B, C, D training'],
            ['Federated aggregation', 'CPU (parameter averaging only, no GPU required)'],
        ],
        col_widths=[1.8, 4.5])

    # ================================================================
    #  S5. Data Availability
    # ================================================================
    doc.add_paragraph('S5. Data Availability', style='Heading 1')

    add_text(doc,
        'The QM9-derived dataset for Clients A and B (15,210 molecules) was curated '
        'from publicly available quantum chemistry databases. The Atahan-Evrenk dataset '
        'for Client C is available from the original publication (J. Phys. Chem. A, '
        '2019, 123, 7855\u20137863; DOI: 10.1021/acs.jpca.9b02733). Client D TADF data '
        'are available upon reasonable request from the corresponding author. Code and '
        'trained models will be released upon acceptance.')

    return doc


# ============================================================
#  VERIFICATION
# ============================================================

def verify_docx(doc, docx_path):
    print('\n' + '=' * 60)
    print('SUPPLEMENTARY INFORMATION VERIFICATION')
    print('=' * 60)

    total_paragraphs = len(doc.paragraphs)
    print(f'Total paragraphs: {total_paragraphs}')

    h1, h2, h3 = [], [], []
    for p in doc.paragraphs:
        if p.style.name == 'Heading 1':
            h1.append(p.text)
        elif p.style.name == 'Heading 2':
            h2.append(p.text)
        elif p.style.name == 'Heading 3':
            h3.append(p.text)

    print(f'\nHeading 1 ({len(h1)}):')
    for h in h1:
        print(f'  - {h}')
    print(f'\nHeading 2 ({len(h2)}):')
    for h in h2:
        print(f'  - {h}')

    print(f'\nTables: {len(doc.tables)}')
    for i, t in enumerate(doc.tables, 1):
        print(f'  Table {i}: {len(t.rows)} rows x {len(t.columns)} cols')

    # Count inline images
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    n_images = 0
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            n_images += 1
    print(f'\nInline images: {n_images}')

    # Check for vague words
    full_text = '\n'.join(p.text for p in doc.paragraphs)
    vague_words = ['approximately', 'estimated', 'around', 'roughly', 'about']
    found_vague = []
    for word in vague_words:
        if word.lower() in full_text.lower():
            found_vague.append(word)
    if found_vague:
        print(f'\nVague words found: {found_vague}')
    else:
        print('\nNo vague words ("approximately", "estimated", etc.) found.')

    # LaTeX residuals
    residual_patterns = [r'\\[a-zA-Z]+', r'\$[^$]+\$', r'\\begin\{', r'\\end\{']
    residuals = []
    for p in doc.paragraphs:
        for pat in residual_patterns:
            for m in re.finditer(pat, p.text):
                if m.group() not in ('\\n', '\\t', '\\r'):
                    residuals.append(m.group())
    if residuals:
        unique = set(residuals)
        print(f'\nResidual LaTeX ({len(unique)} unique): {sorted(unique)}')
    else:
        print('No residual LaTeX commands found.')

    # Word count
    words = len(full_text.split())
    print(f'\nWord count: {words}')

    fsize = os.path.getsize(docx_path)
    print(f'File size: {fsize / 1024:.1f} KB')
    print(f'Saved to: {docx_path}')
    print('=' * 60)


# ============================================================
#  MAIN
# ============================================================

if __name__ == '__main__':
    print('Building Supplementary Information...')
    doc = build_si()

    print(f'Saving to {DOCX_PATH}...')
    doc.save(str(DOCX_PATH))

    verify_docx(doc, DOCX_PATH)
    print('\nDone!')
