"""
Generate Fig1 (Graphical Abstract) and Fig3 (FedPer Workflow).
Nature Machine Intelligence / Patterns style methodology figures.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

OUTDIR = 'results/figures'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
})

# Client colors
C_A = '#1565C0'   # blue
C_B = '#2E7D32'   # green
C_C = '#E65100'   # orange
C_D = '#C62828'   # red
C_A_LIGHT = '#BBDEFB'
C_B_LIGHT = '#C8E6C9'
C_C_LIGHT = '#FFE0B2'
C_D_LIGHT = '#FFCDD2'

C_SHARED = '#E3F2FD'   # light blue for shared encoder
C_PRIVATE = '#FFEBEE'  # light red for private heads
C_SERVER = '#F5F5F5'   # light gray for server
C_ARROW = '#616161'    # gray arrows


# ============================================================
#  FIG 1: GRAPHICAL ABSTRACT
# ============================================================

def draw_fig1():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # ---- TOP LAYER: 4 Client boxes ----
    client_info = [
        ('Client A', 'Non-aromatic', 'n = 6,020', 'λ_cation', C_A, C_A_LIGHT, 0.15),
        ('Client B', 'Aromatic', 'n = 9,190', 'λ_cation', C_B, C_B_LIGHT, 1.90),
        ('Client C', 'Conjugated OSC', 'n = 5,876', 'λ_hole', C_C, C_C_LIGHT, 3.65),
        ('Client D', 'TADF (private)', 'n = 53', 'λ_hole', C_D, C_D_LIGHT, 5.40),
    ]

    top_y = 3.85
    box_w, box_h = 1.50, 1.00

    for name, desc, n_str, lam, color, bg, x in client_info:
        # Rounded rectangle
        box = FancyBboxPatch((x, top_y), box_w, box_h,
                             boxstyle='round,pad=0.05', facecolor=bg,
                             edgecolor=color, linewidth=1.5)
        ax.add_patch(box)
        # Client name
        ax.text(x + box_w/2, top_y + box_h - 0.15, name,
                ha='center', va='top', fontsize=8, fontweight='bold', color=color)
        # Description
        ax.text(x + box_w/2, top_y + box_h/2 + 0.02, desc,
                ha='center', va='center', fontsize=6.5, color='#333333')
        ax.text(x + box_w/2, top_y + box_h/2 - 0.16, n_str,
                ha='center', va='center', fontsize=6, color='#555555')
        ax.text(x + box_w/2, top_y + 0.12, lam,
                ha='center', va='center', fontsize=6.5, color=color, fontstyle='italic')

        # Lock icon (ASCII-safe)
        if name == 'Client D':
            # Closed lock: filled circle
            ax.plot(x + box_w - 0.14, top_y + 0.13, 's',
                    color=color, markersize=5, markeredgecolor=color)
            ax.text(x + box_w - 0.14, top_y + 0.13, 'P',
                    ha='center', va='center', fontsize=4, color='white', fontweight='bold')
        else:
            # Open lock: open circle
            ax.plot(x + box_w - 0.14, top_y + 0.13, 's',
                    color='white', markersize=5, markeredgecolor=color, markeredgewidth=0.8)
            ax.text(x + box_w - 0.14, top_y + 0.13, 'O',
                    ha='center', va='center', fontsize=4, color=color, fontweight='bold')

    # ---- MIDDLE LAYER: Federated Server ----
    srv_x, srv_y = 0.30, 2.05
    srv_w, srv_h = 6.40, 1.45

    # Server background
    srv_box = FancyBboxPatch((srv_x, srv_y), srv_w, srv_h,
                             boxstyle='round,pad=0.08', facecolor=C_SERVER,
                             edgecolor='#9E9E9E', linewidth=1.2)
    ax.add_patch(srv_box)
    ax.text(srv_x + srv_w/2, srv_y + srv_h - 0.12, 'Federated Server',
            ha='center', va='top', fontsize=9, fontweight='bold', color='#424242')

    # Left half: Shared Encoder
    enc_x, enc_y = 0.50, 2.15
    enc_w, enc_h = 2.80, 1.05
    enc_box = FancyBboxPatch((enc_x, enc_y), enc_w, enc_h,
                             boxstyle='round,pad=0.05', facecolor=C_SHARED,
                             edgecolor='#1565C0', linewidth=1.0, linestyle='--')
    ax.add_patch(enc_box)
    ax.text(enc_x + enc_w/2, enc_y + enc_h - 0.12, 'Shared Encoder (θ_enc)',
            ha='center', va='top', fontsize=7.5, fontweight='bold', color='#1565C0')

    # Two encoder options
    for i, (label, lx) in enumerate([('2D GIN', enc_x + 0.55), ('3D SchNet', enc_x + 1.85)]):
        opt_box = FancyBboxPatch((lx, enc_y + 0.12), 1.0, 0.55,
                                 boxstyle='round,pad=0.04', facecolor='white',
                                 edgecolor='#42A5F5', linewidth=0.8)
        ax.add_patch(opt_box)
        ax.text(lx + 0.50, enc_y + 0.40, label,
                ha='center', va='center', fontsize=7, color='#1565C0', fontweight='bold')

    # Right half: Private Heads
    head_x, head_y = 3.60, 2.15
    head_w, head_h = 2.90, 1.05
    head_box = FancyBboxPatch((head_x, head_y), head_w, head_h,
                              boxstyle='round,pad=0.05', facecolor=C_PRIVATE,
                              edgecolor='#C62828', linewidth=1.0, linestyle='--')
    ax.add_patch(head_box)
    ax.text(head_x + head_w/2, head_y + head_h - 0.12, 'Private KAN Heads (θ_head)',
            ha='center', va='top', fontsize=7.5, fontweight='bold', color='#C62828')

    # 4 KAN icons
    kan_colors = [C_A, C_B, C_C, C_D]
    kan_labels = ['K_A', 'K_B', 'K_C', 'K_D']
    for i, (kc, kl) in enumerate(zip(kan_colors, kan_labels)):
        kx = head_x + 0.25 + i * 0.70
        ky = enc_y + 0.18
        # Small box for each KAN
        kan_box = FancyBboxPatch((kx, ky), 0.55, 0.48,
                                 boxstyle='round,pad=0.03', facecolor='white',
                                 edgecolor=kc, linewidth=0.8)
        ax.add_patch(kan_box)
        # Wavy line (B-spline visualization)
        wx = np.linspace(kx + 0.06, kx + 0.49, 30)
        wy = ky + 0.30 + 0.08 * np.sin(4 * np.pi * (wx - kx) / 0.55)
        ax.plot(wx, wy, color=kc, linewidth=1.2)
        ax.text(kx + 0.275, ky + 0.07, kl,
                ha='center', va='center', fontsize=5.5, color=kc, fontweight='bold')

    # ---- ARROWS: Clients ↔ Server ----
    client_centers = [x + box_w/2 for _, _, _, _, _, _, x in client_info]

    for i, cx in enumerate(client_centers):
        # Upload arrow (client → server)
        ax.annotate('', xy=(cx - 0.12, srv_y + srv_h + 0.02),
                    xytext=(cx - 0.12, top_y - 0.02),
                    arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=0.8,
                                    connectionstyle='arc3,rad=0'))
        # Download arrow (server → client)
        ax.annotate('', xy=(cx + 0.12, top_y - 0.02),
                    xytext=(cx + 0.12, srv_y + srv_h + 0.02),
                    arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=0.8,
                                    connectionstyle='arc3,rad=0'))

    # Arrow labels: horizontal text near left pair (A/B) and right pair (C/D)
    mid_y = (top_y + srv_y + srv_h) / 2 + 0.05
    # Between A and B arrows
    ax.text((client_centers[0] + client_centers[1]) / 2, mid_y,
            'θ_enc  ↑↓', ha='center', va='center', fontsize=8, color=C_ARROW)
    # Between C and D arrows
    ax.text((client_centers[2] + client_centers[3]) / 2, mid_y,
            'θ_enc  ↑↓', ha='center', va='center', fontsize=8, color=C_ARROW)

    # ---- BOTTOM LAYER: Results & Diagnosis ----
    bot_y = 0.10

    # Left: mini bar chart
    res_x, res_y = 0.30, bot_y
    res_w, res_h = 2.80, 1.70
    res_box = FancyBboxPatch((res_x, res_y), res_w, res_h,
                             boxstyle='round,pad=0.06', facecolor='#FAFAFA',
                             edgecolor='#BDBDBD', linewidth=0.8)
    ax.add_patch(res_box)
    ax.text(res_x + res_w/2, res_y + res_h - 0.10, 'Key Results (R²)',
            ha='center', va='top', fontsize=7.5, fontweight='bold', color='#424242')

    # Bar chart (simplified)
    bar_labels = ['A', 'B', 'C', 'D\nlocal', 'D\nfed.', 'D\nrigid']
    bar_values = [0.75, 0.85, 0.85, -0.68, 0.18, 0.85]
    bar_colors = [C_A, C_B, C_C, '#FFCDD2', C_D, '#DAA520']
    bar_edge = [C_A, C_B, C_C, C_D, C_D, '#B8860B']

    bar_x_start = res_x + 0.25
    bar_spacing = 0.38
    bar_width = 0.28
    baseline_y = res_y + 0.55  # R²=0 line

    bar_display_vals = ['0.75', '0.85', '0.88', '-0.68', '0.18', '0.85']
    for i, (val, col, ec) in enumerate(zip(bar_values, bar_colors, bar_edge)):
        bx = bar_x_start + i * bar_spacing
        bar_h_px = val * 0.85  # scale
        if val >= 0:
            rect = Rectangle((bx, baseline_y), bar_width, bar_h_px,
                              facecolor=col, edgecolor=ec, linewidth=0.6)
            val_y = baseline_y + bar_h_px + 0.04
        else:
            rect = Rectangle((bx, baseline_y + bar_h_px), bar_width, -bar_h_px,
                              facecolor=col, edgecolor=ec, linewidth=0.6)
            val_y = baseline_y + bar_h_px - 0.06
        ax.add_patch(rect)
        ax.text(bx + bar_width/2, res_y + 0.22, bar_labels[i],
                ha='center', va='center', fontsize=4.5, color='#333333')
        # R² value annotation above/below bar
        ax.text(bx + bar_width/2, val_y, bar_display_vals[i],
                ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=5, color=ec, fontweight='bold')

    # Baseline
    ax.plot([bar_x_start - 0.05, bar_x_start + 6 * bar_spacing],
            [baseline_y, baseline_y], 'k-', lw=0.5)
    ax.text(bar_x_start - 0.12, baseline_y, '0', fontsize=5, ha='right', va='center')

    # Right: Physical Diagnosis
    diag_x, diag_y = 3.60, bot_y
    diag_w, diag_h = 2.90, 1.70
    diag_box = FancyBboxPatch((diag_x, diag_y), diag_w, diag_h,
                              boxstyle='round,pad=0.06', facecolor='#FAFAFA',
                              edgecolor='#BDBDBD', linewidth=0.8)
    ax.add_patch(diag_box)
    ax.text(diag_x + diag_w/2, diag_y + diag_h - 0.10, 'Physical Diagnosis',
            ha='center', va='top', fontsize=7.5, fontweight='bold', color='#424242')

    diag_items = [
        ('x', '2D topology: structural blindness', C_D),
        ('+', 'Federation: cross-domain knowledge', C_A),
        ('+', '3D encoding: torsional resolution', C_B),
        ('+', 'Rigid D\u2013A subset: R\u00b2 = 0.852', '#B8860B'),
    ]
    for i, (icon, text, color) in enumerate(diag_items):
        dy = diag_y + diag_h - 0.42 - i * 0.30
        # Colored circle background
        icon_circle = plt.Circle((diag_x + 0.25, dy), 0.10,
                                  facecolor=color, alpha=0.15,
                                  edgecolor=color, linewidth=0.8)
        ax.add_patch(icon_circle)
        # Draw x as two crossing lines, + as cross lines
        r = 0.055
        cx_, cy_ = diag_x + 0.25, dy
        if icon == 'x':
            ax.plot([cx_-r, cx_+r], [cy_-r, cy_+r], color=color, lw=1.8, solid_capstyle='round')
            ax.plot([cx_-r, cx_+r], [cy_+r, cy_-r], color=color, lw=1.8, solid_capstyle='round')
        else:
            ax.plot([cx_-r, cx_+r], [cy_, cy_], color=color, lw=1.8, solid_capstyle='round')
            ax.plot([cx_, cx_], [cy_-r, cy_+r], color=color, lw=1.8, solid_capstyle='round')
        ax.text(diag_x + 0.45, dy, text, ha='left', va='center',
                fontsize=6.5, color='#333333')

    # ---- Arrows: Server → Bottom ----
    ax.annotate('', xy=(srv_x + srv_w * 0.25, srv_y - 0.02),
                xytext=(srv_x + srv_w * 0.25, res_y + res_h + 0.02),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=0.8))
    ax.annotate('', xy=(srv_x + srv_w * 0.75, srv_y - 0.02),
                xytext=(srv_x + srv_w * 0.75, diag_y + diag_h + 0.02),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=0.8))

    plt.tight_layout(pad=0.3)
    plt.savefig(f'{OUTDIR}/fig1_overview.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f'{OUTDIR}/fig1_overview.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print('Saved fig1_overview.pdf/.png')
    plt.close()


# ============================================================
#  FIG 3: FEDPER WORKFLOW
# ============================================================

def draw_fig3():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 3.5)
    ax.axis('off')

    # Step positions (x-centers)
    steps_x = [0.85, 2.55, 4.25, 5.95]
    step_w = 1.30
    step_labels = [
        'Step 1\nLocal Training',
        'Step 2\nParameter Upload',
        'Step 3\nAggregation',
        'Step 4\nParameter Download',
    ]

    # Top: step labels
    for i, (sx, label) in enumerate(zip(steps_x, step_labels)):
        ax.text(sx, 3.30, label, ha='center', va='top',
                fontsize=7.5, fontweight='bold', color='#424242')

    # ---- Step 1: Local Training (4 client boxes) ----
    s1_x = steps_x[0]
    client_colors = [C_A, C_B, C_C, C_D]
    client_labels_short = ['A', 'B', 'C', 'D']

    for i in range(4):
        bx = s1_x - 0.50
        by = 2.45 - i * 0.52
        bw, bh = 1.00, 0.42

        # Encoder part (blue, left 60%)
        enc_rect = Rectangle((bx, by), bw * 0.60, bh,
                              facecolor=C_SHARED, edgecolor='#1565C0', linewidth=0.7)
        ax.add_patch(enc_rect)
        ax.text(bx + bw * 0.30, by + bh/2, 'Enc',
                ha='center', va='center', fontsize=5.5, color='#1565C0', fontweight='bold')

        # KAN head part (red, right 40%)
        head_rect = Rectangle((bx + bw * 0.60, by), bw * 0.40, bh,
                               facecolor=C_PRIVATE, edgecolor='#C62828', linewidth=0.7)
        ax.add_patch(head_rect)
        ax.text(bx + bw * 0.80, by + bh/2, 'KAN',
                ha='center', va='center', fontsize=5.5, color='#C62828', fontweight='bold')

        # Client label
        ax.text(bx - 0.12, by + bh/2, client_labels_short[i],
                ha='center', va='center', fontsize=7, fontweight='bold',
                color=client_colors[i])

    ax.text(s1_x, 0.42, 'Train E epochs\nlocally',
            ha='center', va='center', fontsize=6, color='#666666', fontstyle='italic')

    # ---- Step 2: Parameter Upload ----
    s2_x = steps_x[1]

    # Show 4 small encoder blocks being sent up
    for i in range(4):
        by_src = 2.45 - i * 0.52 + 0.21  # center of client box
        # Arrow from client to center
        ax.annotate('', xy=(s2_x + 0.10, 1.65),
                    xytext=(s1_x + 0.55, by_src),
                    arrowprops=dict(arrowstyle='->', color='#42A5F5', lw=0.8,
                                    connectionstyle='arc3,rad=0.1'))

    # Central collection point
    coll_box = FancyBboxPatch((s2_x - 0.35, 1.35), 0.90, 0.60,
                              boxstyle='round,pad=0.04', facecolor=C_SHARED,
                              edgecolor='#1565C0', linewidth=1.0)
    ax.add_patch(coll_box)
    ax.text(s2_x + 0.10, 1.65, 'θ_enc\nonly',
            ha='center', va='center', fontsize=7, color='#1565C0', fontweight='bold')

    # KAN stays local annotation
    ax.text(s2_x + 0.10, 0.72, 'KAN heads\nstay local',
            ha='center', va='center', fontsize=6, color='#C62828', fontstyle='italic')
    # Small red boxes showing KAN staying
    for i in range(4):
        by = 2.45 - i * 0.52
        small_rect = Rectangle((s1_x + 0.12, by + 0.05), 0.35, 0.32,
                                facecolor=C_PRIVATE, edgecolor='#C62828',
                                linewidth=0.5, linestyle='--', alpha=0.4)
        ax.add_patch(small_rect)

    # "exclude BN" note
    ax.text(s2_x + 0.10, 0.42, 'Exclude BN\nparameters',
            ha='center', va='center', fontsize=5.5, color='#888888', fontstyle='italic')

    # ---- Step 3: Server Aggregation ----
    s3_x = steps_x[2]

    # Arrow from step 2 to step 3
    ax.annotate('', xy=(s3_x - 0.45, 1.65),
                xytext=(s2_x + 0.55, 1.65),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.2))

    # Server circle
    server_circle = plt.Circle((s3_x, 1.65), 0.45,
                                facecolor=C_SERVER, edgecolor='#757575', linewidth=1.2)
    ax.add_patch(server_circle)
    ax.text(s3_x, 1.78, 'FedAvg', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#424242')
    ax.text(s3_x, 1.52, 'θ = Σ(n_k/N)·θ_k',
            ha='center', va='center', fontsize=5.5, color='#555555', fontstyle='italic')

    # ---- Step 4: Parameter Download ----
    s4_x = steps_x[3]

    # Arrow from step 3 to step 4
    ax.annotate('', xy=(s4_x - 0.35, 1.65),
                xytext=(s3_x + 0.50, 1.65),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.2))

    # Show 4 client boxes receiving updated encoder
    for i in range(4):
        bx = s4_x - 0.50
        by = 2.45 - i * 0.52
        bw, bh = 1.00, 0.42

        # Encoder (updated, highlighted)
        enc_rect = Rectangle((bx, by), bw * 0.60, bh,
                              facecolor='#90CAF9', edgecolor='#1565C0', linewidth=0.9)
        ax.add_patch(enc_rect)
        ax.text(bx + bw * 0.30, by + bh/2, 'Enc*',
                ha='center', va='center', fontsize=5.5, color='#0D47A1', fontweight='bold')

        # KAN head (unchanged)
        head_rect = Rectangle((bx + bw * 0.60, by), bw * 0.40, bh,
                               facecolor=C_PRIVATE, edgecolor='#C62828', linewidth=0.7)
        ax.add_patch(head_rect)
        ax.text(bx + bw * 0.80, by + bh/2, 'KAN',
                ha='center', va='center', fontsize=5.5, color='#C62828', fontweight='bold')

        # Download arrows
        ax.annotate('', xy=(bx + 0.05, by + bh/2),
                    xytext=(s3_x + 0.50, 1.65),
                    arrowprops=dict(arrowstyle='->', color='#42A5F5', lw=0.6,
                                    connectionstyle='arc3,rad=-0.1'))

        # Client label
        ax.text(bx + bw + 0.12, by + bh/2, client_labels_short[i],
                ha='center', va='center', fontsize=7, fontweight='bold',
                color=client_colors[i])

    ax.text(s4_x, 0.42, 'Updated encoder\n+ private KAN',
            ha='center', va='center', fontsize=6, color='#666666', fontstyle='italic')

    # ---- Bottom: repeat note ----
    ax.plot([0.15, 6.85], [0.12, 0.12], color='#BDBDBD', linewidth=0.8, linestyle='-')
    ax.text(3.50, 0.02, 'Repeat for T = 30 communication rounds',
            ha='center', va='center', fontsize=7.5, fontweight='bold',
            color='#757575', fontstyle='italic')

    # Legend — positioned at upper right to avoid blocking Step 1 title
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_SHARED, edgecolor='#1565C0', label='Shared (θ_enc)'),
        Patch(facecolor=C_PRIVATE, edgecolor='#C62828', label='Private (θ_head)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=6.5,
              frameon=True, fancybox=True, framealpha=0.9, ncol=2,
              bbox_to_anchor=(1.0, 1.02))

    plt.tight_layout(pad=0.3)
    plt.savefig(f'{OUTDIR}/fig3_fedper_workflow.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f'{OUTDIR}/fig3_fedper_workflow.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print('Saved fig3_fedper_workflow.pdf/.png')
    plt.close()


# ============================================================
if __name__ == '__main__':
    draw_fig1()
    draw_fig3()
    print('All done.')
