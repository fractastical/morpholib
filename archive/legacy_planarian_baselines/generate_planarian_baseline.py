#!/usr/bin/env python3
"""
Generate a baseline planarian image/graph for use as a reference.
This creates a clean, classic planarian silhouette that can be used
as a baseline for comparisons or overlays.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Import the planarian plotting function from archive
# Note: Python module names can't have hyphens, so we import it differently
import importlib.util
archive_path = os.path.join(os.path.dirname(__file__), 'archive', 'basic-plan.py')
spec = importlib.util.spec_from_file_location("basic_plan", archive_path)
basic_plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(basic_plan)
planarian_outline = basic_plan.planarian_outline

def classic_planarian_params():
    """
    Improved parameters for a classic planarian shape:
    - More prominent auricles (head flares)
    - Wider head relative to tail
    - Better proportions matching Schmidtea mediterranea
    """
    return dict(
        L=1.0,                 # half-length
        W=0.40,                # Increased width for more realistic proportions
        p=1.5,                 # Softer body roundness
        head_taper=1.8,        # Less aggressive head taper (wider head)
        tail_taper=4.0,        # More aggressive tail taper (sharper tail)
        belly_bulge=0.30,      # More pronounced mid-body bulge
        belly_pos=-0.10,       # Bulge slightly toward tail
        belly_sigma=0.50,      # Wider bulge spread
        head_flare=0.50,       # Much more prominent auricles
        head_flare_pos=0.75,   # Auricles positioned near head
        head_flare_sigma=0.15, # Tighter auricle spread for distinct "ears"
        samples=900,
    )

def plot_classic_planarian(params=None, add_eyes=True, ax=None, facecolor="#7ac", edgecolor="k"):
    """Plot a classic planarian with improved parameters."""
    if params is None:
        params = classic_planarian_params()
    X, Y, x, w = planarian_outline(**params)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    ax.fill(X, Y, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)

    # Eyes: positioned on the auricles (head flares)
    if add_eyes:
        # Position eyes on the auricles, slightly forward
        x_eye = params["L"] * 0.70
        idx = np.argmin(np.abs(x - x_eye))
        y_half = w[idx]
        eye_offset = 0.50 * y_half  # More lateral placement on auricles
        eye_r = 0.035 * params["L"]  # Slightly larger eyes
        for y_eye in (+eye_offset, -eye_offset):
            circle = plt.Circle((x_eye, y_eye), eye_r, color="black", zorder=5)
            ax.add_patch(circle)

    # aesthetics
    ax.set_aspect("equal", adjustable="box")
    pad = 0.12 * params["L"]
    ax.set_xlim(-params["L"] - pad, params["L"] + pad)
    ax.set_ylim(-params["W"] * 2.2, params["W"] * 2.2)
    ax.axis("off")
    return ax

def generate_baseline_planarian(output_path="planarian_baseline.png", 
                                format="png", 
                                dpi=300,
                                transparent=True,
                                size=(6, 3),
                                use_classic=True):
    """
    Generate and save a baseline planarian image.
    
    Parameters:
    -----------
    output_path : str
        Path to save the image
    format : str
        Image format: 'png', 'svg', or 'pdf'
    dpi : int
        Resolution for raster formats (png)
    transparent : bool
        Whether to use transparent background
    size : tuple
        Figure size (width, height) in inches
    use_classic : bool
        Use improved classic planarian parameters
    """
    fig, ax = plt.subplots(figsize=size)
    
    # Generate the planarian with improved parameters
    if use_classic:
        plot_classic_planarian(params=None, add_eyes=True, ax=ax, 
                             facecolor="#7ac", edgecolor="k")
    else:
        basic_plan.plot_planarian(params=None, add_eyes=True, ax=ax, 
                                 facecolor="#7ac", edgecolor="k")
    
    # Save the figure
    plt.savefig(output_path, format=format, dpi=dpi, 
                bbox_inches="tight", transparent=transparent)
    print(f"✓ Saved baseline planarian to: {output_path}")
    plt.close()
    
    return output_path

def generate_baseline_with_variants(output_dir="baseline_planarians"):
    """
    Generate multiple baseline variants for comparison.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Classic improved baseline
    generate_baseline_planarian(
        os.path.join(output_dir, "planarian_baseline.png"),
        use_classic=True
    )
    
    # Without eyes
    fig, ax = plt.subplots(figsize=(6, 3))
    plot_classic_planarian(params=None, add_eyes=False, ax=ax, 
                          facecolor="#7ac", edgecolor="k")
    plt.savefig(os.path.join(output_dir, "planarian_baseline_no_eyes.png"),
                dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"✓ Saved baseline without eyes to: {output_dir}/planarian_baseline_no_eyes.png")
    
    # SVG version (vector, scalable)
    fig, ax = plt.subplots(figsize=(6, 3))
    plot_classic_planarian(params=None, add_eyes=True, ax=ax, 
                          facecolor="#7ac", edgecolor="k")
    plt.savefig(os.path.join(output_dir, "planarian_baseline.svg"),
                format="svg", bbox_inches="tight", transparent=True)
    plt.close()
    print(f"✓ Saved SVG baseline to: {output_dir}/planarian_baseline.svg")
    
    # High contrast version (for overlays)
    fig, ax = plt.subplots(figsize=(6, 3))
    plot_classic_planarian(params=None, add_eyes=True, ax=ax, 
                          facecolor="white", edgecolor="black")
    plt.savefig(os.path.join(output_dir, "planarian_baseline_high_contrast.png"),
                dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"✓ Saved high contrast baseline to: {output_dir}/planarian_baseline_high_contrast.png")
    
    # Comparison: original vs improved
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
    basic_plan.plot_planarian(params=None, add_eyes=True, ax=ax1, 
                              facecolor="#7ac", edgecolor="k")
    ax1.set_title("Original Parameters", fontsize=10)
    plot_classic_planarian(params=None, add_eyes=True, ax=ax2, 
                          facecolor="#7ac", edgecolor="k")
    ax2.set_title("Improved Classic Form", fontsize=10)
    plt.savefig(os.path.join(output_dir, "planarian_comparison.png"),
                dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"✓ Saved comparison to: {output_dir}/planarian_comparison.png")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate baseline planarian images")
    parser.add_argument("--output", "-o", default="planarian_baseline.png",
                       help="Output file path")
    parser.add_argument("--format", "-f", choices=["png", "svg", "pdf"], 
                       default="png", help="Output format")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for raster formats")
    parser.add_argument("--variants", action="store_true",
                       help="Generate multiple variants")
    parser.add_argument("--original", action="store_true",
                       help="Use original parameters instead of improved")
    
    args = parser.parse_args()
    
    if args.variants:
        generate_baseline_with_variants()
    else:
        generate_baseline_planarian(args.output, args.format, args.dpi, 
                                  use_classic=not args.original)
