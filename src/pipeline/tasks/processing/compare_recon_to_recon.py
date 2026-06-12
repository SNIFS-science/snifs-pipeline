import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_stats(img_array, name="Image"):
    """Computes and returns a dictionary of key statistical metrics."""
    stats = {
        "Min": np.min(img_array),
        "Max": np.max(img_array),
        "Mean": np.mean(img_array),
        "Median": np.median(img_array),
        "Std Dev (Variation)": np.std(img_array),
        "Total Flux": np.sum(img_array)
    }
    return stats

def print_stats(stats, name):
    print(f"\n--- {name} Statistics ---")
    for key, val in stats.items():
        print(f"  {key:<22}: {val:.4f}")

if __name__ == "__main__":
    v1 = input("Enter the FIRST reconstructed version number to load (e.g., 1): ").strip()
    v2 = input("Enter the SECOND reconstructed version number to load (e.g., 2): ").strip()
    
    real_path = Path(f"quicksaves_test/real_data_{v1}.npy")
    recon_path_1 = Path(f"quicksaves_test/reconstructed_{v1}.npy")
    recon_path_2 = Path(f"quicksaves_test/reconstructed_{v2}.npy")
    

    for path in [real_path, recon_path_1, recon_path_2]:
        if not path.exists():
            print(f"Error: Ensure {path} exists.")
            exit(1)
            

    real_data = np.load(real_path)
    recon_1 = np.load(recon_path_1)
    recon_2 = np.load(recon_path_2)
    

    real_stats = calculate_stats(real_data, "Real Data")
    recon1_stats = calculate_stats(recon_1, f"Reconstructed (v{v1})")
    recon2_stats = calculate_stats(recon_2, f"Reconstructed (v{v2})")
    
    print("\n==============================================================")
    print("                CCD STATISTICAL COMPARISON REPORT               ")
    print("==============================================================")
    

    print_stats(real_stats, "Real Data (Observational Target)")
    print_stats(recon1_stats, f"Reconstructed Model (v{v1})")
    print_stats(recon2_stats, f"Reconstructed Model (v{v2})")
    

    print("\n==============================================================")
    print("                 CROSS-IMAGE DIAGNOSTICS                      ")
    print("==============================================================")

    def print_cross_metrics(target, recon, name):
        mae = np.mean(np.abs(target - recon))
        rmse = np.sqrt(np.mean((target - recon)**2))
        correlation = np.corrcoef(target.ravel(), recon.ravel())[0, 1]
        print(f"\n--- {name} ---")
        print(f"  Mean Absolute Error (MAE)   : {mae:.4f}")
        print(f"  Root Mean Squared Error     : {rmse:.4f}")
        print(f"  Pearson Correlation         : {correlation:.4f}")

    print_cross_metrics(real_data, recon_1, f"Real Data vs Recon (v{v1})")
    print_cross_metrics(real_data, recon_2, f"Real Data vs Recon (v{v2})")
    print_cross_metrics(recon_1, recon_2, f"Recon (v{v1}) vs Recon (v{v2})")
    

    fig1, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
#real
    im_real = axes[0].imshow(real_data, cmap='gray', origin='lower', aspect='auto')
    axes[0].set_title(f"Real Data Target\nRange: [{real_stats['Min']:.1f}, {real_stats['Max']:.1f}]", fontsize=12)
    fig1.colorbar(im_real, ax=axes[0], orientation='vertical', pad=0.02, fraction=0.046)

#recon1
    im_r1 = axes[1].imshow(recon_1, cmap='gray', origin='lower', aspect='auto')
    axes[1].set_title(f"Reconstructed Model v{v1}\nRange: [{recon1_stats['Min']:.1f}, {recon1_stats['Max']:.1f}]", fontsize=12)
    fig1.colorbar(im_r1, ax=axes[1], orientation='vertical', pad=0.02, fraction=0.046)

    # Recon 2
    im_r2 = axes[2].imshow(recon_2, cmap='gray', origin='lower', aspect='auto')
    axes[2].set_title(f"Reconstructed Model v{v2}\nRange: [{recon2_stats['Min']:.1f}, {recon2_stats['Max']:.1f}]", fontsize=12)
    fig1.colorbar(im_r2, ax=axes[2], orientation='vertical', pad=0.02, fraction=0.046)
    
    fig1.tight_layout()
    output_plot_path1 = f"quicksaves_test/comparison_diagnostic_{v1}_vs_{v2}.png"
    fig1.savefig(output_plot_path1, dpi=200, bbox_inches='tight')
    print(f"\n>>> Side-by-side plot saved to: {output_plot_path1}")

    # =========================================================================
    # PLOT 2: Direct RGB Overlay (Red = Recon 1, Green = Recon 2)
    # =========================================================================
    # Map to RGB colors by normalizing the pixel intensities from 0 to 1 using the 99.5th percentile.
    recon1_vmax = np.percentile(recon_1, 99.5)
    recon2_vmax = np.percentile(recon_2, 99.5)
    
    # Clip and normalize values to [0.0, 1.0]
    recon1_norm = np.clip(recon_1 / (recon1_vmax + 1e-8), 0, 1)
    recon2_norm = np.clip(recon_2 / (recon2_vmax + 1e-8), 0, 1)
    
    # Create an empty RGB array shaped (Height, Width, 3 Channels)
    rgb_overlay = np.zeros((*real_data.shape, 3))
    
    # Assign Recon 1 to Red Channel (Index 0)
    rgb_overlay[..., 0] = recon1_norm
    # Assign Recon 2 to Green Channel (Index 1)
    rgb_overlay[..., 1] = recon2_norm
    # Blue channel (Index 2) remains 0
    
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    ax2.imshow(rgb_overlay, origin='lower', aspect='auto')
    
    # Add an informative title indicating the color mix
    ax2.set_title(f"Visual Alignment Overlay\nRED: Recon v{v1} | GREEN: Recon v{v2} | YELLOW: Perfect Overlap", 
                  fontsize=14, pad=15)
    ax2.set_xlabel('X (Pixel Column)')
    ax2.set_ylabel('Y (Pixel Row)')
    
    fig2.tight_layout()
    output_plot_path2 = f"quicksaves_test/alignment_overlay_{v1}_vs_{v2}.png"
    fig2.savefig(output_plot_path2, dpi=200, bbox_inches='tight')
    print(f">>> Alignment overlay plot saved to: {output_plot_path2}")
    
    # Show both figures at the same time
    plt.show()