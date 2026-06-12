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
    test_version = input("Enter the test version number to load (e.g., 1, 2, 3): ").strip()
    

    real_path = Path(f"quicksaves_test/real_data_{test_version}.npy")
    recon_path = Path(f"quicksaves_test/reconstructed_{test_version}.npy")
    
    if not real_path.exists() or not recon_path.exists():
        print(f"Error: Ensure both {real_path} and {recon_path} exist.")
        exit(1)

    real_data = np.load(real_path)
    reconstructed = np.load(recon_path)

    real_stats = calculate_stats(real_data, "Real Data")
    recon_initial_stats = calculate_stats(reconstructed, "Original Reconstructed")
    
    reconstructed_clamped = np.clip(reconstructed, real_stats["Min"], real_stats["Max"])
    recon_clamped_stats = calculate_stats(reconstructed_clamped, "Clamped Reconstructed")
    
    avg_flux_before = recon_initial_stats["Mean"]
    avg_flux_after = recon_clamped_stats["Mean"]
    flux_diff = avg_flux_after - avg_flux_before
    flux_pct_change = (flux_diff / (avg_flux_before + 1e-8)) * 100
    
    print("==============================================================")
    print("                 CCD FLUX TRANSFORMATION REPORT               ")
    print("==============================================================")
    print(f"Average Flux Before Clamping : {avg_flux_before:.4f}")
    print(f"Average Flux After Clamping  : {avg_flux_after:.4f}")
    print(f"Absolute Mean Flux Change    : {flux_diff:.4f}")
    print(f"Percentage Flux Change       : {flux_pct_change:.4f}%")
    

    print_stats(real_stats, "Real Data (Observational Target)")
    print_stats(recon_initial_stats, "Original Reconstructed Model")
    print_stats(recon_clamped_stats, "Clamped Reconstructed Model")
    

    mae = np.mean(np.abs(real_data - reconstructed))
    rmse = np.sqrt(np.mean((real_data - reconstructed)**2))
    correlation = np.corrcoef(real_data.ravel(), reconstructed.ravel())[0, 1]
    
    print("\n--- Cross-Image Comparison Metrics ---")
    print(f"  Mean Absolute Error (MAE)   : {mae:.4f}")
    print(f"  Root Mean Squared Error(RMSE): {rmse:.4f}")
    print(f"  Pearson Correlation (0 to 1): {correlation:.4f} (Measures spatial shape alignment)")
    
    median_bias_diff = real_stats["Median"] - recon_initial_stats["Median"]
    print(f"  Estimated Background Floor Gap: {median_bias_diff:.4f} ADU")
    
    fig1, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    
    im_real = axes[0].imshow(real_data, cmap='gray', origin='lower', aspect='auto')
    axes[0].set_title(f"Real Data Target\nRange: [{real_stats['Min']:.1f}, {real_stats['Max']:.1f}]", fontsize=12)
    cbar_real = fig1.colorbar(im_real, ax=axes[0], orientation='vertical', pad=0.02)
    cbar_real.set_label('Intensity (ADU)', rotation=270, labelpad=15)
    
    im_recon = axes[1].imshow(reconstructed_clamped, cmap='gray', origin='lower', aspect='auto')
    axes[1].set_title(f"Clamped Reconstructed Model\nRange: [{recon_clamped_stats['Min']:.1f}, {recon_clamped_stats['Max']:.1f}]", fontsize=12)
    cbar_recon = fig1.colorbar(im_recon, ax=axes[1], orientation='vertical', pad=0.02)
    cbar_recon.set_label('Intensity (ADU)', rotation=270, labelpad=15)
    
    fig1.tight_layout()
    output_plot_path1 = f"quicksaves_test/clamped_comparison_diagnostic_{test_version}.png"
    fig1.savefig(output_plot_path1, dpi=200, bbox_inches='tight')
    print(f"\n>>> Side-by-side plot saved to: {output_plot_path1}")

    # =========================================================================
    # PLOT 2: Direct RGB Overlay (Red = Real, Green = Reconstructed)
    # =========================================================================
    # To map to RGB colors correctly, we must normalize the pixel intensities from 0 to 1.
    # We use the 99.5th percentile to prevent a single hot pixel from washing out the color scale.
    real_vmax = np.percentile(real_data, 99.5)
    recon_vmax = np.percentile(reconstructed_clamped, 99.5)
    
    # Clip and normalize values to [0.0, 1.0]
    real_norm = np.clip(real_data / (real_vmax + 1e-8), 0, 1)
    recon_norm = np.clip(reconstructed_clamped / (recon_vmax + 1e-8), 0, 1)
    
    # Create an empty RGB array shaped (Height, Width, 3 Channels)
    rgb_overlay = np.zeros((*real_data.shape, 3))
    
    # Assign Real Data to Red Channel (Index 0)
    rgb_overlay[..., 0] = real_norm
    # Assign Reconstructed Data to Green Channel (Index 1)
    rgb_overlay[..., 1] = recon_norm
    # Blue channel (Index 2) remains 0
    
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    ax2.imshow(rgb_overlay, origin='lower', aspect='auto')
    
    # Add an informative title indicating the color mix
    ax2.set_title(f"Visual Alignment Overlay (Test {test_version})\nRED: Real Target | GREEN: Model | YELLOW: Perfect Overlap", 
                  fontsize=14, pad=15)
    ax2.set_xlabel('X (Pixel Column)')
    ax2.set_ylabel('Y (Pixel Row)')
    
    fig2.tight_layout()
    output_plot_path2 = f"quicksaves_test/alignment_overlay_{test_version}.png"
    fig2.savefig(output_plot_path2, dpi=200, bbox_inches='tight')
    print(f">>> Alignment overlay plot saved to: {output_plot_path2}")
    
    # Show both figures at the same time
    plt.show()