import numpy as np
import matplotlib.pyplot as plt


test_version = input("Enter the test version number to load (e.g., 1, 2, 3): ").strip()

file_path = f"quicksaves_test/reconstructed_{test_version}.npy"

try:
    reconstructed_img = np.load(file_path)
    print(f"Successfully loaded image from: {file_path}")
    print(f"Image matrix shape: {reconstructed_img.shape}")

    plt.figure(figsize=(12, 8))

    vmin = 0 
    vmax = np.percentile(reconstructed_img, 99.5) 

    im = plt.imshow(reconstructed_img, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    
    plt.colorbar(im, label='Signal Intensity (ADU)')
    plt.title(f'Reconstructed SNIFS Spectrograph Frame (Test {test_version})')
    plt.xlabel('X (Pixel Column)')
    plt.ylabel('Y (Pixel Row)')
    
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Ensure you typed the correct number and that the test has finished running.")