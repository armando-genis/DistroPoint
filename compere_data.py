import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import glob

def compare_images(folder1, folder2, output_folder="comparison_results"):
    """
    Compare images with the same filename from two different folders.
    
    Parameters:
    - folder1: Path to first folder containing images
    - folder2: Path to second folder containing images
    - output_folder: Folder to save comparison results
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of image files in both folders
    files1 = sorted(glob.glob(os.path.join(folder1, "*.png")))
    
    # Track overall statistics
    all_ssim = []
    all_mse = []
    all_filenames = []
    
    # Set thresholds for determining similarity
    ssim_threshold = 0.99  # SSIM above this is considered similar
    mse_threshold = 0.0001  # MSE below this is considered similar
    
    print(f"Comparing images from:\n - {folder1}\n - {folder2}\n")
    print(f"{'Filename':<15} {'SSIM':<10} {'MSE':<10} {'Similar?':<10}")
    print("-" * 50)
    
    # Loop through files in folder1
    for file_path1 in files1:
        filename = os.path.basename(file_path1)
        file_path2 = os.path.join(folder2, filename)
        
        # Check if file exists in second folder
        if not os.path.exists(file_path2):
            print(f"{filename:<15} {'N/A':<10} {'N/A':<10} {'File missing in folder 2':<10}")
            continue
        
        # Open images
        img1 = np.array(Image.open(file_path1).convert('L'))  # Convert to grayscale
        img2 = np.array(Image.open(file_path2).convert('L'))
        
        # Ensure images are the same size
        if img1.shape != img2.shape:
            print(f"{filename:<15} {'N/A':<10} {'N/A':<10} {'Different sizes':<10}")
            continue
        
        # Calculate SSIM (structural similarity)
        similarity = ssim(img1, img2)
        
        # Calculate MSE (mean squared error)
        error = mean_squared_error(img1, img2)
        
        # Determine if images are similar
        is_similar = similarity > ssim_threshold and error < mse_threshold
        
        # Save results
        all_ssim.append(similarity)
        all_mse.append(error)
        all_filenames.append(filename)
        
        # Print results
        print(f"{filename:<15} {similarity:.6f} {error:.6f} {'Yes' if is_similar else 'No':<10}")
        
        # Create visual comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Display original images
        axes[0].imshow(img1, cmap='viridis')
        axes[0].set_title(f"Original\n{folder1}")
        axes[0].axis('off')
        
        axes[1].imshow(img2, cmap='viridis')
        axes[1].set_title(f"Optimized\n{folder2}")
        axes[1].axis('off')
        
        # Display difference image
        diff = np.abs(img1 - img2)
        axes[2].imshow(diff, cmap='hot')
        axes[2].set_title(f"Difference\nSSIM: {similarity:.4f}, MSE: {error:.6f}")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"compare_{filename}"))
        plt.close()
    
    # Summary statistics
    if all_ssim:
        avg_ssim = np.mean(all_ssim)
        avg_mse = np.mean(all_mse)
        similar_count = sum(1 for s, e in zip(all_ssim, all_mse) if s > ssim_threshold and e < mse_threshold)
        
        print("\nSummary:")
        print(f"Total images compared: {len(all_ssim)}")
        print(f"Similar images: {similar_count} ({similar_count/len(all_ssim)*100:.1f}%)")
        print(f"Average SSIM: {avg_ssim:.6f}")
        print(f"Average MSE: {avg_mse:.6f}")
        
        # Create summary plot
        plt.figure(figsize=(10, 6))
        plt.scatter(all_ssim, all_mse, alpha=0.7)
        for i, filename in enumerate(all_filenames):
            plt.annotate(filename, (all_ssim[i], all_mse[i]))
        
        plt.axhline(y=mse_threshold, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=ssim_threshold, color='r', linestyle='--', alpha=0.5)
        
        plt.xlabel('SSIM (higher is better)')
        plt.ylabel('MSE (lower is better)')
        plt.title('Image Comparison Summary')
        plt.yscale('log')  # Use log scale for MSE to better visualize differences
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "summary_plot.png"))
        plt.close()
    else:
        print("\nNo images were compared.")

if __name__ == "__main__":
    # Folder paths
    original_folder = "scan_dl_images_origianl"
    optimized_folder = "scan_dl_images_second"
    
    # Run comparison
    compare_images(original_folder, optimized_folder)
    
    print("\nComparison complete. Results saved to 'comparison_results' folder.")