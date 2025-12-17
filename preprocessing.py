"""
Advanced Preprocessing Pipeline
Research-grade medical image preprocessing
"""

from typing import Tuple, Optional, Union
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from skimage.feature import graycomatrix, graycoprops


class AdvancedPreprocessor:
    """
    Research-grade preprocessing pipeline with multi-scale analysis.
    
    Attributes:
        target_spacing: Desired voxel spacing in mm
        target_size: Target volume dimensions
    """
    
    def __init__(
        self, 
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        target_size: Tuple[int, int, int] = (128, 128, 64)
    ) -> None:
        """
        Initialize preprocessor with configuration.
        
        Args:
            target_spacing: Target voxel spacing (x, y, z) in mm
            target_size: Target volume size (x, y, z) in voxels
        """
        self.target_spacing = target_spacing
        self.target_size = target_size
    
    def load_nifti(
        self, 
        filepath: Union[str, Path]
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple], Optional[Tuple], Optional[Tuple], Optional[sitk.Image]]:
        """
        Load NIfTI file with comprehensive error handling.
        
        Args:
            filepath: Path to .nii.gz file
            
        Returns:
            Tuple of (array, spacing, origin, direction, sitk_image)
        """
        try:
            img = sitk.ReadImage(str(filepath))
            array = sitk.GetArrayFromImage(img)
            spacing = img.GetSpacing()
            origin = img.GetOrigin()
            direction = img.GetDirection()
            return array, spacing, origin, direction, img
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None, None, None, None
    
    def normalize_hu(
        self,
        image_array: np.ndarray,
        window_center: int = -600,
        window_width: int = 1500
    ) -> np.ndarray:
        """
        Normalize HU values using lung window.
        
        Args:
            image_array: Input CT volume
            window_center: HU window center
            window_width: HU window width
            
        Returns:
            Normalized array in [0, 1]
        """
        min_hu = window_center - window_width // 2
        max_hu = window_center + window_width // 2
        image_array = np.clip(image_array, min_hu, max_hu)
        image_array = (image_array - min_hu) / (max_hu - min_hu)
        return image_array.astype(np.float32)
    
    def extract_nodule_patch(self, image: np.ndarray, center_coords: tuple, diameter_mm: float = 10.0, spacing: tuple = (1.0, 1.0, 1.0)) -> np.ndarray:
        """Extract patch adapted to nodule size."""
        from scipy.ndimage import zoom
        
        z, y, x = center_coords
        
        # Adaptive: 2x diameter + 10mm margin (min 20mm, max 50mm)
        patch_size_mm = max(20, min(50, diameter_mm * 2 + 10))
        patch_size_voxels = int(patch_size_mm / spacing[0])
        half_size = patch_size_voxels // 2
        
        # Extract with bounds check
        z_start = max(0, z - half_size)
        z_end = min(image.shape[0], z + half_size)
        y_start = max(0, y - half_size)
        y_end = min(image.shape[1], y + half_size)
        x_start = max(0, x - half_size)
        x_end = min(image.shape[2], x + half_size)
        
        patch = image[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Resize to 32x32x32 for consistency
        target = (32, 32, 32)
        zoom_factors = [t/p for t, p in zip(target, patch.shape)]
        patch_resized = zoom(patch, zoom_factors, order=1)
        
        return patch_resized
    def extract_radiomics_features(
        self,
        image_array: np.ndarray,
        mask_array: np.ndarray
    ) -> np.ndarray:
        """
        Extract radiomics-like features manually (no PyRadiomics dependency).
        
        Args:
            image_array: Image patch
            mask_array: Binary mask
            
        Returns:
            Feature vector of shape (18,)
        """
        try:
            # Apply mask
            masked_image = image_array[mask_array > 0]
            
            if len(masked_image) == 0:
                return np.zeros(18, dtype=np.float32)
            
            # First-order statistics (5 features)
            mean_intensity = np.mean(masked_image)
            variance = np.var(masked_image)
            std_dev = np.sqrt(variance)
            skewness = np.mean(((masked_image - mean_intensity) / (std_dev + 1e-10)) ** 3)
            kurtosis = np.mean(((masked_image - mean_intensity) / (std_dev + 1e-10)) ** 4)
            
            # GLCM texture features (4 features)
            img_normalized = ((image_array - image_array.min()) / 
                             (image_array.max() - image_array.min() + 1e-10) * 255).astype(np.uint8)
            mid_slice = img_normalized[img_normalized.shape[0] // 2]
            
            try:
                glcm = graycomatrix(mid_slice, distances=[1], angles=[0], 
                                   levels=256, symmetric=True, normed=True)
                
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                correlation = graycoprops(glcm, 'correlation')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            except:
                contrast = correlation = energy = homogeneity = 0.0
            
            # Additional texture measures (9 features)
            entropy = -np.sum(np.histogram(masked_image, bins=50, density=True)[0] * 
                             np.log(np.histogram(masked_image, bins=50, density=True)[0] + 1e-10))
            
            percentile_10 = np.percentile(masked_image, 10)
            percentile_90 = np.percentile(masked_image, 90)
            median = np.median(masked_image)
            iqr = np.percentile(masked_image, 75) - np.percentile(masked_image, 25)
            rms = np.sqrt(np.mean(masked_image ** 2))
            mad = np.mean(np.abs(masked_image - mean_intensity))
            range_val = np.max(masked_image) - np.min(masked_image)
            
            gradient = np.gradient(masked_image)[0] if masked_image.ndim > 0 else 0
            gradient_magnitude = np.abs(gradient).mean() if isinstance(gradient, np.ndarray) else 0
            
            # Combine all features (18 total)
            feature_vector = np.array([
                mean_intensity, variance, skewness, kurtosis, std_dev,
                contrast, correlation, energy, homogeneity,
                entropy, percentile_10, percentile_90, median, iqr,
                rms, mad, range_val, gradient_magnitude
            ], dtype=np.float32)
            
            return feature_vector
            
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            return np.zeros(18, dtype=np.float32)