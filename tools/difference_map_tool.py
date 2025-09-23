from typing import Optional, Dict
from smolagents import Tool
from PIL import Image, ImageOps
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
import re

logger = logging.getLogger(__name__)

class DifferenceMapTool(Tool):
    name = "generate_difference_map"
    description = "Generates a colored difference map between an original chest X-ray and its counterfactual version to visualize the changes made during editing."
    inputs = {
        "original_image_path": {"type": "string", "description": "Path to the original chest X-ray image"},
        "counterfactual_image_path": {"type": "string", "description": "Path to the counterfactual (edited) chest X-ray image"},
        "output_path": {"type": "string", "description": "Path where the difference map should be saved (optional)", "nullable": True},
        "output_prefix": {"type": "string", "description": "Prefix for output filenames to make them unique (e.g., 'scale_7_5')", "nullable": True},
        "session_path": {"type": "string", "description": "Session directory path for organized file storage (optional)", "nullable": True},
        "colormap": {"type": "string", "description": "Colormap to use for visualization (default: 'jet')", "nullable": True}
    }
    output_type = "object"
    
    def __init__(self):
        super().__init__()

    def _load_and_normalize_image(self, image_path: str) -> np.ndarray:
        try:
            image = Image.open(image_path).convert('L')
            original_size = image.size
            
            image = image.resize((512, 512), Image.LANCZOS)
            
            if original_size != (512, 512):
                logger.info(f"Resized image from {original_size} to 512x512 for pixel-perfect comparison")
            else:
                logger.info(f"Image already 512x512, no resizing needed")
            
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise ValueError(f"Could not load image from {image_path}: {e}")

    def _normalize_brightness(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Normalize img2 to match the brightness and contrast of img1.
        This helps with RadEdit's brightness inconsistencies for fairer difference maps.
        
        Args:
            img1: Reference image array (original)
            img2: Image array to normalize (counterfactual)
            
        Returns:
            Normalized img2 array
        """
        try:
            img1_mean = np.mean(img1)
            img1_std = np.std(img1)
            img2_mean = np.mean(img2)
            img2_std = np.std(img2)
            
            # Only normalize if there's sufficient variation in the counterfactual
            if img2_std > 1e-6:  # Avoid division by zero
                # Normalize img2 to match img1's statistics
                normalized_img2 = (img2 - img2_mean) * (img1_std / img2_std) + img1_mean
                # Clamp to valid range [0, 1]
                normalized_img2 = np.clip(normalized_img2, 0.0, 1.0)
                
                logger.info(f"🔧 BRIGHTNESS NORMALIZATION: Original mean={img2_mean:.3f}, std={img2_std:.3f} → Normalized mean={np.mean(normalized_img2):.3f}, std={np.std(normalized_img2):.3f}")
                return normalized_img2
            else:
                logger.warning(f"⚠️ Skipping brightness normalization - insufficient variation in counterfactual (std={img2_std:.6f})")
                return img2
                
        except Exception as e:
            logger.warning(f"⚠️ Error in brightness normalization: {e}")
            return img2

    def _create_difference_map(self, original: np.ndarray, counterfactual: np.ndarray, colormap: str = 'jet') -> np.ndarray:
        """Create a colored difference map between two images."""
        if original.shape != counterfactual.shape:
            logger.warning(f"🔧 SAFETY: Shape mismatch detected - Original: {original.shape}, Counterfactual: {counterfactual.shape}")
            if original.shape != (512, 512):
                original = cv2.resize(original, (512, 512))
                logger.info(f"🔧 SAFETY: Resized original to 512x512")
            if counterfactual.shape != (512, 512):
                counterfactual = cv2.resize(counterfactual, (512, 512))
                logger.info(f"🔧 SAFETY: Resized counterfactual to 512x512")
        
        assert original.shape == (512, 512), f"Original image shape should be (512, 512), got {original.shape}"
        assert counterfactual.shape == (512, 512), f"Counterfactual image shape should be (512, 512), got {counterfactual.shape}"
        logger.info(f"✅ Both images confirmed to be 512x512 for pixel-perfect difference calculation")
        
        diff = np.abs(original - counterfactual)
        
        # Normalize difference to [0, 1] for better visualization
        if diff.max() > 0:
            diff = diff / diff.max()
        
        cmap = cm.get_cmap(colormap)
        colored_diff = cmap(diff)
        
        colored_diff_rgb = (colored_diff[:, :, :3] * 255).astype(np.uint8)
        
        return colored_diff_rgb, diff

    def _extract_descriptive_prefix_from_counterfactual(self, counterfactual_path: str) -> str:
        """Extract descriptive prefix from counterfactual image path for consistent naming."""
        try:
            filename = os.path.basename(counterfactual_path).replace('.png', '').replace('.jpg', '')
            
            # Remove timestamp pattern (e.g., _20250610_165722)
            # Pattern to match timestamp at the end: _YYYYMMDD_HHMMSS
            timestamp_pattern = r'_\d{8}_\d{6}$'
            descriptive_part = re.sub(timestamp_pattern, '', filename)
            
            if descriptive_part == filename:
                if descriptive_part.startswith('counterfactual_'):
                    descriptive_part = descriptive_part[len('counterfactual_'):]
                    
                    parts = descriptive_part.split('_')
                    if len(parts) > 1 and '.png' in parts[0]:
                        descriptive_part = '_'.join(parts[1:])
            
            logger.info(f"Extracted descriptive prefix: '{descriptive_part}' from counterfactual path: {counterfactual_path}")
            return descriptive_part
            
        except Exception as e:
            logger.warning(f"Could not extract descriptive prefix from {counterfactual_path}: {e}")
            return "difference_map"

    def forward(self, 
                original_image_path: str, 
                counterfactual_image_path: str,
                output_path: Optional[str] = None,
                output_prefix: Optional[str] = None,
                session_path: Optional[str] = None,
                colormap: Optional[str] = 'jet') -> Dict:
        """
        Generate a colored difference map between original and counterfactual images.
        
        Args:
            original_image_path: Path to the original image
            counterfactual_image_path: Path to the counterfactual image
            output_path: Optional path to save the difference map
            output_prefix: Optional prefix for output filenames to make them unique
            colormap: Colormap for visualization (default: 'jet')
            
        Returns:
            Dict with path to difference map and detailed statistics
        """
        logger.info(f"Generating difference map between {original_image_path} and {counterfactual_image_path}")
        
        try:
            original = self._load_and_normalize_image(original_image_path)
            counterfactual = self._load_and_normalize_image(counterfactual_image_path)
            
            counterfactual = self._normalize_brightness(original, counterfactual)
            
            diff_colored, diff_raw = self._create_difference_map(original, counterfactual, colormap)
            
            if output_path is None:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if not output_prefix:
                    output_prefix = self._extract_descriptive_prefix_from_counterfactual(counterfactual_image_path)
                    logger.info(f"🔧 AUTO-NAMING: Using extracted prefix '{output_prefix}' for consistent counterfactual-difference map pairing")
                
                if output_prefix:
                    if "/" in output_prefix:
                        output_dir = os.path.dirname(output_prefix)
                        os.makedirs(output_dir, exist_ok=True)
                        base_name = f"{os.path.basename(output_prefix)}_{timestamp}"
                        diff_path = os.path.join(output_dir, f"{base_name}.png")
                    else:
                        if session_path:
                            base_dir = os.path.join(session_path, "difference_maps")
                        else:
                            base_dir = "statics/output_counterfactuals"
                        os.makedirs(base_dir, exist_ok=True)
                        base_name = f"{output_prefix}_{timestamp}"
                        diff_path = os.path.join(base_dir, f"{base_name}.png")
                else:
                    if session_path:
                        base_dir = os.path.join(session_path, "difference_maps")
                    else:
                        base_dir = "statics/output_counterfactuals"
                    os.makedirs(base_dir, exist_ok=True)
                    base_name = f"difference_map_{timestamp}"
                    diff_path = os.path.join(base_dir, f"{base_name}.png")
            else:
                diff_path = output_path
            
            diff_image = Image.fromarray(diff_colored)
            diff_image.save(diff_path)
            logger.info(f"Saved difference map to {diff_path}")
            
            max_diff = float(diff_raw.max())
            mean_diff = float(diff_raw.mean())
            median_diff = float(np.median(diff_raw))
            std_diff = float(np.std(diff_raw))
            
            significant_pixels = int(np.sum(diff_raw > 0.25))  # Pixels with significant difference (>20% change)
            moderate_pixels = int(np.sum((diff_raw > 0.1) & (diff_raw <= 0.2)))  # Moderate change (10-20%)
            minor_pixels = int(np.sum((diff_raw > 0.05) & (diff_raw <= 0.1)))  # Minor change (5-10%)
            total_pixels = int(diff_raw.size)
            
            change_percentage = (significant_pixels / total_pixels) * 100
            moderate_percentage = (moderate_pixels / total_pixels) * 100
            minor_percentage = (minor_pixels / total_pixels) * 100
            
            return {
                "difference_map_path": diff_path,
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "median_difference": median_diff,
                "std_difference": std_diff,
                "changed_pixels_significant": significant_pixels,
                "changed_pixels_moderate": moderate_pixels,
                "changed_pixels_minor": minor_pixels,
                "total_pixels": total_pixels,
                "change_percentage": change_percentage,
                "moderate_change_percentage": moderate_percentage,
                "minor_change_percentage": minor_percentage,
                "colormap_used": colormap
            }
            
        except Exception as e:
            logger.error(f"Error in DifferenceMapTool: {e}")
            return {"error": str(e)} 