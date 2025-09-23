from typing import Optional, Dict, List, Union, Tuple
from smolagents import Tool
from PIL import Image, ImageDraw
import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torchxrayvision as xrv
import skimage.io

logger = logging.getLogger(__name__)

class AnatomicalSegmentationTool(Tool):
    name = "get_anatomical_mask"
    description = """Generates anatomical region masks using TorchXrayVision's PSPNet segmentation model.
    This tool is specifically designed for "add" operations where you need to target specific anatomical regions.
    
    Available anatomical regions:
    'Left Clavicle', 'Right Clavicle', 'Left Scapula', 'Right Scapula', 'Left Lung', 'Right Lung', 
    'Left Hilus Pulmonis', 'Right Hilus Pulmonis', 'Heart', 'Aorta', 'Facies Diaphragmatica', 
    'Mediastinum', 'Weasand', 'Spine'
    
    For "add" operations:
    - If target regions are specified, returns a mask covering those regions
    - If no target regions specified, defaults to union of Left Lung and Right Lung
    - Mask is returned as a PIL Image suitable for use with RadEdit
    """
    
    inputs = {
        "image_path": {"type": "string", "description": "Path to the chest X-ray image file"},
        "target_regions": {"type": "object", "description": "List of anatomical region names to create mask for (optional)", "nullable": True},
        "user_prompt": {"type": "string", "description": "User's original prompt to help infer target regions (optional)", "nullable": True},
        "output_path": {"type": "string", "description": "Path to save the generated mask (optional)", "nullable": True}
    }
    output_type = "object"
    
    def __init__(self):
        """Initialize the anatomical segmentation tool."""
        super().__init__()
        self.model = None
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Available anatomical regions from PSPNet model
        self.available_regions = [
            'Left Clavicle', 'Right Clavicle', 'Left Scapula', 'Right Scapula',
            'Left Lung', 'Right Lung', 'Left Hilus Pulmonis', 'Right Hilus Pulmonis',
            'Heart', 'Aorta', 'Facies Diaphragmatica', 'Mediastinum', 'Weasand', 'Spine'
        ]
        
        # Default regions for unspecified "add" operations
        self.default_regions = ['Left Lung', 'Right Lung']
        
    def _load_model(self):
        """Load the TorchXrayVision PSPNet segmentation model."""
        if not self.model_loaded:
            try:
                logger.info("Loading TorchXrayVision PSPNet segmentation model...")
                self.model = xrv.baseline_models.chestx_det.PSPNet()
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"PSPNet model loaded successfully on {self.device}")
                logger.info(f"Available anatomical regions: {self.model.targets}")
                
                self.model_loaded = True
                
            except Exception as e:
                logger.error(f"Failed to load PSPNet model: {e}")
                raise
    
    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess the chest X-ray image for the PSPNet model."""
        try:
            # Load image using skimage (as per torchxrayvision convention)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            img = skimage.io.imread(image_path)
            
            if len(img.shape) == 3:
                if img.shape[2] == 4: 
                    img = img[:, :, :3]  # Remove alpha channel
                img = img.mean(2)  # Convert to grayscale
            elif len(img.shape) == 2:
                # Already grayscale
                pass
            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")
            
            # Normalize to [-1024, 1024] range as required by torchxrayvision
            img = xrv.datasets.normalize(img, 255)
            
            # Resize to 512x512 (PSPNet expects this)
            img = xrv.datasets.XRayResizer(512)(img[None, ...])[0]
            
            # Convert to tensor and add batch dimension
            img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # Shape: (1, 512, 512)
            
            logger.info(f"Preprocessed image shape for PSPNet: {img_tensor.shape}")
            return img_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def _infer_target_regions(self, user_prompt: str) -> List[str]:
        """
        Infer target anatomical regions from user prompt for "add" operations.
        Uses intelligent priority-based parsing to avoid redundant combinations.
        
        Args:
            user_prompt: The user's original prompt (e.g., "add pacemaker", "add device to left lung")
            
        Returns:
            List of inferred anatomical region names
        """
        if not user_prompt:
            return self.default_regions
        
        prompt_lower = user_prompt.lower()
        
        # Priority-based region mappings
        # Higher priority mappings will override lower priority ones
        specific_anatomical_mappings = {
            # Specific anatomical regions (HIGHEST PRIORITY)
            'left lung': ['Left Lung'],
            'right lung': ['Right Lung'],
            'left clavicle': ['Left Clavicle'],
            'right clavicle': ['Right Clavicle'],
            'left scapula': ['Left Scapula'],
            'right scapula': ['Right Scapula'],
            'left hilus': ['Left Hilus Pulmonis'],
            'right hilus': ['Right Hilus Pulmonis'],
            'left hilum': ['Left Hilus Pulmonis'],
            'right hilum': ['Right Hilus Pulmonis'],
        }
        
        general_anatomical_mappings = {
            # General anatomical regions (MEDIUM PRIORITY)
            'heart': ['Heart'],
            'cardiac': ['Heart'],
            'mediastinum': ['Mediastinum'],
            'mediastinal': ['Mediastinum'],
            'spine': ['Spine'],
            'aorta': ['Aorta'],
            'lung': ['Left Lung', 'Right Lung'],  # Only if no specific lung mentioned
            'lungs': ['Left Lung', 'Right Lung'],
            'pulmonary': ['Left Lung', 'Right Lung'],
            'clavicle': ['Left Clavicle', 'Right Clavicle'],
            'scapula': ['Left Scapula', 'Right Scapula'],
            'hilus': ['Left Hilus Pulmonis', 'Right Hilus Pulmonis'],
            'hilum': ['Left Hilus Pulmonis', 'Right Hilus Pulmonis'],
        }
        
        device_default_mappings = {
            # Device-specific defaults (LOWEST PRIORITY - only used if no anatomical region specified)
            'pacemaker': ['Heart', 'Left Lung'],
            'defibrillator': ['Heart'],
            'icd': ['Heart'],
            'catheter': ['Heart'],
            'tube': ['Left Lung', 'Right Lung'],
            'line': ['Mediastinum'],
            'wire': ['Heart'],
        }
        
        # Step 1: Check for specific anatomical regions first (highest priority)
        inferred_regions = []
        found_specific = False
        
        for term, regions in specific_anatomical_mappings.items():
            if term in prompt_lower:
                inferred_regions.extend(regions)
                found_specific = True
                logger.info(f"Found specific anatomical region '{term}': {regions}")
        
        # Step 2: If no specific regions found, check general anatomical terms
        if not found_specific:
            found_general = False
            for term, regions in general_anatomical_mappings.items():
                if term in prompt_lower:
                    inferred_regions.extend(regions)
                    found_general = True
                    logger.info(f"Found general anatomical region '{term}': {regions}")
            
            # Step 3: If no anatomical regions found, use device defaults
            if not found_general:
                for term, regions in device_default_mappings.items():
                    if term in prompt_lower:
                        inferred_regions.extend(regions)
                        logger.info(f"Using device default for '{term}': {regions}")
        
        # Handle directional modifiers for bilateral structures
        if not found_specific and inferred_regions:
            # Apply directional modifiers only if we haven't already found specific regions
            if 'left' in prompt_lower:
                # Filter to left-side structures only
                left_filtered = []
                for region in inferred_regions:
                    if region in ['Left Lung', 'Right Lung']:
                        left_filtered.append('Left Lung')
                    elif region in ['Left Clavicle', 'Right Clavicle']:
                        left_filtered.append('Left Clavicle')
                    elif region in ['Left Scapula', 'Right Scapula']:
                        left_filtered.append('Left Scapula')
                    elif region in ['Left Hilus Pulmonis', 'Right Hilus Pulmonis']:
                        left_filtered.append('Left Hilus Pulmonis')
                    elif 'Left' not in region and 'Right' not in region:
                        left_filtered.append(region)  # Keep non-bilateral structures like Heart
                    elif 'Left' in region:
                        left_filtered.append(region)  # Keep already left-specific regions
                # Remove duplicates
                left_filtered = list(dict.fromkeys(left_filtered))
                if left_filtered:
                    inferred_regions = left_filtered
                    logger.info(f"Applied 'left' modifier: {inferred_regions}")
            
            elif 'right' in prompt_lower:
                # Filter to right-side structures only
                right_filtered = []
                for region in inferred_regions:
                    if region in ['Left Lung', 'Right Lung']:
                        right_filtered.append('Right Lung')
                    elif region in ['Left Clavicle', 'Right Clavicle']:
                        right_filtered.append('Right Clavicle')
                    elif region in ['Left Scapula', 'Right Scapula']:
                        right_filtered.append('Right Scapula')
                    elif region in ['Left Hilus Pulmonis', 'Right Hilus Pulmonis']:
                        right_filtered.append('Right Hilus Pulmonis')
                    elif 'Left' not in region and 'Right' not in region:
                        right_filtered.append(region)  # Keep non-bilateral structures like Heart
                    elif 'Right' in region:
                        right_filtered.append(region)  # Keep already right-specific regions
                # Remove duplicates
                right_filtered = list(dict.fromkeys(right_filtered))
                if right_filtered:
                    inferred_regions = right_filtered
                    logger.info(f"Applied 'right' modifier: {inferred_regions}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_regions = []
        for region in inferred_regions:
            if region not in seen:
                seen.add(region)
                unique_regions.append(region)
        
        # If no regions found, use default lung regions for "add" operations
        if not unique_regions:
            unique_regions = self.default_regions
            logger.info(f"No specific regions inferred, using default: {unique_regions}")
        
        # Validate that all inferred regions are available
        valid_regions = [region for region in unique_regions if region in self.available_regions]
        
        if not valid_regions:
            logger.warning(f"No valid regions found, falling back to default: {self.default_regions}")
            return self.default_regions
        
        logger.info(f"Final inferred target regions: {valid_regions}")
        return valid_regions
    
    def _create_combined_mask(self, segmentation_output: torch.Tensor, target_regions: List[str]) -> Image.Image:
        """
        Create a combined binary mask from segmentation output for specified regions.
        
        Args:
            segmentation_output: PSPNet output tensor of shape [1, 14, 512, 512]
            target_regions: List of anatomical region names to include in mask
            
        Returns:
            PIL Image mask (512x512, mode 'L')
        """
        try:
            # segmentation_output shape: [1, 14, 512, 512]
            seg_probs = segmentation_output[0]  # Remove batch dimension: [14, 512, 512]
            
            # Create empty mask on the same device as segmentation output
            combined_mask = torch.zeros(512, 512, dtype=torch.float32, device=seg_probs.device)
            
            # Add each target region to the mask
            for region_name in target_regions:
                if region_name in self.model.targets:
                    region_index = self.model.targets.index(region_name)
                    region_mask = seg_probs[region_index]
                    
                    # Threshold the segmentation probabilities (you can adjust this threshold)
                    threshold = 0.5
                    binary_region_mask = (region_mask > threshold).float()
                    
                    # Add to combined mask (union operation)
                    combined_mask = torch.maximum(combined_mask, binary_region_mask)
                    
                    logger.info(f"Added region '{region_name}' (index {region_index}) to mask")
                else:
                    logger.warning(f"Region '{region_name}' not found in model targets: {self.model.targets}")
            
            # Convert to numpy and then to PIL Image
            mask_np = (combined_mask.cpu().numpy() * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np, mode='L')
            
            # Optional: Apply morphological operations to clean up the mask
            # You can uncomment these if you want smoother masks
            # from scipy import ndimage
            # mask_np = ndimage.binary_opening(mask_np > 127, structure=np.ones((3,3))).astype(np.uint8) * 255
            # mask_pil = Image.fromarray(mask_np, mode='L')
            
            logger.info(f"Created combined mask for regions: {target_regions}")
            logger.info(f"Mask coverage: {(mask_np > 0).sum()} / {mask_np.size} pixels ({100 * (mask_np > 0).sum() / mask_np.size:.1f}%)")
            
            return mask_pil
            
        except Exception as e:
            logger.error(f"Error creating combined mask: {e}")
            raise
    
    def forward(self, 
                image_path: str,
                target_regions: Optional[List[str]] = None,
                user_prompt: Optional[str] = None,
                output_path: Optional[str] = None) -> Dict:
        """
        Generate anatomical region mask for "add" operations.
        
        Args:
            image_path: Path to the chest X-ray image
            target_regions: Optional list of specific anatomical regions to mask
            user_prompt: Optional user prompt to infer target regions from
            output_path: Optional path to save the generated mask
            
        Returns:
            Dict containing mask information and file paths
        """
        logger.info(f"Generating anatomical mask for: {image_path}")
        
        try:
            # Load model if not already loaded
            self._load_model()
            
            # Determine target regions
            if target_regions:
                # Use explicitly provided regions
                final_target_regions = [region for region in target_regions if region in self.available_regions]
                if not final_target_regions:
                    logger.warning(f"None of the specified regions {target_regions} are available. Using default.")
                    final_target_regions = self.default_regions
            else:
                # Infer from user prompt or use default
                final_target_regions = self._infer_target_regions(user_prompt)
            
            logger.info(f"Using target regions: {final_target_regions}")
            
            # Preprocess the image
            img_tensor = self._preprocess_image(image_path)
            img_tensor = img_tensor.to(self.device)
            
            # Run segmentation
            with torch.no_grad():
                segmentation_output = self.model(img_tensor)
            
            logger.info(f"Segmentation output shape: {segmentation_output.shape}")
            
            # Create combined mask for target regions
            mask_image = self._create_combined_mask(segmentation_output, final_target_regions)
            
            # Save mask if output path provided
            mask_path = None
            if output_path:
                mask_image.save(output_path)
                mask_path = output_path
                logger.info(f"Saved anatomical mask to: {output_path}")
            
            # Calculate mask statistics
            mask_array = np.array(mask_image)
            total_pixels = mask_array.size
            masked_pixels = (mask_array > 127).sum()  # Count pixels above threshold
            coverage_percentage = (masked_pixels / total_pixels) * 100
            
            result = {
                "image_path": image_path,
                "mask_image": mask_image,  # PIL Image object
                "mask_path": mask_path,
                "target_regions": final_target_regions,
                "available_regions": self.available_regions,
                "coverage_percentage": float(coverage_percentage),
                "masked_pixels": int(masked_pixels),
                "total_pixels": int(total_pixels),
                "mask_size": mask_image.size,
                "segmentation_successful": True
            }
            
            logger.info(f"✅ Anatomical mask generation successful")
            logger.info(f"   Target regions: {final_target_regions}")
            logger.info(f"   Coverage: {coverage_percentage:.1f}% ({masked_pixels}/{total_pixels} pixels)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in anatomical mask generation: {e}")
            return {
                "error": str(e),
                "image_path": image_path,
                "segmentation_successful": False
            } 