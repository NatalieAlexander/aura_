from typing import Optional, Dict, List
from smolagents import Tool
from PIL import Image
import os
import logging
import numpy as np
import torch
import torchvision
import skimage
import torchxrayvision as xrv

logger = logging.getLogger(__name__)

class TorchXrayVisionTool(Tool):
    name = "detect_pathologies"
    description = """Detects pathological conditions in chest X-ray images using TorchXrayVision DenseNet model.
    Can perform single image analysis OR comparative analysis between original and counterfactual images.
    
    This tool uses the DenseNet121 model trained on CheXpert dataset to identify multiple pathological conditions
    and returns probability scores for each pathology. When a counterfactual image is provided, it performs
    comprehensive delta analysis to quantify the effectiveness of counterfactual generation.
    
    SINGLE IMAGE MODE - Returns:
    {
        "pathologies": {"pathology_name": probability_score, ...},
        "top_pathologies": [{"name": "pathology_name", "probability": score}, ...],
        "positive_pathologies": [...],  // Above threshold
        "model_used": "densenet121-res224-chex"
    }
    
    COMPARISON MODE - Additional returns:
    {
        "comparison_mode": true,
        "counterfactual_pathologies": {...},
        "pathology_deltas": {"pathology_name": delta_value, ...},  // original - counterfactual
        "significant_changes": [...],  // Changes > 5%
        "focused_analysis": [...],  // Analysis for target pathologies
        "target_improvement": {"pathology": "name", "delta": float, "effectiveness": "IMPROVED/WORSENED/MINIMAL_CHANGE", ...},  // Primary target pathology result
        "effectiveness_metrics": {
            "net_improvement": float,
            "num_improved": int,
            "num_worsened": int
        },
        "best_improvement": {"pathology": "name", "delta": float},
        "worst_change": {"pathology": "name", "delta": float}
    }
    
    The pathologies detected include: Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema,
    Emphysema, Fibrosis, Effusion, Pneumonia, Pleural_Thickening, Cardiomegaly, Nodule, Mass, Hernia,
    Lung Lesion, Fracture, Lung Opacity, Enlarged Cardiomediastinum.
    """
    
    inputs = {
        "image_path": {"type": "string", "description": "Path to the chest X-ray image file"},
        "threshold": {"type": "number", "description": "Probability threshold for considering a pathology positive (default: 0.5)", "nullable": True},
        "top_k": {"type": "integer", "description": "Number of top pathologies to return (default: 5)", "nullable": True},
        "counterfactual_image_path": {"type": "string", "description": "Path to counterfactual image for comparison analysis (optional)", "nullable": True},
        "target_pathologies": {"type": "object", "description": "List of specific pathologies to focus delta analysis on (optional)", "nullable": True}
    }
    output_type = "object"
    
    def __init__(self):
        """Initialize the TorchXrayVision tool."""
        super().__init__()
        self.model = None
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = None
        
    def _load_model(self):
        """Load the TorchXrayVision model if not already loaded."""
        if not self.model_loaded:
            try:
                logger.info("Loading TorchXrayVision DenseNet121 model with CheXpert weights...")
                
                self.model = xrv.models.DenseNet(weights="densenet121-res224-chex")
                self.model.to(self.device)
                self.model.eval()
                
                self.transform = torchvision.transforms.Compose([
                    xrv.datasets.XRayCenterCrop(),
                    xrv.datasets.XRayResizer(224)
                ])
                
                self.model_loaded = True
                logger.info(f"TorchXrayVision model loaded successfully on {self.device}")
                logger.info(f"Model pathologies: {self.model.pathologies}")
                
            except Exception as e:
                logger.error(f"Failed to load TorchXrayVision model: {e}")
                raise
                
    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess the chest X-ray image for TorchXrayVision model."""
        try:
            # Load image using skimage (as per torchxrayvision example)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
                
            img = skimage.io.imread(image_path)
            
            # Handle different image formats
            if len(img.shape) == 3:
                # RGB/RGBA image - convert to grayscale
                if img.shape[2] == 4:  # RGBA
                    img = img[:, :, :3]  # Remove alpha channel
                img = img.mean(2)  # Convert to grayscale
            elif len(img.shape) == 2:
                pass
            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")
            
            # Normalize to [-1024, 1024] range as required by torchxrayvision
            img = xrv.datasets.normalize(img, 255)
            
            # Add channel dimension
            img = img[None, ...]  # Shape: (1, H, W)
            
            # Apply transforms (center crop and resize to 224x224)
            img = self.transform(img)
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img).float()
            
            logger.info(f"Preprocessed image shape: {img_tensor.shape}")
            return img_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def _normalize_images_for_comparison(self, img1_path: str, img2_path: str) -> tuple:
        """
        Load and normalize two images to ensure consistent preprocessing for fair comparison.
        This ensures both images have the same size and pixel intensity distribution.
        """
        try:
            img1_tensor = self._preprocess_image(img1_path)
            img2_tensor = self._preprocess_image(img2_path)
            
            # Ensure both images have the same shape
            if img1_tensor.shape != img2_tensor.shape:
                logger.warning(f"Shape mismatch detected - Original: {img1_tensor.shape}, Counterfactual: {img2_tensor.shape}")
                # Both should be (1, 224, 224) after preprocessing, but double-check
                target_shape = (1, 224, 224)
                if img1_tensor.shape != target_shape:
                    img1_tensor = torch.nn.functional.interpolate(
                        img1_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
                    ).squeeze(0)
                if img2_tensor.shape != target_shape:
                    img2_tensor = torch.nn.functional.interpolate(
                        img2_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
                    ).squeeze(0)
                logger.info(f"🔧 Normalized both images to shape: {img1_tensor.shape}")
            
            # Optional: Normalize pixel intensity distributions to match
            # This helps account for any brightness/contrast differences from counterfactual generation
            img1_flat = img1_tensor.flatten()
            img2_flat = img2_tensor.flatten()
            
            img1_mean, img1_std = img1_flat.mean(), img1_flat.std()
            img2_mean, img2_std = img2_flat.mean(), img2_flat.std()
            
            # Only normalize if there's sufficient variation in the counterfactual
            if img2_std > 1e-6:
                # Normalize img2 to match img1's statistics
                img2_normalized = (img2_tensor - img2_mean) * (img1_std / img2_std) + img1_mean
                # Clamp to maintain reasonable range (based on torchxrayvision's expected input range)
                img2_normalized = torch.clamp(img2_normalized, -1024, 1024)
                logger.info(f"🔧 Intensity normalization applied - Original std: {img2_std:.3f} -> {img1_std:.3f}")
                return img1_tensor, img2_normalized
            else:
                logger.warning(f"⚠️ Skipping intensity normalization - insufficient variation (std={img2_std:.6f})")
                return img1_tensor, img2_tensor
                
        except Exception as e:
            logger.error(f"Error normalizing images for comparison: {e}")
            raise
    
    def _map_report_to_pathologies(self, medical_report: str) -> List[str]:
        """
        Map findings from a medical report to TorchXrayVision pathology classes.
        Returns a list of relevant pathologies to focus analysis on.
        """
        if not medical_report:
            return []
        
        # Create mapping from common medical terms to TorchXrayVision pathologies
        pathology_mapping = {
            # Effusion mappings
            'effusion': ['Effusion'],
            'pleural effusion': ['Effusion'],
            'fluid': ['Effusion'],
            
            # Pneumonia/Consolidation mappings
            'pneumonia': ['Pneumonia', 'Infiltration'],
            'consolidation': ['Consolidation'],
            'infiltration': ['Infiltration', 'Pneumonia'],
            'infection': ['Pneumonia', 'Infiltration'],
            
            # Pneumothorax mappings
            'pneumothorax': ['Pneumothorax'],
            'collapsed lung': ['Pneumothorax'],
            
            # Cardiac mappings
            'cardiomegaly': ['Cardiomegaly'],
            'enlarged heart': ['Cardiomegaly'],
            'heart': ['Cardiomegaly', 'Enlarged Cardiomediastinum'],
            
            # Lung opacity mappings
            'opacity': ['Lung Opacity', 'Infiltration'],
            'opacification': ['Lung Opacity'],
            
            # Atelectasis mappings
            'atelectasis': ['Atelectasis'],
            'collapse': ['Atelectasis'],
            
            # Nodule/Mass mappings
            'nodule': ['Nodule'],
            'mass': ['Mass'],
            'lesion': ['Lung Lesion', 'Mass', 'Nodule'],
            
            # Edema mappings
            'edema': ['Edema'],
            'pulmonary edema': ['Edema'],
            
            # Other mappings
            'emphysema': ['Emphysema'],
            'fibrosis': ['Fibrosis'],
            'thickening': ['Pleural_Thickening'],
            'fracture': ['Fracture'],
            'hernia': ['Hernia']
        }
        
        # Convert report to lowercase for matching
        report_lower = medical_report.lower()
        
        # Find matching pathologies
        relevant_pathologies = set()
        for term, pathologies in pathology_mapping.items():
            if term in report_lower:
                relevant_pathologies.update(pathologies)
                logger.info(f"📋 Found '{term}' in report, mapping to: {pathologies}")
        
        result = list(relevant_pathologies)
        logger.info(f"📋 Mapped medical report to {len(result)} relevant pathologies: {result}")
        return result
    
    def forward(self, 
                image_path: str,
                threshold: Optional[float] = 0.5,
                top_k: Optional[int] = 5,
                counterfactual_image_path: Optional[str] = None,
                target_pathologies: Optional[List[str]] = None) -> Dict:
        """
        Detect pathological conditions in a chest X-ray image.
        Can also perform comparison analysis with a counterfactual image.
        
        Args:
            image_path: Path to the chest X-ray image
            threshold: Probability threshold for considering a pathology positive (default: 0.5)
            top_k: Number of top pathologies to return (default: 5)
            counterfactual_image_path: Optional path to counterfactual image for comparison
            target_pathologies: Optional list of specific pathologies to focus analysis on
            
        Returns:
            Dict with pathology predictions and comparison analysis if counterfactual provided
        """
        logger.info(f"TorchXrayVision pathology detection for: {image_path}")
        
        try:
            # Load model if not already loaded
            self._load_model()
            
            # Check if this is a comparison analysis
            is_comparison = counterfactual_image_path is not None
            
            if is_comparison:
                logger.info(f"🔬 COMPARISON MODE: Analyzing original vs counterfactual")
                # Normalize both images for fair comparison
                img1_tensor, img2_tensor = self._normalize_images_for_comparison(image_path, counterfactual_image_path)
                
                # Move to device and add batch dimensions
                img1_tensor = img1_tensor.to(self.device)
                img2_tensor = img2_tensor.to(self.device)
                if len(img1_tensor.shape) == 3:
                    img1_tensor = img1_tensor[None, ...]
                if len(img2_tensor.shape) == 3:
                    img2_tensor = img2_tensor[None, ...]
                
                with torch.no_grad():
                    outputs1 = self.model(img1_tensor)  # Original
                    outputs2 = self.model(img2_tensor)  # Counterfactual
                
                # Convert outputs to numpy
                predictions1 = outputs1[0].detach().cpu().numpy()  # Original predictions
                predictions2 = outputs2[0].detach().cpu().numpy()  # Counterfactual predictions
                
                # Filter out empty pathology names and create dictionaries
                valid_pathologies = [(i, name) for i, name in enumerate(self.model.pathologies) if name.strip()]
                
                pathologies_dict1 = {name: predictions1[i] for i, name in valid_pathologies}
                pathologies_dict2 = {name: predictions2[i] for i, name in valid_pathologies}
                
                # Calculate deltas (original - counterfactual)
                # Positive delta means pathology was reduced in counterfactual
                pathology_deltas = {name: float(predictions1[i] - predictions2[i]) 
                                  for i, name in valid_pathologies}
                
                # Primary predictions are from the original image
                predictions = predictions1
                pathologies_dict = pathologies_dict1
                
            else:
                logger.info(f"🔍 SINGLE IMAGE MODE: Analyzing pathologies")
                # Standard single image analysis
                img_tensor = self._preprocess_image(image_path)
                
                                 # Move to device and add batch dimension
                img_tensor = img_tensor.to(self.device)
                if len(img_tensor.shape) == 3:
                    img_tensor = img_tensor[None, ...]  # Add batch dimension
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(img_tensor)
                    
                # Convert outputs to numpy and create pathology dictionary
                predictions = outputs[0].detach().cpu().numpy()
                # Filter out empty pathology names
                valid_pathologies = [(i, name) for i, name in enumerate(self.model.pathologies) if name.strip()]
                pathologies_dict = {name: predictions[i] for i, name in valid_pathologies}
            
            # Sort pathologies by probability (highest first)
            sorted_pathologies = sorted(pathologies_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Get top-k pathologies
            top_pathologies = [
                {"name": name, "probability": float(prob)} 
                for name, prob in sorted_pathologies[:top_k]
            ]
            
            # Get pathologies above threshold
            positive_pathologies = [
                {"name": name, "probability": float(prob)}
                for name, prob in pathologies_dict.items()
                if prob >= threshold
            ]
            
            # Sort positive pathologies by probability
            positive_pathologies.sort(key=lambda x: x["probability"], reverse=True)
            
            # Format all pathologies with float conversion
            all_pathologies = {name: float(prob) for name, prob in pathologies_dict.items()}
            
            logger.info(f"Detected {len(positive_pathologies)} pathologies above threshold {threshold}")
            logger.info(f"Top pathology: {sorted_pathologies[0][0]} ({sorted_pathologies[0][1]:.3f})")
            
            # Build base result
            result = {
                "image_path": image_path,
                "pathologies": all_pathologies,
                "top_pathologies": top_pathologies,
                "positive_pathologies": positive_pathologies,
                "threshold_used": threshold,
                "model_used": "densenet121-res224-chex",
                "total_pathologies": len([name for name in self.model.pathologies if name.strip()]),
                "max_probability": float(max(predictions)),
                "mean_probability": float(np.mean(predictions)),
                "pathology_list": [name for name in self.model.pathologies if name.strip()]
            }
            
            # Add comparison analysis if counterfactual was provided
            if is_comparison:
                logger.info(f"🔬 Adding comparison analysis data")
                
                all_pathologies_cf = {name: float(prob) for name, prob in pathologies_dict2.items()}
                
                significant_deltas = [
                    {"name": name, "original": float(pathologies_dict1[name]), 
                     "counterfactual": float(pathologies_dict2[name]), "delta": float(delta)}
                    for name, delta in pathology_deltas.items()
                    if abs(delta) > 0.05  # Only show meaningful changes (>5%)
                ]
                
                significant_deltas.sort(key=lambda x: abs(x["delta"]), reverse=True)
                
                focused_analysis = []
                if target_pathologies:
                    for target_pathology in target_pathologies:
                        # Try to find matching pathology in model pathologies
                        matched_pathology = None
                        
                        if target_pathology in pathology_deltas:
                            matched_pathology = target_pathology
                        else:
                            # Try to find partial matches (e.g., "Right pleural effusion." -> "Effusion")
                            target_lower = target_pathology.lower()
                            for model_pathology in pathology_deltas.keys():
                                if model_pathology.lower() in target_lower or target_lower in model_pathology.lower():
                                    matched_pathology = model_pathology
                                    break
                        
                        if matched_pathology:
                            focused_analysis.append({
                                "name": matched_pathology,
                                "original_target": target_pathology,  # Keep track of original target
                                "original": float(pathologies_dict1[matched_pathology]),
                                "counterfactual": float(pathologies_dict2[matched_pathology]),
                                "delta": float(pathology_deltas[matched_pathology]),
                                "effectiveness": "IMPROVED" if pathology_deltas[matched_pathology] > 0.05 else 
                                               "WORSENED" if pathology_deltas[matched_pathology] < -0.05 else "MINIMAL_CHANGE"
                            })
                            logger.info(f"📋 Mapped target '{target_pathology}' to model pathology '{matched_pathology}'")
                        else:
                            logger.warning(f"Target pathology '{target_pathology}' could not be matched to any model pathologies")
                            logger.info(f"Available pathologies: {list(pathology_deltas.keys())}")
                
                # Calculate overall effectiveness metrics
                total_positive_delta = sum(max(0, delta) for delta in pathology_deltas.values())
                total_negative_delta = sum(min(0, delta) for delta in pathology_deltas.values())
                net_improvement = total_positive_delta + total_negative_delta
                
                # Identify best and worst changes
                best_improvement = max(pathology_deltas.items(), key=lambda x: x[1])
                worst_change = min(pathology_deltas.items(), key=lambda x: x[1])
                
                # Identify target pathology improvement (what the user cares about)
                target_improvement = None
                if target_pathologies and focused_analysis:
                    # Find the first target pathology that was successfully analyzed
                    for analysis in focused_analysis:
                        target_improvement = {
                            "pathology": analysis['name'],
                            "delta": analysis['delta'],
                            "effectiveness": analysis['effectiveness'],
                            "original": analysis['original'],
                            "counterfactual": analysis['counterfactual']
                        }
                        break  # Use the first (primary) target pathology
                
                # Add comparison data to result
                result.update({
                    "comparison_mode": True,
                    "counterfactual_image_path": counterfactual_image_path,
                    "counterfactual_pathologies": all_pathologies_cf,
                    "pathology_deltas": {name: float(delta) for name, delta in pathology_deltas.items()},
                    "significant_changes": significant_deltas,
                    "focused_analysis": focused_analysis,
                    "target_improvement": target_improvement,  # NEW: Highlight user's target pathology
                    "effectiveness_metrics": {
                        "net_improvement": float(net_improvement),
                        "total_positive_delta": float(total_positive_delta),
                        "total_negative_delta": float(total_negative_delta),
                        "num_improved": sum(1 for delta in pathology_deltas.values() if delta > 0.05),
                        "num_worsened": sum(1 for delta in pathology_deltas.values() if delta < -0.05),
                        "num_minimal_change": sum(1 for delta in pathology_deltas.values() if abs(delta) <= 0.05)
                    },
                    "best_improvement": {
                        "pathology": best_improvement[0],
                        "delta": float(best_improvement[1])
                    },
                    "worst_change": {
                        "pathology": worst_change[0],
                        "delta": float(worst_change[1])
                    }
                })
                
                logger.info(f"✅ Comparison analysis complete - Net improvement: {net_improvement:.3f}")
                logger.info(f"🏆 Best improvement: {best_improvement[0]} ({best_improvement[1]:+.3f})")
                logger.info(f"⚠️ Worst change: {worst_change[0]} ({worst_change[1]:+.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in TorchXrayVision pathology detection: {e}")
            return {"error": str(e)} 