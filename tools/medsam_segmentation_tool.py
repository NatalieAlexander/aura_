from typing import Optional, Any, Dict, List, Union, Tuple
from smolagents import Tool
from PIL import Image, ImageDraw
import os, requests, logging, torch
from io import BytesIO
import numpy as np
from transformers import SamModel, SamProcessor
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class MedSAMSegmentationTool(Tool):
    name = "segment_with_medsam"
    description = "Generates precise segmentation masks for medical findings using MedSAM. Use this to get accurate masks for specific regions identified by bounding boxes."
    inputs = {
        "image_path": {"type": "string", "description": "Path or URL to the CXR image file"},
        "bounding_boxes": {"type": "object", "description": "List of bounding box coordinates [x1, y1, x2, y2] to segment"},
        "finding_texts": {"type": "object", "description": "List of text descriptions for each bounding box (optional)", "nullable": True},
        "output_path": {"type": "string", "description": "Path where the segmentation mask should be saved (optional)", "nullable": True}
    }
    output_type = "object"
    
    def __init__(self):
        """Initialize the MedSAM segmentation tool."""
        super().__init__()
        self.model_loaded = False
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        """Load the MedSAM model and processor."""
        if self.model_loaded:
            return

        try:
            logger.info("Loading MedSAM model from cache...")
            
            # Load the MedSAM model and processor - USE CACHE ONLY to avoid HuggingFace requests
            model_name = "flaviagiammarino/medsam-vit-base"
            logger.info(f"Loading MedSAM model from cache (local_files_only=True): {model_name}")
            
            self.model = SamModel.from_pretrained(
                model_name,
                local_files_only=True,  # CACHE ONLY - no HuggingFace requests
                use_safetensors=True
            ).to(self.device)
            
            self.processor = SamProcessor.from_pretrained(
                model_name,
                local_files_only=True,  # CACHE ONLY - no HuggingFace requests
            )
            
            self.model_loaded = True
            logger.info(f"✅ MedSAM model loaded successfully from CACHE ONLY on {self.device} (no HuggingFace requests)")
            
        except Exception as e:
            logger.error(f"❌ Error loading MedSAM model from cache: {e}")
            logger.error("This likely means the MedSAM model is not cached yet. Please run the cache setup script first.")
            logger.error("Try: python setup_cache_env.py --download-models")
            raise RuntimeError(f"Failed to load MedSAM model from cache: {e}")

    def _load_image(self, image_path: str) -> Image.Image:
        """Load an image from a path or URL."""
        if image_path.startswith(('http://', 'https://')):
            logger.info(f"Downloading image from URL: {image_path}")
            try:
                response = requests.get(image_path)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                logger.error(f"Error loading image from URL: {e}")
                raise ValueError(f"Could not load image from URL: {image_path}. Error: {e}")
        else:
            logger.info(f"Loading image from local path: {image_path}")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = Image.open(image_path).convert("RGB")
            
        return image

    def _normalize_box_coordinates(self, box: List[float], image_size: Tuple[int, int]) -> List[float]:
        """Normalize bounding box coordinates to absolute pixel values."""
        width, height = image_size
        
        # If coordinates are already normalized (0-1), convert to pixel coordinates
        if all(0 <= coord <= 1 for coord in box):
            return [
                box[0] * width,   # x1
                box[1] * height,  # y1
                box[2] * width,   # x2
                box[3] * height   # y2
            ]
        else:
            # Already in pixel coordinates
            return box

    def _segment_region(self, image: Image.Image, box: List[float]) -> np.ndarray:
        """Segment a specific region using MedSAM."""
        try:
            # Normalize box coordinates
            normalized_box = self._normalize_box_coordinates(box, image.size)
            
            # Prepare inputs for MedSAM
            inputs = self.processor(
                image, 
                input_boxes=[[normalized_box]], 
                return_tensors="pt"
            ).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs, multimask_output=False)
            
            # Post-process the mask
            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.sigmoid().cpu(), 
                inputs["original_sizes"].cpu(), 
                inputs["reshaped_input_sizes"].cpu(), 
                binarize=False
            )
            
            # Get the mask as a numpy array
            mask = masks[0].squeeze().numpy()
            
            # Binarize the mask (threshold at 0.5)
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
            
            return binary_mask
            
        except Exception as e:
            logger.error(f"Error segmenting region with MedSAM: {e}")
            # Fallback: create a simple rectangular mask
            width, height = image.size
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Convert normalized coordinates to pixel coordinates if needed
            if all(0 <= coord <= 1 for coord in box):
                x1, y1, x2, y2 = [
                    int(box[0] * width),
                    int(box[1] * height),
                    int(box[2] * width),
                    int(box[3] * height)
                ]
            else:
                x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            mask[y1:y2, x1:x2] = 255
            return mask

    def forward(self, 
                image_path: str, 
                bounding_boxes: List[List[float]],
                finding_texts: Optional[List[str]] = None,
                output_path: Optional[str] = None) -> Dict:
        """
        Generate segmentation masks for specified bounding boxes using MedSAM.
        
        Args:
            image_path: Path or URL to the CXR image
            bounding_boxes: List of bounding box coordinates [x1, y1, x2, y2]
            finding_texts: Optional list of text descriptions for each box
            output_path: Optional path to save the combined mask
            
        Returns:
            Dict with segmentation results and mask paths
        """
        logger.info(f"MedSAMSegmentationTool called with image_path: {image_path}")
        logger.info(f"Number of bounding boxes: {len(bounding_boxes)}")
        
        try:
            # Load the model if not already loaded
            self._load_model()
            
            # Load the input image
            try:
                input_image = self._load_image(image_path)
                logger.info(f"Successfully loaded image from: {image_path}")
            except Exception as e:
                error_msg = f"Failed to load image from {image_path}: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg}
            
            # Validate bounding boxes
            if not bounding_boxes or len(bounding_boxes) == 0:
                return {"error": "No bounding boxes provided"}
            
            # Initialize results
            segmentation_results = []
            combined_mask = np.zeros((input_image.size[1], input_image.size[0]), dtype=np.uint8)
            
            # Process each bounding box
            for i, box in enumerate(bounding_boxes):
                if len(box) != 4:
                    logger.warning(f"Invalid bounding box at index {i}: {box}")
                    continue
                
                finding_text = finding_texts[i] if finding_texts and i < len(finding_texts) else f"Finding {i+1}"
                logger.info(f"Segmenting region {i+1}: {finding_text}")
                
                # Generate segmentation mask for this box
                mask = self._segment_region(input_image, box)
                
                # Add to combined mask
                combined_mask = np.maximum(combined_mask, mask)
                
                # Save individual mask if needed
                individual_mask_path = None
                if output_path:
                    base_dir = os.path.dirname(output_path) if output_path else "statics/output_segmentations"
                    os.makedirs(base_dir, exist_ok=True)
                    
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_name = os.path.splitext(os.path.basename(image_path))[0] if not image_path.startswith(('http://', 'https://')) else "image"
                    individual_mask_path = os.path.join(base_dir, f"mask_{img_name}_{finding_text.replace(' ', '_')}_{timestamp}.png")
                    
                    # Save individual mask
                    mask_image = Image.fromarray(mask, mode='L')
                    mask_image.save(individual_mask_path)
                    logger.info(f"Saved individual mask to: {individual_mask_path}")
                
                segmentation_results.append({
                    "finding_text": finding_text,
                    "bounding_box": box,
                    "mask_path": individual_mask_path,
                    "mask_area": np.sum(mask > 0)
                })
            
            # Save combined mask
            combined_mask_path = None
            if output_path:
                combined_mask_path = output_path
            else:
                base_dir = "statics/output_segmentations"
                os.makedirs(base_dir, exist_ok=True)
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                img_name = os.path.splitext(os.path.basename(image_path))[0] if not image_path.startswith(('http://', 'https://')) else "image"
                combined_mask_path = os.path.join(base_dir, f"combined_mask_{img_name}_{timestamp}.png")
            
            # Save combined mask
            combined_mask_image = Image.fromarray(combined_mask, mode='L')
            combined_mask_image.save(combined_mask_path)
            logger.info(f"Saved combined mask to: {combined_mask_path}")
            
            return {
                "image_path": image_path,
                "combined_mask_path": combined_mask_path,
                "individual_results": segmentation_results,
                "total_findings": len(segmentation_results),
                "combined_mask_area": np.sum(combined_mask > 0)
            }
            
        except Exception as e:
            logger.error(f"Error in MedSAMSegmentationTool: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)} 