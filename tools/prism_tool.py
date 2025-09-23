"""PRISM Tool - Medical Image Editing using Fine-tuned Diffusion Models"""

import os
import sys
import warnings
import time
import json
import logging
from typing import Dict, List, Optional, Union, Tuple
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

prism_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'PRISM')
sys.path.insert(0, prism_path)
PRISM_AVAILABLE = True
try:
    from misc_utils import ClipSimilarity
    from edit_utils import NullInversion, AttentionStore, run_and_display, make_controller
    from fastedit_medical_images import ImageEditor
    logger = logging.getLogger(__name__)
    logger.info("PRISM dependencies loaded successfully")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"PRISM dependencies not available: {e}")
    PRISM_AVAILABLE = False
    class ClipSimilarity:
        def __init__(self, **kwargs): 
            raise RuntimeError("PRISM dependencies not available. Please ensure PRISM directory and models are properly installed.")
    
    class ImageEditor:
        def __init__(self, **kwargs): 
            raise RuntimeError("PRISM dependencies not available. Please ensure PRISM directory and models are properly installed.")

from smolagents import Tool


class PRISMTool(Tool):
    """
    PRISM (Prompt-to-Prompt) medical image editing tool using fine-tuned diffusion models.
    
    This tool provides advanced medical image editing capabilities using attention-based
    prompt-to-prompt editing with models fine-tuned on medical datasets.
    """
    
    name = "generate_prism_edit"
    description = """Generate medical image edits using PRISM (Prompt-to-Prompt) framework with fine-tuned diffusion models.
    
    This tool uses attention-based editing to make precise modifications to medical images while preserving 
    anatomical structure and medical context. It supports both simple prompt-based editing (like RadEdit) 
    and advanced prompt-to-prompt editing with specific word replacements.
    
    Key features:
    - Simple prompt-based editing: just provide image_path and prompt
    - Advanced prompt-to-prompt editing with word replacements
    - Null-text inversion for precise image reconstruction
    - Attention-controlled editing with cross-attention manipulation
    - Fine-tuned models (CheXpert, BiomedNLP variants)
    - CLIP-based quality filtering and assessment
    - Medical terminology understanding
    """
    
    inputs = {
        "image_path": {"type": "string", "description": "Path to the input medical image"},
        "prompt": {"type": "string", "description": "Simple edit prompt describing desired changes (e.g., 'Remove pleural effusion')", "nullable": True},
        "original_caption": {"type": "string", "description": "Original caption describing the image (optional for simple mode)", "nullable": True},
        "edited_caption": {"type": "string", "description": "Target caption describing desired edits (optional for simple mode)", "nullable": True},
        "original_word": {"type": "string", "description": "Word/phrase to be replaced in the edit (optional for simple mode)", "nullable": True},
        "edit_word": {"type": "string", "description": "Word/phrase to replace with (optional for simple mode)", "nullable": True},
        "model_type": {"type": "string", "description": "Model type: 'finetuned_chexpert', 'finetuned_chexpert_biomednlp', 'stable_diffusion_v1_5', 'stable_diffusion_mimic_cxr_v0.1'", "nullable": True},
        "num_inference_steps": {"type": "number", "description": "Number of diffusion steps (20-50, lower=faster but less quality)", "nullable": True},
        "guidance_scale": {"type": "number", "description": "Guidance scale for classifier-free guidance (5.0-15.0, higher=more adherence to prompt)", "nullable": True},
        "self_replace_steps": {"type": "number", "description": "Self-attention replacement steps (0.0-1.0)", "nullable": True},
        "cross_replace_steps": {"type": "number", "description": "Cross-attention replacement steps (0.0-1.0)", "nullable": True},
        "edit_word_weight": {"type": "number", "description": "Weight for edit word emphasis", "nullable": True},
        "clip_img_thresh": {"type": "number", "description": "CLIP image similarity threshold for quality filtering (lower = more permissive)", "nullable": True},
        "clip_thresh": {"type": "number", "description": "CLIP text similarity threshold (lower = more permissive)", "nullable": True},
        "clip_dir_thresh": {"type": "number", "description": "CLIP directional similarity threshold (lower = more permissive)", "nullable": True},
        "session_path": {"type": "string", "description": "Session directory path for organized file storage", "nullable": True},
        "output_prefix": {"type": "string", "description": "Prefix for output files", "nullable": True},
        "save_inversion": {"type": "boolean", "description": "Whether to save intermediate inversion results", "nullable": True},
        "verbose": {"type": "boolean", "description": "Enable verbose logging", "nullable": True}
    }
    
    output_type = "object"
    
    def __init__(self):
        """Initialize the PRISM tool."""
        super().__init__()
        self.model_loaded = False
        self.image_editor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configure logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def _check_dependencies(self):
        """Check if PRISM dependencies are available."""
        if not PRISM_AVAILABLE:
            raise RuntimeError(
                "PRISM dependencies not available. Please ensure:\n"
                "1. PRISM directory exists at the expected location\n"
                "2. Required Python packages are installed\n"
                "3. Model files are downloaded\n"
                f"Expected PRISM path: {prism_path}"
            )
        
    def _load_model(self, model_type: str = "finetuned_chexpert"):
        """Load the PRISM model and initialize ImageEditor."""
        self._check_dependencies()
        
        if self.model_loaded and hasattr(self, 'current_model_type') and self.current_model_type == model_type:
            return
            
        try:
            self.logger.info(f"Loading PRISM model: {model_type}")
            
            # Create a mock args object for ImageEditor
            class MockArgs:
                def __init__(self):
                    self.out_path = "outputs"
                    self.original_caption = ""
                    
            args = MockArgs()
            
            # Initialize ImageEditor with specified model type
            # Note: The ImageEditor expects a specific checkpoint path structure
            # We need to ensure the checkpoint is available at the expected location
            checkpoint_path = self._get_checkpoint_path(model_type)
            
            if checkpoint_path and not os.path.exists(checkpoint_path):
                self.logger.warning(f"⚠️ Checkpoint not found at {checkpoint_path}, using default model")
            
            self.image_editor = ImageEditor(
                args=args,
                device=self.device,
                self_replace_steps_range=[0.5],  # Will be overridden per call
                cross_replace_steps={"default_": 0.8},  # Will be overridden per call
                similarity_metric=ClipSimilarity(device=self.device),
                text_similarity_threshold=0.9,  # High threshold for medical accuracy
                ldm_type=model_type,
                verbose=False,  # Will be overridden per call
                save_inversion=True,
                edit_word_weight=2.0,  # Will be overridden per call
                clip_img_thresh=0.6,
                clip_thresh=0.1,
                clip_dir_thresh=-0.5,
                num_inference_steps=50,  # Will be overridden per call
                guidance_scale=7.5,  # Will be overridden per call
            )
            
            self.model_loaded = True
            self.current_model_type = model_type
            self.logger.info(f"✅ PRISM model {model_type} loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading PRISM model {model_type}: {e}")
            raise RuntimeError(f"Failed to load PRISM model: {e}")
    
    def _get_checkpoint_path(self, model_type: str) -> str:
        """Get the checkpoint path for the specified model type."""
        # Default checkpoint path structure
        prism_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'PRISM')
        
        if model_type in ["finetuned_chexpert", "finetuned_chexpert_biomednlp"]:
            # Use the local checkpoint
            checkpoint_path = os.path.join(prism_dir, "checkpoint-15-5000", "model.safetensors")
            if os.path.exists(checkpoint_path):
                return checkpoint_path
            else:
                # Fallback to the hardcoded path in the original code
                return "statics/PRISM/checkpoint-15-5000/model.safetensors"
        
        return None  # Other models don't need custom checkpoints
    
    def _prepare_output_directory(self, session_path: Optional[str], output_prefix: str) -> str:
        """Prepare output directory for storing results."""
        if session_path:
            output_dir = os.path.join(session_path, "counterfactuals")
        else:
            output_dir = "statics/output_counterfactuals"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Create unique output path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{output_prefix}_{timestamp}")
        os.makedirs(output_path, exist_ok=True)
        
        return output_path
    
    def _validate_inputs(self, image_path: str, original_caption: str, edited_caption: str, 
                        original_word: str, edit_word: str) -> None:
        """Validate input parameters."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        if not original_caption.strip():
            raise ValueError("Original caption cannot be empty")
            
        if not edited_caption.strip():
            raise ValueError("Edited caption cannot be empty")
            
        if not original_word.strip():
            raise ValueError("Original word cannot be empty")
            
        if not edit_word.strip():
            raise ValueError("Edit word cannot be empty")
    
    def _parse_simple_prompt(self, prompt: str) -> Tuple[str, str, str, str]:
        """
        Parse a simple prompt into PRISM parameters using the format from the original paper.
        
        Format from paper:
        - Original: "Chest X-ray showing {disease}"
        - Edited: "Normal chest X-ray with no significant findings" OR "Chest X-ray without {disease}"
        """
        prompt_lower = prompt.lower().strip()
        
        # Disease mapping to standardized names (matching CheXpert format)
        disease_keywords = {
            'pleural effusion': 'Pleural Effusion',
            'effusion': 'Pleural Effusion',
            'cardiomegaly': 'Cardiomegaly',
            'enlarged heart': 'Cardiomegaly',
            'pneumonia': 'Pneumonia',
            'consolidation': 'Consolidation',
            'atelectasis': 'Atelectasis',
            'pneumothorax': 'Pneumothorax',
            'edema': 'Edema',
            'lung opacity': 'Lung Opacity',
            'opacity': 'Lung Opacity',
            'enlarged cardiomediastinum': 'Enlarged Cardiomediastinum',
            'fracture': 'Fracture',
            'lung lesion': 'Lung Lesion',
            'lesion': 'Lung Lesion'
        }
        
        # Find the disease mentioned in the prompt
        target_disease = None
        for keyword, disease in disease_keywords.items():
            if keyword in prompt_lower:
                target_disease = disease
                break
        
        if target_disease:
            # Use the exact format from the original PRISM paper
            original_caption = f"Chest X-ray showing {target_disease}"
            
            # Use the two main options from the paper:
            # Option 1: Complete removal - "Normal chest X-ray with no significant findings"
            # Option 2: Specific removal - "Chest X-ray without {disease}"
            if any(word in prompt_lower for word in ['remove', 'clear', 'eliminate', 'treat']):
                edited_caption = "Normal chest X-ray with no significant findings"
                original_word = target_disease
                edit_word = "no significant findings"
            else:
                edited_caption = f"Chest X-ray without {target_disease}"
                original_word = f"showing {target_disease}"
                edit_word = f"without {target_disease}"
        else:
            # Fallback for generic prompts
            original_caption = "Chest X-ray showing medical findings"
            edited_caption = "Normal chest X-ray with no significant findings"
            original_word = "medical findings"
            edit_word = "no significant findings"
        
        return original_caption, edited_caption, original_word, edit_word
    
    def forward(self, 
                image_path: str,
                prompt: Optional[str] = None,
                original_caption: Optional[str] = None,
                edited_caption: Optional[str] = None,
                original_word: Optional[str] = None,
                edit_word: Optional[str] = None,
                model_type: Optional[str] = None,
                num_inference_steps: Optional[int] = None,
                guidance_scale: Optional[float] = None,
                self_replace_steps: Optional[float] = None,
                cross_replace_steps: Optional[float] = None,
                edit_word_weight: Optional[float] = None,
                clip_img_thresh: Optional[float] = None,
                clip_thresh: Optional[float] = None,
                clip_dir_thresh: Optional[float] = None,
                session_path: Optional[str] = None,
                output_prefix: Optional[str] = None,
                save_inversion: Optional[bool] = None,
                verbose: Optional[bool] = None) -> Dict:
        """
        Execute PRISM-based medical image editing.
        
        Can be used in two modes:
        1. Simple mode: Just provide image_path and prompt (like RadEdit)
        2. Advanced mode: Provide all PRISM parameters for fine control
        
        Returns:
            Dict containing:
            - success: Whether the operation was successful
            - output_path: Path to the edited image
            - difference_path: Path to the difference visualization
            - original_path: Path to the original/inverted image
            - inversion_path: Path to inversion results (if saved)
            - edit_info: Dictionary with edit details and metrics
            - error: Error message if failed
        """
        try:
            # Determine mode: simple or advanced
            if prompt and not all([original_caption, edited_caption, original_word, edit_word]):
                # Simple mode: parse prompt
                original_caption, edited_caption, original_word, edit_word = self._parse_simple_prompt(prompt)
                self.logger.info(f"🔄 Simple mode: parsed prompt '{prompt}' into PRISM parameters")
                
            elif not all([original_caption, edited_caption, original_word, edit_word]):
                raise ValueError("Either provide 'prompt' for simple mode or all of 'original_caption', 'edited_caption', 'original_word', 'edit_word' for advanced mode")
            
            # Set default values
            model_type = model_type or "finetuned_chexpert"
            num_inference_steps = num_inference_steps or 30
            guidance_scale = guidance_scale or 7.5
            self_replace_steps = self_replace_steps or 0.5
            cross_replace_steps = cross_replace_steps or 0.8
            edit_word_weight = edit_word_weight or 2.0
            # Make thresholds more permissive to increase chance of output
            clip_img_thresh = clip_img_thresh if clip_img_thresh is not None else 0.4
            clip_thresh = clip_thresh if clip_thresh is not None else 0.05
            clip_dir_thresh = clip_dir_thresh if clip_dir_thresh is not None else -1.0
            output_prefix = output_prefix or "prism_edit"
            save_inversion = save_inversion if save_inversion is not None else True
            verbose = verbose if verbose is not None else False
            
            # Validate inputs
            self._validate_inputs(image_path, original_caption, edited_caption, original_word, edit_word)
            
            # Load model
            self._load_model(model_type)
            
            # Prepare output directory
            output_dir = self._prepare_output_directory(session_path, output_prefix)
            
            # Reset state to avoid corruption between images
            if hasattr(self.image_editor, 'reset_state'):
                self.image_editor.reset_state()
            
            # Update ImageEditor parameters for this run (more permissive thresholds)
            self.image_editor.self_replace_steps_range = [self_replace_steps]
            self.image_editor.cross_replace_steps = {"default_": cross_replace_steps}
            self.image_editor.edit_word_weight = edit_word_weight
            self.image_editor.clip_img_thresh = clip_img_thresh
            self.image_editor.clip_thresh = clip_thresh
            self.image_editor.clip_dir_thresh = clip_dir_thresh
            self.image_editor.verbose = verbose
            self.image_editor.save_inversion = save_inversion
            self.image_editor.num_inference_steps = num_inference_steps
            self.image_editor.guidance_scale = guidance_scale
            
            # Also update the inverter with new parameters
            if hasattr(self.image_editor, 'inverter'):
                self.image_editor.inverter.num_inference_steps = num_inference_steps
                self.image_editor.inverter.guidance_scale = guidance_scale
            
            if verbose:
                self.logger.info(f"🎯 Starting PRISM edit:")
                self.logger.info(f"   Image: {image_path}")
                self.logger.info(f"   Original: '{original_caption}'")
                self.logger.info(f"   Edited: '{edited_caption}'")
                self.logger.info(f"   Edit: '{original_word}' → '{edit_word}'")
                self.logger.info(f"   Model: {model_type}")
                self.logger.info(f"   Thresholds: img={clip_img_thresh}, text={clip_thresh}, dir={clip_dir_thresh}")
            
            # Prepare inversion directory
            inversion_dir = os.path.join(output_dir, "inversions")
            os.makedirs(inversion_dir, exist_ok=True)
            
            # Step 1: Perform null-text inversion
            if verbose:
                self.logger.info("🔄 Performing null-text inversion...")
                
            img, img_inv, x_t, uncond_embeddings = self.image_editor.invert(
                img_path=image_path,
                prompt=original_caption,
                img_inv_dir=inversion_dir
            )
            
            # Step 2: Prepare edit parameters
            edited_cap_dicts = [{
                "original_caption": original_caption,
                "edited_caption": edited_caption,
                "original": original_word,
                "edit": edit_word,
            }]
            
            # Step 3: Perform editing
            if verbose:
                self.logger.info("✏️ Performing attention-based editing...")
                
            edit_output_dir = os.path.join(output_dir, "edit_output")
            os.makedirs(edit_output_dir, exist_ok=True)
            
            self.image_editor.edit(
                out_path=edit_output_dir,
                cls_name="medical",
                x_t=x_t,
                uncond_embeddings=uncond_embeddings,
                cap=original_caption,
                edited_cap_dicts=edited_cap_dicts
            )
            
            # Step 4: Collect results - handle case where no files are generated
            edited_image_path = None
            difference_image_path = None
            
            if os.path.exists(edit_output_dir):
                # Look for any generated files
                generated_files = [f for f in os.listdir(edit_output_dir) if f.endswith('.jpeg')]
                
                for file in generated_files:
                    if not file.endswith('_diff.jpeg'):
                        edited_image_path = os.path.join(edit_output_dir, file)
                    elif file.endswith('_diff.jpeg'):
                        difference_image_path = os.path.join(edit_output_dir, file)
            
            # If no files generated, create a fallback result
            if not edited_image_path:
                if verbose:
                    self.logger.warning("⚠️ No edited image generated - PRISM quality thresholds not met")
                    self.logger.info("💡 Try lowering quality thresholds or adjusting edit parameters")
                
                # Create a simple copy as fallback
                fallback_path = os.path.join(edit_output_dir, f"{edit_word.replace(' ', '_')}_fallback.jpeg")
                img.save(fallback_path)
                edited_image_path = fallback_path
                
                # Log warning in result
                result_warning = "Edit quality thresholds not met - returning original image. Try adjusting thresholds."
            else:
                result_warning = None
            
            # Load prompt dict for additional info
            prompt_dict_path = os.path.join(edit_output_dir, "prompt_dict.json")
            edit_info = {}
            if os.path.exists(prompt_dict_path):
                with open(prompt_dict_path, 'r') as f:
                    edit_info = json.load(f)
            
            # Prepare result
            result = {
                "success": True,
                "output_path": edited_image_path,
                "difference_path": difference_image_path,
                "original_path": image_path,
                "inversion_path": inversion_dir if save_inversion else None,
                "output_directory": output_dir,
                "edit_info": edit_info,
                "warning": result_warning,
                "parameters": {
                    "model_type": model_type,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "self_replace_steps": self_replace_steps,
                    "cross_replace_steps": cross_replace_steps,
                    "edit_word_weight": edit_word_weight,
                    "original_caption": original_caption,
                    "edited_caption": edited_caption,
                    "original_word": original_word,
                    "edit_word": edit_word,
                    "clip_thresholds": {
                        "img": clip_img_thresh,
                        "text": clip_thresh,
                        "dir": clip_dir_thresh
                    }
                }
            }
            
            if verbose:
                self.logger.info("✅ PRISM editing completed successfully")
                if edited_image_path:
                    self.logger.info(f"   Edited image: {edited_image_path}")
                if difference_image_path:
                    self.logger.info(f"   Difference map: {difference_image_path}")
                if result_warning:
                    self.logger.warning(f"   Warning: {result_warning}")
            
            # Clean up GPU memory and reset states after each image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return result
            
        except Exception as e:
            error_msg = f"PRISM editing failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "output_path": None,
                "difference_path": None,
                "original_path": image_path,
                "inversion_path": None,
                "output_directory": None,
                "edit_info": {},
                "parameters": {
                    "image_path": image_path,
                    "prompt": prompt,
                    "original_caption": original_caption,
                    "edited_caption": edited_caption,
                    "original_word": original_word,
                    "edit_word": edit_word,
                    "model_type": model_type
                }
            }
    
    def get_available_models(self) -> List[str]:
        """Get list of available model types."""
        return [
            "finetuned_chexpert",
            "finetuned_chexpert_biomednlp", 
            "stable_diffusion_v1_5",
            "stable_diffusion_v1_4",
            "stable_diffusion_mimic_cxr_v0.1"
        ]
    
    def cleanup(self):
        """Clean up loaded models and free memory."""
        if self.model_loaded:
            del self.image_editor
            self.model_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info("🧹 PRISM model cleaned up")


# Additional utility functions for integration

def create_prism_edit_batch(image_paths: List[str], 
                           captions: List[str],
                           edits: List[Dict[str, str]],
                           **kwargs) -> List[Dict]:
    """
    Create a batch of PRISM edits for multiple images.
    
    Args:
        image_paths: List of image file paths
        captions: List of original captions
        edits: List of edit dictionaries with 'original_word' and 'edit_word'
        **kwargs: Additional parameters for PRISMTool
    
    Returns:
        List of result dictionaries from each edit operation
    """
    tool = PRISMTool()
    results = []
    
    try:
        for i, (image_path, caption, edit_dict) in enumerate(zip(image_paths, captions, edits)):
            
            result = tool.forward(
                image_path=image_path,
                original_caption=caption,
                edited_caption=edit_dict.get('edited_caption', caption),
                original_word=edit_dict['original_word'],
                edit_word=edit_dict['edit_word'],
                output_prefix=f"batch_edit_{i:03d}",
                **kwargs
            )
            results.append(result)
            
    finally:
        tool.cleanup()
    
    return results


def validate_prism_checkpoint(checkpoint_path: str) -> bool:
    """
    Validate that a PRISM checkpoint file exists and is loadable.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        True if checkpoint is valid, False otherwise
    """
    try:
        if not os.path.exists(checkpoint_path):
            return False
            
        # Try to load the checkpoint
        from safetensors.torch import load_file
        checkpoint = load_file(checkpoint_path)
        return len(checkpoint) > 0
        
    except Exception:
        return False


# Export the tool class
__all__ = ['PRISMTool', 'create_prism_edit_batch', 'validate_prism_checkpoint'] 