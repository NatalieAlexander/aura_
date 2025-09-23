from typing import Optional, Any, Dict, List, Union, Tuple
from smolagents import Tool
from PIL import Image, ImageDraw, ImageOps
import os, requests, logging, torch
from io import BytesIO
import numpy as np
from transformers import AutoModel, AutoTokenizer
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DiffusionPipeline
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class RadEditTool(Tool):
    name = "generate_counterfactual"
    description = "Generates a counterfactual version of a chest X-ray by editing specific regions based on a prompt. Use this to visualize how the image would look with or without specific findings."
    inputs = {
        "image_path": {"type": "string", "description": "Path or URL to the CXR image file"},
        "prompt": {"type": "string", "description": "Text describing the desired edit (e.g., 'No pleural effusion' or 'Add pneumothorax')"},
        "findings": {"type": "object", "description": "List of findings with text and box coordinates to use as edit masks", "nullable": True},
        "finding_index": {"type": "integer", "description": "Index of the specific finding to edit (if multiple findings are present)", "nullable": True},
        "anatomical_mask": {"type": "object", "description": "PIL Image mask from anatomical segmentation for ADD operations", "nullable": True},
        "invert_prompt": {"type": "string", "description": "Optional negative prompt to guide the generation away from certain features", "nullable": True},
        "output_path": {"type": "string", "description": "Path where the output image should be saved (optional)", "nullable": True},
        "output_prefix": {"type": "string", "description": "Prefix for output filenames to ensure consistent naming with difference maps (e.g., 'sex_variation_1')", "nullable": True},
        "session_path": {"type": "string", "description": "Session directory path for organized file storage (optional)", "nullable": True},
        "use_medsam": {"type": "boolean", "description": "Whether to use MedSAM for precise segmentation masks (default: True)", "nullable": True},
        "weights": {"type": "number", "description": "Guidance scale for the diffusion process (default: 7.5, higher values follow prompt more closely)", "nullable": True},
        "num_inference_steps": {"type": "integer", "description": "Number of diffusion inference steps (default: 100, more steps = better quality but slower)", "nullable": True},
        "skip_ratio": {"type": "number", "description": "Skip ratio for the diffusion process (default: 0.3, controls noise injection)", "nullable": True}
    }
    output_type = "object"
    
    def __init__(self):
        """Initialize the RadEdit tool."""
        super().__init__()
        self.model_loaded = False
        self.generation_pipeline = None
        self.radedit_pipeline = None
        self.medsam_tool = None

    def _load_model(self):
        """Load the RadEdit model and pipeline components."""
        if self.model_loaded:
            return

        try:
            logger.info("Loading RadEdit model components...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            def load_with_fallback(load_func, model_name, **kwargs):
                try:
                    logger.info(f"Trying to load {model_name} from cache...")
                    return load_func(**kwargs, local_files_only=True)
                except Exception as cache_error:
                    logger.warning(f"Cache loading failed for {model_name}: {cache_error}")
                    logger.info(f"Downloading {model_name} from HuggingFace...")
                    try:
                        kwargs.pop('local_files_only', None)  # Remove local_files_only if present
                        return load_func(**kwargs, local_files_only=False)
                    except Exception as download_error:
                        logger.error(f"Both cache and download failed for {model_name}")
                        raise download_error
            
            # Load the UNet model
            logger.info("Loading UNet...")
            unet_loaded = load_with_fallback(
                UNet2DConditionModel.from_pretrained,
                "RadEdit UNet",
                pretrained_model_name_or_path="microsoft/radedit",
                subfolder="unet",
                use_safetensors=True
            )
            
            # Load VAE
            logger.info("Loading VAE...")
            vae = load_with_fallback(
                AutoencoderKL.from_pretrained,
                "SDXL VAE",
                pretrained_model_name_or_path="stabilityai/sdxl-vae",
                use_safetensors=True
            )
            
            # Load text encoder
            logger.info("Loading text encoder...")
            text_encoder = load_with_fallback(
                AutoModel.from_pretrained,
                "BiomedVLP Text Encoder",
                pretrained_model_name_or_path="microsoft/BiomedVLP-BioViL-T",
                trust_remote_code=True
            )
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            tokenizer = load_with_fallback(
                AutoTokenizer.from_pretrained,
                "BiomedVLP Tokenizer",
                pretrained_model_name_or_path="microsoft/BiomedVLP-BioViL-T",
                model_max_length=128,
                trust_remote_code=True
            )
            
            scheduler = DDIMScheduler(
                beta_schedule="linear",
                clip_sample=False,
                prediction_type="epsilon",
                timestep_spacing="trailing",
                steps_offset=1,
            )
            
            # Create the generation pipeline
            logger.info("Creating StableDiffusionPipeline...")
            self.generation_pipeline = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet_loaded,
                scheduler=scheduler,
                safety_checker=None,
                requires_safety_checker=False,
                feature_extractor=None,
            )
            self.generation_pipeline.to(device)
            
            # Convert to RadEdit pipeline
            logger.info("Converting to RadEdit pipeline...")
            self.radedit_pipeline = DiffusionPipeline.from_pipe(
                self.generation_pipeline,
                custom_pipeline="microsoft/radedit",
            )
            
            self.model_loaded = True
            logger.info("✅ RadEdit model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading RadEdit model: {e}")
            logger.error("This could be due to network issues or missing model files.")
            logger.error("If you're in an offline environment, please run: python setup_cache_env.py --download-models")
            raise RuntimeError(f"Failed to load RadEdit model: {e}")

    def _load_medsam_tool(self):
        """Load the MedSAM tool for precise segmentation."""
        if self.medsam_tool is None:
            try:
                from .medsam_segmentation_tool import MedSAMSegmentationTool
                self.medsam_tool = MedSAMSegmentationTool()
                logger.info("MedSAM tool loaded for precise segmentation")
            except Exception as e:
                logger.warning(f"Could not load MedSAM tool: {e}. Will use basic rectangular masks.")
                self.medsam_tool = None

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
            
        original_size = image.size
        logger.info(f"Original image size: {original_size}")
        
        # Resize to 512x512 for the model (as per RadEdit documentation)
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512)
        ])
        transformed_image = transform(image)
        logger.info(f"Transformed image size: {transformed_image.size}")
        
        return transformed_image

    def _create_mask_from_box(self, box: List[float], image_size: Tuple[int, int]) -> Image.Image:
        """Create a binary mask from a bounding box."""
        width, height = image_size
        
        # Convert normalized coordinates to pixel coordinates if needed
        if all(0 <= coord <= 1 for coord in box):
            x0, y0, x1, y1 = [
                int(box[0] * width),
                int(box[1] * height),
                int(box[2] * width),
                int(box[3] * height)
            ]
        else:
            x0, y0, x1, y1 = [int(coord) for coord in box]
        
        x0 = max(0, x0 - 30)
        y0 = max(0, y0 - 30)
        x1 = min(width, x1 + 30)
        y1 = min(height, y1 + 30) 
            
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)
        
        draw.rectangle([x0, y0, x1, y1], fill=255)
        
        return mask

    def _create_medsam_mask(self, image_path: str, findings: List[Dict], finding_index: Optional[int] = None) -> Optional[Image.Image]:
        """Create a precise segmentation mask using MedSAM."""
        try:
            if self.medsam_tool is None:
                return None
            
           
            if finding_index is not None and 0 <= finding_index < len(findings):
                # Use only the specified finding
                target_finding = findings[finding_index]
                bounding_boxes = [target_finding.get('box')]
                finding_texts = [target_finding.get('text', f'Finding {finding_index}')]
                logger.info(f"Using MedSAM for specific finding: {finding_texts[0]}")
            else:
                # Use all findings
                bounding_boxes = [finding.get('box') for finding in findings if finding.get('box')]
                finding_texts = [finding.get('text', f'Finding {i}') for i, finding in enumerate(findings)]
                logger.info(f"Using MedSAM for {len(bounding_boxes)} findings")
            
            valid_boxes = []
            valid_texts = []
            for box, text in zip(bounding_boxes, finding_texts):
                if box and len(box) == 4:
                    valid_boxes.append(box)
                    valid_texts.append(text)
            
            if not valid_boxes:
                logger.warning("No valid bounding boxes found for MedSAM")
                return None
            
            result = self.medsam_tool.forward(
                image_path=image_path,
                bounding_boxes=valid_boxes,
                finding_texts=valid_texts
            )
            
            if "error" in result:
                logger.warning(f"MedSAM segmentation failed: {result['error']}")
                return None
            
            mask_path = result.get('combined_mask_path')
            if mask_path and os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
                logger.info(f"Successfully created MedSAM mask with area: {result.get('combined_mask_area', 'unknown')}")
                return mask
            else:
                logger.warning("MedSAM mask path not found or invalid")
                return None
                
        except Exception as e:
            logger.warning(f"Error creating MedSAM mask: {e}")
            return None

    def _generate_descriptive_prefix(self, prompt: str, findings: Optional[List[Dict]], finding_index: Optional[int], weights: float, use_medsam: bool, mask_creation_method: str) -> str:
        """Generate a descriptive prefix based on the prompt, findings, and configuration."""
        import re
        
        base_prompt = prompt.lower().strip()
        base_prompt = re.sub(r'[^a-z0-9\s]', '', base_prompt)  # Remove special chars
        base_prompt = re.sub(r'\s+', '_', base_prompt)  # Replace spaces with underscores
        base_prompt = base_prompt[:50]  # Limit length
        
        prefix = base_prompt
        
        if findings and finding_index is not None and 0 <= finding_index < len(findings):
            finding = findings[finding_index]
            finding_text = finding.get('text', '').lower()
            finding_clean = re.sub(r'[^a-z0-9\s]', '', finding_text)
            finding_clean = re.sub(r'\s+', '_', finding_clean)
            finding_clean = finding_clean[:30]  # Limit length
            if finding_clean:
                prefix += f"_{finding_clean}"
        
        prefix += f"_w{weights:.1f}"  
        
        if mask_creation_method != "rectangular":
            prefix += f"_{mask_creation_method}"
        
        prefix = re.sub(r'_+', '_', prefix)
        prefix = prefix.strip('_')
        
        logger.info(f"Generated descriptive prefix: '{prefix}'")
        return prefix

    def forward(self, 
                image_path: str, 
                prompt: str, 
                findings: Optional[List[Dict]] = None,
                finding_index: Optional[int] = None,
                anatomical_mask: Optional[Image.Image] = None,
                invert_prompt: Optional[str] = '',
                output_path: Optional[str] = None,
                output_prefix: Optional[str] = None,
                session_path: Optional[str] = None,
                use_medsam: Optional[bool] = True,
                weights: Optional[float] = 7.5,
                num_inference_steps: Optional[int] = 100,
                skip_ratio: Optional[float] = 0.3) -> Dict:
        """
        Generate a counterfactual version of a chest X-ray.
        
        Args:
            image_path: Path or URL to the CXR image
            prompt: Text describing the desired edit
            findings: Optional list of findings with text and box coordinates
            finding_index: Optional index of specific finding to edit
            anatomical_mask: Optional PIL Image mask from anatomical segmentation (for ADD operations)
            invert_prompt: Optional negative prompt
            output_path: Optional path to save the output image
            output_prefix: Optional prefix for output filename to ensure consistent naming with difference maps
            session_path: Session directory path for organized file storage
            use_medsam: Whether to use MedSAM for precise segmentation (default: True)
            weights: Guidance scale for the diffusion process (default: 7.5, higher values follow prompt more closely)
            num_inference_steps: Number of diffusion inference steps (default: 100, more steps = better quality but slower)
            skip_ratio: Skip ratio for the diffusion process (default: 0.3, controls noise injection)
            
        Returns:
            Dict with paths to original and edited images
        """
        if not prompt.strip():
            logger.error("Empty prompt provided!")
            return {"error": "Empty prompt provided"}
        
        try:
           
            self._load_model()
            
           
            if use_medsam:
                self._load_medsam_tool()
            
           
            try:
                input_image = self._load_image(image_path)
                logger.info(f"Successfully loaded image from: {image_path}")
            except Exception as e:
                error_msg = f"Failed to load image from {image_path}: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg}
            
            edit_mask = None
            mask_creation_method = "none"
            
            # Priority 1: Use anatomical mask if provided (for ADD operations)
            if anatomical_mask is not None:
                edit_mask = anatomical_mask.resize(input_image.size, Image.LANCZOS)
                mask_creation_method = "anatomical_segmentation"
                logger.info("Using provided anatomical segmentation mask for ADD operation")
            
            # Priority 2: Use findings for EDIT/REMOVE operations
            elif findings and len(findings) > 0:
                if use_medsam and self.medsam_tool is not None:
                    edit_mask = self._create_medsam_mask(image_path, findings, finding_index)
                    if edit_mask is not None:
                        mask_creation_method = "medsam"
                        # Resize mask to match the processed image size (512x512)
                        edit_mask = edit_mask.resize(input_image.size, Image.LANCZOS)
                
                # Fallback to rectangular masks if MedSAM failed or not used
                if edit_mask is None:
                    mask_creation_method = "rectangular"
                    if finding_index is not None and 0 <= finding_index < len(findings):
                        # Use a specific finding
                        finding = findings[finding_index]
                        logger.info(f"Using finding at index {finding_index}: {finding.get('text', '')}")
                        box = finding.get('box')
                        if box and len(box) == 4:
                            edit_mask = self._create_mask_from_box(box, input_image.size)
                    else:
                        # Combine all findings
                        edit_mask = Image.new("L", input_image.size, 0)
                        draw = ImageDraw.Draw(edit_mask)
                        
                        for i, finding in enumerate(findings):
                            box = finding.get('box')
                            if box and len(box) == 4:
                                # Convert normalized coordinates to pixel coordinates if needed
                                width, height = input_image.size
                                if all(0 <= coord <= 1 for coord in box):
                                    x0, y0, x1, y1 = [
                                        int(box[0] * width),
                                        int(box[1] * height),
                                        int(box[2] * width),
                                        int(box[3] * height)
                                    ]
                                else:
                                    x0, y0, x1, y1 = [int(coord) for coord in box]
                                    
                                x0 = max(0, x0 - 30)
                                y0 = max(0, y0 - 30)
                                x1 = min(width, x1 + 30)
                                y1 = min(height, y1 + 30)
                                    
                                draw.rectangle([x0, y0, x1, y1], fill=255)
                                logger.info(f"Added finding {i} to edit mask: {finding.get('text', '')}")
            
            if edit_mask is None:
                logger.info("No specific findings to edit, using the whole image")
                edit_mask = Image.new("L", input_image.size, 255)
                mask_creation_method = "full_image"
            
            keep_mask = ImageOps.invert(edit_mask)
            
            if output_path is None:   
                if session_path:
                    base_dir = os.path.join(session_path, "counterfactuals")
                else:
                    base_dir = "statics/output_counterfactuals"
                os.makedirs(base_dir, exist_ok=True)
                
                if output_prefix:
                    descriptive_prefix = output_prefix
                    logger.info(f"🔧 NAMING: Using explicit output_prefix: '{output_prefix}'")
                else:
                    descriptive_prefix = self._generate_descriptive_prefix(prompt, findings, finding_index, weights, use_medsam, mask_creation_method)
                    logger.info(f"🔧 NAMING: Generated descriptive prefix: '{descriptive_prefix}'")
                
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                output_path = os.path.join(base_dir, f"{descriptive_prefix}_{timestamp}.png")
            
            if "counterfactual_" in output_path:
                transformed_input_path = output_path.replace("counterfactual_", "transformed_input_")
            else:
                base_name = os.path.splitext(output_path)[0]
                ext = os.path.splitext(output_path)[1]
                transformed_input_path = f"{base_name}_input{ext}"
            
            input_image.save(transformed_input_path)
            logger.info(f"Saved transformed input image to {transformed_input_path}")
            
            original_image_backup_path = image_path.replace(".png", "_original_backup.png").replace(".jpg", "_original_backup.jpg")
            try:
                if not os.path.exists(original_image_backup_path):
                    import shutil
                    shutil.copy2(image_path, original_image_backup_path)
                    logger.info(f"Created backup of original image: {original_image_backup_path}")
                
                input_image.save(image_path)
                logger.info(f"🔧 SAFETY: Replaced original image with 512x512 transformed version: {image_path}")
            except Exception as e:
                logger.warning(f"Could not replace original image with transformed version: {e}")
            
            logger.info(f"Running RadEdit with prompt: '{prompt}' using {mask_creation_method} mask")
            logger.info(f"Parameters: weights={weights}, num_inference_steps={num_inference_steps}, skip_ratio={skip_ratio}")
            with torch.no_grad():
                torch.manual_seed(42)
                 
                edited_image = self.radedit_pipeline(
                    prompt,  
                    weights=[weights],
                    image=input_image,
                    edit_mask=edit_mask,
                    keep_mask=keep_mask,
                    num_inference_steps=num_inference_steps,  
                    invert_prompt=invert_prompt,  
                    skip_ratio=skip_ratio,  
                    output_type="pil",
                )[0]
            
            edited_image.save(output_path)
            logger.info(f"Saved counterfactual image to {output_path}")
            
            base_name = os.path.splitext(output_path)[0]
            ext = os.path.splitext(output_path)[1]
            mask_path = f"{base_name}_mask{ext}"
            edit_mask.save(mask_path)
            
            return {
                "original_image_path": image_path,  
                "transformed_input_path": transformed_input_path,  
                "counterfactual_image_path": output_path,
                "edit_mask_path": mask_path,
                "prompt": prompt,
                "mask_creation_method": mask_creation_method,
                "used_medsam": mask_creation_method == "medsam",
                "original_backup_path": original_image_backup_path  
            }
            
        except Exception as e:
            logger.error(f"Error in RadEditTool: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)} 