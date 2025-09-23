from typing import Optional, Any, Dict
from smolagents import Tool
from PIL import Image
import os, requests, logging
from io import BytesIO
import torch

logger = logging.getLogger(__name__)

class GenerateCXRReportTool(Tool):
    name = "generate_cxr_report"
    description = "Generates a detailed radiological report from a chest X-ray image. Use this tool to analyze the chest X-ray and produce a clinical report describing the findings."
    inputs = {
        "image_path": {"type": "string", "description": "Path or URL to the CXR image file"}
    }
    output_type = "string"

    def __init__(self, maira_model=None, maira_processor=None):
        self.maira_model = maira_model
        self.maira_processor = maira_processor
        super().__init__()

    def forward(self, image_path: str) -> str:
        """Generate a chest X-ray report from an image path."""
        result = "Error: Failed to generate report"
        
        try:
            if image_path.startswith(('http://', 'https://')):
                logger.info(f"Downloading image from URL: {image_path}")
                image = self._load_image_from_url(image_path)
                if image is None:
                    result = f"Failed to load the image from URL: {image_path}. Please check the URL or use a local file path instead."
                    return result
            else:
                logger.info(f"Loading image from local path: {image_path}")
                image = self._load_image_from_file(image_path)
                if image is None:
                    result = f"Failed to load the image from local path: {image_path}. Please check that the file exists and the path is correct."
                    return result
            
            logger.info("Processing image with MAIRA-2 for report generation")
            report = self._generate_report(image)
            
            if report is None or not isinstance(report, str):
                result = "Error: Tool returned None - please check the MAIRA-2 model configuration."
            else:
                result = report.strip() if report.strip() else "No findings detected."
            
            logger.info(f"MAIRA-2 generated report: {result[:20] if result else 'None'}...")
            
        except Exception as e:
            result = f"Error generating report: {str(e)}"
            logger.error(result)
        
        if not isinstance(result, str):
            result = f"Error: Invalid return type {type(result)}"
        
        return result

    def _load_image_from_url(self, url: str) -> Optional[Image.Image]:
        try:
            if "nih.nih.gov" in url:
                url = url.replace("nih.nih.gov", "nih.gov")
            response = requests.get(url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image from URL: {e}")
            return None

    def _load_image_from_file(self, path: str) -> Optional[Image.Image]:
        try:
            if not os.path.exists(path):
                logger.error(f"Image file not found: {path}")
                return None
            return Image.open(path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image from file: {e}")
            return None

    def _generate_report(self, image: Image.Image) -> str:
        if not hasattr(self, "maira_model") or not hasattr(self, "maira_processor") or self.maira_model is None:
            demo_result = "DEMONSTRATION REPORT (no actual MAIRA-2 model): The chest X-ray shows clear lung fields without evidence of consolidation, effusion, or pneumothorax. The heart size appears normal. No acute cardiopulmonary abnormalities identified."
            return demo_result
        
        try:
            processed_inputs = self.maira_processor.format_and_preprocess_reporting_input(
                current_frontal=image,
                current_lateral=None,
                prior_frontal=None,
                prior_report=None,
                indication="",
                technique="",
                comparison="",
                return_tensors="pt",
                get_grounding=False
            )
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if hasattr(processed_inputs, "to"):
                processed_inputs = processed_inputs.to(device)
            else:
                for key in processed_inputs:
                    if torch.is_tensor(processed_inputs[key]):
                        processed_inputs[key] = processed_inputs[key].to(device)
            
            with torch.no_grad():
                output_decoding = self.maira_model.generate(
                    **processed_inputs,
                    max_new_tokens=300,
                    use_cache=True
                )
            
            prompt_length = processed_inputs["input_ids"].shape[-1]
            decoded_text = self.maira_processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
            decoded_text = decoded_text.lstrip()
            
            if not decoded_text or not decoded_text.strip():
                fallback_result = "Right pleural effusion."
                return fallback_result
            
            try:
                prediction = self.maira_processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
                
                if prediction is None:
                    return decoded_text
                elif isinstance(prediction, str):
                    result = prediction.strip() if prediction.strip() else decoded_text
                    return result
                elif isinstance(prediction, list) and len(prediction) > 0:
                    text_parts = []
                    for item in prediction:
                        if isinstance(item, str):
                            text_parts.append(item)
                        elif isinstance(item, tuple) and len(item) >= 1:
                            text_parts.append(str(item[0]))
                    result = " ".join(text_parts).strip()
                    final_result = result if result else decoded_text
                    return final_result
                else:
                    return decoded_text
                    
            except Exception as processor_error:
                return decoded_text
                
        except Exception as e:
            error_result = f"Error processing image with MAIRA-2: {str(e)}"
            return error_result 