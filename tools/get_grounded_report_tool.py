from typing import Optional, Any, Dict, Tuple, List
from smolagents import Tool
from PIL import Image
import os, requests, logging
from io import BytesIO
import torch

logger = logging.getLogger(__name__)

class GetGroundedReportTool(Tool):
    name = "get_grounded_report"
    description = "Generates a complete radiological report with visual groundings from a chest X-ray image. This combines report generation and visual grounding in one step."
    inputs = {
        "image_path": {"type": "string", "description": "Path or URL to the CXR image file"}
    }
    output_type = "object"

    def __init__(self, maira_model=None, maira_processor=None):
        self.maira_model = maira_model
        self.maira_processor = maira_processor
        super().__init__()

    def forward(self, image_path: str) -> dict:
        logger.info(f"GetGroundedReportTool called with image_path: {image_path}")
        try:
            if image_path.startswith(('http://', 'https://')):
                logger.info(f"Downloading image from URL: {image_path}")
                image = self._load_image_from_url(image_path)
                if image is None:
                    return {"error": "Failed to load the image from URL. Please check the URL or use a local file path instead."}
            else:
                logger.info(f"Loading image from local path: {image_path}")
                image = self._load_image_from_file(image_path)
                if image is None:
                    return {"error": f"Failed to load the image from local path: {image_path}. Please check that the file exists and the path is correct."}
            logger.info("Processing image with MAIRA-2 for grounded report generation")
            report, groundings = self._generate_grounded_report(image)
            logger.info(f"MAIRA-2 generated grounded report with {len(groundings)} groundings")
            result = {"report": report, "groundings": groundings}
            return result
        except Exception as e:
            error_msg = f"Error generating grounded report: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

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

    def _generate_grounded_report(self, image: Image.Image) -> Tuple[str, List[Dict]]:
        if not hasattr(self, "maira_model") or not hasattr(self, "maira_processor") or self.maira_model is None:
            return ("DEMONSTRATION REPORT (no actual MAIRA-2 model): The chest X-ray shows clear lung fields without evidence of consolidation, effusion, or pneumothorax.", 
                   [{"text": "Clear lung fields", "box": [100, 100, 300, 300]}, {"text": "Normal heart size", "box": [200, 200, 400, 400]}])
        original_width, original_height = image.size
        processed_inputs = self.maira_processor.format_and_preprocess_reporting_input(
            current_frontal=image,
            current_lateral=None,
            prior_frontal=None,
            prior_report=None,
            indication="",
            technique="",
            comparison="",
            return_tensors="pt",
            get_grounding=True
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
                max_new_tokens=450,
                use_cache=True
            )
        prompt_length = processed_inputs["input_ids"].shape[-1]
        decoded_text = self.maira_processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
        decoded_text = decoded_text.lstrip()
        prediction = self.maira_processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
        report_text = ""
        groundings = []
        for item in prediction:
            if isinstance(item, tuple) and len(item) == 2:
                text, boxes = item
                report_text += text + " "
                if boxes:
                    if isinstance(boxes, list) and len(boxes) > 0:
                        if isinstance(boxes[0], tuple):
                            box_values = list(boxes[0])
                        else:
                            box_values = boxes[0] if isinstance(boxes, list) and boxes else boxes
                        adjusted_box = self.maira_processor.adjust_box_for_original_image_size(
                            box_values,
                            width=original_width, 
                            height=original_height
                        )
                        groundings.append({"text": text, "box": list(adjusted_box)})
                    else:
                        box_values = list(boxes) if isinstance(boxes, tuple) else boxes
                        adjusted_box = self.maira_processor.adjust_box_for_original_image_size(
                            box_values,
                            width=original_width, 
                            height=original_height
                        )
                        groundings.append({"text": text, "box": list(adjusted_box)})
            else:
                report_text += str(item) + " "
        return report_text.strip(), groundings 