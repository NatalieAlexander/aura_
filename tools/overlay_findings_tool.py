from typing import Optional, Any, Dict, List
from smolagents import Tool
from PIL import Image, ImageDraw, ImageFont
import os, requests, logging
from io import BytesIO

logger = logging.getLogger(__name__)

class OverlayFindingsTool(Tool):
    name = "overlay_findings"
    description = "Overlays the visual findings (bounding boxes) on the chest X-ray image and saves the result. Use this to create a visualization of where abnormalities are located in the image."
    inputs = {
        "image_path": {"type": "string", "description": "Path or URL to the CXR image file"},
        "findings": {"type": "object", "description": "List of findings with text and box coordinates"},
        "output_path": {"type": "string", "description": "Path where the output image should be saved (optional)", "nullable": True},
        "max_findings": {"type": "integer", "description": "Maximum number of findings to display (optional, displays all if not specified)", "nullable": True}
    }
    output_type = "string"

    def forward(self, image_path: str, findings: Any, output_path: Optional[str] = None, max_findings: Optional[int] = None) -> str:
        logger.info(f"OverlayFindingsTool called with image_path: {image_path}, findings: {findings[:3] if findings else []}")
        try:
            if isinstance(findings, dict) and 'groundings' in findings:
                findings = findings['groundings']
            if image_path.startswith(('http://', 'https://')):
                logger.info(f"Downloading image from URL: {image_path}")
                image = self._load_image_from_url(image_path)
                if image is None:
                    print(f"ERROR: Could not download image from URL: {image_path}")
                    print("HINT: Try using a local file path instead, e.g., ./CXR145_IM-0290-1001.png")
                    return "Failed to load the image from URL. Please check the URL or use a local file path instead."
            else:
                logger.info(f"Loading image from local path: {image_path}")
                image = self._load_image_from_file(image_path)
                if image is None:
                    print(f"ERROR: Could not load image from local path: {image_path}")
                    print("HINT: Check that the file exists and the path is correct")
                    return f"Failed to load the image from local path: {image_path}. Please check that the file exists and the path is correct."
            if output_path is None:
                base_dir = "output_groundings"
                os.makedirs(base_dir, exist_ok=True)
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                img_name = os.path.basename(image_path) if not image_path.startswith(('http://', 'https://')) else "image"
                output_path = os.path.join(base_dir, f"grounded_{img_name}_{timestamp}.png")
            if max_findings is not None and isinstance(findings, list):
                findings = findings[:max_findings]
            image_with_boxes = self._overlay_findings(image, findings)
            image_with_boxes.save(output_path)
            logger.info(f"Saved overlay image to {output_path}")
            print(f"OVERLAY IMAGE SAVED: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error in OverlayFindingsTool: {e}")
            print(f"ERROR: {e}")
            return f"Error overlaying findings: {e}"

    def _load_image_from_url(self, url: str) -> Optional[Image.Image]:
        try:
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

    def _overlay_findings(self, image: Image.Image, findings: List[Dict]) -> Image.Image:
        image_with_boxes = image.copy()
        draw = ImageDraw.Draw(image_with_boxes)
        
        font = None
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception as e:
            logger.warning(f"Could not load font: {e}")
            font = ImageFont.load_default()
        
        width, height = image.size
        
        colors = {
            "effusion": (255, 0, 0),      # Red
            "pneumothorax": (0, 255, 0),  # Green
            "nodule": (0, 0, 255),        # Blue
            "cardiomegaly": (255, 165, 0), # Orange
            "infiltrate": (128, 0, 128),  # Purple
            "mass": (255, 0, 255),        # Magenta
            "opacity": (0, 255, 255),     # Cyan
            "consolidation": (165, 42, 42), # Brown
            "default": (255, 255, 0)      # Yellow
        }
        
        logger.info(f"Drawing {len(findings)} findings on image of size {width}x{height}")
        valid_findings_count = 0
        
        for i, finding in enumerate(findings):
            if not isinstance(finding, dict) or 'box' not in finding or 'text' not in finding:
                logger.warning(f"Skipping invalid finding at index {i}: {finding}")
                continue
                
            box = finding['box']
            if not box or not isinstance(box, (list, tuple)) or len(box) != 4:
                logger.warning(f"Skipping finding with invalid box format at index {i}: {box}")
                continue
                
            x1, y1, x2, y2 = box
            
            # IMPORTANT: MAIRA-2 groundings are ALWAYS normalized (0-1 range)
            # Force conversion to absolute pixels
            x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
            
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))
            
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Skipping invalid box at index {i}: coordinates result in empty or negative area: [{x1}, {y1}, {x2}, {y2}]")
                continue
            
            color = colors["default"]
            text = finding['text'].lower()
            for key in colors:
                if key in text:
                    color = colors[key]
                    break
            
            box_width = max(3, int(min(width, height) * 0.005))  
            draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
            
            # Draw label
            label = finding['text']
            if len(label) > 25:  
                label = label[:22] + "..."
                
            text_width = len(label) * 8  
            text_height = 20
            text_position = (int(x1), max(0, int(y1 - text_height - 2)))  
            
            draw.rectangle([
                text_position[0], 
                text_position[1], 
                text_position[0] + text_width, 
                text_position[1] + text_height
            ], fill=color)
            
            draw.text(
                text_position, 
                label, 
                fill=(0, 0, 0), 
                font=font
            )
            
            logger.info(f"Successfully drew finding {i+1}: '{label}' at coordinates [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
            valid_findings_count += 1
        
        logger.info(f"Visualization complete: {valid_findings_count} valid findings drawn out of {len(findings)} total")
        return image_with_boxes 