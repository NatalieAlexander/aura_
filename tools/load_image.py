from typing import Optional, Union
from PIL import Image, UnidentifiedImageError
import os, requests, logging, traceback
from io import BytesIO
import numpy as np

logger = logging.getLogger(__name__)

def load_image(image_path_or_url: str) -> Union[Image.Image, None]:
    """
    Load an image from a file path or URL.
    Args:
        image_path_or_url: Local file path or URL to the image
    Returns:
        PIL Image object or None if loading fails
    """
    if not image_path_or_url:
        logger.error(f"Empty image path provided")
        return None
        
    try:
        if image_path_or_url.startswith(('http://', 'https://')):
            try:
                logger.info(f"Downloading image from URL: {image_path_or_url}")
                response = requests.get(image_path_or_url)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                if image.mode != "L":
                    image = image.convert("RGB")
                logger.info(f"Successfully loaded image from URL: {image_path_or_url} (size: {image.size}, mode: {image.mode})")
                return image
            except requests.RequestException as e:
                logger.error(f"Error downloading image from URL: {image_path_or_url}, error: {e}")
                return None
            except UnidentifiedImageError as e:
                logger.error(f"Invalid image format from URL: {image_path_or_url}, error: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error loading image from URL: {image_path_or_url}, error: {e}")
                logger.debug(traceback.format_exc())
                return None
        else:
            try:
                logger.info(f"Loading image from local path: {image_path_or_url}")
                if not os.path.exists(image_path_or_url):
                    logger.error(f"Image file not found: {image_path_or_url}")
                    return None
                
                image = Image.open(image_path_or_url)
                
                if "_mask" in image_path_or_url or image_path_or_url.endswith(".mask.png"):
                    if image.mode != "L":
                        image = image.convert("L")
                elif image.mode != "RGB":
                    image = image.convert("RGB")
                
                logger.info(f"Successfully loaded image from: {image_path_or_url} (size: {image.size}, mode: {image.mode})")
                return image
            except UnidentifiedImageError as e:
                logger.error(f"Invalid image format: {image_path_or_url}, error: {e}")
                return None
            except Exception as e:
                logger.error(f"Error loading image from path: {image_path_or_url}, error: {e}")
                logger.debug(traceback.format_exc())
                return None
    except Exception as e:
        logger.error(f"Unexpected error in load_image: {e}")
        logger.debug(traceback.format_exc())
        return None 