"""
CheXagent VQA Tool for Visual Question Answering on chest X-ray images.
This tool uses Stanford AIMI's CheXagent-8b model for answering questions about demographics,
abnormalities, and clinical findings in chest X-ray images.
"""

import torch
import logging
import os
import io
from PIL import Image
from typing import Dict, Any, Optional, Union
from smolagents import Tool
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

logger = logging.getLogger(__name__)

class CheXagentVQATool(Tool):
    """
    Tool for Visual Question Answering using CheXagent-8b model.
    Can answer questions about demographics, abnormalities, and clinical findings.
    """
    
    name = "chexagent_vqa"
    description = """
    Answers visual questions about chest X-ray images using CheXagent-8b model.
    Can provide information about:
    - Demographics (age, sex, race)
    - Clinical findings and abnormalities
    - Anatomical descriptions
    - Disease characteristics
    
    Args:
        image_path (str): Path to the chest X-ray image
        question (str): Question to ask about the image
        
    Returns:
        dict: Contains the question, answer, and metadata
    """
    
    inputs = {
        "image_path": {
            "type": "string", 
            "description": "Path to the chest X-ray image file"
        },
        "question": {
            "type": "string", 
            "description": "Question to ask about the image (e.g., 'What is the age of this patient?', 'What abnormalities are visible?')"
        }
    }
    
    output_type = "string"
    
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = None
        self.processor = None
        self.generation_config = None
        
        # Initialize model on first use (lazy loading)
        self._model_loaded = False
        
    def _load_model(self):
        """Load CheXagent model components."""
        if self._model_loaded:
            return
            
        try:
            logger.info("Loading CheXagent-8b model for VQA...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                "StanfordAIMI/CheXagent-8b", 
                trust_remote_code=True
            )
            
            # Load generation config
            self.generation_config = GenerationConfig.from_pretrained(
                "StanfordAIMI/CheXagent-8b"
            )
            
            # Load model with device mapping
            self.model = AutoModelForCausalLM.from_pretrained(
                "StanfordAIMI/CheXagent-8b",
                torch_dtype=self.dtype,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            logger.info(f"CheXagent model loaded successfully on {self.device}")
            self._model_loaded = True
            
        except Exception as e:
            logger.error(f"Error loading CheXagent model: {e}")
            raise RuntimeError(f"Failed to load CheXagent model: {e}")
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image."""
        try:
            if image_path.startswith('http'):
                # Handle URL
                import requests
                response = requests.get(image_path)
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
            else:
                # Handle local file
                image = Image.open(image_path).convert("RGB")
            
            logger.info(f"Successfully loaded image from: {image_path}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image from {image_path}: {e}")
            raise ValueError(f"Failed to load image: {e}")
    
    def _format_prompt(self, question: str) -> str:
        """Format the question according to CheXagent's expected format."""
        # CheXagent expects: USER: <s>{prompt} ASSISTANT: <s>
        formatted_prompt = f"USER: <s>{question} ASSISTANT: <s>"
        return formatted_prompt
    
    def _generate_answer(self, image: Image.Image, question: str) -> str:
        """Generate answer using CheXagent model."""
        try:
            # Format the prompt
            prompt = self._format_prompt(question)
            
            # Process inputs
            inputs = self.processor(
                images=[image], 
                text=prompt, 
                return_tensors="pt"
            ).to(device=self.device, dtype=self.dtype)
            
            with torch.no_grad():
                output = self.model.generate(**inputs, generation_config=self.generation_config)[0]
            
            response = self.processor.tokenizer.decode(output, skip_special_tokens=True)
            
            if "ASSISTANT:" in response:
                answer = response.split("ASSISTANT:")[-1].strip()
                # Remove any remaining special tokens
                answer = answer.replace("<s>", "").replace("</s>", "").strip()
            else:
                answer = response.strip()
            
            logger.info(f"Generated answer: {answer[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise RuntimeError(f"Failed to generate answer: {e}")
    
    def forward(self, image_path: str, question: str) -> str:
        """
        Main forward function for VQA.
        
        Args:
            image_path: Path to the chest X-ray image
            question: Question to ask about the image
            
        Returns:
            str: JSON string containing the VQA result
        """
        try:
            logger.info(f"CheXagent VQA called with question: {question}")
            
            # Lazy load model
            if not self._model_loaded:
                self._load_model()
            
            image = self._load_image(image_path)
            
            answer = self._generate_answer(image, question)
            
            result = {
                "question": question,
                "answer": answer,
                "image_path": image_path,
                "model": "CheXagent-8b",
                "status": "success"
            }
            
            logger.info(f"VQA completed successfully: {question} -> {answer[:50]}...")
            
            import json
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"Error in CheXagent VQA: {e}")
            error_result = {
                "question": question,
                "answer": f"Error: {str(e)}",
                "image_path": image_path,
                "model": "CheXagent-8b",
                "status": "error"
            }
            import json
            return json.dumps(error_result)

    def __call__(self, image_path: str, question: str) -> str:
        """Callable interface for the tool."""
        return self.forward(image_path, question) 