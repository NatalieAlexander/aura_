"""
CXR Analysis Application

This script uses an AI agent to analyze chest X-ray images and display
the results, including the report and visual groundings.
"""

import asyncio
import argparse
import logging
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from agents.cxr_analyst_agent import CXRAnalystAgent
from tools import load_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_findings(image_path_or_url, findings, output_path=None):
    """
    Visualize the findings by drawing bounding boxes on the original image.
    
    Args:
        image_path_or_url: Path or URL to the CXR image
        findings: List of findings with bounding boxes
        output_path: Optional path to save visualization results
    """
    # Load the image
    image = load_image(image_path_or_url)
    draw = ImageDraw.Draw(image)
    
    # Use a default font
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw each finding with its box
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, finding in enumerate(findings):
        box = finding.get("box")
        text = finding.get("text", "")
        
        if box:
            color = colors[i % len(colors)]
            
            # Draw rectangle
            draw.rectangle(box, outline=color, width=3)
            
            # Draw text label
            text_position = (box[0], box[1] - 15) if box[1] > 15 else (box[0], box[3] + 5)
            draw.text(text_position, text[:20], fill=color, font=font)
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    
    # Save if output path is provided
    if output_path:
        image.save(output_path)
        logger.info(f"Visualization saved to {output_path}")
    
    plt.show()

async def run_agent_analysis(image_path_or_url, custom_prompt=None, get_groundings=False):
    """
    Run the CXRAnalystAgent to analyze the image with visible reasoning steps.
    
    Args:
        image_path_or_url: Path or URL to the image
        custom_prompt: Optional custom prompt to guide the analysis
        get_groundings: Whether to request groundings
        
    Returns:
        Analysis results
    """
    logger.info("Initializing CXR Analyst Agent...")
    agent = CXRAnalystAgent()
    
    # Create a prompt that shows what we want the agent to do
    if not custom_prompt:
        if get_groundings:
            prompt = (
                "Please analyze this chest X-ray image carefully. "
                "First, generate a detailed radiology report of your findings. "
                "Then, identify important findings and visually ground them in the image. "
                "Explain your reasoning as you analyze the image."
            )
        else:
            prompt = (
                "Please analyze this chest X-ray image carefully. "
                "Generate a detailed radiology report of your findings. "
                "Explain your reasoning as you analyze the image."
            )
    else:
        prompt = custom_prompt
    
    logger.info(f"Agent prompt: {prompt}")
    
    # Run the agent
    print("\n=== CXR Analysis Agent Starting ===")
    print(f"Agent will now analyze the image with the following prompt: \n'{prompt}'\n")
    
    # Load the image for analysis
    image = load_image(image_path_or_url)
    
    # Run the agent and capture the result
    result = await agent.analyze_image(image_path_or_url, prompt)
    
    print("\n=== Agent Analysis Complete ===\n")
    
    return result

async def main_async():
    """Async main function to run the CXR analysis."""
    parser = argparse.ArgumentParser(description="CXR Analysis Application")
    parser.add_argument("--image", type=str, required=True, 
                        help="Path or URL to the chest X-ray image")
    parser.add_argument("--grounding", action="store_true",
                        help="Request visual groundings for findings")
    parser.add_argument("--output", type=str, 
                        help="Path to save the visualization output")
    parser.add_argument("--prompt", type=str,
                        help="Custom prompt for the agent")
    args = parser.parse_args()
    
    # Check if the image exists or is a valid URL
    if not args.image.startswith(('http://', 'https://')) and not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
        return
    
    logger.info(f"Starting analysis of chest X-ray image: {args.image}")
    
    # Run the agent analysis
    result = await run_agent_analysis(args.image, args.prompt, args.grounding)
    
    # Check for errors
    if "error" in result:
        logger.error(f"Analysis error: {result['error']}")
        return
    
    # Display the report
    if "report" in result and result["report"]:
        print("\n=== CXR Report ===")
        print(result["report"])
        print("==================\n")
    
    # Visualize findings/groundings if we have them
    groundings = result.get("groundings", [])
    if groundings:
        logger.info(f"Visualizing {len(groundings)} groundings")
        visualize_findings(args.image, groundings, args.output)
    elif args.grounding:
        logger.warning("No groundings detected in the image")

def main():
    """Main function wrapper to handle async."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 