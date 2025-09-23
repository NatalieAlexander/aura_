import gradio as gr
import os
import logging
import re
from typing import Optional

import setup_cache_env

from agents.aura_agent import AURAAgent
from PIL import Image, ImageDraw, ImageFont
import tempfile
import shutil
import time
import hashlib
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the agent
agent = None

# Create served images directory for file serving
SERVED_IMAGE_DIR = "gradio_served_images"
os.makedirs(SERVED_IMAGE_DIR, exist_ok=True)

def initialize_agent():
    """Initialize the visual explainability agent."""
    global agent
    try:
        agent = AURAAgent()
        return "✅ Visual Explainability Agent initialized successfully!"
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return f"❌ Failed to initialize agent: {str(e)}"

def copy_image_to_served_dir(src_path):
    """Copy image to served directory for display."""
    if not os.path.exists(src_path):
        print(f"❌ Source image does not exist: {src_path}")
        return None
    
    import uuid
    # Create unique filename to avoid conflicts
    filename = f"{uuid.uuid4().hex}_{os.path.basename(src_path)}"
    dst_path = os.path.join(SERVED_IMAGE_DIR, filename)
    
    try:
        shutil.copy2(src_path, dst_path)
        print(f"✅ Successfully copied: {src_path} -> {dst_path}")
        return dst_path
    except Exception as e:
        logger.error(f"Failed to copy image {src_path} to {dst_path}: {e}")
        print(f"❌ Failed to copy: {src_path} -> {dst_path}: {e}")
        return None

def extract_session_id_from_response(response_text: str) -> Optional[str]:
    """Extract session ID from agent response."""
    import re
    # Look for session creation messages (multiple formats)
    session_patterns = [
        r'Created session: ([a-f0-9]{8}_\d{8}_\d{6})',
        r'✅ Created session: ([a-f0-9]{8}_\d{8}_\d{6})',
        r'Session ID:\*\* ([a-f0-9]{8}_\d{8}_\d{6})',
        r'- \*\*Session ID:\*\* ([a-f0-9]{8}_\d{8}_\d{6})'
    ]
    
    for pattern in session_patterns:
        session_match = re.search(pattern, response_text)
        if session_match:
            return session_match.group(1)
    
    # Look for session paths
    path_match = re.search(r'output_counterfactuals/sessions/([a-f0-9]{8}_\d{8}_\d{6})/', response_text)
    if path_match:
        return path_match.group(1)
    
    return None

def format_change_percentage(change_pct_str: str) -> str:
    """Format change percentage for display, only if reasonable."""
    try:
        change_pct = float(change_pct_str)
        # Only include percentage if it's reasonable (between 0.1% and 90%)
        if 0.1 <= change_pct <= 90:
            return f" ({change_pct:.1f}% change)"
        else:
            # If percentage is unreasonable, don't show it
            return ""
    except (ValueError, TypeError):
        return ""

def extract_variations_from_session(session_id: str, original_image_path: str) -> list:
    """
    Extract ALL variations by scanning the session directory structure.
    This works for demographic, pathology removal, or any other counterfactual type.
    
    Args:
        session_id: The session ID to scan
        original_image_path: Path to the original image (can be None)
        
    Returns:
        List of variations with file paths and metadata
    """
    import glob
    import os
    import re
    from PIL import Image
    
    session_dir = f"output_counterfactuals/sessions/{session_id}"
    if not os.path.exists(session_dir):
        print(f"❌ Session directory not found: {session_dir}")
        return []
    
    counterfactuals_dir = os.path.join(session_dir, "counterfactuals")
    difference_maps_dir = os.path.join(session_dir, "difference_maps")
    
    if not os.path.exists(counterfactuals_dir) or not os.path.exists(difference_maps_dir):
        print(f"❌ Missing subdirectories in session: {session_dir}")
        return []
    
    # Find all counterfactual files (excluding input files and mask files)
    cf_pattern = os.path.join(counterfactuals_dir, "*.png")
    cf_files = [f for f in glob.glob(cf_pattern) 
                if not f.endswith('_input.png') and not f.endswith('_mask.png')]
    
    # Find all difference map files
    dm_pattern = os.path.join(difference_maps_dir, "*.png")
    dm_files = glob.glob(dm_pattern)
    
    
    variations = {
        'sex': {'counterfactuals': [], 'difference_maps': []},
        'age': {'counterfactuals': [], 'difference_maps': []},
        'race': {'counterfactuals': [], 'difference_maps': []},
        'pathology': {'counterfactuals': [], 'difference_maps': []},
        'general': {'counterfactuals': [], 'difference_maps': []}  
    }
    
    for cf_file in cf_files:
        filename = os.path.basename(cf_file).lower()
        categorized = False
        
        if any(word in filename for word in ['female', 'male', 'gender', 'sex']):
            variations['sex']['counterfactuals'].append(cf_file)
            categorized = True
        elif any(word in filename for word in ['older', 'elderly', 'younger', 'age']):
            variations['age']['counterfactuals'].append(cf_file)
            categorized = True
        elif any(word in filename for word in ['black', 'white', 'asian', 'race', 'ethnicity']):
            variations['race']['counterfactuals'].append(cf_file)
            categorized = True
        elif any(word in filename for word in ['remove', 'healthy', 'normal', 'pneumonia', 'effusion', 'opacity', 'lesion', 'pathology']):
            variations['pathology']['counterfactuals'].append(cf_file)
            categorized = True
        
        if not categorized:
            variations['general']['counterfactuals'].append(cf_file)
    
    for dm_file in dm_files:
        filename = os.path.basename(dm_file).lower()
        categorized = False
        
        if any(word in filename for word in ['sex_variation', 'gender_variation']):
            variations['sex']['difference_maps'].append(dm_file)
            categorized = True
        elif any(word in filename for word in ['age_variation']):
            variations['age']['difference_maps'].append(dm_file)
            categorized = True
        elif any(word in filename for word in ['race_variation']):
            variations['race']['difference_maps'].append(dm_file)
            categorized = True
        elif any(word in filename for word in ['removal_variation', 'healthy_variation', 'pathology_variation', 'effusion_removal', 'pneumonia_removal']):
            variations['pathology']['difference_maps'].append(dm_file)
            categorized = True
        elif any(word in filename for word in ['remove', 'healthy', 'normal', 'pneumonia', 'effusion', 'opacity', 'lesion', 'pathology', 'addition']):
            variations['pathology']['difference_maps'].append(dm_file)
            categorized = True
        elif any(word in filename for word in ['female', 'male', 'gender', 'sex']):
            variations['sex']['difference_maps'].append(dm_file)
            categorized = True
        elif any(word in filename for word in ['older', 'elderly', 'younger', 'age']):
            variations['age']['difference_maps'].append(dm_file)
            categorized = True
        elif any(word in filename for word in ['black', 'white', 'asian', 'race', 'ethnicity']):
            variations['race']['difference_maps'].append(dm_file)
            categorized = True
        
        if not categorized:
            variations['general']['difference_maps'].append(dm_file)
    

    def calculate_change_percentage(dm_path):
        try:
            dm_img = Image.open(dm_path).convert('RGB')
            dm_array = np.array(dm_img)
            
            
            total_pixels = dm_array.shape[0] * dm_array.shape[1]
            
            dm_gray = np.mean(dm_array, axis=2)
            
            changed_pixels = np.sum(dm_gray > 10) 
            
            percentage = (changed_pixels / total_pixels) * 100
            
            
            if percentage > 95:    
                changed_pixels_high = np.sum(dm_gray > 50)
                percentage_high = (changed_pixels_high / total_pixels) * 100
                                
                if percentage_high > 95:
                    dm_variance = np.var(dm_array, axis=2)
                    changed_pixels_var = np.sum(dm_variance > 100)  # Variance threshold
                    percentage_var = (changed_pixels_var / total_pixels) * 100
                    
                    
                    if percentage_var < 95:
                        return round(percentage_var, 2)
                    else:
                        print(f"⚠️ Warning: All approaches give >95% change for {dm_path}")
                        return round(percentage_high, 2)
                else:
                    return round(percentage_high, 2)
            else:
                return round(percentage, 2)
                
        except Exception as e:
            print(f"⚠️ Error calculating change percentage for {dm_path}: {e}")
            return 0.0
    
    result_variations = []
    
    for var_type, files in variations.items():
        if files['counterfactuals'] and files['difference_maps']:
            
            cf_with_prefixes = []
            for cf_path in files['counterfactuals']:
                cf_filename = os.path.basename(cf_path)
                parts = cf_filename.split('_')
                if len(parts) >= 2:
                    prefix = '_'.join(parts[:-1])
                    cf_with_prefixes.append((prefix, cf_path))
                else:
                    cf_with_prefixes.append((cf_filename.replace('.png', ''), cf_path))
            
            dm_with_prefixes = []
            for dm_path in files['difference_maps']:
                dm_filename = os.path.basename(dm_path)
                parts = dm_filename.split('_')
                if len(parts) >= 2:
                    prefix = '_'.join(parts[:-1])
                    dm_with_prefixes.append((prefix, dm_path))
                else:
                    dm_with_prefixes.append((dm_filename.replace('.png', ''), dm_path))
            
            cf_prefix_map = {prefix: path for prefix, path in cf_with_prefixes}
            dm_prefix_map = {prefix: path for prefix, path in dm_with_prefixes}
            
            paired_files = []
            all_prefixes = set(cf_prefix_map.keys()) | set(dm_prefix_map.keys())
            
            for prefix in sorted(all_prefixes):
                cf_path = cf_prefix_map.get(prefix)
                dm_path = dm_prefix_map.get(prefix)
                
                if cf_path and dm_path:
                    change_pct = calculate_change_percentage(dm_path)
                    
                    description_details = ""
                    if var_type == 'sex':
                        if 'female' in prefix.lower():
                            description_details = " → Female"
                        elif 'male' in prefix.lower():
                            description_details = " → Male"
                        import re
                        var_match = re.search(r'variation_(\d+)', prefix)
                        if var_match:
                            description_details += f" (Variation {var_match.group(1)})"
                    elif var_type == 'age':
                        if 'older' in prefix.lower() or 'elderly' in prefix.lower():
                            description_details = " → Elderly"
                        elif 'younger' in prefix.lower():
                            description_details = " → Younger"
                        var_match = re.search(r'variation_(\d+)', prefix)
                        if var_match:
                            description_details += f" (Variation {var_match.group(1)})"
                    elif var_type == 'race':
                        if 'black' in prefix.lower():
                            description_details = " → Black"
                        elif 'white' in prefix.lower():
                            description_details = " → White"
                        elif 'asian' in prefix.lower():
                            description_details = " → Asian"
                        var_match = re.search(r'variation_(\d+)', prefix)
                        if var_match:
                            description_details += f" (Variation {var_match.group(1)})"
                    elif var_type == 'pathology':
                        if 'remove' in prefix.lower():
                            description_details = " (Removal)"
                        elif 'healthy' in prefix.lower():
                            description_details = " (Healthy version)"
                        elif 'addition' in prefix.lower():
                            description_details = " (Addition)"
                        elif 'effusion' in prefix.lower():
                            if 'addition' in prefix.lower():
                                description_details = " (Pleural Effusion Addition)"
                            else:
                                description_details = " (Pleural Effusion)"
                        elif 'pneumonia' in prefix.lower():
                            description_details = " (Pneumonia)"
                    elif var_type == 'general':
                        if 'remove' in prefix.lower() or 'healthy' in prefix.lower():
                            description_details = " (Pathology modification)"
                        elif 'addition' in prefix.lower():
                            description_details = " (Addition)"
                        elif 'effusion' in prefix.lower():
                            description_details = " (Pleural Effusion)"
                        elif 'pneumonia' in prefix.lower():
                            description_details = " (Pneumonia)"
                    
                    weight_match = re.search(r'w([\d.]+)', prefix)
                    if weight_match:
                        description_details += f" (w={weight_match.group(1)})"
                    
                    paired_files.append({
                        'counterfactual': cf_path,
                        'difference_map': dm_path,
                        'change_percentage': str(change_pct),
                        'description_details': description_details,
                        'prefix': prefix
                    })
                    
                    print(f"✅ Perfect prefix match: '{prefix}' -> CF: {os.path.basename(cf_path)} + DM: {os.path.basename(dm_path)}")
                    
                elif cf_path and not dm_path:
                    print(f"⚠️ Orphaned counterfactual with prefix '{prefix}': {os.path.basename(cf_path)}")
                    
                elif dm_path and not cf_path:
                    change_pct = calculate_change_percentage(dm_path)
                    paired_files.append({
                        'counterfactual': None,
                        'difference_map': dm_path,
                        'change_percentage': str(change_pct),
                        'description_details': f" (Orphaned {var_type} variation)",
                        'prefix': prefix
                    })
                    print(f"⚠️ Orphaned difference map with prefix '{prefix}': {os.path.basename(dm_path)}")
            
            if paired_files:
                if var_type == 'sex':
                    description = "Changed patient sex"
                elif var_type == 'age':
                    description = "Changed patient age"
                elif var_type == 'race':
                    description = "Changed patient race/ethnicity"
                elif var_type == 'pathology':
                    description = "Pathology removal/modification"
                elif var_type == 'general':
                    description = "Counterfactual variation"
                else:
                    description = f"{var_type.capitalize()} variation"
                
                result_variations.append({
                    'type': var_type,
                    'description': description,
                    'files': paired_files
                })
                
                print(f"✅ Added {len(paired_files)} {var_type} variations with PREFIX-BASED pairing")
    
    return result_variations

def extract_image_paths_from_response(response_text: str):
    """Extract counterfactual and difference map paths from agent response with session support."""
    print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
    
    counterfactuals = []
    difference_maps = []
    best_counterfactual = None
    
    # First, try to extract session ID
    session_id = extract_session_id_from_response(response_text)    
    # Extract best counterfactual first (highest priority)
    import re
    best_cf_pattern = r'BEST_COUNTERFACTUAL:\s*([^\s\n]+\.png)'
    best_cf_matches = re.findall(best_cf_pattern, response_text)
    if best_cf_matches:
        best_counterfactual = best_cf_matches[0].strip()
        if not os.path.exists(best_counterfactual):
            best_counterfactual = None
    
    lines = response_text.split('\\n')
    for line in lines:
        if 'counterfactual' in line.lower() and '.png' in line:
            import re
            if session_id:
                path_matches = re.findall(r'(?:output_counterfactuals/sessions/[^\\s,]*counterfactual[^\\s,]*\\.png)', line)
            else:
                path_matches = re.findall(r'(?:output_counterfactuals/[^\\s,]*counterfactual[^\\s,]*\\.png)', line)
            
            for path_match in path_matches:
                if ('diff_map' not in path_match and 
                    'difference_map' not in path_match and
                    'mask' not in path_match and 
                    os.path.exists(path_match) and 
                    path_match not in counterfactuals):
                    counterfactuals.append(path_match)
    
        # Find difference maps - look for actual difference map files  
        if 'difference map' in line.lower() and '.png' in line:
            import re
            # Look for difference map files with various naming patterns
            if session_id:
                path_matches = re.findall(r'(?:output_counterfactuals/sessions/[^\s,]*(?:diff_map|difference_map|grid_search|removal_variation|healthy_variation)[^\s,]*\.png)', line)
            else:
                path_matches = re.findall(r'(?:output_counterfactuals/[^\s,]*(?:diff_map|difference_map|grid_search|removal_variation|healthy_variation|effusion_removal_variation)[^\s,]*\.png)', line)
            
            for path_match in path_matches:
                if 'overlay' not in path_match and os.path.exists(path_match) and path_match not in difference_maps:
                    difference_maps.append(path_match)
                            
        if '.png' in line and any(pattern in line for pattern in ['effusion_removal_variation', 'removal_variation', 'healthy_variation']):
            import re
            variation_patterns = [
                r'(output_counterfactuals/effusion_removal_variation_\d+_\d+\.png)',
                r'(output_counterfactuals/removal_variation_\d+_\d+\.png)',
                r'(output_counterfactuals/healthy_variation_\d+_\d+\.png)'
            ]
            
            for pattern in variation_patterns:
                path_matches = re.findall(pattern, line)
                for path_match in path_matches:
                    if os.path.exists(path_match) and path_match not in difference_maps:
                        difference_maps.append(path_match)
    
    
    
    if session_id and (len(counterfactuals) < 2 or len(difference_maps) < 2):
        session_dir = f"output_counterfactuals/sessions/{session_id}"
        
        if os.path.exists(session_dir):
            import glob
            cf_pattern = os.path.join(session_dir, "counterfactuals", "*.png")
            session_counterfactuals = glob.glob(cf_pattern)
            for cf_path in session_counterfactuals:
                if '_mask.png' not in cf_path and cf_path not in counterfactuals:
                    counterfactuals.append(cf_path)
            
            dm_pattern = os.path.join(session_dir, "difference_maps", "*.png")
            session_diff_maps = glob.glob(dm_pattern) 
            for dm_path in session_diff_maps:
                if dm_path not in difference_maps:
                    difference_maps.append(dm_path)
    
    if len(counterfactuals) == 0 or len(difference_maps) == 0:
        
        import re
        
        cf_pattern = r'output_counterfactuals/counterfactual_[^\\s]*\.png'
        response_counterfactuals = re.findall(cf_pattern, response_text)
        
        diff_patterns = [
            r'output_counterfactuals/[^\s]*(?:healthy_variation|removal_variation|effusion_removal_variation|diff_map|difference_map)[^\s]*\.png',
            r'output_counterfactuals/[^\s]*variation_\d+[^\s]*\.png'
        ]
        
        response_difference_maps = []
        for pattern in diff_patterns:
            response_difference_maps.extend(re.findall(pattern, response_text))
        
        
        for cf_path in response_counterfactuals:
            if os.path.exists(cf_path) and '_mask.png' not in cf_path and cf_path not in counterfactuals:
                counterfactuals.append(cf_path)
        
        for diff_path in response_difference_maps:
            if os.path.exists(diff_path) and diff_path not in difference_maps:
                difference_maps.append(diff_path)
        
        if len(counterfactuals) == 0 or len(difference_maps) == 0:
            
            base_dir = "output_counterfactuals"
            if os.path.exists(base_dir):
                current_time = time.time()
                
                for root, dirs, files in os.walk(base_dir):
                    for file in files:
                        if file.endswith('.png'):
                            file_path = os.path.join(root, file)
                            try:
                                file_time = os.path.getmtime(file_path)
                                if current_time - file_time < 300:  # Only last 5 minutes (current session)
                                    # Prioritize difference maps (removal_variation, diff_map, etc.)
                                    if any(x in file for x in ['diff_map', 'difference_map', 'grid_search', 'removal_variation', 'healthy_variation']) and file_path not in difference_maps:
                                        difference_maps.append(file_path)
                                    elif 'counterfactual' in file and '_mask.png' not in file and file_path not in counterfactuals:
                                        counterfactuals.append(file_path)
                            except:
                                continue
    
    for i, cf in enumerate(counterfactuals):
        print(f"   CF {i+1}: {cf}")
    for i, dm in enumerate(difference_maps):
        print(f"   DM {i+1}: {dm}")
    
    return counterfactuals, difference_maps, best_counterfactual

def process_message(message, history):
    """Process user message and return response with images."""
    global agent
    
    if agent is None:
        init_result = initialize_agent()
        if "❌" in init_result:
            return f"❌ Agent not initialized: {init_result}"
    
    image_path = None
    user_text = ""
    
    if isinstance(message, dict):
        if "files" in message and message["files"]:
            raw_path = message["files"][0]
            if isinstance(raw_path, str) and os.path.exists(raw_path):
                image_path = raw_path
            else:
                image_path = None
            user_text = message.get("text", "Analyze this chest X-ray and provide comprehensive visual explainability analysis including demographics and counterfactuals.")
        else:
            user_text = message.get("text", "")
    elif isinstance(message, str):
        if message.startswith("Image:"):
            parts = message.split("Image:", 1)
            if len(parts) == 2:
                image_path = parts[1].strip()
                user_text = "Analyze this chest X-ray and provide comprehensive visual explainability analysis including demographics and counterfactuals."
        else:
            user_text = message
    else:
        user_text = str(message)
    
    try:
        if image_path and not os.path.exists(image_path):
            return f"❌ **Image file not found:** The specified image path does not exist: `{image_path}`\n\nPlease upload a valid chest X-ray image."
        
        if user_text and user_text.strip():
            if image_path:
                full_request = f"{user_text.strip()} Image path: {image_path}"
            else:
                full_request = user_text.strip()
        else:
            if image_path:
                full_request = f"Analyze this chest X-ray and provide comprehensive visual explainability analysis including demographics and counterfactuals. Image path: {image_path}"
            else:
                full_request = "Please upload a chest X-ray image to analyze."
        
        
        response = agent.run_chat(full_request)
        
        counterfactuals, difference_maps, best_counterfactual = extract_image_paths_from_response(response)
        
        session_id = extract_session_id_from_response(response)
        
        if session_id:
            print(f"🎨 Attempting to create triplet displays for session: {session_id}")
            
            variations = extract_variations_from_session(session_id, image_path)
            
            if variations:
                print(f"✅ Found {len(variations)} variations from session filesystem")
                
                triplet_images = []
                
                for variation in variations:
                    var_type = variation['type']
                    var_description = variation['description'] 
                    
                    for i, file_info in enumerate(variation['files']):
                        cf_path = file_info['counterfactual']
                        dm_path = file_info['difference_map']
                        change_pct = file_info['change_percentage']
                        description_details = file_info.get('description_details', '')
                        
                        caption = f"{var_description}{description_details}{format_change_percentage(change_pct)}"
                        
                        if cf_path and os.path.exists(cf_path) and os.path.exists(dm_path):
                            triplet_path = create_triplet_display(
                                original_path=image_path,
                                counterfactual_path=cf_path,
                                diff_map_path=dm_path,
                                caption=caption
                            )
                            
                            if triplet_path:
                                triplet_images.append(triplet_path)
                                print(f"✅ Created triplet for {var_type} variation {i+1}")
                            else:
                                print(f"❌ Failed to create triplet for {var_type} variation {i+1}")
                        elif cf_path is None:
                            print(f"⚠️ Skipping orphaned difference map {i+1} (no counterfactual): {os.path.basename(dm_path)}")
                        else:
                            print(f"❌ Skipping triplet {i+1} - missing files: cf={cf_path and os.path.exists(cf_path)}, dm={os.path.exists(dm_path)}")
                
                if triplet_images:
                    print(f"🎨 Created {len(triplet_images)} enhanced triplet displays")
                    return {"text": response, "files": triplet_images}
                else:
                    print("❌ No triplet images created, falling back to standard pairing")
            else:
                print("❌ No variations found in session, falling back to standard pairing")
        else:
            print("❌ No session ID found, falling back to standard pairing")
        
        print("📁 Using standard image pairing for non-demographic analysis...")
        
        served_counterfactuals = []
        served_difference_maps = []
        served_best_counterfactual = None
        
        for cf_path in counterfactuals:
            served_path = copy_image_to_served_dir(cf_path)
            if served_path:
                served_counterfactuals.append(served_path)
                print(f"✅ Served counterfactual: {cf_path} -> {served_path}")
        
        for diff_path in difference_maps:
            served_path = copy_image_to_served_dir(diff_path)
            if served_path:
                served_difference_maps.append(served_path)
                print(f"✅ Served difference map: {diff_path} -> {served_path}")
        
        if best_counterfactual:
            served_best_counterfactual = copy_image_to_served_dir(best_counterfactual)
            if served_best_counterfactual:
                print(f"✅ Served best counterfactual: {best_counterfactual} -> {served_best_counterfactual}")
        
        if served_counterfactuals or served_difference_maps or served_best_counterfactual:
            paired_images = []
            
            actual_counterfactuals = [cf for cf in served_counterfactuals if not cf.endswith('_input.png')]
            input_images = [cf for cf in served_counterfactuals if cf.endswith('_input.png')]
            
            
            if len(actual_counterfactuals) == len(served_difference_maps):
                print("✅ Using smart semantic pairing (counts match)")
                cf_matches = {}
                
                for cf_path in actual_counterfactuals:
                    cf_name = os.path.basename(cf_path)
                    
                    timestamp_match = re.search(r'_(\d{8}_\d{6})', cf_name)
                    if timestamp_match:
                        timestamp = timestamp_match.group(1)
                        
                        best_match = None
                        best_score = 0
                        
                        for dm_path in served_difference_maps:
                            dm_name = os.path.basename(dm_path)
                            
                            score = 0
                            if 'female' in cf_name.lower() and 'sex' in dm_name.lower():
                                score += 10
                            elif 'older' in cf_name.lower() and 'age' in dm_name.lower():
                                score += 10  
                            elif 'black' in cf_name.lower() and 'race' in dm_name.lower():
                                score += 10
                                
                            dm_timestamp_match = re.search(r'_(\d{8}_\d{6})', dm_name)
                            if dm_timestamp_match:
                                dm_timestamp = dm_timestamp_match.group(1)
                                if abs(int(timestamp.replace('_', '')) - int(dm_timestamp.replace('_', ''))) < 1000:  # Within ~10 minutes
                                    score += 5
                                    
                            if score > best_score and dm_path not in cf_matches.values():
                                best_match = dm_path
                                best_score = score
                        
                        if best_match:
                            cf_matches[cf_path] = best_match
                
                for cf_path in actual_counterfactuals:
                    paired_images.append(cf_path)
                    if cf_path in cf_matches:
                        paired_images.append(cf_matches[cf_path])
                        print(f"✅ Paired: {os.path.basename(cf_path)} -> {os.path.basename(cf_matches[cf_path])}")
                    else:
                        print(f"⚠️ No match found for: {os.path.basename(cf_path)}")
                        
                for dm_path in served_difference_maps:
                    if dm_path not in cf_matches.values():
                        paired_images.append(dm_path)
                        print(f"⚠️ Unmatched difference map: {os.path.basename(dm_path)}")
                        
                for input_path in input_images:
                    paired_images.append(input_path)
                    print(f"📄 Added input image: {os.path.basename(input_path)}")
                        
            if served_best_counterfactual:
                paired_images.append(served_best_counterfactual)
                print(f"🏆 Added best counterfactual to end: {os.path.basename(served_best_counterfactual)}")
                        
            else:
                print(f"⚠️ Mismatched counts: {len(actual_counterfactuals)} actual CFs, {len(served_difference_maps)} DMs - using alternating pattern")
                for i in range(max(len(served_counterfactuals), len(served_difference_maps))):
                    if i < len(served_counterfactuals):
                        paired_images.append(served_counterfactuals[i])
                    if i < len(served_difference_maps):
                        paired_images.append(served_difference_maps[i])
                
                if served_best_counterfactual:
                    paired_images.append(served_best_counterfactual)
                    print(f"🏆 Added best counterfactual to end: {os.path.basename(served_best_counterfactual)}")
            
            response_text = response
            
            for i, img_path in enumerate(paired_images):
                print(f"   Image {i+1}: {img_path}")
            
            return {"text": response_text, "files": paired_images}
        else:
            return response
            
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return f"❌ **Error processing request:**\n\n```\n{str(e)}\n```"

def respond(message, history):
    """Respond function for ChatInterface."""
    result = process_message(message, history)
    
    if isinstance(result, dict):
        return result
    else:
        return result

def create_triplets_from_last_session():
    """Create triplet displays from the most recent session."""
    try:
        sessions_dir = "output_counterfactuals/sessions"
        if not os.path.exists(sessions_dir):
            return "❌ No sessions directory found"
        
        session_dirs = [d for d in os.listdir(sessions_dir) if os.path.isdir(os.path.join(sessions_dir, d))]
        if not session_dirs:
            return "❌ No sessions found"
        
        session_dirs.sort(reverse=True)
        latest_session = session_dirs[0]
        
        print(f"🔍 Creating triplets from latest session: {latest_session}")
        
        variations = extract_variations_from_session(latest_session, None)
        
        if not variations:
            return "❌ No variations found in latest session"
        
        triplet_images = []
        
        original_image_path = None
        temp_dirs = ["/tmp/gradio"]
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            file_path = os.path.join(root, file)
                            if original_image_path is None or os.path.getmtime(file_path) > os.path.getmtime(original_image_path):
                                original_image_path = file_path
        
        if not original_image_path:
            return "❌ Could not find original image for triplet creation"
        
        for variation in variations:
            var_type = variation['type']
            var_description = variation['description']
            
            for i, file_info in enumerate(variation['files']):
                cf_path = file_info['counterfactual']
                dm_path = file_info['difference_map']
                change_pct = file_info['change_percentage']
                description_details = file_info.get('description_details', '')
                
                caption = f"{var_description}{description_details}{format_change_percentage(change_pct)}"
                
                if cf_path and os.path.exists(cf_path) and os.path.exists(dm_path):
                    triplet_path = create_triplet_display(
                        original_path=original_image_path,
                        counterfactual_path=cf_path,
                        diff_map_path=dm_path,
                        caption=caption
                    )
                    
                    if triplet_path:
                        triplet_images.append(triplet_path)
                        print(f"✅ Created triplet for {var_type} variation {i+1}")
        
        if triplet_images:
            return f"✅ Created {len(triplet_images)} triplet displays from session {latest_session}", triplet_images
        else:
            return "❌ Failed to create any triplet displays"
            
    except Exception as e:
        return f"❌ Error creating triplets: {str(e)}"

# Create the Gradio interface using ChatInterface
def create_interface():
    # ChatInterface with file upload support
    chat_interface = gr.ChatInterface(
        fn=respond,
        title="🧠 Visual Explainability Agent - Advanced CXR Analysis",
        description="""
        **Advanced AI Agent for Comprehensive Chest X-Ray Visual Explainability**
        
        Upload a chest X-ray image and ask for sophisticated analysis including:
        
        🔬 **Core Capabilities:**
        - **VQA (Visual Question Answering)**: Ask about demographics, findings, or clinical details
        - **Smart Counterfactuals**: Generate optimal counterfactuals with parameter optimization  
        - **Demographic Explanations**: Visual explanations of how age, sex, race affect imaging
        - **Grid Search Optimization**: Find best parameters for counterfactual generation
        
        💡 **Example Requests:**
        - *"What are the patient demographics and show me visual explanations"*
        - *"Generate optimal counterfactuals using grid search to remove pneumonia"*
        - *"Show me how this image would look with different patient demographics"*
        - *"Find the best parameters for removing pleural effusion"*
        """,
        examples=[
            "Analyze demographics and generate visual explanations",
            "Generate optimal counterfactuals using grid search for abnormality removal",
            "Show me how patient age and sex affect the imaging appearance",
            "What clinical findings are present and create counterfactuals",
            "Use parameter optimization to find the best counterfactual settings"
        ],
        multimodal=True,  # Enable file uploads
        cache_examples=False,
        textbox=gr.MultimodalTextbox(
            placeholder="Ask about chest X-ray analysis, demographics, or counterfactuals...",
            file_count="multiple",
            file_types=["image"],
            sources=["upload"]
        )
    )
    
    with gr.Blocks(fill_height=True, css="""
        .chatbot {
            height: 600px !important;
        }
        """) as wrapper:
        gr.Markdown("# 🧠 Visual Explainability Agent - Advanced CXR Analysis")
        
        with gr.Row():
            init_btn = gr.Button("🚀 Initialize Agent", variant="primary") 
            triplet_btn = gr.Button("🎨 Create Triplet Displays", variant="secondary")
            init_status = gr.Textbox(label="Status", interactive=False, value="Click to initialize agent")
        
        triplet_gallery = gr.Gallery(
            label="Triplet Displays (Original | Counterfactual | Difference Map)",
            show_label=True,
            elem_id="triplet_gallery",
            columns=1,
            rows=3,
            height="auto",
            visible=False
        )
        
        triplet_status = gr.Textbox(label="Triplet Creation Status", interactive=False, visible=False)
        
        def display_triplets():
            """Handle triplet display button click."""
            result = create_triplets_from_last_session()
            
            if isinstance(result, tuple) and len(result) == 2:
                status_msg, triplet_paths = result
                return {
                    triplet_status: gr.update(value=status_msg, visible=True),
                    triplet_gallery: gr.update(value=triplet_paths, visible=True)
                }
            else:
                return {
                    triplet_status: gr.update(value=result, visible=True),
                    triplet_gallery: gr.update(visible=False)
                }
        
        init_btn.click(initialize_agent, outputs=init_status)
        triplet_btn.click(
            display_triplets,
            outputs=[triplet_status, triplet_gallery]
        )
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### 🔬 **Advanced Features**
                - **CheXagent VQA**: Answer questions about patient demographics and findings
                - **Smart Parameter Search**: Grid search for optimal counterfactual generation
                - **Demographic Counterfactuals**: Visual explanations of how demographics affect imaging
                - **Enhanced RadEdit Control**: Fine-tune inference steps, skip ratio, and guidance scales
                """)
            with gr.Column():
                gr.Markdown("""
                ### 💡 **Smart Workflows**  
                - **Automatic Operation Detection**: Intelligently detects adding vs editing operations
                - **MedSAM Decision Logic**: Automatically chooses optimal segmentation approach
                - **Batch Processing**: Efficient generation of multiple counterfactuals
                - **Comprehensive Analysis**: Combines VQA insights with visual explanations
                """)
        
        gr.Markdown("---")
        
        chat_interface.render()
        
        wrapper.load(initialize_agent, outputs=init_status)
        
        gr.Markdown("""
        ---
        ### 🔧 **Advanced Usage Tips:**
        
        **For VQA + Demographics:**
        - Ask: *"What is the age, sex, and race of this patient? Then show me visual explanations."*
        
        **For Parameter Optimization:**
        - Ask: *"Use grid search to find optimal parameters for removing [abnormality]"*
        
        **For Demographic Analysis:**
        - Ask: *"Show me how this image would look with different patient demographics"*
        
        **For Pathology Removal:**
        - Ask: *"Remove the pneumonia from this chest X-ray"*
        
        **For Any Counterfactual:**
        - Ask: *"Generate counterfactuals for this image"* then click **🎨 Create Triplet Displays**
        
        📊 The agent automatically uses **CheXagent-8b** for VQA, **MAIRA-2** for medical analysis, and **RadEdit** with enhanced parameter control for counterfactual generation.
        
        🎨 **Triplet Display Feature:** After any counterfactual generation, click the **"Create Triplet Displays"** button to see beautiful side-by-side comparisons showing Original | Counterfactual | Difference Map with change percentages.
        """)
    
    return wrapper

def create_triplet_display(original_path: str, counterfactual_path: str, diff_map_path: str, 
                          caption: str, output_dir: str = "gradio_served_images") -> str:
    """
    Create a horizontal triplet display: Original | Counterfactual | Difference Map
    with labels and caption. Brightness normalization is now handled by the difference map tool.
    
    Args:
        original_path: Path to the original/factual image
        counterfactual_path: Path to the counterfactual image  
        diff_map_path: Path to the difference map
        caption: Caption describing the transformation
        output_dir: Directory to save the composite image
        
    Returns:
        Path to the created composite image
    """
    try:
        original = Image.open(original_path).convert('RGB')
        counterfactual = Image.open(counterfactual_path).convert('RGB')
        diff_map = Image.open(diff_map_path).convert('RGB')
        
        
        size = (512, 512)
        original = original.resize(size, Image.LANCZOS)
        counterfactual = counterfactual.resize(size, Image.LANCZOS)
        diff_map = diff_map.resize(size, Image.LANCZOS)
        
        label_height = 40
        caption_height = 60
        spacing = 10
        
        total_width = 3 * size[0] + 2 * spacing  # 3 images + 2 spacings
        total_height = label_height + size[1] + caption_height
        
        composite = Image.new('RGB', (total_width, total_height), 'white')
        
        y_offset = label_height
        composite.paste(original, (0, y_offset))
        composite.paste(counterfactual, (size[0] + spacing, y_offset))
        composite.paste(diff_map, (2 * size[0] + 2 * spacing, y_offset))
        
        draw = ImageDraw.Draw(composite)
        
        try:
            font_label = ImageFont.truetype("arial.ttf", 28)
            font_caption = ImageFont.truetype("arial.ttf", 24)
        except:
            try:
                font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
                font_caption = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                try:
                    font_label = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 28)
                    font_caption = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 24)
                except:
                    try:
                        font_label = ImageFont.load_default()
                        font_caption = ImageFont.load_default()
                    except:
                        font_label = ImageFont.load_default()
                        font_caption = ImageFont.load_default()
        
        labels = ["Factual", "Counterfactual", "Difference Map"]
        label_positions = [
            size[0] // 2, 
            size[0] + spacing + size[0] // 2,  
            2 * size[0] + 2 * spacing + size[0] // 2  
        ]
        
        for i, (label, x_pos) in enumerate(zip(labels, label_positions)):
            bbox = draw.textbbox((0, 0), label, font=font_label)
            text_width = bbox[2] - bbox[0]
            
            draw.text((x_pos - text_width // 2, 5), label, 
                     fill='black', font=font_label)
        
        caption_y = label_height + size[1] + 10
        
        words = caption.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font_caption)
            if bbox[2] - bbox[0] <= total_width - 20:  
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)  
        
        if current_line:
            lines.append(' '.join(current_line))
        
       
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font_caption)
            text_width = bbox[2] - bbox[0]
            x_pos = (total_width - text_width) // 2  # Center the text
            draw.text((x_pos, caption_y + i * 25), line, 
                     fill='black', font=font_caption)
        
        
        os.makedirs(output_dir, exist_ok=True)
        
        
        hash_input = f"{original_path}{counterfactual_path}{diff_map_path}{caption}".encode()
        file_hash = hashlib.md5(hash_input).hexdigest()[:12]
        composite_filename = f"triplet_{file_hash}.png"
        composite_path = os.path.join(output_dir, composite_filename)
        
        composite.save(composite_path)
        print(f"✅ Created triplet display: {composite_path}")
        
        return composite_path
        
    except Exception as e:
        print(f"❌ Error creating triplet display: {e}")
        return None

def parse_demographic_analysis(response_text: str) -> list:
    """
    Parse the agent's response to extract demographic variation information.
    
    Returns:
        List of dicts with variation info: [{'type': 'sex', 'description': '...', 'files': [...]}]
    """
    variations = []
    
    
    demographics = {}
    
    
    demo_match = re.search(r'EXTRACTED DEMOGRAPHICS:\s*Age:\s*(\d+),\s*Sex:\s*(\w+),\s*Race:\s*(\w+)', response_text)
    
    
    if not demo_match:
        demo_section = re.search(r'PATIENT DEMOGRAPHICS:\s*Age:\s*(\d+)\s*Sex:\s*(\w+)\s*Race:\s*(\w+)', response_text, re.DOTALL)
        if demo_section:
            demographics = {
                'age': demo_section.group(1),
                'sex': demo_section.group(2),
                'race': demo_section.group(3)
            }
    else:
        demographics = {
            'age': demo_match.group(1),
            'sex': demo_match.group(2), 
            'race': demo_match.group(3)
        }
    
    
    sex_section = re.search(r'Sex Variations:(.*?)(?=Age Variations:|Race Variations:|$)', response_text, re.DOTALL)
    if sex_section:
        sex_text = sex_section.group(1).strip()
        
        sex_files = []
        
        
        sex_clean = re.sub(r'\n(?=output_counterfactuals/)', ' ', sex_text)  # Join split paths
        sex_lines = sex_clean.split('\n')
        
        for line in sex_lines:
            if 'Counterfactual:' in line and '|' in line:
                
                parts = line.split('|')
                cf_match = None
                dm_match = None
                change_match = None
                
                for part in parts:
                    part = part.strip()
                    if 'Counterfactual:' in part:
                        cf_match = re.search(r'Counterfactual:\s*([^\s\|]+\.png)', part)
                    elif 'Difference Map:' in part:
                        dm_match = re.search(r'Difference Map:\s*([^\s\|]+\.png)', part)
                    elif 'Change:' in part:
                        change_match = re.search(r'Change:\s*([\d.]+)%', part)
                
                if cf_match and dm_match:
                    sex_files.append({
                        'counterfactual': cf_match.group(1),
                        'difference_map': dm_match.group(1),
                        'change_percentage': change_match.group(1) if change_match else 'N/A'
                    })
        if len(sex_files) == 0:
            for block_match in re.finditer(r'Sex Variation \d+:(.*?)(?=Sex Variation \d+:|Age Variations:|Race Variations:|$)', sex_text, re.DOTALL):
                block_text = block_match.group(1)
                
                cf_match = re.search(r'Counterfactual Image Path:\s*([^\s\n]+\.png)', block_text)
                dm_match = re.search(r'Difference Map Path:\s*([^\s\n]+\.png)', block_text)  
                change_match = re.search(r'Change Percentage:\s*([\d.]+)%', block_text)
                
                if cf_match and dm_match:
                    sex_files.append({
                        'counterfactual': cf_match.group(1),
                        'difference_map': dm_match.group(1),
                        'change_percentage': change_match.group(1) if change_match else 'N/A'
                    })
        
        if sex_files:
            target_sex = "female" if demographics.get('sex', '').lower().startswith('m') else "male"
            variations.append({
                'type': 'sex',
                'description': f"Changed sex from {demographics.get('sex', 'unknown')} to {target_sex}",
                'files': sex_files
            })
            print(f"✅ Added {len(sex_files)} sex variations")
    
    # Parse Age Variations  
    age_section = re.search(r'Age Variations:(.*?)(?=Race Variations:|$)', response_text, re.DOTALL)
    if age_section:
        age_text = age_section.group(1).strip()
        
        age_files = []
        
        # Handle new format: "Counterfactual: path | Difference Map: \npath | Change: 44.09%" (multiline)
        # First, join lines that are part of the same entry
        age_clean = re.sub(r'\n(?=output_counterfactuals/)', ' ', age_text)  # Join split paths
        age_lines = age_clean.split('\n')
        
        for line in age_lines:
            if 'Counterfactual:' in line and '|' in line:
                parts = line.split('|')
                cf_match = None
                dm_match = None
                change_match = None
                
                for part in parts:
                    part = part.strip()
                    if 'Counterfactual:' in part:
                        cf_match = re.search(r'Counterfactual:\s*([^\s\|]+\.png)', part)
                    elif 'Difference Map:' in part:
                        dm_match = re.search(r'Difference Map:\s*([^\s\|]+\.png)', part)
                    elif 'Change:' in part:
                        change_match = re.search(r'Change:\s*([\d.]+)%', part)
                
                if cf_match and dm_match:
                    age_files.append({
                        'counterfactual': cf_match.group(1),
                        'difference_map': dm_match.group(1),
                        'change_percentage': change_match.group(1) if change_match else 'N/A'
                    })
        
        if len(age_files) == 0:
            for block_match in re.finditer(r'Age Variation \d+:(.*?)(?=Age Variation \d+:|Race Variations:|$)', age_text, re.DOTALL):
                block_text = block_match.group(1)
                
                cf_match = re.search(r'Counterfactual Image Path:\s*([^\s\n]+\.png)', block_text)
                dm_match = re.search(r'Difference Map Path:\s*([^\s\n]+\.png)', block_text)
                change_match = re.search(r'Change Percentage:\s*([\d.]+)%', block_text)
                
                if cf_match and dm_match:
                    age_files.append({
                        'counterfactual': cf_match.group(1),
                        'difference_map': dm_match.group(1),
                        'change_percentage': change_match.group(1) if change_match else 'N/A'
                    })
        
        if age_files:
            current_age = int(demographics.get('age', '0'))
            target_age = "elderly (70s)" if current_age < 50 else "younger (30s)"
            variations.append({
                'type': 'age',
                'description': f"Changed age from {demographics.get('age', 'unknown')} to {target_age}",
                'files': age_files
            })
            print(f"✅ Added {len(age_files)} age variations")
    
    # Parse Race Variations
    race_section = re.search(r'Race Variations:(.*?)$', response_text, re.DOTALL)
    if race_section:
        race_text = race_section.group(1).strip()        
        race_files = []
        
        race_clean = re.sub(r'\n(?=output_counterfactuals/)', ' ', race_text)  # Join split paths
        race_lines = race_clean.split('\n')
        
        for line in race_lines:
            if 'Counterfactual:' in line and '|' in line:
                parts = line.split('|')
                cf_match = None
                dm_match = None
                change_match = None
                
                for part in parts:
                    part = part.strip()
                    if 'Counterfactual:' in part:
                        cf_match = re.search(r'Counterfactual:\s*([^\s\|]+\.png)', part)
                    elif 'Difference Map:' in part:
                        dm_match = re.search(r'Difference Map:\s*([^\s\|]+\.png)', part)
                    elif 'Change:' in part:
                        change_match = re.search(r'Change:\s*([\d.]+)%', part)
                
                if cf_match and dm_match:
                    race_files.append({
                        'counterfactual': cf_match.group(1),
                        'difference_map': dm_match.group(1),
                        'change_percentage': change_match.group(1) if change_match else 'N/A'
                    })
        
        if len(race_files) == 0:   
            for block_match in re.finditer(r'Race Variation \d+:(.*?)(?=Race Variation \d+:|$)', race_text, re.DOTALL):
                block_text = block_match.group(1)
                
                cf_match = re.search(r'Counterfactual Image Path:\s*([^\s\n]+\.png)', block_text)
                dm_match = re.search(r'Difference Map Path:\s*([^\s\n]+\.png)', block_text)
                change_match = re.search(r'Change Percentage:\s*([\d.]+)%', block_text)
                
                if cf_match and dm_match:
                    race_files.append({
                        'counterfactual': cf_match.group(1),
                        'difference_map': dm_match.group(1),
                        'change_percentage': change_match.group(1) if change_match else 'N/A'
                    })
        
        if race_files:
            current_race = demographics.get('race', 'unknown')
            if current_race.lower() in ['white', 'caucasian']:
                target_race = "Black"
            elif current_race.lower() in ['black', 'african american']:
                target_race = "Asian"
            else:
                target_race = "White"
                
            variations.append({
                'type': 'race',
                'description': f"Changed race from {current_race} to {target_race}",
                'files': race_files
            })
            print(f"✅ Added {len(race_files)} race variations")
    
    return variations

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,  
        share=True,
        show_error=True,
        allowed_paths=[SERVED_IMAGE_DIR, "output_counterfactuals"] 
    ) 