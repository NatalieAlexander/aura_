#!/usr/bin/env python3
"""
Cache Environment Setup Module
This MUST be imported before any transformers or huggingface_hub imports.
"""
import os
import sys
import argparse
from typing import List, Dict

def setup_cache_environment():
    """Set up the cache environment variables before any HF imports."""
    
    # Get the correct scratch path
    scratch_path = os.environ.get('SCRATCH')
    if scratch_path and os.path.exists(scratch_path):
        cache_dir = os.path.join(scratch_path, 'hugging_face_hub')
    else:
        # Try alternative paths
        username = os.environ.get('USER', os.environ.get('USERNAME', ''))
        if username:
            potential_paths = [
                f'/network/scratch/n/{username}/hugging_face_hub',
                f'/scratch/{username}/hugging_face_hub',
            ]
            cache_dir = None
            for path in potential_paths:
                parent_dir = os.path.dirname(path)
                if os.path.exists(parent_dir):
                    cache_dir = path
                    break
            
            if cache_dir is None:
                # Fallback to current directory
                cache_dir = os.path.join(os.getcwd(), 'hugging_face_hub')
        else:
            cache_dir = os.path.join(os.getcwd(), 'hugging_face_hub')
    
    # Create the cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set ALL the HuggingFace environment variables
    os.environ['HF_HOME'] = cache_dir
    os.environ['HF_HUB_CACHE'] = cache_dir  # This is the key one!
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    os.environ['HF_METRICS_CACHE'] = cache_dir
    
    # Print confirmation (you can comment this out in production)
    print(f"🔧 Cache configured: {cache_dir}")
    
    return cache_dir

def download_models_to_cache():
    """Download all required models to local cache to avoid rate limiting."""
    print("🚀 Starting model download to local cache...")
    
    # Import after setting up cache environment
    from transformers import (
        AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel,
        UNet2DConditionModel, AutoencoderKL
    )
    from diffusers import DiffusionPipeline
    from transformers import SamModel, SamProcessor
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    
    # List of all models used in the codebase
    models_to_download = [
        # RadEdit components
        {
            'model_id': 'microsoft/radedit',
            'model_class': UNet2DConditionModel,
            'kwargs': {'subfolder': 'unet', 'use_safetensors': True}
        },
        {
            'model_id': 'stabilityai/sdxl-vae',
            'model_class': AutoencoderKL,
            'kwargs': {'use_safetensors': True}
        },
        {
            'model_id': 'microsoft/BiomedVLP-BioViL-T',
            'model_class': AutoModel,
            'kwargs': {'trust_remote_code': True}
        },
        {
            'model_id': 'microsoft/BiomedVLP-BioViL-T',
            'model_class': AutoTokenizer,
            'kwargs': {'trust_remote_code': True, 'model_max_length': 128}
        },
        
        # MedSAM model
        {
            'model_id': 'flaviagiammarino/medsam-vit-base',
            'model_class': SamModel,
            'kwargs': {'use_safetensors': True}
        },
        {
            'model_id': 'flaviagiammarino/medsam-vit-base',
            'model_class': SamProcessor,
            'kwargs': {}
        },
        
        # MAIRA-2 model components
        {
            'model_id': 'microsoft/maira-2',
            'model_class': AutoProcessor,
            'kwargs': {'trust_remote_code': True, 'num_additional_image_tokens': 1}
        },
        {
            'model_id': 'microsoft/maira-2',
            'model_class': AutoModelForCausalLM,
            'kwargs': {'trust_remote_code': True, 'torch_dtype': 'auto'}
        }
    ]
    
    downloaded_count = 0
    failed_count = 0
    
    for model_info in models_to_download:
        model_id = model_info['model_id']
        model_class = model_info['model_class']
        kwargs = model_info.get('kwargs', {})
        
        try:
            print(f"📥 Downloading {model_class.__name__}: {model_id}")
            
            # Add token if available
            if hf_token:
                kwargs['token'] = hf_token
            
            # Download the model/processor/tokenizer
            model_class.from_pretrained(model_id, **kwargs)
            
            print(f"✅ Successfully downloaded: {model_id} ({model_class.__name__})")
            downloaded_count += 1
            
        except Exception as e:
            print(f"❌ Failed to download {model_id} ({model_class.__name__}): {e}")
            failed_count += 1
    
    # Try to download RadEdit custom pipeline
    try:
        print("📥 Downloading RadEdit custom pipeline...")
        DiffusionPipeline.from_pretrained("microsoft/radedit", custom_pipeline="microsoft/radedit")
        print("✅ Successfully downloaded RadEdit custom pipeline")
        downloaded_count += 1
    except Exception as e:
        print(f"❌ Failed to download RadEdit custom pipeline: {e}")
        failed_count += 1
    
    print(f"\n📊 DOWNLOAD SUMMARY:")
    print(f"✅ Successfully downloaded: {downloaded_count} models")
    print(f"❌ Failed downloads: {failed_count} models")
    
    if failed_count == 0:
        print("🎉 All models successfully cached! You can now run the application offline.")
    else:
        print("⚠️  Some models failed to download. Check your HF_TOKEN and internet connection.")
        print("💡 You may still be able to run with partially cached models.")
    
    return downloaded_count, failed_count

def verify_cache():
    """Verify that all required models are available in cache."""
    print("🔍 Verifying cached models...")
    
    # Import after setting up cache environment
    from transformers import (
        AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel,
        UNet2DConditionModel, AutoencoderKL
    )
    from transformers import SamModel, SamProcessor
    
    # List of models to verify
    models_to_verify = [
        ('microsoft/radedit', UNet2DConditionModel, {'subfolder': 'unet', 'local_files_only': True}),
        ('stabilityai/sdxl-vae', AutoencoderKL, {'local_files_only': True}),
        ('microsoft/BiomedVLP-BioViL-T', AutoModel, {'trust_remote_code': True, 'local_files_only': True}),
        ('microsoft/BiomedVLP-BioViL-T', AutoTokenizer, {'trust_remote_code': True, 'local_files_only': True}),
        ('flaviagiammarino/medsam-vit-base', SamModel, {'local_files_only': True}),
        ('flaviagiammarino/medsam-vit-base', SamProcessor, {'local_files_only': True}),
        ('microsoft/maira-2', AutoProcessor, {'trust_remote_code': True, 'local_files_only': True}),
        ('microsoft/maira-2', AutoModelForCausalLM, {'trust_remote_code': True, 'local_files_only': True}),
    ]
    
    available_count = 0
    missing_count = 0
    
    for model_id, model_class, kwargs in models_to_verify:
        try:
            # Try to load from cache only
            model_class.from_pretrained(model_id, **kwargs)
            print(f"✅ Available in cache: {model_id} ({model_class.__name__})")
            available_count += 1
        except Exception as e:
            print(f"❌ Missing from cache: {model_id} ({model_class.__name__})")
            missing_count += 1
    
    print(f"\n📊 CACHE VERIFICATION SUMMARY:")
    print(f"✅ Available in cache: {available_count} models")
    print(f"❌ Missing from cache: {missing_count} models")
    
    if missing_count == 0:
        print("🎉 All required models are cached and ready for offline use!")
    else:
        print("⚠️  Some models are missing from cache. Run with --download-models to cache them.")
    
    return available_count, missing_count

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Setup cache environment and download models")
    parser.add_argument('--download-models', action='store_true', 
                       help='Download all required models to local cache')
    parser.add_argument('--verify-cache', action='store_true',
                       help='Verify all required models are in cache')
    
    args = parser.parse_args()
    
    # Always set up cache environment
    cache_dir = setup_cache_environment()
    
    if args.download_models:
        print("🔄 Downloading models to cache...")
        download_models_to_cache()
    elif args.verify_cache:
        print("🔍 Verifying cache...")
        verify_cache()
    else:
        print("ℹ️  Cache environment configured.")
        print("ℹ️  Use --download-models to cache all required models")
        print("ℹ️  Use --verify-cache to check which models are cached")

if __name__ == "__main__":
    main()
else:
    # Set up the environment immediately when this module is imported
    setup_cache_environment() 