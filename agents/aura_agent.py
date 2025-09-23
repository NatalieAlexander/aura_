import setup_cache_env
from smolagents import CodeAgent, TransformersModel, InferenceClientModel
from smolagents import PromptTemplates, PlanningPromptTemplate, ManagedAgentPromptTemplate, FinalAnswerPromptTemplate
from tools import (
    GenerateCXRReportTool, 
    GroundCXRFindingsTool, 
    GetGroundedReportTool,
    OverlayFindingsTool,
    RadEditTool,
    MedSAMSegmentationTool,
    DifferenceMapTool,
    SessionManagerTool,
    CheXagentVQATool,
    TorchXrayVisionTool
)
import logging
import yaml
import os
import torch
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForCausalLM

load_dotenv()
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = os.getenv('MODEL_NAME', 'microsoft/maira-2')
hf_token = os.getenv('HF_TOKEN')

using_mock_model = False
maira_processor = None
maira_model = None

try:
    logger.info(f"Loading MAIRA-2 model on {device}...")
    maira_processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token,
    )
    
    maira_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token,
    ).eval()
    
    maira_model = maira_model.to(device)
    logger.info("MAIRA-2 model loaded successfully")
    
except Exception as e:
    logger.error(f"Error loading MAIRA-2 model: {e}")
    logger.info("Using mock model for demonstration purposes.")
    using_mock_model = True

class AURAAgent:
    def __init__(self, model_id: str = None):
        self.name = "aura_medical_agent"
        self.description = "AURA: A Multi-Modal Medical Agent for Understanding, Reasoning & Annotation. Provides comprehensive medical image analysis through counterfactuals, VQA, demographic analysis, and intelligent parameter search."
        
        try:
            if not model_id:
                model_id = 'Qwen/Qwen2.5-Coder-32B-Instruct'  
                
            hf_token = os.getenv('HF_TOKEN')
            
            if not hf_token:
                logger.warning("No HF_TOKEN found in environment. Some features may be limited.")
            
            logger.info(f"Initializing Visual Explainability Agent with model: {model_id}")
            
            try:
                logger.info(f"Initializing TransformersModel with model_id: {model_id}")
                
                self.model = TransformersModel(
                    model_id=model_id,
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True,
                )
                
                logger.info(f"Successfully initialized TransformersModel: {model_id}")
                    
            except Exception as e:
                logger.error(f"Error initializing model with auto device map: {e}")
                logger.warning("Trying again with CPU fallback...")
                
                try:
                    self.model = TransformersModel(
                        model_id=model_id,
                        device_map="cpu",
                        torch_dtype="auto",
                        trust_remote_code=True,
                        token=hf_token
                    )
                    logger.info(f"Successfully initialized TransformersModel on CPU: {model_id}")
                except Exception as e1:
                    logger.error(f"Error initializing with CPU fallback: {e1}")
                    logger.warning("Falling back to meta-llama model for compatibility")
                    
                    try:
                        fallback_model = "meta-llama/Meta-Llama-3-8B-Instruct"
                        logger.info(f"Attempting to initialize fallback model: {fallback_model}")
                        
                        self.model = TransformersModel(
                            model_id=fallback_model,
                            device_map="auto",
                            torch_dtype="auto",
                            token=hf_token
                        )
                        
                        logger.info(f"Successfully initialized fallback TransformersModel: {fallback_model}")
                        
                    except Exception as e2:
                        logger.error(f"Error initializing fallback model: {e2}")
                        raise RuntimeError("Failed to initialize any suitable model for the agent")
        
            self.prompt_config = self._load_prompt_config()
            prompt_templates = self._create_prompt_templates()
            
            session_tool = SessionManagerTool()
            report_tool = GenerateCXRReportTool(maira_model=maira_model, maira_processor=maira_processor)
            grounding_tool = GroundCXRFindingsTool(maira_model=maira_model, maira_processor=maira_processor)
            grounded_report_tool = GetGroundedReportTool(maira_model=maira_model, maira_processor=maira_processor)
            overlay_tool = OverlayFindingsTool()
            counterfactual_tool = RadEditTool()
            segmentation_tool = MedSAMSegmentationTool()
            difference_map_tool = DifferenceMapTool()
            vqa_tool = CheXagentVQATool()
            pathology_tool = TorchXrayVisionTool()
            
            try:
                from tools.anatomical_segmentation_tool import AnatomicalSegmentationTool
                anatomical_tool = AnatomicalSegmentationTool()
                logger.info("AnatomicalSegmentationTool loaded successfully for ADD operations")
            except ImportError as e:
                logger.warning(f"Could not load AnatomicalSegmentationTool: {e}")
                anatomical_tool = None
            
            tools_list = [session_tool, report_tool, grounding_tool, grounded_report_tool, overlay_tool, 
                         counterfactual_tool, segmentation_tool, difference_map_tool, vqa_tool, pathology_tool]
            
            if anatomical_tool is not None:
                tools_list.append(anatomical_tool)
                logger.info("Added AnatomicalSegmentationTool to available tools")
            
            self.tools = tools_list
            
            self.agent = CodeAgent(
                name="visual_explainability_analyst",
                description="A sophisticated agent that provides visual explainability for chest X-ray images through counterfactuals, VQA, demographic analysis, and intelligent parameter search",
                tools=self.tools,
                model=self.model,
                prompt_templates=prompt_templates,
                max_steps=25,
                verbosity_level=1,
                additional_authorized_imports=["json", "math", "re", "base64", "datetime", "io", "typing", 
                                              "requests", "PIL", "urllib", "urllib.request", "os", "matplotlib", "matplotlib.pyplot", "posixpath", "ntpath", "itertools", "time", "glob", "numpy", "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont"]
            )
            
            logger.info("Visual Explainability Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Visual Explainability Agent: {e}")
            raise
    
    def _load_prompt_config(self):
        try:
            prompt_file = "prompt.yml"
            if os.path.exists(prompt_file):
                with open(prompt_file, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info("Loaded specialized visual explainability prompt")
                return config
            else:
                logger.warning(f"Prompt file {prompt_file} not found, using default prompt")
                return {}
        except Exception as e:
            logger.error(f"Error loading prompt config: {e}")
            return {}
    
    def _create_prompt_templates(self):
        try:
            if self.prompt_config and 'system_prompt' in self.prompt_config:
                system_prompt = self.prompt_config['system_prompt']
                
                planning_config = self.prompt_config.get('planning', {})
                planning_template = PlanningPromptTemplate(
                    initial_plan=planning_config.get('initial_plan', 
                        "I need to provide comprehensive visual explainability analysis including VQA, counterfactuals, and demographic analysis."),
                    update_plan_pre_messages=planning_config.get('update_plan_pre_messages', 
                        "Let me review what I've learned about this chest X-ray and the user's specific explainability needs."),
                    update_plan_post_messages=planning_config.get('update_plan_post_messages', 
                        "Based on my analysis so far, I'll continue with the most appropriate explainability approach.")
                )
                
                managed_agent_config = self.prompt_config.get('managed_agent', {})
                managed_agent_template = ManagedAgentPromptTemplate(
                    task=managed_agent_config.get('task', 
                        "Provide comprehensive visual explainability for chest X-ray images using advanced counterfactual generation, VQA, demographic analysis, and parameter optimization."),
                    report=managed_agent_config.get('report', 
                        "I've completed the visual explainability analysis with the following capabilities:")
                )
                
                final_answer_config = self.prompt_config.get('final_answer', {})
                final_answer_template = FinalAnswerPromptTemplate(
                    pre_messages=final_answer_config.get('pre_messages', 
                        "I have completed all analysis steps including VQA, counterfactual generation, and visual explainability. I will now consolidate the results and provide the final comprehensive analysis."),
                    post_messages=final_answer_config.get('post_messages', 
                        "This concludes my comprehensive visual explainability analysis of the chest X-ray image.")
                )
                
            else:
                system_prompt = """You are VisualExplainabilityAnalyst, an advanced AI radiologist specializing in comprehensive visual explainability for chest X-ray images. You combine counterfactual generation, visual question answering, parameter optimization, and demographic analysis to provide deep insights into medical imaging."""

                planning_template = PlanningPromptTemplate(
                    initial_plan="I need to provide comprehensive visual explainability analysis including VQA, counterfactuals, and demographic analysis.",
                    update_plan_pre_messages="Let me review what I've learned about this chest X-ray and the user's specific explainability needs.",
                    update_plan_post_messages="Based on my analysis so far, I'll continue with the most appropriate explainability approach."
                )
                
                managed_agent_template = ManagedAgentPromptTemplate(
                    task="Provide comprehensive visual explainability for chest X-ray images using advanced counterfactual generation, VQA, demographic analysis, and parameter optimization.",
                    report="I've completed the visual explainability analysis with the following capabilities:"
                )
                
                final_answer_template = FinalAnswerPromptTemplate(
                    pre_messages="I have completed all analysis steps including VQA, counterfactual generation, and visual explainability. I will now consolidate the results and provide the final comprehensive analysis.",
                    post_messages="This concludes my comprehensive visual explainability analysis of the chest X-ray image."
                )
            
            return PromptTemplates(
                system_prompt=system_prompt,
                planning=planning_template,
                managed_agent=managed_agent_template,
                final_answer=final_answer_template
            )
            
        except Exception as e:
            logger.error(f"Error creating prompt templates: {e}")
            return PromptTemplates(
                system_prompt="You are a visual explainability expert for chest X-rays.",
                planning=PlanningPromptTemplate(
                    initial_plan="Provide comprehensive visual explainability analysis.",
                    update_plan_pre_messages="Reviewing progress...",
                    update_plan_post_messages="Continuing with next steps..."
                ),
                managed_agent=ManagedAgentPromptTemplate(
                    task="Provide visual explainability for chest X-ray images.",
                    report="Analysis complete:"
                ),
                final_answer=FinalAnswerPromptTemplate(
                    pre_messages="Final analysis:",
                    post_messages="Analysis complete."
                )
            )
    
    def generate_counterfactuals_with_grid_search(self, image_path: str, prompt: str, 
                                                 search_space: dict = None, max_combinations: int = 6) -> dict:
        try:
            if search_space is None:
                search_space = {
                    'weights': [5.0, 7.5, 10.0],
                    'num_inference_steps': [70, 100, 150],
                    'skip_ratio': [0.2, 0.3, 0.4]
                }
            
            logger.info(f"Starting grid search with space: {search_space}")
            
            import itertools
            param_names = list(search_space.keys())
            param_values = list(search_space.values())
            
            all_combinations = list(itertools.product(*param_values))
            
            if len(all_combinations) > max_combinations:
                import random
                random.seed(42)
                combinations = random.sample(all_combinations, max_combinations)
                logger.info(f"Limited to {max_combinations} random combinations from {len(all_combinations)} total")
            else:
                combinations = all_combinations
                logger.info(f"Using all {len(combinations)} combinations")
            
            results = []
            
            for i, combination in enumerate(combinations):
                logger.info(f"Testing combination {i+1}/{len(combinations)}: {dict(zip(param_names, combination))}")
                
                params = dict(zip(param_names, combination))
                
                cf_result = self.agent.run(
                    f"Generate a single counterfactual with specific parameters: {params}. "
                    f"Image: {image_path}, Prompt: {prompt}"
                )
                
                results.append({
                    'parameters': params,
                    'combination_index': i,
                    'result': cf_result
                })
            
            return {
                'search_space': search_space,
                'total_combinations': len(combinations),
                'results': results,
                'best_combination': self._select_best_combination(results)
            }
            
        except Exception as e:
            logger.error(f"Error in grid search: {e}")
            return {'error': str(e)}
    
    def _select_best_combination(self, results: list) -> dict:
        for result in results:
            if 'error' not in result.get('result', {}):
                return result
        
        return results[0] if results else {}
    
    def generate_demographic_explanations(self, image_path: str, demographics: dict) -> dict:
        try:
            explanations = {}
            
            current_age = demographics.get('age', 'unknown')
            current_sex = demographics.get('sex', 'unknown')
            current_race = demographics.get('race', 'unknown')
            
            logger.info(f"Generating demographic explanations for: {demographics}")
            
            if current_age != 'unknown':
                age_prompts = []
                if 'young' not in current_age.lower() and 'child' not in current_age.lower():
                    age_prompts.append("make this patient appear younger, like a young adult in their 20s")
                if 'old' not in current_age.lower() and 'elderly' not in current_age.lower():
                    age_prompts.append("make this patient appear older, like an elderly person in their 70s")
                
                explanations['age_variations'] = []
                for prompt in age_prompts:
                    result = self.agent.run(f"Generate counterfactual: {prompt}. Image: {image_path}")
                    explanations['age_variations'].append({
                        'prompt': prompt,
                        'result': result
                    })
            
            if current_sex.lower() in ['male', 'female']:
                opposite_sex = 'female' if current_sex.lower() == 'male' else 'male'
                sex_prompt = f"make this patient appear as a {opposite_sex}"
                
                explanations['sex_variation'] = {
                    'prompt': sex_prompt,
                    'result': self.agent.run(f"Generate counterfactual: {sex_prompt}. Image: {image_path}")
                }
            
            if current_race != 'unknown':
                race_variations = ['white', 'black', 'asian', 'hispanic']
                current_race_lower = current_race.lower()
                
                explanations['race_variations'] = []
                for race in race_variations:
                    if race != current_race_lower:
                        race_prompt = f"make this patient appear as a {race} person"
                        result = self.agent.run(f"Generate counterfactual: {race_prompt}. Image: {image_path}")
                        explanations['race_variations'].append({
                            'race': race,
                            'prompt': race_prompt,
                            'result': result
                        })
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating demographic explanations: {e}")
            return {'error': str(e)}
    
    def run_chat(self, user_message: str) -> str:
        try:
            logger.info(f"Visual Explainability Agent received message: {user_message[:100]}...")
            return self.agent.run(user_message)
        except Exception as e:
            logger.error(f"Error in VisualExpAgent chat: {e}")
            return f"Error: {str(e)}" 