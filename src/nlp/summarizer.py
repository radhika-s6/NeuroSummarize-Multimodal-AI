import openai
from openai import OpenAI
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import requests
import json
import os
from dotenv import load_dotenv

# Import Groq with error handling
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Groq library not installed. Install with: pip install groq")

load_dotenv()

class MultiLLMSummarizer:
    def __init__(self, config):
        self.config = config
        self.openai_client = None
        self.groq_client = None
        self.local_model = None
        self.local_tokenizer = None
        self.setup_clients()

    def setup_clients(self):
        """Initialize available LLM clients with comprehensive error handling"""
        
        # OpenAI setup
        try:
            openai_key = self.config.get("api_keys", {}).get("openai", "") or os.getenv("OPENAI_API_KEY", "")
            if openai_key and openai_key.strip() and len(openai_key) > 10:
                self.openai_client = OpenAI(api_key=openai_key)
                print("OpenAI client initialized successfully")
            else:
                print("OpenAI not configured - no valid API key")
                self.openai_client = None
        except Exception as e:
            print(f"OpenAI setup failed: {e}")
            self.openai_client = None

        # Groq setup with detailed debugging
        try:
            if not GROQ_AVAILABLE:
                print("Groq library not available. Install with: pip install groq")
                self.groq_client = None
                return
            
            # Get API key from multiple sources
            groq_key = None
            
            # Try config file first
            if self.config and "api_keys" in self.config:
                groq_key = self.config["api_keys"].get("groq", "")
                print(f"Config file groq key: {'Found' if groq_key else 'Not found'}")
            
            # Try environment variable
            if not groq_key:
                groq_key = os.getenv("GROQ_API_KEY", "")
                print(f"Environment groq key: {'Found' if groq_key else 'Not found'}")
            
            # Validate key
            if groq_key and groq_key.strip() and len(groq_key) > 10 and groq_key != "your-groq-key-here":
                # Test the key by initializing client
                self.groq_client = Groq(api_key=groq_key.strip())
                
                # Test with a simple call
                test_response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="llama3-8b-8192",
                    max_tokens=10
                )
                print("Groq client initialized and tested successfully")
                
            else:
                print("Groq not configured - invalid or missing API key")
                print(f"Key length: {len(groq_key) if groq_key else 0}")
                self.groq_client = None
                
        except Exception as e:
            print(f"Groq setup failed: {e}")
            print(f"Error type: {type(e).__name__}")
            self.groq_client = None

        # Local model setup
        try:
            print("Local model ready (fallback)")
        except Exception as e:
            print(f"Local model setup failed: {e}")

    def is_groq_available(self):
        """Check if Groq client is properly initialized"""
        return self.groq_client is not None

    def create_prompts(self, text, summary_type="both"):
        """Create medical prompts"""
        return f"""You are a medical AI assistant. Analyze this neuroimaging report and provide:

1. CLINICAL SUMMARY: Concise technical summary for healthcare professionals
2. PATIENT SUMMARY: Simple explanation for patients/families  
3. STRUCTURED DATA in JSON format:
{{
    "modality": "MRI/CT/PET/etc",
    "brain_regions": ["region1", "region2"],
    "findings": ["finding1", "finding2"],
    "severity": "mild/moderate/severe",
    "recommendations": ["rec1", "rec2"]
}}

Report: {text}"""

    def summarize_with_openai(self, text, model="gpt-3.5-turbo"):
        """OpenAI summarization with error handling"""
        if not self.openai_client:
            return {'error': 'OpenAI client not initialized', 'model': model}
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a medical AI assistant."},
                    {"role": "user", "content": self.create_prompts(text)}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            return {
                'summary': response.choices[0].message.content,
                'model': model,
                'tokens_used': response.usage.total_tokens
            }
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower():
                return {'error': 'OpenAI quota exceeded. Add billing or use Groq (free)', 'model': model}
            else:
                return {'error': f'OpenAI error: {error_msg}', 'model': model}
            

    def summarize_with_groq(self, text, model="llama-3.1-8b-instant"):
        """Groq summarization with current working models"""
    
        # Check if client is initialized
        if not self.is_groq_available():
            return {
                'error': 'Groq client not initialized. Check API key configuration.', 
                'model': model,
                'debug_info': {
                    'groq_available': GROQ_AVAILABLE,
                    'client_status': 'None' if self.groq_client is None else 'Initialized'
                }
            }
    
        # List of current working groq models
        working_models = [
            "llama-3.1-8b-instant",      # Current default - fast and reliable
            "llama-3.1-70b-versatile",   # More capable but slower
            "llama-3.2-11b-text-preview", # Latest model
            "llama-3.2-3b-preview",      # Lightweight option
            "mixtral-8x7b-32768",        # Alternative model
            "gemma2-9b-it"               # Google's Gemma model
        ]
    
        # Use working model
        model_to_use = model if model in working_models else working_models[0]
    
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a medical AI assistant."},
                    {"role": "user", "content": self.create_prompts(text)}
                ],
                model=model_to_use,
                max_tokens=1000,
                temperature=0.3
            )
            return {
                'summary': response.choices[0].message.content,
                'model': model_to_use,
                'tokens_used': getattr(response.usage, 'total_tokens', 0)
            }
        except Exception as e:
            error_message = str(e)
        
            # If model is decommissioned, try next available model
            if "decommissioned" in error_message or "model" in error_message.lower():
                for backup_model in working_models[1:]:
                    try:
                        response = self.groq_client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": "You are a medical AI assistant."},
                                {"role": "user", "content": self.create_prompts(text)}
                            ],
                            model=backup_model,
                            max_tokens=1000,
                            temperature=0.3
                        )
                        return {
                            'summary': response.choices[0].message.content,
                            'model': backup_model,
                            'tokens_used': getattr(response.usage, 'total_tokens', 0)
                        }
                    except:
                        continue
        
            return {
                'error': f'Groq API error: {error_message}', 
                'model': model_to_use,
                'debug_info': {
                    'error_type': type(e).__name__,
                    'client_status': 'Initialized' if self.groq_client else 'Not initialized'
                }
            }

    def summarize_with_local_model(self, text):
        """Enhanced local model with medical focus"""
        try:
            # Extract medical information using keywords
            modality_keywords = ['MRI', 'CT', 'PET', 'fMRI', 'DTI', 'SPECT']
            region_keywords = ['frontal', 'temporal', 'parietal', 'occipital', 'cerebellum', 'hippocampus', 'thalamus', 'brainstem']
            finding_keywords = ['lesion', 'mass', 'edema', 'hemorrhage', 'infarct', 'atrophy', 'enhancement', 'stroke', 'tumor']
            
            text_lower = text.lower()
            detected_modality = next((kw for kw in modality_keywords if kw.lower() in text_lower), "Unknown")
            detected_regions = [kw for kw in region_keywords if kw.lower() in text_lower]
            detected_findings = [kw for kw in finding_keywords if kw.lower() in text_lower]
            
            # Generate summaries
            if detected_findings:
                clinical = f"Neuroimaging study ({detected_modality}) reveals {', '.join(detected_findings)}"
                if detected_regions:
                    clinical += f" involving {', '.join(detected_regions)} region(s)"
                clinical += ". Clinical correlation recommended."
                
                layperson = f"Brain scan shows changes in {', '.join(detected_regions) if detected_regions else 'brain tissue'}. Doctor will explain significance."
            else:
                clinical = f"Neuroimaging study ({detected_modality}) reviewed. No acute abnormalities noted."
                layperson = "Brain scan appears normal based on available information."
            
            structured = {
                "modality": detected_modality,
                "brain_regions": detected_regions,
                "findings": detected_findings if detected_findings else ["normal study"],
                "severity": "mild" if detected_findings else "normal",
                "recommendations": ["Clinical correlation", "Follow-up as needed"]
            }
            
            summary = f"""CLINICAL SUMMARY:
{clinical}

PATIENT SUMMARY:
{layperson}

STRUCTURED DATA:
{json.dumps(structured, indent=2)}"""
            
            return {
                'summary': summary,
                'model': 'local_enhanced',
                'tokens_used': len(text.split())
            }
        except Exception as e:
            return {'error': f'Local model error: {str(e)}', 'model': 'local'}

    def ensemble_summarization(self, text):
        """Try all models with status reporting"""
        results = {}
        
        # Always include local (most reliable)
        results['local'] = self.summarize_with_local_model(text)
        
        # Try Groq if available
        if self.is_groq_available():
            results['groq'] = self.summarize_with_groq(text)
        else:
            results['groq'] = {
                'error': 'Groq client not initialized. Check your API key in config.yaml', 
                'model': 'groq',
                'debug_info': {
                    'groq_library': GROQ_AVAILABLE,
                    'client_initialized': False
                }
            }
        
        # Try OpenAI if available
        if self.openai_client:
            results['openai'] = self.summarize_with_openai(text)
        else:
            results['openai'] = {'error': 'OpenAI not configured', 'model': 'openai'}
        
        return results