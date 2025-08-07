import openai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import requests
import json
from groq import Groq
import os
from huggingface_hub import login, whoami
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

class MultiLLMSummarizer:
    def __init__(self, config):
        self.config = config
        self.openai_client = None
        self.groq_client = None
        self.local_model = None
        self.local_tokenizer = None

        self.setup_clients()

    def setup_clients(self):
        """Initialize available LLM clients"""
        # OpenAI
        openai_key = self.config.get("openai_api_key")
        if openai_key:
            openai.api_key = openai_key
            self.openai_client = openai

        # Groq
        groq_key = self.config.get("groq_api_key")
        if groq_key:
            self.groq_client = Groq(api_key=groq_key)

        # Local model (if specified)
        if self.config.get('models', {}).get('local', {}).get('model_name'):
            self.setup_local_model()


    def setup_local_model(self):
        """Setup local Hugging Face model"""
        try:
            # Check login status
            try:
                user_info = whoami()
                print(f"Hugging Face: Logged in as {user_info['name']}")
            except Exception:
                print("Not logged in to Hugging Face. Attempting login...")
                login(token=HF_TOKEN, add_to_git_credential=False)

            # Load the model
            local_cfg = self.config.get("models", {}).get("local", {})
            model_name = local_cfg.get("model_name", "microsoft/BioGPT-Large")

            self.local_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if local_cfg.get("use_gpu", True) and torch.cuda.is_available() else None
            )
        except Exception as e:
            print(f"Failed to load local model: {e}")


    def create_prompts(self, text, summary_type="both"):
        """Create prompts for different summary types"""
        base_prompt = f"""
        You are a medical AI assistant specializing in neuroimaging report analysis.

        Original Report:
        {text}

        Please provide:
        """

        clinical_prompt = ""
        layperson_prompt = ""

        if summary_type == "clinical" or summary_type == "both":
            clinical_prompt = """
            1. CLINICAL SUMMARY: A concise technical summary for healthcare professionals including:
               - Key findings
               - Imaging modality used
               - Affected brain regions
               - Clinical significance
               - Recommendations
            """

        if summary_type == "layperson" or summary_type == "both":
            layperson_prompt = """
            2. PATIENT-FRIENDLY SUMMARY: A simple explanation for patients/families including:
               - What was found in plain language
               - What this means for the patient
               - Next steps in simple terms
            """

        structured_prompt = """
        3. STRUCTURED DATA: Extract the following in JSON format:
        {
            "modality": "MRI/CT/PET/etc",
            "brain_regions": ["region1", "region2"],
            "findings": ["finding1", "finding2"],
            "severity": "mild/moderate/severe",
            "recommendations": ["rec1", "rec2"]
        }
        """

        return base_prompt + clinical_prompt + layperson_prompt + structured_prompt

    def summarize_with_openai(self, text, model="gpt-3.5-turbo"):
        """Summarize using OpenAI API"""
        try:
            prompt = self.create_prompts(text)

            response = self.openai_client.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a medical AI assistant."},
                    {"role": "user", "content": prompt}
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
            return {'error': str(e), 'model': model}

    def summarize_with_groq(self, text, model="llama3-8b-8192"):
        """Summarize using Groq API (free tier)"""
        try:
            prompt = self.create_prompts(text)

            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a medical AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                model=model,
                max_tokens=1000,
                temperature=0.3
            )

            return {
                'summary': chat_completion.choices[0].message.content,
                'model': model,
                'tokens_used': chat_completion.usage.total_tokens
            }
        except Exception as e:
            return {'error': str(e), 'model': model}

    def summarize_with_local_model(self, text):
        """Summarize using local Hugging Face model"""
        try:
            prompt = self.create_prompts(text)

            inputs = self.local_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)

            with torch.no_grad():
                outputs = self.local_model.generate(
                    inputs,
                    max_length=1000,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.local_tokenizer.eos_token_id
                )

            summary = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)

            return {
                'summary': summary,
                'model': 'local_biogpt',
                'tokens_used': len(inputs[0])
            }
        except Exception as e:
            return {'error': str(e), 'model': 'local'}

    def ensemble_summarization(self, text):
        """Try multiple models and return best result"""
        results = {}

        if self.openai_client:
            results['openai'] = self.summarize_with_openai(text)

        if self.groq_client:
            results['groq'] = self.summarize_with_groq(text)

        if self.local_model:
            results['local'] = self.summarize_with_local_model(text)

        return results
