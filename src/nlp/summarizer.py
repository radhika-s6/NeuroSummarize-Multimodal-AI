import openai
from openai import OpenAI
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import requests
import json
from groq import Groq
import os
from huggingface_hub import login, whoami
from dotenv import load_dotenv

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
        """Initialize available LLM clients"""
        # OpenAI
        openai_key = self.config.get("api_keys", {}).get("openai")
        if openai_key:
            openai.api_key = openai_key
            self.openai_client = openai

        # Groq
        groq_key = self.config.get("api_keys", {}).get("groq")
        if groq_key:
            self.groq_client = Groq(api_key=groq_key)

    # ----------------------------
    # Prompt creation
    # ----------------------------
    def create_prompts(self, text, summary_type="both"):
        base_prompt = f"""
        You are a medical AI assistant specializing in neuroimaging report analysis.

        Original Report:
        {text}

        Please provide:
        """

        clinical_prompt = ""
        layperson_prompt = ""

        if summary_type in ("clinical", "both"):
            clinical_prompt = """
            1. CLINICAL SUMMARY: A concise technical summary for healthcare professionals including:
               - Key findings
               - Imaging modality used
               - Affected brain regions
               - Clinical significance
               - Recommendations
            """

        if summary_type in ("layperson", "both"):
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
        """Summarize using Groq API"""
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

    def ensemble_summarization(self, text):
        """Try multiple models and return best result"""
        results = {}
        if self.openai_client:
            results['openai'] = self.summarize_with_openai(text)
        if self.groq_client:
            results['groq'] = self.summarize_with_groq(text)
        
        return results