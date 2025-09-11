import numpy as np
import pandas as pd
import json
import re
from sklearn.metrics import f1_score, precision_recall_fscore_support, cohen_kappa_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from textstat import flesch_reading_ease, flesch_kincaid_grade
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import time
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class NeuroSummarizeEvaluator:
    """
    Comprehensive evaluation framework for NeuroSummarize system
    Based on ToR specifications and referenced papers
    """
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1
        
        # Target metrics from ToR
        self.target_f1 = 0.90
        self.target_rouge_l = 0.45  # ClinicalBERT baseline
        self.target_kappa = 0.85
        self.target_hallucination = 0.12  # 12% baseline from Chien et al.
        
        # Baseline systems for comparison
        self.baseline_systems = {
            'tesseract_gpt4': {'name': 'Tesseract + GPT-4', 'color': '#FF6B6B'},
            'layoutlm_clinical': {'name': 'LayoutLMv3 + ClinicalBERT', 'color': '#4ECDC4'},
            'textract_gpt4': {'name': 'Amazon Textract + GPT-4', 'color': '#45B7D1'},
            'neurosummarize': {'name': 'NeuroSummarize (Ours)', 'color': '#96CEB4'}
        }
        
        # Initialize results storage
        self.evaluation_results = {}
        self.metrics_history = []
        
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text using domain-specific patterns
        Based on RadGraph schema mentioned in ToR
        """
        entities = {
            'brain_regions': [],
            'modalities': [],
            'findings': [],
            'anatomical_locations': [],
            'abnormalities': []
        }
        
        text_lower = text.lower()
        
        # Brain regions (comprehensive list)
        brain_regions = [
            'frontal lobe', 'temporal lobe', 'parietal lobe', 'occipital lobe',
            'cerebellum', 'brainstem', 'hippocampus', 'amygdala', 'thalamus',
            'basal ganglia', 'corpus callosum', 'ventricles', 'cortex',
            'white matter', 'gray matter', 'cerebellar', 'frontal', 'temporal',
            'parietal', 'occipital'
        ]
        
        # Imaging modalities
        modalities = [
            'mri', 'ct', 'pet', 'spect', 'fmri', 'dti', 'dwi',
            'magnetic resonance', 'computed tomography', 'positron emission'
        ]
        
        # Clinical findings
        findings = [
            'lesion', 'mass', 'tumor', 'hemorrhage', 'infarct', 'stroke',
            'edema', 'atrophy', 'enhancement', 'signal', 'density',
            'abnormality', 'pathology', 'inflammation', 'ischemia'
        ]
        
        # Extract entities
        for region in brain_regions:
            if region in text_lower:
                entities['brain_regions'].append(region)
        
        for modality in modalities:
            if modality in text_lower:
                entities['modalities'].append(modality)
        
        for finding in findings:
            if finding in text_lower:
                entities['findings'].append(finding)
        
        return entities
    
    def calculate_entity_f1(self, predicted_entities: Dict, gold_entities: Dict) -> Dict[str, float]:
        """
        Calculate entity-level F1 scores as specified in ToR
        """
        f1_scores = {}
        
        for entity_type in predicted_entities.keys():
            if entity_type in gold_entities:
                pred_set = set(predicted_entities[entity_type])
                gold_set = set(gold_entities[entity_type])
                
                if len(gold_set) == 0 and len(pred_set) == 0:
                    f1_scores[entity_type] = 1.0
                elif len(gold_set) == 0:
                    f1_scores[entity_type] = 0.0
                else:
                    intersection = len(pred_set.intersection(gold_set))
                    precision = intersection / len(pred_set) if len(pred_set) > 0 else 0
                    recall = intersection / len(gold_set) if len(gold_set) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    f1_scores[entity_type] = f1
            else:
                f1_scores[entity_type] = 0.0
        
        # Overall F1 (macro average)
        f1_scores['overall'] = np.mean(list(f1_scores.values()))
        return f1_scores
    
    def calculate_rouge_scores(self, generated_summary: str, reference_summary: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores for summary quality assessment
        """
        scores = self.rouge_scorer.score(reference_summary, generated_summary)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge1_p': scores['rouge1'].precision,
            'rouge1_r': scores['rouge1'].recall,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rouge2_p': scores['rouge2'].precision,
            'rouge2_r': scores['rouge2'].recall,
            'rougeL_f': scores['rougeL'].fmeasure,
            'rougeL_p': scores['rougeL'].precision,
            'rougeL_r': scores['rougeL'].recall
        }
    
    def calculate_bleu_score(self, generated_summary: str, reference_summary: str) -> float:
        """
        Calculate BLEU score for summary quality
        """
        reference_tokens = reference_summary.split()
        generated_tokens = generated_summary.split()
        
        try:
            score = sentence_bleu([reference_tokens], generated_tokens, 
                                smoothing_function=self.smoothing_function)
            return score
        except:
            return 0.0
    
    def detect_hallucinations(self, generated_text: str, source_text: str) -> Dict[str, Any]:
        """
        Detect hallucinations in generated summaries
        Based on Chien et al. (2024) methodology
        """
        gen_entities = self.extract_entities(generated_text)
        source_entities = self.extract_entities(source_text)
        
        hallucinated_entities = []
        total_generated = 0
        
        for entity_type in gen_entities:
            for entity in gen_entities[entity_type]:
                total_generated += 1
                if entity not in source_entities.get(entity_type, []):
                    hallucinated_entities.append({
                        'entity': entity,
                        'type': entity_type,
                        'context': self._find_entity_context(entity, generated_text)
                    })
        
        hallucination_rate = len(hallucinated_entities) / total_generated if total_generated > 0 else 0
        
        return {
            'hallucination_rate': hallucination_rate,
            'hallucinated_entities': hallucinated_entities,
            'total_generated_entities': total_generated,
            'hallucinated_count': len(hallucinated_entities)
        }
    
    def _find_entity_context(self, entity: str, text: str, context_window: int = 50) -> str:
        """Find context around an entity in text"""
        text_lower = text.lower()
        entity_lower = entity.lower()
        
        index = text_lower.find(entity_lower)
        if index == -1:
            return ""
        
        start = max(0, index - context_window)
        end = min(len(text), index + len(entity) + context_window)
        return text[start:end]
    
    def calculate_readability_scores(self, text: str) -> Dict[str, float]:
        """
        Calculate readability scores for clinical utility assessment
        """
        return {
            'flesch_reading_ease': flesch_reading_ease(text),
            'flesch_kincaid_grade': flesch_kincaid_grade(text)
        }
    
    def evaluate_clinical_utility(self, summaries: List[str], ratings: List[int]) -> Dict[str, float]:
        """
        Evaluate clinical utility using 5-point Likert scale
        As specified in ToR
        """
        ratings_array = np.array(ratings)
        
        return {
            'mean_rating': np.mean(ratings_array),
            'std_rating': np.std(ratings_array),
            'median_rating': np.median(ratings_array),
            'utility_score': np.mean(ratings_array) / 5.0,  # Normalized to 0-1
            'high_utility_percentage': np.sum(ratings_array >= 4) / len(ratings_array)
        }
    
    def calculate_inter_annotator_agreement(self, annotations1: List, annotations2: List) -> float:
        """
        Calculate Cohen's Kappa for inter-annotator agreement
        Target: κ > 0.85 as specified in ToR
        """
        try:
            return cohen_kappa_score(annotations1, annotations2)
        except:
            return 0.0
    
    def benchmark_against_baselines(self, test_data: List[Dict]) -> Dict[str, Dict]:
        """
        Comprehensive benchmarking against baseline systems
        """
        results = {}
        
        for system_name in self.baseline_systems:
            results[system_name] = {
                'f1_scores': [],
                'rouge_scores': [],
                'bleu_scores': [],
                'hallucination_rates': [],
                'processing_times': [],
                'readability_scores': []
            }
        
        # Simulate baseline performance (replace with actual implementation)
        for item in test_data:
            # NeuroSummarize results (from your actual system)
            if 'neurosummarize_summary' in item:
                ns_entities = self.extract_entities(item['neurosummarize_summary'])
                gold_entities = self.extract_entities(item['gold_summary'])
                
                f1_result = self.calculate_entity_f1(ns_entities, gold_entities)
                rouge_result = self.calculate_rouge_scores(item['neurosummarize_summary'], item['gold_summary'])
                bleu_result = self.calculate_bleu_score(item['neurosummarize_summary'], item['gold_summary'])
                hallucination_result = self.detect_hallucinations(item['neurosummarize_summary'], item['source_text'])
                
                results['neurosummarize']['f1_scores'].append(f1_result['overall'])
                results['neurosummarize']['rouge_scores'].append(rouge_result['rougeL_f'])
                results['neurosummarize']['bleu_scores'].append(bleu_result)
                results['neurosummarize']['hallucination_rates'].append(hallucination_result['hallucination_rate'])
        
        # Simulate baseline results based on literature
        num_samples = len(test_data)
        
        # Tesseract + GPT-4 (estimated performance)
        results['tesseract_gpt4']['f1_scores'] = np.random.normal(0.75, 0.08, num_samples).clip(0, 1)
        results['tesseract_gpt4']['rouge_scores'] = np.random.normal(0.42, 0.05, num_samples).clip(0, 1)
        results['tesseract_gpt4']['hallucination_rates'] = np.random.normal(0.15, 0.03, num_samples).clip(0, 1)
        
        # LayoutLMv3 + ClinicalBERT (from referenced papers)
        results['layoutlm_clinical']['f1_scores'] = np.random.normal(0.82, 0.06, num_samples).clip(0, 1)
        results['layoutlm_clinical']['rouge_scores'] = np.random.normal(0.45, 0.04, num_samples).clip(0, 1)
        results['layoutlm_clinical']['hallucination_rates'] = np.random.normal(0.08, 0.02, num_samples).clip(0, 1)
        
        # Amazon Textract + GPT-4
        results['textract_gpt4']['f1_scores'] = np.random.normal(0.78, 0.07, num_samples).clip(0, 1)
        results['textract_gpt4']['rouge_scores'] = np.random.normal(0.48, 0.05, num_samples).clip(0, 1)
        results['textract_gpt4']['hallucination_rates'] = np.random.normal(0.12, 0.03, num_samples).clip(0, 1)
        
        return results
    
    def generate_comprehensive_report(self, test_data: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        """
        benchmark_results = self.benchmark_against_baselines(test_data)
    
        # Calculate summary statistics - FIXED VERSION
        summary_stats = {}
        for system_name, results in benchmark_results.items():
            # Fix: Check array length instead of truthiness
            if len(results['f1_scores']) > 0:  # FIXED LINE
                summary_stats[system_name] = {
                    'mean_f1': np.mean(results['f1_scores']),
                    'std_f1': np.std(results['f1_scores']),
                    'mean_rouge_l': np.mean(results['rouge_scores']) if len(results['rouge_scores']) > 0 else 0,
                    'std_rouge_l': np.std(results['rouge_scores']) if len(results['rouge_scores']) > 0 else 0,
                    'mean_hallucination': np.mean(results['hallucination_rates']) if len(results['hallucination_rates']) > 0 else 0,
                    'std_hallucination': np.std(results['hallucination_rates']) if len(results['hallucination_rates']) > 0 else 0
                }
    
        # Performance comparison against targets
        target_analysis = {}
        if 'neurosummarize' in summary_stats:
            ns_stats = summary_stats['neurosummarize']
            target_analysis = {
                'f1_target_met': ns_stats['mean_f1'] >= self.target_f1,
                'rouge_improvement': ns_stats['mean_rouge_l'] - self.target_rouge_l,
                'hallucination_below_target': ns_stats['mean_hallucination'] <= self.target_hallucination,
                'overall_performance_grade': self._calculate_performance_grade(ns_stats)
            }
    
        return {
            'benchmark_results': benchmark_results,
            'summary_statistics': summary_stats,
            'target_analysis': target_analysis,
            'evaluation_timestamp': datetime.now().isoformat(),
            'num_test_samples': len(test_data)
        }
    
    def _calculate_performance_grade(self, stats: Dict[str, float]) -> str:
        """Calculate overall performance grade"""
        score = 0
        
        # F1 score component (40%)
        if stats['mean_f1'] >= 0.90:
            score += 40
        elif stats['mean_f1'] >= 0.85:
            score += 35
        elif stats['mean_f1'] >= 0.80:
            score += 30
        elif stats['mean_f1'] >= 0.75:
            score += 25
        
        # ROUGE-L component (30%)
        if stats['mean_rouge_l'] >= 0.50:
            score += 30
        elif stats['mean_rouge_l'] >= 0.45:
            score += 25
        elif stats['mean_rouge_l'] >= 0.40:
            score += 20
        
        # Hallucination component (30%)
        if stats['mean_hallucination'] <= 0.05:
            score += 30
        elif stats['mean_hallucination'] <= 0.10:
            score += 25
        elif stats['mean_hallucination'] <= 0.15:
            score += 20
        
        if score >= 85:
            return "A (Excellent)"
        elif score >= 75:
            return "B (Good)"
        elif score >= 65:
            return "C (Satisfactory)"
        else:
            return "D (Needs Improvement)"