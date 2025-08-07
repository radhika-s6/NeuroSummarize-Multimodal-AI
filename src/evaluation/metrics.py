import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
from sklearn.metrics import f1_score, precision_recall_fscore_support
import re
import json
from collections import Counter
class EvaluationMetrics:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def calculate_rouge_scores(self, generated_summaries, reference_summaries):
        """Calculate ROUGE scores"""
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for gen, ref in zip(generated_summaries, reference_summaries):
            scores = self.rouge_scorer.score(ref, gen)
            for metric in rouge_scores:
                rouge_scores[metric].append(scores[metric].fmeasure)
        
        # Calculate means
        return {metric: np.mean(scores) for metric, scores in rouge_scores.items()}
    
    def calculate_bleu_score(self, generated_summaries, reference_summaries):
        """Calculate BLEU score"""
        references = [[ref.split()] for ref in reference_summaries]
        hypotheses = [gen.split() for gen in generated_summaries]
        
        bleu_score = corpus_bleu(hypotheses, references)
        return bleu_score.score
    
    def extract_entities(self, text):
        """Extract medical entities (simplified)"""
        # Brain regions
        brain_regions = re.findall(r'\b(?:frontal|temporal|parietal|occipital|cerebellum|hippocampus|amygdala)\b', text.lower())
        
        # Imaging modalities
        modalities = re.findall(r'\b(?:MRI|CT|PET|fMRI|DTI)\b', text.upper())
        
        # Findings
        findings = re.findall(r'\b(?:lesion|tumor|edema|hemorrhage|infarct|atrophy|mass)\b', text.lower())
        
        return {
            'brain_regions': list(set(brain_regions)),
            'modalities': list(set(modalities)),
            'findings': list(set(findings))
        }
    
    def calculate_entity_f1(self, generated_texts, reference_texts):
        """Calculate entity-level F1 score"""
        all_generated_entities = []
        all_reference_entities = []
        
        for gen, ref in zip(generated_texts, reference_texts):
            gen_entities = self.extract_entities(gen)
            ref_entities = self.extract_entities(ref)
            
            # Flatten all entities
            gen_flat = gen_entities['brain_regions'] + gen_entities['modalities'] + gen_entities['findings']
            ref_flat = ref_entities['brain_regions'] + ref_entities['modalities'] + ref_entities['findings']
            
            all_generated_entities.extend(gen_flat)
            all_reference_entities.extend(ref_flat)
        
        # Create binary vectors for F1 calculation
        all_entities = list(set(all_generated_entities + all_reference_entities))
        
        gen_vector = [1 if entity in all_generated_entities else 0 for entity in all_entities]
        ref_vector = [1 if entity in all_reference_entities else 0 for entity in all_entities]
        
        f1 = f1_score(ref_vector, gen_vector, average='weighted')
        return f1
    
    def detect_hallucinations(self, generated_text, source_text):
        """Detect potential hallucinations"""
        gen_entities = self.extract_entities(generated_text)
        source_entities = self.extract_entities(source_text)
        
        hallucinations = []
        
        # Check for entities in generated text not in source
        for category in gen_entities:
            for entity in gen_entities[category]:
                if entity not in source_entities[category]:
                    hallucinations.append(f"{category}: {entity}")
        
        hallucination_rate = len(hallucinations) / (len(generated_text.split()) / 100)  # Per 100 words
        
        return {
            'hallucinations': hallucinations,
            'hallucination_rate': hallucination_rate
        }
    
    def calculate_readability_score(self, text):
        """Calculate simple readability metrics"""
        sentences = text.split('.')
        words = text.split()
        
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Simple complexity score based on medical terms
        medical_terms = len(re.findall(r'\b(?:neuroimaging|radiological|pathological|anatomical)\b', text.lower()))
        complexity_score = medical_terms / len(words) * 100 if words else 0
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'complexity_score': complexity_score,
            'word_count': len(words)
        }
    
    def comprehensive_evaluation(self, results_df):
        """Run comprehensive evaluation"""
        evaluation_results = {}
        
        if 'generated_summary' in results_df.columns and 'reference_summary' in results_df.columns:
            # ROUGE scores
            rouge_scores = self.calculate_rouge_scores(
                results_df['generated_summary'].tolist(),
                results_df['reference_summary'].tolist()
            )
            evaluation_results['rouge'] = rouge_scores
            
            # BLEU score
            bleu_score = self.calculate_bleu_score(
                results_df['generated_summary'].tolist(),
                results_df['reference_summary'].tolist()
            )
            evaluation_results['bleu'] = bleu_score
            
            # Entity F1
            entity_f1 = self.calculate_entity_f1(
                results_df['generated_summary'].tolist(),
                results_df['reference_summary'].tolist()
            )
            evaluation_results['entity_f1'] = entity_f1
        
        # Hallucination analysis
        if 'generated_summary' in results_df.columns and 'source_text' in results_df.columns:
            hallucination_results = []
            for _, row in results_df.iterrows():
                hall_result = self.detect_hallucinations(row['generated_summary'], row['source_text'])
                hallucination_results.append(hall_result['hallucination_rate'])
            
            evaluation_results['avg_hallucination_rate'] = np.mean(hallucination_results)
        
        return evaluation_results