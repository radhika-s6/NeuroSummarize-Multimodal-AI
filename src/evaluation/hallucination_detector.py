from src.nlp.entity_extractor import extract_entities

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