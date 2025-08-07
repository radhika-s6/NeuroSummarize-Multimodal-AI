import re

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