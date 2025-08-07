import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import random

class DataLoader:
    def __init__(self, data_dir="data/"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.synthetic_dir = self.data_dir / "synthetic"

        for dir_path in [self.raw_dir, self.processed_dir, self.synthetic_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def load_text_reports(self):
        """Load text-based neuroimaging reports from .txt files"""
        reports = []
        for file in self.raw_dir.glob("*.txt"):
            try:
                text = file.read_text(encoding="utf-8")
                reports.append({
                    "filename": file.name,
                    "text": text.strip(),
                    "source": "txt"
                })
            except Exception as e:
                print(f"[WARN] Skipping {file.name}: {e}")
        return pd.DataFrame(reports)

    def load_acrin_metadata(self):
        """Load tabular ACRIN metadata (xlsx)"""
        acrin_file = self.raw_dir / "ACRIN-DSC-MR-Brain-NBIA-manifes-nbia-digest.xlsx"
        if acrin_file.exists():
            df = pd.read_excel(acrin_file)
            if 'Comments' in df.columns:
                df['text'] = df['Comments']
            else:
                df['text'] = df.apply(lambda row: " ".join(str(v) for v in row.values), axis=1)
            df['source'] = 'acrin'
            df['filename'] = acrin_file.name
            return df[['filename', 'text', 'source']]
        else:
            return pd.DataFrame()

    def load_openneuro_json_reports(self):
        """Load OpenNeuro JSON reports with textual fields"""
        base_path = self.raw_dir / "openneuro"
        json_files = list(base_path.rglob("*.json"))
        reports = []

        for file in json_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    flat_text = self.flatten_json(data)
                    reports.append({
                        "filename": file.name,
                        "text": flat_text,
                        "source": "openneuro-json"
                    })
            except Exception as e:
                print(f"[WARN] Failed to parse {file.name}: {e}")

        return pd.DataFrame(reports)

    def load_openneuro_tsv_reports(self):
        """Extract diagnostic info and notes from OpenNeuro .tsv files"""
        base_path = self.raw_dir / "openneuro"
        tsv_files = list(base_path.rglob("*.tsv"))
        records = []

        for tsv_file in tsv_files:
            try:
                df = pd.read_csv(tsv_file, sep='\t')
                for _, row in df.iterrows():
                    row_text = []
                    for col in df.columns:
                        val = row[col]
                        if isinstance(val, str) and len(val.strip()) > 5:
                            row_text.append(f"{col}: {val}")
                    if row_text:
                        records.append({
                            "filename": tsv_file.name,
                            "text": "\n".join(row_text),
                            "source": "openneuro-tsv"
                        })
            except Exception as e:
                print(f"[WARN] Skipped {tsv_file.name} due to: {e}")

        return pd.DataFrame(records)

    def load_all_structured_reports(self):
        """Aggregate all reports into one DataFrame"""
        dfs = [
            self.load_text_reports(),
            self.load_acrin_metadata(),
            self.load_openneuro_json_reports(),
            self.load_openneuro_tsv_reports()
        ]
        combined = pd.concat(dfs, ignore_index=True)
        combined.dropna(subset=['text'], inplace=True)
        return combined

    def flatten_json(self, obj, prefix=''):
        """Recursively flatten nested JSON to plain text"""
        text_lines = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                full_key = f"{prefix}.{k}" if prefix else k
                text_lines.append(self.flatten_json(v, full_key))
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                text_lines.append(self.flatten_json(item, f"{prefix}[{idx}]"))
        else:
            if isinstance(obj, str) and len(obj.strip()) > 3:
                text_lines.append(f"{prefix}: {obj}")
        return "\n".join([t for t in text_lines if t])
    



# Enhanced Data Loader Class for Neuroimaging Reports

class EnhancedDataLoader:
    def __init__(self, data_dir="data/"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.synthetic_dir = self.data_dir / "synthetic"
        
        # Neuroimaging-specific vocabularies
        self.modalities = ["MRI", "CT", "PET", "fMRI", "DTI", "SPECT"]
        self.sequences = ["T1-weighted", "T2-weighted", "FLAIR", "DWI", "ADC", "SWI", "T2*"]
        self.brain_regions = [
            "frontal lobe", "temporal lobe", "parietal lobe", "occipital lobe",
            "cerebellum", "brainstem", "hippocampus", "amygdala", "thalamus",
            "basal ganglia", "corpus callosum", "ventricular system",
            "frontal cortex", "motor cortex", "visual cortex", "cerebellar hemispheres"
            ]
        self.findings = [
            "mass lesion", "edema", "hemorrhage", "infarct", "atrophy",
            "white matter hyperintensities", "microbleeds", "calcifications",
            "enhancement", "restricted diffusion", "signal abnormality",
            "volume loss", "gliosis", "demyelination", "ischemic changes"
        ]
        self.severities = ["mild", "moderate", "severe", "extensive", "focal", "diffuse"]
        self.contrasts = ["gadolinium", "iodinated contrast", "non-contrast"]
        
    def generate_realistic_neuroimaging_reports(self, num_samples=100):
        """Generate realistic neuroimaging reports with proper medical terminology"""
        reports = []
        
        report_templates = [
            self._generate_mri_brain_template,
            self._generate_ct_brain_template,
            self._generate_stroke_template,
            self._generate_tumor_template,
            self._generate_normal_template
            ]
        
        for i in range(num_samples):
            template_func = random.choice(report_templates)
            report_data = template_func(i)
            reports.append(report_data)
            
        return pd.DataFrame(reports)
    
    def _generate_mri_brain_template(self, report_id):
        """Generate MRI brain report"""
        modality = "MRI"
        sequences = random.sample(self.sequences, random.randint(2, 4))
        contrast = random.choice(self.contrasts)
        region = random.choice(self.brain_regions)
        finding = random.choice(self.findings)
        severity = random.choice(self.severities)
        
        clinical_text = f"""
        EXAMINATION: {modality} Brain {contrast}
        SEQUENCES: {', '.join(sequences)}
        CLINICAL HISTORY: {random.choice(['Headache', 'Neurological symptoms', 'Follow-up', 'Screening'])}

        FINDINGS:
        The brain parenchyma demonstrates {severity} {finding} involving the {region}. 
        {random.choice(['No', 'Mild', 'Moderate'])} mass effect is observed.
        The ventricular system appears {random.choice(['normal', 'mildly enlarged', 'stable'])}.
        {random.choice(['No', 'Subtle'])} midline shift is present.

        IMPRESSION:
        {severity.title()} {finding} in the {region}, {random.choice(['likely', 'possibly', 'consistent with'])} 
        {random.choice(['ischemic changes', 'inflammatory process', 'degenerative changes', 'post-traumatic changes'])}.
        {random.choice(['Follow-up recommended', 'Clinical correlation advised', 'Further evaluation suggested'])}.
        
        """.strip()
        
        # Embedded JSON
        structured_data = {
            "report_id": f"MRI_{report_id:04d}",
            "modality": modality,
            "sequences": sequences,
            "brain_regions": [region],
            "findings": [finding],
            "severity": severity,
            "contrast": contrast,
            "recommendations": ["Follow-up", "Clinical correlation"],
            "date": (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")
        }
        
        json_embedded = f"\n\nSTRUCTURED_DATA: {json.dumps(structured_data)}"
        
        return {
            'report_id': f'synthetic_mri_{report_id:04d}',
            'text': clinical_text + json_embedded,
            'modality': modality,
            'source': 'synthetic',
            'type': 'neuroimaging',
            'structured_data': structured_data
        }
    
    def _generate_ct_brain_template(self, report_id):
        """Generate CT brain report"""
        modality = "CT"
        contrast = random.choice(["non-contrast", "with contrast"])
        region = random.choice(self.brain_regions)
        finding = random.choice(["hemorrhage", "infarct", "mass", "edema", "calcification"])
        
        clinical_text = f"""
        EXAMINATION: CT Brain {contrast}
        TECHNIQUE: Axial images obtained from skull base to vertex
        CLINICAL INDICATION: {random.choice(['Acute neurological deficit', 'Head trauma', 'Altered mental status'])}

        FINDINGS:
        There is evidence of {finding} in the {region}. 
        {random.choice(['No', 'Mild', 'Significant'])} surrounding edema is noted.
        The gray-white matter differentiation is {random.choice(['preserved', 'slightly blurred', 'maintained'])}.
        Ventricular size and configuration appear {random.choice(['normal', 'stable', 'within normal limits'])}.

        IMPRESSION:
        Acute {finding} involving {region}. 
        {random.choice(['Recommend urgent neurology consultation', 'Follow-up imaging suggested', 'Clinical correlation recommended'])}.
        
        """.strip()
        
        structured_data = {
            "report_id": f"CT_{report_id:04d}",
            "modality": modality,
            "brain_regions": [region],
            "findings": [finding],
            "severity": random.choice(self.severities),
            "contrast": contrast,
            "recommendations": ["Neurology consultation", "Follow-up imaging"],
            "urgency": "acute"
        }
        
        json_embedded = f"\n\nSTRUCTURED_DATA: {json.dumps(structured_data)}"
        
        return {
            'report_id': f'synthetic_ct_{report_id:04d}',
            'text': clinical_text + json_embedded,
            'modality': modality,
            'source': 'synthetic',
            'type': 'neuroimaging',
            'structured_data': structured_data
        }
    
    def _generate_stroke_template(self, report_id):
        """Generate stroke-specific report"""
        modality = random.choice(["MRI", "CT"])
        vessel = random.choice(["middle cerebral artery", "anterior cerebral artery", "posterior cerebral artery"])
        territory = random.choice(["left MCA territory", "right MCA territory", "bilateral watershed"])
        
        clinical_text = f"""
        EXAMINATION: {modality} Brain - Stroke Protocol
        CLINICAL HISTORY: Acute stroke symptoms, onset {random.randint(1, 12)} hours ago

        FINDINGS:
        Acute infarction in the {territory} with restricted diffusion on DWI sequences.
        The {vessel} territory shows evidence of cytotoxic edema.
        {random.choice(['No', 'Mild', 'Moderate'])} hemorrhagic transformation is present.
        ASPECTS score: {random.randint(6, 10)}

        IMPRESSION:
        Acute ischemic stroke in {territory}.
        {random.choice(['Thrombolysis candidate', 'Consider endovascular therapy', 'Medical management recommended'])}.
        
        """.strip()
        
        structured_data = {
            "report_id": f"STROKE_{report_id:04d}",
            "modality": modality,
            "brain_regions": [territory],
            "findings": ["acute infarction", "restricted diffusion"],
            "vessel_territory": vessel,
            "aspects_score": random.randint(6, 10),
            "hemorrhage": random.choice([True, False]),
            "recommendations": ["Stroke team consultation"]
        }
        
        json_embedded = f"\n\nSTRUCTURED_DATA: {json.dumps(structured_data)}"
        
        return {
            'report_id': f'synthetic_stroke_{report_id:04d}',
            'text': clinical_text + json_embedded,
            'modality': modality,
            'source': 'synthetic',
            'type': 'stroke',
            'structured_data': structured_data
        }
    
    def _generate_tumor_template(self, report_id):
        """Generate tumor/mass report"""
        modality = "MRI"
        location = random.choice(self.brain_regions)
        size = f"{random.randint(10, 50)} x {random.randint(10, 50)} mm"
        
        clinical_text = f"""
        EXAMINATION: MRI Brain with and without gadolinium
        CLINICAL HISTORY: Known brain mass, follow-up

        FINDINGS:
        There is a {size} enhancing mass in the {location}.
        The lesion demonstrates heterogeneous enhancement with central necrosis.
        Surrounding vasogenic edema extends into adjacent white matter.
        Mass effect with {random.choice(['mild', 'moderate', 'significant'])} midline shift of {random.randint(2, 8)} mm.

        IMPRESSION:
        Enhancing mass in {location}, suspicious for {random.choice(['glioblastoma', 'metastasis', 'meningioma'])}.
        Recommend tissue sampling and oncology consultation.
       
         """.strip()
        
        structured_data = {
            "report_id": f"TUMOR_{report_id:04d}",
            "modality": modality,
            "brain_regions": [location],
            "findings": ["enhancing mass", "vasogenic edema"],
            "size": size,
            "enhancement": True,
            "mass_effect": True,
            "differential": ["glioblastoma", "metastasis", "meningioma"]
        }
        
        json_embedded = f"\n\nSTRUCTURED_DATA: {json.dumps(structured_data)}"
        
        return {
            'report_id': f'synthetic_tumor_{report_id:04d}',
            'text': clinical_text + json_embedded,
            'modality': modality,
            'source': 'synthetic',
            'type': 'tumor',
            'structured_data': structured_data
        }
    
    def _generate_normal_template(self, report_id):
        """Generate normal study report"""
        modality = random.choice(["MRI", "CT"])
        
        clinical_text = f"""
        EXAMINATION: {modality} Brain
        CLINICAL HISTORY: {random.choice(['Headache evaluation', 'Screening', 'Follow-up'])}

        FINDINGS:
        The brain parenchyma appears normal without evidence of acute pathology.
        Gray-white matter differentiation is preserved.
        Ventricular size and configuration are within normal limits.
        No mass, hemorrhage, or acute infarction is identified.

        IMPRESSION:
        Normal {modality} brain study.
        No acute intracranial abnormality.
        """.strip()
        
        structured_data = {
            "report_id": f"NORMAL_{report_id:04d}",
            "modality": modality,
            "brain_regions": [],
            "findings": ["normal study"],
            "pathology": False,
            "recommendations": ["Routine follow-up as clinically indicated"]
        }
        
        json_embedded = f"\n\nSTRUCTURED_DATA: {json.dumps(structured_data)}"
        
        return {
            'report_id': f'synthetic_normal_{report_id:04d}',
            'text': clinical_text + json_embedded,
            'modality': modality,
            'source': 'synthetic',
            'type': 'normal',
            'structured_data': structured_data
        }
    
    def generate_txt_reports_with_embedded_json(self, num_reports=50, output_dir="data/synthetic/txt_reports"):
        """
        Generate realistic .txt neuroimaging reports with embedded JSON metadata
        Args:
        num_reports (int): Number of reports to generate
        output_dir (str): Directory to save the .txt files
        
        Returns:
        Path: Directory path where files were saved
        """

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
        # Sample report templates with embedded JSON
        report_templates = [
            {
                "type": "mri_brain",
                "template": """RADIOLOGY REPORT

                PATIENT: [PATIENT_ID]
                DOB: [DOB]
                EXAM DATE: [EXAM_DATE]
                REFERRING PHYSICIAN: Dr. [PHYSICIAN]

                EXAMINATION: MRI Brain without and with contrast
                CLINICAL INDICATION: [INDICATION]

                TECHNIQUE:
                Multiplanar multisequence MRI of the brain was performed including T1, T2, FLAIR, and DWI sequences before and after IV gadolinium administration.

                FINDINGS:
                [FINDINGS_TEXT]

                The cerebral hemispheres demonstrate normal gray-white matter differentiation. The ventricular system is normal in size and configuration. No midline shift is present.

                IMPRESSION:
                [IMPRESSION_TEXT]

                RECOMMENDATIONS:
                [RECOMMENDATIONS]

                --- STRUCTURED_DATA_JSON_START ---
                {
                    "report_type": "mri_brain",
                    "patient_id": "[PATIENT_ID]",
                    "exam_date": "[EXAM_DATE]",
                    "modality": "MRI",
                    "contrast": "[CONTRAST]",
                    "sequences": ["T1", "T2", "FLAIR", "DWI"],
                    "brain_regions": [BRAIN_REGIONS],
                    "findings": [FINDINGS_LIST],
                    "pathology_present": [PATHOLOGY_BOOL],
                    "severity": "[SEVERITY]",
                    "recommendations": [RECOMMENDATIONS_LIST],
                    "radiologist": "Dr. [RADIOLOGIST]",
                    "report_status": "final"
                }
                --- STRUCTURED_DATA_JSON_END ---
                """
            },
            {
                "type": "ct_brain",
                "template": """COMPUTED TOMOGRAPHY REPORT

                PATIENT ID: [PATIENT_ID]
                STUDY DATE: [EXAM_DATE]
                ORDERING PROVIDER: Dr. [PHYSICIAN]

                EXAMINATION: CT Head without contrast
                CLINICAL HISTORY: [INDICATION]

                TECHNIQUE:
                Non-contrast axial CT images of the head were obtained from the skull base to the vertex using standard technique.

                FINDINGS:
                [FINDINGS_TEXT]

                No acute hemorrhage or mass effect is identified. The gray-white matter differentiation is preserved. Ventricular size is within normal limits.

                IMPRESSION:
                [IMPRESSION_TEXT]

                --- STRUCTURED_DATA_JSON_START ---
                {
                    "report_type": "ct_brain",
                    "patient_id": "[PATIENT_ID]",
                    "exam_date": "[EXAM_DATE]",
                    "modality": "CT",
                    "contrast": "none",
                    "brain_regions": [BRAIN_REGIONS],
                    "findings": [FINDINGS_LIST],
                    "hemorrhage_present": [HEMORRHAGE_BOOL],
                    "mass_effect": [MASS_EFFECT_BOOL],
                    "severity": "[SEVERITY]",
                    "emergency_findings": [EMERGENCY_BOOL],
                    "radiologist": "Dr. [RADIOLOGIST]"
                }
                --- STRUCTURED_DATA_JSON_END ---
                """
                }]
    
        # Medical vocabularies for realistic content
        indications = [
            "Headache, rule out intracranial pathology",
            "Altered mental status",
            "Seizure disorder, follow-up",
            "Head trauma evaluation",
            "Memory loss workup",
            "Stroke symptoms evaluation",
            "Known brain mass, surveillance"
            ]
    
        findings_options = {
            "normal": [
                "No acute intracranial abnormality identified.",
                "Brain parenchyma appears within normal limits.",
                "No evidence of acute infarction, hemorrhage, or mass."
                ],
            "pathological": [
                "Small vessel ischemic changes in bilateral cerebral white matter.",
                "Mild cortical atrophy consistent with age-related changes.",
                "Chronic lacunar infarcts in the basal ganglia region.",
                "White matter hyperintensities suggesting small vessel disease.",
                "Focal area of T2/FLAIR hyperintensity in the frontal lobe.",
                "Microhemorrhages consistent with cerebral amyloid angiopathy.",
                "Enlarged ventricles suggesting mild hydrocephalus."
                ]}
    
        brain_regions = [
            "frontal lobe", "temporal lobe", "parietal lobe", "occipital lobe",
            "cerebellum", "brainstem", "basal ganglia", "thalamus", "hippocampus"
            ]
    
        severities = ["mild", "moderate", "severe", "minimal", "extensive"]
        radiologists = ["Smith", "Johnson", "Williams", "Brown", "Davis", "Miller"]
        physicians = ["Anderson", "Wilson", "Moore", "Taylor", "Thomas", "Jackson"]
    
        # Generate reports
        for i in range(num_reports):
            template_info = random.choice(report_templates)
            template = template_info["template"]
        
            # Generate random data
            patient_id = f"MRN{random.randint(100000, 999999)}"
            exam_date = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")
            dob = (datetime.now() - timedelta(days=random.randint(18*365, 80*365))).strftime("%Y-%m-%d")
            indication = random.choice(indications)
            radiologist = random.choice(radiologists)
            physician = random.choice(physicians)
        
            # Determine if pathology is present
            has_pathology = random.choice([True, False, False])  # 33% chance of pathology
        
            if has_pathology:
                findings_text = random.choice(findings_options["pathological"])
                impression_text = findings_text.replace(".", "") + ". Clinical correlation recommended."
                findings_list = ["white matter changes", "ischemic changes"]
                affected_regions = random.sample(brain_regions, random.randint(1, 2))
                severity = random.choice(severities)
                recommendations = ["Follow-up imaging", "Clinical correlation"]
            else:
                findings_text = random.choice(findings_options["normal"])
                impression_text = "Normal brain MRI study."
                findings_list = ["normal study"]
                affected_regions = []
                severity = "none"
                recommendations = ["Routine follow-up as clinically indicated"]
        
        # Prepare JSON data
        json_data = {
            "report_type": template_info["type"],
            "patient_id": patient_id,
            "exam_date": exam_date,
            "modality": "MRI" if template_info["type"] == "mri_brain" else "CT",
            "brain_regions": affected_regions,
            "findings": findings_list,
            "pathology_present": has_pathology,
            "severity": severity,
            "recommendations": recommendations,
            "radiologist": f"Dr. {radiologist}",
            "report_status": "final"
        }
        
        if template_info["type"] == "mri_brain":
            json_data.update({
                "contrast": "gadolinium" if random.choice([True, False]) else "none",
                "sequences": ["T1", "T2", "FLAIR", "DWI"]
            })
        else:  # CT
            json_data.update({
                "contrast": "none",
                "hemorrhage_present": False,
                "mass_effect": False,
                "emergency_findings": False
            })
        
        # Replace placeholders
        report_text = template
        replacements = {
            "[PATIENT_ID]": patient_id,
            "[DOB]": dob,
            "[EXAM_DATE]": exam_date,
            "[PHYSICIAN]": physician,
            "[INDICATION]": indication,
            "[FINDINGS_TEXT]": findings_text,
            "[IMPRESSION_TEXT]": impression_text,
            "[RECOMMENDATIONS]": ", ".join(recommendations),
            "[CONTRAST]": json_data.get("contrast", "none"),
            "[BRAIN_REGIONS]": json.dumps(affected_regions),
            "[FINDINGS_LIST]": json.dumps(findings_list),
            "[PATHOLOGY_BOOL]": json.dumps(has_pathology),
            "[SEVERITY]": severity,
            "[RECOMMENDATIONS_LIST]": json.dumps(recommendations),
            "[RADIOLOGIST]": f"Dr. {radiologist}",
            "[HEMORRHAGE_BOOL]": json.dumps(False),
            "[MASS_EFFECT_BOOL]": json.dumps(False),
            "[EMERGENCY_BOOL]": json.dumps(False)
        }
        
        for placeholder, value in replacements.items():
            report_text = report_text.replace(placeholder, str(value))
            
            # Save as .txt file
            filename = f"neuroimaging_report_{i+1:04d}_{template_info['type']}.txt"
            filepath = output_path / filename
        
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_text)
    
        print(f"Generated {num_reports} synthetic neuroimaging reports with embedded JSON in {output_path}")
        return output_path

    def extract_json_from_txt_report(self, txt_file_path):
        """
        Extract embedded JSON from a .txt report file
    
        Args:
            txt_file_path (str or Path): Path to the .txt file
    
        Returns:
            dict: Extracted JSON data, or None if not found
        """
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
            # Look for JSON between markers
            start_marker = "--- STRUCTURED_DATA_JSON_START ---"
            end_marker = "--- STRUCTURED_DATA_JSON_END ---"
        
            start_idx = content.find(start_marker)
            end_idx = content.find(end_marker)
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx + len(start_marker):end_idx].strip()
                return json.loads(json_str)
            else:
                print(f"No JSON markers found in {txt_file_path}")
                return None
            
        except Exception as e:
            print(f"Error extracting JSON from {txt_file_path}: {e}")
            return None
    
    
    def save_synthetic_reports_as_txt(self, reports_df, output_dir="data/synthetic/txt_reports"):
        """Save synthetic reports as individual .txt files with embedded JSON"""
    
        # Convert to Path object and handle Windows paths properly
        output_path = Path(output_dir).resolve()
    
        # Create directory with robust error handling
        try:
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
            elif not output_path.is_dir():
                # If it exists but is not a directory, create a new path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_path.parent / f"{output_path.name}_{timestamp}"
                output_path.mkdir(parents=True, exist_ok=True)
        
        except OSError as e:
            # Fallback: create in current directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"synthetic_reports_{timestamp}")
            output_path.mkdir(exist_ok=True)
            print(f"[INFO] Created fallback directory: {output_path}")
    
        saved_count = 0
        failed_count = 0
    
        for _, row in reports_df.iterrows():
            try:
                # Clean filename for Windows compatibility
                report_id = str(row['report_id']).replace('/', '_').replace('\\', '_')
                filename = f"{report_id}.txt"
                filepath = output_path / filename
            
                # Write file with proper encoding
                with open(filepath, 'w', encoding='utf-8', newline='') as f:
                    f.write(row['text'])
            
                saved_count += 1
            
            except Exception as e:
                failed_count += 1
                print(f"[WARN] Failed to save {row.get('report_id', 'unknown')}: {e}")
    
        print(f"Successfully saved {saved_count}/{len(reports_df)} synthetic reports to {output_path}")
        if failed_count > 0:
            print(f"[WARN] Failed to save {failed_count} reports")
    
        return output_path