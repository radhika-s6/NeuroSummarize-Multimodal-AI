import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import yaml
import json
import re
from datetime import datetime
from datetime import timedelta
import plotly.express as px
import zipfile
import io
from src.utils.data_loader import EnhancedDataLoader
from src.utils.enhanced_dataset_processor import EnhancedDatasetProcessor

# Import custom modules
from src.ocr.text_extractor import MultiModalOCR
from src.nlp.summarizer import MultiLLMSummarizer
from src.evaluation.metrics import EvaluationMetrics
from src.utils.data_loader import DataLoader
from src.visualisation.brain_mapper import show_affected_regions, plot_brain_heatmap

def batch_analysis_page(ocr_system, summarizer, evaluator):
    st.header("Batch Analysis")

    uploaded_files = st.file_uploader("Upload neuroimaging reports", type=['png', 'jpg', 'jpeg', 'txt'], accept_multiple_files=True)

    if uploaded_files:
        st.write("Select options for processing")
        ocr_method = st.selectbox("OCR Method", ["ensemble", "easyocr", "tesseract"])
        llm_model = st.selectbox("LLM Model", ["ensemble", "openai", "local"])
        summary_type = st.selectbox("Summary Type", ["both", "clinical", "layperson"])

        if st.button("Process Batch"):
            results = []
            all_detected_regions = []

            for uploaded_file in uploaded_files:
                if uploaded_file.type.startswith('image') or uploaded_file.type == 'text/plain':
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    try:
                        if uploaded_file.type.startswith('image'):
                            ocr_result = ocr_system.ensemble_extraction(temp_path)
                            file_text = ocr_result['final_text']
                        else:
                            file_text = Path(temp_path).read_text(encoding='utf-8')

                        result = summarizer.ensemble_summarization(file_text)
                        summary_data = result.get('ensemble', {})
                        summary = summary_data.get('summary', '')

                        try:
                            json_match = next((line for line in summary.splitlines() if line.strip().startswith('{')), None)
                            if json_match:
                                structured = json.loads(json_match)
                                regions = structured.get('brain_regions', [])
                                all_detected_regions.extend(regions)
                                show_affected_regions(regions)
                        except Exception as e:
                            st.warning(f"Failed to extract structured data from summary: {e}")

                        results.append({
                            'filename': uploaded_file.name,
                            'summary': summary,
                            'text_length': len(file_text),
                            'summary_length': len(summary)
                        })

                    except Exception as e:
                        st.error(f"Failed to process {uploaded_file.name}: {e}")
                    finally:
                        Path(temp_path).unlink()

            df = pd.DataFrame(results)
            st.session_state.batch_results = df
            st.dataframe(df)

            if all_detected_regions:
                plot_brain_heatmap(all_detected_regions)

def model_comparison_page(summarizer, evaluator):
    st.header("Model Comparison")

    sample_text = st.text_area("Paste a neuroimaging report for comparison")

    if st.button("Compare Models"):
        result = summarizer.ensemble_summarization(sample_text)
        records = []

        for name, output in result.items():
            summary = output.get('summary', '')
            score = evaluator.calculate_readability_score(summary)

            try:
                json_match = next((line for line in summary.splitlines() if line.strip().startswith('{')), None)
                if json_match:
                    structured = json.loads(json_match)
                    regions = structured.get('brain_regions', [])
                    show_affected_regions(regions)
            except Exception as e:
                st.warning(f"Structured data parsing error: {e}")

            records.append({
                'Model': name,
                'Summary Length': len(summary),
                'Complexity': score['complexity_score'],
                'Average Sentence Length': score['avg_sentence_length']
            })

        df = pd.DataFrame(records)
        st.dataframe(df)
        st.plotly_chart(px.bar(df, x='Model', y='Summary Length', title='Model Summary Length'))

def evaluation_dashboard_page(evaluator):
    st.header("Evaluation Dashboard")
    if 'batch_results' in st.session_state:
        df = st.session_state.batch_results
        st.write("Statistics of summaries generated")
        st.metric("Documents", len(df))
        st.metric("Avg Summary Length", df['summary_length'].mean())

        fig = px.histogram(df, x='summary_length', nbins=20, title="Summary Length Distribution")
        st.plotly_chart(fig)
    else:
        st.warning("No batch results found. Please process batch first.")


def data_generation_page(data_loader):
    st.header("Data Generation")
    
    # Enhanced data loader with neuroimaging capabilities
    enhanced_loader = EnhancedDataLoader(data_loader.data_dir)
    
    tab1, tab2, tab3 = st.tabs(["Realistic Reports", "TXT with JSON", "Load Existing"])
    
    with tab1:
        st.subheader("Generate Realistic Neuroimaging Reports")
        
        col1, col2 = st.columns(2)
        with col1:
            num_samples = st.slider("Number of reports", 10, 200, 50)
            report_types = st.multiselect(
                "Report types to include",
                ["MRI Brain", "CT Brain", "Stroke", "Tumor", "Normal"],
                default=["MRI Brain", "CT Brain", "Normal"]
            )
        
        with col2:
            include_json = st.checkbox("Include structured JSON", value=True)
            save_as_csv = st.checkbox("Save as CSV", value=True)
            save_as_txt = st.checkbox("Save individual TXT files", value=True)
        
        if st.button("Generate Realistic Reports"):
            with st.spinner("Generating realistic neuroimaging reports..."):
                try:
                    reports_df = enhanced_loader.generate_realistic_neuroimaging_reports(num_samples)
                    
                    st.success(f"Generated {len(reports_df)} realistic reports!")
                    
                    # Display sample
                    st.subheader("Sample Generated Report")
                    sample_report = reports_df.iloc[0]
                    st.text_area("Sample Report Text", sample_report['text'], height=300)
                    
                    if include_json:
                        st.json(sample_report['structured_data'])
                    
                    # Save options
                    if save_as_csv:
                        csv_data = reports_df.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            data=csv_data,
                            file_name=f"realistic_neuroimaging_reports_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    
                    if save_as_txt:
                        output_dir = enhanced_loader.save_synthetic_reports_as_txt(reports_df)
                        st.success(f"Saved individual TXT files to {output_dir}")
                    
                    # Display statistics
                    st.subheader("Generation Statistics")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    
                    with stats_col1:
                        st.metric("Total Reports", len(reports_df))
                    with stats_col2:
                        avg_length = reports_df['text'].str.len().mean()
                        st.metric("Avg Report Length", f"{avg_length:.0f} chars")
                    with stats_col3:
                        modality_counts = reports_df['modality'].value_counts()
                        st.metric("Most Common Modality", modality_counts.index[0])
                    
                    # Distribution charts
                    fig_modality = px.pie(
                        values=modality_counts.values,
                        names=modality_counts.index,
                        title="Distribution by Modality"
                    )
                    st.plotly_chart(fig_modality)
                    
                except Exception as e:
                    st.error(f"Error generating reports: {e}")
    
    with tab2:
        st.subheader("Generate TXT Files with Embedded JSON")
        
        col1, col2 = st.columns(2)
        with col1:
            num_txt_reports = st.slider("Number of TXT reports", 10, 100, 25)
            output_dir = st.text_input("Output directory", "data/synthetic/txt_reports")
        
        with col2:
            report_format = st.selectbox(
                "Report format",
                ["Mixed (MRI + CT)", "MRI Only", "CT Only"]
            )
        
        if st.button("Generate TXT Reports with JSON"):
            with st.spinner("Creating TXT files with embedded JSON..."):
                try:
                    output_path = enhanced_loader.generate_txt_reports_with_embedded_json(
                        num_reports=num_txt_reports,
                        output_dir=output_dir
                    )
                    
                    st.success(f"Generated {num_txt_reports} TXT files with embedded JSON!")
                    st.info(f"Files saved to: {output_path}")
                    
                    # Show sample file content
                    txt_files = list(output_path.glob("*.txt"))
                    if txt_files:
                        sample_file = txt_files[0]
                        with open(sample_file, 'r', encoding='utf-8') as f:
                            sample_content = f.read()
                        
                        st.subheader("Sample TXT File Content")
                        st.text_area("Content", sample_content, height=400)
                        
                        # Extract and show JSON
                        json_data = enhanced_loader.extract_json_from_txt_report(sample_file)
                        if json_data:
                            st.subheader("Extracted JSON Data")
                            st.json(json_data)
                
                except Exception as e:
                    st.error(f"Error generating TXT files: {e}")
    
    with tab3:
        st.subheader("Load and Process Existing Data")
        
        if st.button("Load All Structured Reports"):
            with st.spinner("Loading structured reports..."):
                try:
                    structured_data = data_loader.load_all_structured_reports()
                    st.write(f"Loaded {len(structured_data)} structured records")
                    st.dataframe(structured_data.head())
                    
                    # Show data distribution
                    if 'source' in structured_data.columns:
                        source_counts = structured_data['source'].value_counts()
                        fig_sources = px.bar(
                            x=source_counts.index,
                            y=source_counts.values,
                            title="Data Sources Distribution",
                            labels={'x': 'Source', 'y': 'Count'}
                        )
                        st.plotly_chart(fig_sources)
                    
                    # Download option
                    csv_data = structured_data.to_csv(index=False)
                    st.download_button(
                        "Download Structured Data CSV",
                        data=csv_data,
                        file_name=f"structured_reports_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error loading structured reports: {e}")
        
        # TXT files with JSON extraction
        st.subheader("Extract JSON from TXT Reports")
        uploaded_txt = st.file_uploader(
            "Upload TXT report with embedded JSON",
            type=['txt'],
            help="Upload a TXT file containing embedded JSON data"
        )
        
        if uploaded_txt:
            try:
                # Save temporarily and extract JSON
                temp_path = f"temp_{uploaded_txt.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_txt.getvalue())
                
                # Show file content
                with open(temp_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                st.text_area("File Content", content, height=300)
                
                # Extract JSON
                json_data = enhanced_loader.extract_json_from_txt_report(temp_path)
                if json_data:
                    st.subheader("Extracted JSON Metadata")
                    st.json(json_data)
                    
                    # Show key information
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Modality", json_data.get('modality', 'N/A'))
                    with col2:
                        st.metric("Pathology Present", json_data.get('pathology_present', 'N/A'))
                    with col3:
                        st.metric("Brain Regions", len(json_data.get('brain_regions', [])))
                else:
                    st.warning("No embedded JSON found in the uploaded file")
                
                # Cleanup
                Path(temp_path).unlink()
                
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")
    
    # Additional utilities section
    st.markdown("---")
    st.subheader("Data Utilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Quick Actions:**")
        if st.button("🔍 Analyze Existing Synthetic Data"):
            synthetic_dir = Path("data/synthetic")
            if synthetic_dir.exists():
                txt_files = list(synthetic_dir.rglob("*.txt"))
                csv_files = list(synthetic_dir.rglob("*.csv"))
                
                st.write(f"Found {len(txt_files)} TXT files and {len(csv_files)} CSV files")
                
                if txt_files:
                    # Analyze TXT files with JSON
                    json_data_list = []
                    for txt_file in txt_files[:10]:  # Limit to first 10 for performance
                        json_data = enhanced_loader.extract_json_from_txt_report(txt_file)
                        if json_data:
                            json_data_list.append(json_data)
                    
                    if json_data_list:
                        df_json = pd.DataFrame(json_data_list)
                        st.write("**JSON Metadata Analysis:**")
                        st.dataframe(df_json.head())
                        
                        # Show modality distribution
                        if 'modality' in df_json.columns:
                            modality_dist = df_json['modality'].value_counts()
                            st.bar_chart(modality_dist)
            else:
                st.info("No synthetic data directory found. Generate some data first!")
    
    with col2:
        st.write("**Data Validation:**")
        if st.button("✅ Validate Generated Reports"):
            # Quick validation of generated reports
            validation_results = {
                "total_files": 0,
                "valid_json": 0,
                "missing_fields": 0,
                "errors": []
            }
            
            synthetic_dir = Path("data/synthetic")
            if synthetic_dir.exists():
                txt_files = list(synthetic_dir.rglob("*.txt"))
                validation_results["total_files"] = len(txt_files)
                
                required_fields = ["modality", "brain_regions", "findings", "report_type"]
                
                for txt_file in txt_files[:20]:  # Validate first 20 files
                    try:
                        json_data = enhanced_loader.extract_json_from_txt_report(txt_file)
                        if json_data:
                            validation_results["valid_json"] += 1
                            missing = [field for field in required_fields if field not in json_data]
                            if missing:
                                validation_results["missing_fields"] += 1
                                validation_results["errors"].append(f"{txt_file.name}: Missing {missing}")
                    except Exception as e:
                        validation_results["errors"].append(f"{txt_file.name}: {str(e)}")
                
                # Display validation results
                st.write("**Validation Results:**")
                st.metric("Total Files", validation_results["total_files"])
                st.metric("Valid JSON", validation_results["valid_json"])
                st.metric("Missing Fields", validation_results["missing_fields"])
                
                if validation_results["errors"]:
                    with st.expander("Validation Errors"):
                        for error in validation_results["errors"][:10]:  # Show first 10 errors
                            st.text(error)
            else:
                st.warning("No synthetic data found to validate")

# def data_generation_page(data_loader):
#     st.header("Data Generation")

#     num_samples = st.slider("Number of synthetic reports", 10, 100, 20)
#     if st.button("Generate"):
#         synthetic_data = data_loader.generate_synthetic_reports(base_reports=[], num_samples=num_samples)
#         st.write(synthetic_data.head())
#         csv = synthetic_data.to_csv(index=False)
#         st.download_button("Download CSV", data=csv, file_name="synthetic_reports.csv")

#     if st.button("Load Structured Reports"):
#         structured_data = data_loader.load_all_structured_reports()
#         st.write("Loaded structured records:")
#         st.dataframe(structured_data.head())

# st.set_page_config(
#     page_title="NeuroSummarize",
#     page_icon=" ",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

def dataset_processing_page(data_loader, ocr_system, summarizer, evaluator):
    """Enhanced dataset processing page with comprehensive analysis"""
    st.header("Dataset Processing & Analysis")
    st.markdown("Upload and analyze neuroimaging datasets for processing")
    
    # Initialize enhanced processor
    dataset_processor = EnhancedDatasetProcessor(data_loader)
    
    # Sidebar for upload options
    with st.sidebar:
        st.subheader("Upload Options")
        upload_type = st.radio(
            "Choose upload method:",
            ["Multiple Files", "ZIP Archive", "Folder Structure"]
        )
        
        max_files = st.slider("Max files to process", 1, 100, 20)
        process_images = st.checkbox("Process images with OCR", value=True)
        generate_summaries = st.checkbox("Generate AI summaries", value=False)
    
    # Main upload interface
    if upload_type == "Multiple Files":
        uploaded_files = st.file_uploader(
            "Upload neuroimaging dataset files",
            type=['txt', 'csv', 'xlsx', 'json', 'png', 'jpg', 'jpeg', 'pdf', 'tsv', 'xml'],
            accept_multiple_files=True,
            help="Supported: Text reports, CSV/Excel data, JSON metadata, Images, PDFs"
        )
    elif upload_type == "ZIP Archive":
        uploaded_zip = st.file_uploader(
            "Upload ZIP archive containing dataset",
            type=['zip'],
            help="ZIP file containing multiple neuroimaging files"
        )
        uploaded_files = None
        if uploaded_zip:
            # Extract ZIP and process
            uploaded_files = extract_zip_files(uploaded_zip, max_files)
    else:
        st.info("Folder structure upload coming soon...")
        uploaded_files = None
    
    if uploaded_files and len(uploaded_files) > 0:
        # Limit files for performance
        if len(uploaded_files) > max_files:
            st.warning(f"Processing first {max_files} files out of {len(uploaded_files)} uploaded")
            uploaded_files = uploaded_files[:max_files]
        
        # Dataset analysis tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview", "Text Analysis", "Data Tables", "Medical Content", "Processing"
        ])
        
        with tab1:
            st.subheader("Dataset Overview")
            
            # Quick analysis
            with st.spinner("Analyzing uploaded dataset..."):
                analysis = dataset_processor.analyze_uploaded_dataset(uploaded_files)
            
            # Display file summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Files", analysis['file_summary']['total_files'])
            with col2:
                st.metric("Total Size (MB)", f"{analysis['file_summary']['total_size_mb']:.1f}")
            with col3:
                supported_count = len(analysis['file_summary']['supported_formats']['supported'])
                st.metric("Supported Formats", supported_count)
            with col4:
                unsupported_count = len(analysis['file_summary']['supported_formats']['unsupported'])
                st.metric("Unsupported Formats", unsupported_count)
            
            # File type distribution
            file_types = analysis['file_summary']['file_types']
            if file_types:
                fig_types = px.pie(
                    values=list(file_types.values()),
                    names=list(file_types.keys()),
                    title="File Type Distribution"
                )
                st.plotly_chart(fig_types, use_container_width=True)
            
            # Format support status
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Supported Formats:**")
                for fmt in analysis['file_summary']['supported_formats']['supported']:
                    st.write(f"- {fmt}")
            
            with col2:
                if analysis['file_summary']['supported_formats']['unsupported']:
                    st.write("**Unsupported Formats:**")
                    for fmt in analysis['file_summary']['supported_formats']['unsupported']:
                        st.write(f"- {fmt}")
                else:
                    st.success("All uploaded formats are supported!")
        
        with tab2:
            st.subheader("Text Content Analysis")
            
            # Process text files
            with st.spinner("Processing text files..."):
                text_df = dataset_processor.process_text_files(uploaded_files)
            
            if not text_df.empty:
                # Text statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Text Files", len(text_df))
                with col2:
                    st.metric("Avg Word Count", f"{text_df['word_count'].mean():.0f}")
                with col3:
                    st.metric("Total Words", f"{text_df['word_count'].sum():,}")
                with col4:
                    files_with_entities = (text_df['medical_entities'].apply(lambda x: sum(len(v) for v in x.values())) > 0).sum()
                    st.metric("Files w/ Medical Terms", files_with_entities)
                
                # Word count distribution
                fig_words = px.histogram(
                    text_df, x='word_count', nbins=20,
                    title="Word Count Distribution",
                    labels={'word_count': 'Words per Document', 'count': 'Number of Documents'}
                )
                st.plotly_chart(fig_words, use_container_width=True)
                
                # File details table
                st.subheader("File Details")
                display_df = text_df[['filename', 'word_count', 'char_count', 'has_json']].copy()
                display_df['has_medical_entities'] = text_df['medical_entities'].apply(
                    lambda x: sum(len(v) for v in x.values()) > 0
                )
                st.dataframe(display_df, use_container_width=True)
                
                # Sample content
                st.subheader("Sample Content")
                selected_file = st.selectbox("Select file to preview:", text_df['filename'].tolist())
                if selected_file:
                    file_content = text_df[text_df['filename'] == selected_file]['content'].iloc[0]
                    st.text_area("Content Preview", file_content[:1000] + "..." if len(file_content) > 1000 else file_content, height=200)
            else:
                st.info("No text files found in upload")
        
        with tab3:
            st.subheader("Tabular Data Analysis")
            
            # Process CSV/Excel files
            with st.spinner("Processing tabular files..."):
                csv_files = dataset_processor.process_csv_files(uploaded_files)
            
            if csv_files:
                # Summary metrics
                total_rows = sum(info['shape'][0] for info in csv_files)
                total_cols = sum(info['shape'][1] for info in csv_files)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tabular Files", len(csv_files))
                with col2:
                    st.metric("Total Rows", f"{total_rows:,}")
                with col3:
                    st.metric("Total Columns", total_cols)
                
                # File details
                for i, file_info in enumerate(csv_files):
                    with st.expander(f"📋 {file_info['filename']} ({file_info['shape'][0]} rows, {file_info['shape'][1]} cols)"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write("**Data Preview:**")
                            st.dataframe(file_info['dataframe'].head(), use_container_width=True)
                        
                        with col2:
                            st.write("**Quality Metrics:**")
                            quality = file_info['quality_metrics']
                            st.metric("Completeness", f"{quality['completeness']:.1f}%")
                            st.metric("Duplicates", quality['duplicate_rows'])
                            st.metric("Empty Rows", quality['empty_rows'])
                        
                        if file_info['medical_columns']:
                            st.write("**Medical Columns Detected:**")
                            st.write(", ".join(file_info['medical_columns']))
            else:
                st.info("No tabular files found in upload")
        
        with tab4:
            st.subheader("Medical Content Analysis")
            
            # Create unified dataset
            with st.spinner("Creating unified dataset..."):
                processed_files = {
                    'text_files': dataset_processor.process_text_files(uploaded_files),
                    'csv_files': dataset_processor.process_csv_files(uploaded_files),
                    'json_files': dataset_processor.process_medical_json_files(uploaded_files)
                }
                
                unified_dataset = dataset_processor.create_unified_dataset(processed_files)
                dataset_report = dataset_processor.generate_dataset_report(processed_files, unified_dataset)
            
            if not unified_dataset.empty:
                # Medical content overview
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", dataset_report['summary']['total_records'])
                with col2:
                    st.metric("Avg Words/Record", f"{dataset_report['summary']['avg_word_count']:.0f}")
                with col3:
                    st.metric("Records w/ Medical Entities", dataset_report['summary']['files_with_medical_entities'])
                
                # Entity analysis
                st.subheader("Medical Entity Distribution")
                
                # Collect all entities
                all_entities = {}
                for _, row in unified_dataset.iterrows():
                    for category, entities in row['medical_entities'].items():
                        if category not in all_entities:
                            all_entities[category] = []
                        all_entities[category].extend(entities)
                
                # Display entity distributions
                for category, entities in all_entities.items():
                    if entities:
                        entity_counts = pd.Series(entities).value_counts().head(10)
                        if not entity_counts.empty:
                            fig_entities = px.bar(
                                x=entity_counts.values,
                                y=entity_counts.index,
                                orientation='h',
                                title=f"Most Common {category.replace('_', ' ').title()}"
                            )
                            fig_entities.update_layout(height=300)
                            st.plotly_chart(fig_entities, use_container_width=True)
                
                # Content type distribution
                if 'content_type' in unified_dataset.columns:
                    content_dist = unified_dataset['content_type'].value_counts()
                    fig_content = px.pie(
                        values=content_dist.values,
                        names=content_dist.index,
                        title="Content Type Distribution"
                    )
                    st.plotly_chart(fig_content, use_container_width=True)
                
                # Recommendations
                st.subheader("📋 Recommendations")
                for i, recommendation in enumerate(dataset_report['recommendations'], 1):
                    st.write(f"{i}. {recommendation}")
            else:
                st.warning("No medical content detected in uploaded files")
        
        with tab5:
            st.subheader("⚡ Batch Processing")
            
            # Processing options
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Processing Options:**")
                extract_text = st.checkbox("Extract text from images", value=process_images)
                generate_summaries_flag = st.checkbox("Generate AI summaries", value=generate_summaries)
                extract_entities = st.checkbox("Extract medical entities", value=True)
                create_embeddings = st.checkbox("Create text embeddings", value=False)
            
            with col2:
                st.write("**Output Options:**")
                save_csv = st.checkbox("Save processed data as CSV", value=True)
                save_json = st.checkbox("Save metadata as JSON", value=True)
                generate_report = st.checkbox("Generate analysis report", value=True)
            
            # Processing button
            if st.button("🚀 Start Batch Processing", type="primary"):
                processing_results = process_uploaded_dataset(
                    uploaded_files, 
                    ocr_system, 
                    summarizer, 
                    evaluator,
                    dataset_processor,
                    {
                        'extract_text': extract_text,
                        'generate_summaries': generate_summaries_flag,
                        'extract_entities': extract_entities,
                        'create_embeddings': create_embeddings,
                        'save_csv': save_csv,
                        'save_json': save_json,
                        'generate_report': generate_report
                    }
                )
                
                if processing_results:
                    st.success("✅ Batch processing completed!")
                    
                    # Display results summary
                    st.subheader("Processing Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Files Processed", processing_results['files_processed'])
                    with col2:
                        st.metric("Text Extracted", processing_results['text_extracted'])
                    with col3:
                        st.metric("Summaries Generated", processing_results['summaries_generated'])
                    with col4:
                        st.metric("Errors", processing_results['errors'])
                    
                    # Download processed data
                    if processing_results.get('processed_data') is not None:
                        processed_df = processing_results['processed_data']
                        
                        # CSV download
                        if save_csv:
                            csv_data = processed_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Processed Data (CSV)",
                                data=csv_data,
                                file_name=f"processed_dataset_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                        
                        # JSON download
                        if save_json and processing_results.get('metadata'):
                            json_data = json.dumps(processing_results['metadata'], indent=2)
                            st.download_button(
                                "📥 Download Metadata (JSON)",
                                data=json_data,
                                file_name=f"dataset_metadata_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                                mime="application/json"
                            )
                        
                        # Display processed data preview
                        st.subheader("Processed Data Preview")
                        st.dataframe(processed_df.head(10), use_container_width=True)
                        
                        # Processing statistics
                        if processing_results.get('statistics'):
                            st.subheader("Processing Statistics")
                            stats = processing_results['statistics']
                            
                            # Word count distribution
                            if 'word_counts' in stats:
                                fig_words = px.histogram(
                                    x=stats['word_counts'], 
                                    nbins=20,
                                    title="Word Count Distribution (Processed Data)",
                                    labels={'x': 'Word Count', 'y': 'Frequency'}
                                )
                                st.plotly_chart(fig_words, use_container_width=True)
                            
                            # Processing time by file type
                            if 'processing_times' in stats:
                                fig_time = px.bar(
                                    x=list(stats['processing_times'].keys()),
                                    y=list(stats['processing_times'].values()),
                                    title="Average Processing Time by File Type",
                                    labels={'x': 'File Type', 'y': 'Time (seconds)'}
                                )
                                st.plotly_chart(fig_time, use_container_width=True)
                        
                        # Error analysis
                        if processing_results.get('error_details'):
                            with st.expander("❌ Error Details"):
                                for error in processing_results['error_details']:
                                    st.error(f"File: {error['filename']} - Error: {error['error']}")
    
    else:
        # No files uploaded - show instructions
        st.info("👆 Upload dataset files to begin analysis")
        
        # Example dataset structure
        with st.expander("💡 Supported Dataset Formats"):
            st.markdown("""
            **Text Files (.txt, .md, .rtf):**
            - Neuroimaging reports
            - Clinical notes
            - Radiological findings
            
            **Tabular Data (.csv, .xlsx, .xls):**
            - Patient metadata
            - Study parameters
            - Clinical measurements
            
            **JSON Files (.json):**
            - DICOM metadata
            - Study annotations
            - Structured reports
            
            **Images (.png, .jpg, .jpeg, .tiff):**
            - Scanned reports
            - Handwritten notes
            - Medical images
            
            **Archives (.zip):**
            - Combined datasets
            - Multi-format collections
            """)
        
        # Sample data generation
        st.subheader("🎯 Don't have data? Generate sample dataset")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Sample Text Reports"):
                # Generate sample reports using existing functionality
                enhanced_loader = EnhancedDataLoader("data/")
                sample_reports = enhanced_loader.generate_realistic_neuroimaging_reports(10)
                
                st.success("Generated 10 sample reports!")
                st.dataframe(sample_reports[['report_id', 'modality', 'type']].head())
                
                # Save sample files
                sample_dir = enhanced_loader.save_synthetic_reports_as_txt(sample_reports)
                st.info(f"Sample files saved to: {sample_dir}")
        
        with col2:
            if st.button("Generate Sample CSV Data"):
                # Generate sample CSV data
                sample_csv_data = generate_sample_csv_dataset(50)
                
                st.success("Generated sample CSV dataset!")
                st.dataframe(sample_csv_data.head())
                
                # Download option
                csv_data = sample_csv_data.to_csv(index=False)
                st.download_button(
                    "Download Sample CSV",
                    data=csv_data,
                    file_name="sample_neuroimaging_dataset.csv",
                    mime="text/csv"
                )


def extract_zip_files(uploaded_zip, max_files=50):
    """Extract files from uploaded ZIP archive"""
    extracted_files = []
    
    try:
        with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as zip_file:
            file_list = zip_file.namelist()[:max_files]  # Limit for performance
            
            for filename in file_list:
                if not filename.endswith('/'):  # Skip directories
                    file_data = zip_file.read(filename)
                    
                    # Create a file-like object
                    file_obj = io.BytesIO(file_data)
                    file_obj.name = filename
                    file_obj.getvalue = lambda: file_data
                    
                    extracted_files.append(file_obj)
    
    except Exception as e:
        st.error(f"Error extracting ZIP file: {e}")
    
    return extracted_files


def process_uploaded_dataset(uploaded_files, ocr_system, summarizer, evaluator, dataset_processor, options):
    """Process uploaded dataset with specified options"""
    processed_records = []
    processing_stats = {
        'files_processed': 0,
        'text_extracted': 0,
        'summaries_generated': 0,
        'errors': 0,
        'error_details': [],
        'processing_times': {},
        'word_counts': []
    }
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            start_time = datetime.now()
            
            status_text.text(f"Processing {uploaded_file.name}...")
            
            file_ext = Path(uploaded_file.name).suffix.lower()
            record = {
                'filename': uploaded_file.name,
                'file_type': file_ext,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Extract text content
            text_content = ""
            
            if file_ext in ['.txt', '.md', '.rtf']:
                text_content = str(uploaded_file.read(), 'utf-8')
                processing_stats['text_extracted'] += 1
                
            elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff'] and options['extract_text']:
                # Save temporarily for OCR
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                ocr_result = ocr_system.ensemble_extraction(temp_path)
                text_content = ocr_result.get('final_text', '')
                record['ocr_confidence'] = ocr_result.get('confidence', 0)
                record['ocr_method'] = ocr_result.get('best_method', 'unknown')
                
                Path(temp_path).unlink()  # Cleanup
                processing_stats['text_extracted'] += 1
                
            elif file_ext in ['.csv', '.xlsx', '.xls']:
                # Process tabular data
                if file_ext == '.csv':
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Convert to text representation
                text_rows = []
                for _, row in df.iterrows():
                    row_text = ' '.join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    text_rows.append(row_text)
                
                text_content = '\n'.join(text_rows)
                record['rows_processed'] = len(df)
                record['columns'] = list(df.columns)
                
            elif file_ext == '.json':
                json_data = json.loads(uploaded_file.read())
                # Extract text from JSON
                text_fields = dataset_processor._extract_text_from_json(json_data)
                text_content = ' '.join(text_fields.values())
                record['json_structure'] = list(text_fields.keys())
            
            # Add text content to record
            record['text_content'] = text_content
            record['word_count'] = len(text_content.split()) if text_content else 0
            processing_stats['word_counts'].append(record['word_count'])
            
            # Extract medical entities if requested
            if options['extract_entities'] and text_content:
                entities = dataset_processor._extract_medical_entities(text_content)
                record['medical_entities'] = entities
                record['has_medical_content'] = any(len(v) > 0 for v in entities.values())
            
            # Generate summary if requested
            if options['generate_summaries'] and text_content and len(text_content.split()) > 20:
                try:
                    summary_result = summarizer.ensemble_summarization(text_content)
                    if summary_result:
                        # Get the best summary result
                        best_result = max(summary_result.values(), 
                                        key=lambda x: len(x.get('summary', '')) if 'error' not in x else 0)
                        
                        if 'error' not in best_result:
                            record['ai_summary'] = best_result.get('summary', '')
                            record['summary_model'] = best_result.get('model', 'unknown')
                            record['tokens_used'] = best_result.get('tokens_used', 0)
                            processing_stats['summaries_generated'] += 1
                        
                except Exception as e:
                    record['summary_error'] = str(e)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            record['processing_time'] = processing_time
            
            # Update processing time stats
            if file_ext not in processing_stats['processing_times']:
                processing_stats['processing_times'][file_ext] = []
            processing_stats['processing_times'][file_ext].append(processing_time)
            
            processed_records.append(record)
            processing_stats['files_processed'] += 1
            
        except Exception as e:
            processing_stats['errors'] += 1
            processing_stats['error_details'].append({
                'filename': uploaded_file.name,
                'error': str(e)
            })
        
        # Update progress
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    # Calculate average processing times
    for file_type, times in processing_stats['processing_times'].items():
        processing_stats['processing_times'][file_type] = sum(times) / len(times)
    
    # Create final DataFrame
    processed_df = pd.DataFrame(processed_records)
    
    # Generate metadata
    metadata = {
        'processing_date': datetime.now().isoformat(),
        'total_files': len(uploaded_files),
        'successfully_processed': processing_stats['files_processed'],
        'processing_options': options,
        'statistics': processing_stats
    }
    
    status_text.text("Processing complete!")
    
    return {
        'processed_data': processed_df,
        'metadata': metadata,
        'statistics': processing_stats,
        'files_processed': processing_stats['files_processed'],
        'text_extracted': processing_stats['text_extracted'],
        'summaries_generated': processing_stats['summaries_generated'],
        'errors': processing_stats['errors'],
        'error_details': processing_stats['error_details']
    }


def generate_sample_csv_dataset(num_records=50):
    """Generate sample CSV dataset for demonstration"""
    np.random.seed(42)  # For reproducibility
    
    patients = []
    modalities = ['MRI', 'CT', 'PET', 'fMRI', 'DTI']
    findings = ['Normal', 'Lesion', 'Atrophy', 'Edema', 'Hemorrhage', 'Infarct']
    regions = ['Frontal', 'Temporal', 'Parietal', 'Occipital', 'Cerebellum']
    
    for i in range(num_records):
        patient = {
            'PatientID': f'P{i+1:04d}',
            'Age': np.random.randint(18, 85),
            'Gender': np.random.choice(['M', 'F']),
            'StudyDate': (datetime.now() - timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d'),
            'Modality': np.random.choice(modalities),
            'Finding': np.random.choice(findings),
            'Region': np.random.choice(regions),
            'Severity': np.random.choice(['Mild', 'Moderate', 'Severe']),
            'Report_Summary': f"Patient shows {np.random.choice(findings).lower()} in {np.random.choice(regions).lower()} region. {np.random.choice(['Recommend follow-up', 'No action needed', 'Further evaluation required'])}.",
            'Radiologist': f"Dr. {np.random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Davis'])}",
            'StudyInstanceUID': f"1.2.3.{np.random.randint(100, 999)}.{np.random.randint(1000, 9999)}"
        }
        patients.append(patient)
    
    return pd.DataFrame(patients)


@st.cache_data
def load_config():
    with open('config/config.yaml', 'r') as file:
        return yaml.safe_load(file)

@st.cache_resource
def initialize_components():
    config = load_config()
    ocr_system = MultiModalOCR()
    summarizer = MultiLLMSummarizer(config)
    evaluator = EvaluationMetrics()
    data_loader = DataLoader()
    return ocr_system, summarizer, evaluator, data_loader, config

def main():
    st.title("NeuroSummarize: AI-Powered Neuroimaging Report Analysis")
    st.markdown("---")

    ocr_system, summarizer, evaluator, data_loader, config = initialize_components()

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Document Processing", "Batch Analysis", "Model Comparison", "Evaluation Dashboard", "Data Generation"]
    )

    if page == "Document Processing":
        document_processing_page(ocr_system, summarizer)
    elif page == "Batch Analysis":
        batch_analysis_page(ocr_system, summarizer, evaluator)
    elif page == "Model Comparison":
        model_comparison_page(summarizer, evaluator)
    elif page == "Evaluation Dashboard":
        evaluation_dashboard_page(evaluator)
    elif page == "Data Generation":
        data_generation_page(data_loader)

def document_processing_page(ocr_system, summarizer):
    st.header("Document Processing")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a neuroimaging report",
            type=['png', 'jpg', 'jpeg', 'pdf', 'txt'],
            help="Upload scanned, handwritten, or typed neuroimaging reports"
        )

        if uploaded_file is not None:
            if uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                if st.button("Extract Text", type="primary"):
                    with st.spinner("Extracting text..."):
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())

                        ocr_result = ocr_system.ensemble_extraction(temp_path)

                        st.session_state.extracted_text = ocr_result['final_text']
                        st.session_state.ocr_confidence = ocr_result['confidence']
                        st.session_state.best_method = ocr_result['best_method']

                        Path(temp_path).unlink()

            elif uploaded_file.type == 'text/plain':
                text_content = str(uploaded_file.read(), "utf-8")
                st.session_state.extracted_text = text_content
                st.session_state.ocr_confidence = 1.0
                st.session_state.best_method = "direct_text"

    with col2:
        st.subheader("Extracted Text")
        if 'extracted_text' in st.session_state:
            st.text_area(
                "Extracted Content",
                st.session_state.extracted_text,
                height=300,
                help=f"Confidence: {st.session_state.ocr_confidence:.2f} | Method: {st.session_state.best_method}"
            )

            st.subheader("Summarization Options")
            summary_type = st.selectbox("Summary Type", ["both", "clinical", "layperson"])
            model_choice = st.selectbox("Model Selection", ["ensemble", "openai", "groq", "local"])

            if st.button("Generate Summary", type="primary"):
                with st.spinner("Generating summary..."):
                    if model_choice == "ensemble":
                        results = summarizer.ensemble_summarization(st.session_state.extracted_text)
                        st.session_state.summary_results = results
                    else:
                        if model_choice == "openai":
                            result = summarizer.summarize_with_openai(st.session_state.extracted_text)
                        elif model_choice == "groq":
                            result = summarizer.summarize_with_groq(st.session_state.extracted_text)
                        else:
                            result = summarizer.summarize_with_local_model(st.session_state.extracted_text)

                        st.session_state.summary_results = {model_choice: result}

    if 'summary_results' in st.session_state:
        st.markdown("---")
        st.header("Results")

        for model_name, result in st.session_state.summary_results.items():
            if 'error' not in result:
                with st.expander(f"{model_name.upper()} Results", expanded=True):
                    st.markdown(f"**Model:** {result['model']}")
                    st.markdown(f"**Tokens Used:** {result.get('tokens_used', 'N/A')}")

                    summary_text = result['summary']

                    try:
                        json_match = re.search(r'\{.*\}', summary_text, re.DOTALL)
                        if json_match:
                            json_data = json.loads(json_match.group())

                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.json(json_data)

                            with col2:
                                if 'brain_regions' in json_data:
                                    regions = json_data['brain_regions']
                                    if isinstance(regions, list) and regions:
                                        st.subheader("Brain Region Visualization")
                                        show_affected_regions(regions)
                                        plot_brain_heatmap(regions)
                    except:
                        pass

                    st.text_area(
                        "Generated Summary",
                        summary_text,
                        height=200,
                        key=f"summary_{model_name}"
                    )
            else:
                st.error(f"Error with {model_name}: {result['error']}")

if __name__ == "__main__":
    main()