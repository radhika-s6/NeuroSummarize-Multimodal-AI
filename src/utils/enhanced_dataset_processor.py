"""
This file provides comprehensive dataset processing capabilities for neuroimaging reports,
including batch processing, quality evaluation, and data export functionality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedDatasetProcessor:
    """
    Enhanced processor for handling large datasets of neuroimaging reports
    with batch processing, evaluation, and export capabilities.
    """
    
    def __init__(self, data_loader, ocr_system, summarizer, evaluator):
        """
        Initialize the enhanced dataset processor
        
        Args:
            data_loader: DataLoader instance for data operations
            ocr_system: OCR system for text extraction
            summarizer: Summarization system
            evaluator: Quality evaluation system
        """
        self.data_loader = data_loader
        self.ocr_system = ocr_system
        self.summarizer = summarizer
        self.evaluator = evaluator
        
        # Processing statistics
        self.processing_stats = {
            'total_processed': 0,
            'successful_summaries': 0,
            'failed_summaries': 0,
            'total_processing_time': 0,
            'average_processing_time': 0
        }
        
        # Cache for processed results
        self.results_cache = []
    
    def process_batch(self, df_batch: pd.DataFrame, include_evaluation: bool = True) -> List[Dict[str, Any]]:
        """
        Process a batch of reports
        
        Args:
            df_batch: DataFrame containing batch of reports to process
            include_evaluation: Whether to include quality evaluation
            
        Returns:
            List of processing results for the batch
        """
        batch_results = []
        batch_start_time = time.time()
        
        logger.info(f"Processing batch of {len(df_batch)} reports...")
        
        for idx, row in df_batch.iterrows():
            try:
                # Process individual report
                result = self._process_single_report(row, include_evaluation)
                batch_results.append(result)
                
                # Update statistics
                self.processing_stats['total_processed'] += 1
                if result.get('summary'):
                    self.processing_stats['successful_summaries'] += 1
                else:
                    self.processing_stats['failed_summaries'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing report {idx}: {str(e)}")
                
                # Add error result
                error_result = {
                    'filename': row.get('filename', f'report_{idx}'),
                    'original_text': row.get('text', ''),
                    'summary': None,
                    'error': str(e),
                    'processing_time': 0,
                    'timestamp': datetime.now().isoformat()
                }
                batch_results.append(error_result)
                self.processing_stats['failed_summaries'] += 1
        
        # Update timing statistics
        batch_time = time.time() - batch_start_time
        self.processing_stats['total_processing_time'] += batch_time
        
        if self.processing_stats['total_processed'] > 0:
            self.processing_stats['average_processing_time'] = (
                self.processing_stats['total_processing_time'] / 
                self.processing_stats['total_processed']
            )
        
        logger.info(f"Batch completed in {batch_time:.2f} seconds")
        return batch_results
    
    def _process_single_report(self, row: pd.Series, include_evaluation: bool = True) -> Dict[str, Any]:
        """
        Process a single report
        
        Args:
            row: Pandas Series containing report data
            include_evaluation: Whether to include quality evaluation
            
        Returns:
            Dictionary containing processing results
        """
        start_time = time.time()
        
        # Extract text content
        text_content = row.get('text', '')
        filename = row.get('filename', 'unknown')
        
        result = {
            'filename': filename,
            'original_text': text_content,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Generate summary
            if text_content.strip():
                summary_result = self.summarizer.summarize_text(text_content)
                result['summary'] = summary_result.get('summary', '')
                result['summary_metadata'] = summary_result
            else:
                result['summary'] = ''
                result['error'] = 'Empty text content'
            
            # Extract entities if available
            if hasattr(self.summarizer, 'extract_entities'):
                try:
                    entities = self.summarizer.extract_entities(text_content)
                    result['entities'] = entities
                except Exception as e:
                    logger.warning(f"Entity extraction failed: {str(e)}")
                    result['entities'] = []
            
            # Perform quality evaluation
            if include_evaluation and result.get('summary'):
                try:
                    evaluation = self.evaluator.evaluate_summary(
                        original_text=text_content,
                        summary=result['summary']
                    )
                    result['evaluation'] = evaluation
                except Exception as e:
                    logger.warning(f"Evaluation failed: {str(e)}")
                    result['evaluation'] = {'error': str(e)}
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            result['error'] = str(e)
            result['summary'] = None
        
        # Calculate processing time
        result['processing_time'] = time.time() - start_time
        
        return result
    
    def process_dataset(self, df: pd.DataFrame, batch_size: int = 10, 
                       include_evaluation: bool = True) -> List[Dict[str, Any]]:
        """
        Process entire dataset with batch processing
        
        Args:
            df: DataFrame containing reports to process
            batch_size: Size of each processing batch
            include_evaluation: Whether to include quality evaluation
            
        Returns:
            List of all processing results
        """
        all_results = []
        total_reports = len(df)
        
        logger.info(f"Starting dataset processing: {total_reports} reports in batches of {batch_size}")
        
        # Reset statistics
        self.processing_stats = {
            'total_processed': 0,
            'successful_summaries': 0,
            'failed_summaries': 0,
            'total_processing_time': 0,
            'average_processing_time': 0
        }
        
        # Process in batches
        for i in range(0, total_reports, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_results = self.process_batch(batch_df, include_evaluation)
            all_results.extend(batch_results)
            
            logger.info(f"Completed batch {i//batch_size + 1}/{(total_reports + batch_size - 1)//batch_size}")
        
        # Cache results
        self.results_cache = all_results
        
        logger.info(f"Dataset processing completed. Processed {len(all_results)} reports")
        return all_results
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return self.processing_stats.copy()
    
    def export_results(self, results: List[Dict[str, Any]], 
                      export_format: str = 'csv', 
                      output_path: Optional[str] = None,
                      include_options: List[str] = None) -> str:
        """
        Export processing results to various formats
        
        Args:
            results: List of processing results
            export_format: Format to export ('csv', 'excel', 'json')
            output_path: Output file path (if None, generates timestamp-based name)
            include_options: List of data types to include in export
            
        Returns:
            Path to exported file
        """
        if include_options is None:
            include_options = ["Summaries", "Evaluations"]
        
        # Prepare export data
        export_data = []
        
        for result in results:
            row = {
                'filename': result.get('filename', 'unknown'),
                'processing_time': result.get('processing_time', 0),
                'timestamp': result.get('timestamp', '')
            }
            
            if "Original text" in include_options:
                row['original_text'] = result.get('original_text', '')
            
            if "Summaries" in include_options:
                row['summary'] = result.get('summary', '')
            
            if "Evaluations" in include_options and 'evaluation' in result:
                eval_data = result['evaluation']
                if isinstance(eval_data, dict):
                    for key, value in eval_data.items():
                        if key != 'error':
                            row[f'eval_{key}'] = value
            
            if "Entities" in include_options and 'entities' in result:
                entities = result['entities']
                if entities:
                    row['entities_count'] = len(entities)
                    row['entities'] = json.dumps(entities) if entities else ''
                else:
                    row['entities_count'] = 0
                    row['entities'] = ''
            
            if "Processing metadata" in include_options:
                if 'summary_metadata' in result:
                    metadata = result['summary_metadata']
                    if isinstance(metadata, dict):
                        for key, value in metadata.items():
                            if key not in ['summary']:
                                row[f'meta_{key}'] = value
                
                if 'error' in result:
                    row['error'] = result['error']
            
            export_data.append(row)
        
        # Create DataFrame
        export_df = pd.DataFrame(export_data)
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"neuroimaging_analysis_{timestamp}"
        
        # Export based on format
        if export_format.lower() == 'csv':
            file_path = f"{output_path}.csv"
            export_df.to_csv(file_path, index=False)
        
        elif export_format.lower() == 'excel':
            file_path = f"{output_path}.xlsx"
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                export_df.to_excel(writer, sheet_name='Analysis Results', index=False)
                
                # Add summary sheet
                summary_data = {
                    'Metric': ['Total Processed', 'Successful Summaries', 'Failed Summaries', 
                              'Average Processing Time', 'Total Processing Time'],
                    'Value': [
                        self.processing_stats['total_processed'],
                        self.processing_stats['successful_summaries'],
                        self.processing_stats['failed_summaries'],
                        f"{self.processing_stats['average_processing_time']:.2f}s",
                        f"{self.processing_stats['total_processing_time']:.2f}s"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Processing Summary', index=False)
        
        elif export_format.lower() == 'json':
            file_path = f"{output_path}.json"
            export_data_with_stats = {
                'processing_statistics': self.processing_stats,
                'results': export_data,
                'export_timestamp': datetime.now().isoformat(),
                'total_results': len(export_data)
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data_with_stats, f, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        logger.info(f"Results exported to {file_path}")
        return file_path
    
    def analyze_dataset_quality(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze overall dataset quality metrics
        
        Args:
            results: List of processing results
            
        Returns:
            Dictionary containing quality analysis
        """
        analysis = {
            'total_reports': len(results),
            'successful_processing': 0,
            'failed_processing': 0,
            'average_summary_length': 0,
            'quality_scores': {
                'overall': [],
                'completeness': [],
                'accuracy': [],
                'clarity': []
            },
            'processing_times': [],
            'error_types': {},
            'entity_extraction_success': 0
        }
        
        summary_lengths = []
        
        for result in results:
            # Count processing success/failure
            if result.get('summary') and not result.get('error'):
                analysis['successful_processing'] += 1
                
                # Collect summary lengths
                summary_length = len(result.get('summary', ''))
                summary_lengths.append(summary_length)
                
            else:
                analysis['failed_processing'] += 1
                
                # Categorize errors
                error = result.get('error', 'Unknown error')
                error_type = error.split(':')[0] if ':' in error else error
                analysis['error_types'][error_type] = analysis['error_types'].get(error_type, 0) + 1
            
            # Collect quality scores
            if 'evaluation' in result and isinstance(result['evaluation'], dict):
                eval_data = result['evaluation']
                if 'overall_score' in eval_data:
                    analysis['quality_scores']['overall'].append(eval_data['overall_score'])
                if 'completeness_score' in eval_data:
                    analysis['quality_scores']['completeness'].append(eval_data['completeness_score'])
                if 'accuracy_score' in eval_data:
                    analysis['quality_scores']['accuracy'].append(eval_data['accuracy_score'])
                if 'clarity_score' in eval_data:
                    analysis['quality_scores']['clarity'].append(eval_data['clarity_score'])
            
            # Collect processing times
            if 'processing_time' in result:
                analysis['processing_times'].append(result['processing_time'])
            
            # Count entity extraction success
            if result.get('entities'):
                analysis['entity_extraction_success'] += 1
        
        # Calculate averages
        if summary_lengths:
            analysis['average_summary_length'] = np.mean(summary_lengths)
        
        # Calculate quality score statistics
        for score_type, scores in analysis['quality_scores'].items():
            if scores:
                analysis['quality_scores'][score_type] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'count': len(scores)
                }
            else:
                analysis['quality_scores'][score_type] = {
                    'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0
                }
        
        # Calculate processing time statistics
        if analysis['processing_times']:
            analysis['processing_time_stats'] = {
                'mean': np.mean(analysis['processing_times']),
                'std': np.std(analysis['processing_times']),
                'min': np.min(analysis['processing_times']),
                'max': np.max(analysis['processing_times']),
                'total': np.sum(analysis['processing_times'])
            }
        
        return analysis
    
    def get_cached_results(self) -> List[Dict[str, Any]]:
        """Get cached processing results"""
        return self.results_cache.copy()
    
    def clear_cache(self):
        """Clear cached results and reset statistics"""
        self.results_cache = []
        self.processing_stats = {
            'total_processed': 0,
            'successful_summaries': 0,
            'failed_summaries': 0,
            'total_processing_time': 0,
            'average_processing_time': 0
        }