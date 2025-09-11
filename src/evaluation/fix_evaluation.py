import numpy as np

def fix_benchmark_results(benchmark_results):
    """
    Fix benchmark results to ensure consistent data types
    """
    for system_name, results in benchmark_results.items():
        for metric_name, metric_data in results.items():
            if isinstance(metric_data, np.ndarray):
                # Convert to list if it's a numpy array
                results[metric_name] = metric_data.tolist()
            elif not isinstance(metric_data, list):
                # Ensure it's a list
                results[metric_name] = list(metric_data) if hasattr(metric_data, '__iter__') else [metric_data]
    
    return benchmark_results

# Test the fix
if __name__ == "__main__":
    print("Testing benchmark results fix...")
    
    # Create sample data like in the original
    test_data = [{'id': f'test_{i}'} for i in range(10)]
    
    sample_results = {
        'system1': {
            'f1_scores': np.random.normal(0.8, 0.1, 10),
            'rouge_scores': np.random.normal(0.5, 0.1, 10),
            'hallucination_rates': np.random.normal(0.1, 0.02, 10)
        }
    }
    
    print("Before fix:")
    print(f"f1_scores type: {type(sample_results['system1']['f1_scores'])}")
    
    fixed_results = fix_benchmark_results(sample_results)
    
    print("After fix:")
    print(f"f1_scores type: {type(fixed_results['system1']['f1_scores'])}")
    print("Fix successful!")