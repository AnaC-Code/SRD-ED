from sdv.datasets.demo import download_demo
from sdv.evaluation.multi_table import run_diagnostic, evaluate_quality
import pandas as pd
from sdv.multi_table import HMASynthesizer


def get_synthetic_data(real_data, metadata, dataset_name):
    """
    Load synthetic data from CSV files and adjust column types based on metadata.
    
    Parameters:
    real_data (dict): The real dataset as a dictionary of DataFrames.
    metadata (sdv.Metadata): Metadata describing the dataset structure.
    dataset_name (str): Name of the dataset used.
    
    Returns:
    dict: A dictionary containing synthetic data tables.
    """
    synthetic_data = {}
    
    for table_name in metadata.to_dict()["tables"].keys():
        file_path = f"datasets/{dataset_name}/synthethic_data/{table_name}.csv"
        df = pd.read_csv(file_path)
        
        # Convert ID columns to object type for consistency
        for column_name, column_details in metadata.to_dict()["tables"][table_name]["columns"].items():
            if column_details["sdtype"] == "id":
                df[column_name] = df[column_name].astype('object')
                real_data[table_name][column_name] = real_data[table_name][column_name].astype('object')
        
        synthetic_data[table_name] = df.copy()
    
    return synthetic_data


def evaluate_synthetic_data(real_data, synthetic_data, metadata):
    """
    Evaluate the quality of synthetic data using SDV diagnostic and quality reports.
    
    Parameters:
    real_data (dict): The real dataset as a dictionary of DataFrames.
    synthetic_data (dict): The synthetic dataset as a dictionary of DataFrames.
    metadata (sdv.Metadata): Metadata describing the dataset structure.
    """
    diagnostic_report = run_diagnostic(real_data, synthetic_data, metadata)
    quality_report = evaluate_quality(real_data, synthetic_data, metadata)
    
    print("Diagnostic Report Score:", diagnostic_report.get_score())
    print("Quality Report Score:", quality_report.get_score())


# Define dataset name
dataset_name = "DCG_v1"

# Download real data and metadata
real_data, metadata = download_demo(modality='multi_table', dataset_name=dataset_name)

# Initialize the synthesizer with metadata
synthesizer = HMASynthesizer(metadata)

# Train the synthesizer on real data
synthesizer.fit(real_data)

# Generate synthetic data using HMASynthesizer
hma_synthetic_data = synthesizer.sample(scale=0.1)

# Load synthetic data using the alternative method
new_method_synthetic_data = get_synthetic_data(real_data, metadata, dataset_name)

# Evaluate synthetic data
print("Evaluating New Method Synthetic Data")
evaluate_synthetic_data(real_data, new_method_synthetic_data, metadata)

print("Evaluating HMA Synthetic Data")
evaluate_synthetic_data(real_data, hma_synthetic_data, metadata)