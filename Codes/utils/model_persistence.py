"""
Model Persistence Module
Handles saving and loading of trained models with metadata
"""

import joblib
import pickle
import os
from datetime import datetime
import json


def save_model(model, results, model_name, save_dir="saved_models", use_joblib=True):
    """
    Save model and its results to disk
    
    Args:
        model: Trained model object
        results: Dictionary containing model evaluation results
        model_name: Name of the model for file naming
        save_dir: Directory to save models
        use_joblib: Whether to use joblib or pickle for serialization
    
    Returns:
        Dictionary containing save information
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{model_name}_{timestamp}"
    
    # Choose serialization method
    if use_joblib:
        model_file = os.path.join(save_dir, f"{base_filename}.joblib")
        joblib.dump(model, model_file)
        serialization_method = "joblib"
    else:
        model_file = os.path.join(save_dir, f"{base_filename}.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        serialization_method = "pickle"
    
    # Save results and metadata
    metadata = {
        "model_name": model_name,
        "timestamp": timestamp,
        "serialization_method": serialization_method,
        "model_file": model_file,
        "results": results,
        "model_params": model.get_params() if hasattr(model, 'get_params') else str(model),
        "save_date": datetime.now().isoformat()
    }
    
    metadata_file = os.path.join(save_dir, f"{base_filename}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    save_info = {
        "model_file": model_file,
        "metadata_file": metadata_file,
        "timestamp": timestamp,
        "success": True
    }
    
    print(f"✅ Model saved successfully:")
    print(f"   Model: {model_file}")
    print(f"   Metadata: {metadata_file}")
    print(f"   Test Accuracy: {results.get('test_accuracy', 'N/A'):.4f}")
    
    return save_info


def load_model(model_file, metadata_file=None):
    """
    Load model and its metadata from disk
    
    Args:
        model_file: Path to the saved model file
        metadata_file: Optional path to metadata file
    
    Returns:
        Tuple of (model, metadata)
    """
    # Determine serialization method from file extension
    if model_file.endswith('.joblib'):
        model = joblib.load(model_file)
    elif model_file.endswith('.pkl'):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    else:
        raise ValueError("Unsupported file format. Use .joblib or .pkl files.")
    
    # Load metadata if available
    metadata = None
    if metadata_file and os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    elif metadata_file is None:
        # Try to find metadata file automatically
        base_name = os.path.splitext(model_file)[0]
        auto_metadata_file = f"{base_name}_metadata.json"
        if os.path.exists(auto_metadata_file):
            with open(auto_metadata_file, 'r') as f:
                metadata = json.load(f)
    
    print(f"✅ Model loaded successfully: {model_file}")
    if metadata:
        print(f"   Model Name: {metadata.get('model_name', 'Unknown')}")
        print(f"   Save Date: {metadata.get('save_date', 'Unknown')}")
        print(f"   Test Accuracy: {metadata.get('results', {}).get('test_accuracy', 'N/A')}")
    
    return model, metadata


def list_saved_models(save_dir="saved_models"):
    """
    List all saved models in the directory
    
    Args:
        save_dir: Directory containing saved models
    
    Returns:
        List of dictionaries containing model information
    """
    if not os.path.exists(save_dir):
        print(f"❌ Directory {save_dir} does not exist.")
        return []
    
    models = []
    
    # Find all model files
    for file in os.listdir(save_dir):
        if file.endswith(('.joblib', '.pkl')):
            model_file = os.path.join(save_dir, file)
            metadata_file = os.path.join(save_dir, file.replace('.joblib', '_metadata.json').replace('.pkl', '_metadata.json'))
            
            model_info = {
                "model_file": model_file,
                "metadata_file": metadata_file if os.path.exists(metadata_file) else None,
                "filename": file
            }
            
            # Load metadata if available
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    model_info.update(metadata)
                except:
                    model_info["metadata_error"] = True
            
            models.append(model_info)
    
    # Sort by timestamp/save date
    models.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return models


def display_saved_models(save_dir="saved_models"):
    """
    Display all saved models in a formatted table
    
    Args:
        save_dir: Directory containing saved models
    """
    models = list_saved_models(save_dir)
    
    if not models:
        print(f"📁 No saved models found in {save_dir}")
        return
    
    print(f"\n📁 Saved Models in {save_dir}:")
    print("=" * 100)
    print(f"{'Model Name':<20} {'Timestamp':<15} {'Accuracy':<10} {'File':<30}")
    print("-" * 100)
    
    for model in models:
        name = model.get('model_name', 'Unknown')[:19]
        timestamp = model.get('timestamp', 'Unknown')[:14]
        accuracy = model.get('results', {}).get('test_accuracy', 'N/A')
        if isinstance(accuracy, (int, float)):
            accuracy = f"{accuracy:.4f}"
        filename = model.get('filename', '')[:29]
        
        print(f"{name:<20} {timestamp:<15} {str(accuracy):<10} {filename:<30}")
    
    print("=" * 100)


def save_all_models(all_results, save_dir="saved_models"):
    """
    Save all trained models in batch
    
    Args:
        all_results: Dictionary containing all model results
        save_dir: Directory to save models
    
    Returns:
        Dictionary mapping model names to save information
    """
    save_summary = {}
    
    print("💾 Saving all models...")
    print("=" * 50)
    
    for model_name, result_data in all_results.items():
        if 'model' in result_data and 'results' in result_data:
            try:
                save_info = save_model(
                    model=result_data['model'],
                    results=result_data['results'],
                    model_name=model_name,
                    save_dir=save_dir
                )
                save_summary[model_name] = save_info
            except Exception as e:
                print(f"❌ Failed to save {model_name}: {str(e)}")
                save_summary[model_name] = {"success": False, "error": str(e)}
    
    print("=" * 50)
    print(f"✅ Batch save completed. {len([s for s in save_summary.values() if s.get('success')])} models saved successfully.")
    
    return save_summary


def load_model_by_name(model_name, save_dir="saved_models"):
    """
    Load the most recent model by name
    
    Args:
        model_name: Name of the model to load
        save_dir: Directory containing saved models
    
    Returns:
        Tuple of (model, metadata) or (None, None) if not found
    """
    models = list_saved_models(save_dir)
    
    # Find models with matching name
    matching_models = [m for m in models if m.get('model_name') == model_name]
    
    if not matching_models:
        print(f"❌ No saved models found with name: {model_name}")
        return None, None
    
    # Get the most recent one (first in sorted list)
    latest_model = matching_models[0]
    
    return load_model(latest_model['model_file'], latest_model.get('metadata_file'))


def compare_saved_models(save_dir="saved_models", metric='test_accuracy'):
    """
    Compare performance of all saved models
    
    Args:
        save_dir: Directory containing saved models
        metric: Metric to compare (default: test_accuracy)
    
    Returns:
        Sorted list of models by performance
    """
    models = list_saved_models(save_dir)
    
    if not models:
        print(f"📁 No saved models found in {save_dir}")
        return []
    
    # Extract performance data
    performance_data = []
    for model in models:
        results = model.get('results', {})
        if metric in results:
            performance_data.append({
                'model_name': model.get('model_name', 'Unknown'),
                'timestamp': model.get('timestamp', 'Unknown'),
                'performance': results[metric],
                'file': model.get('filename', '')
            })
    
    # Sort by performance (descending)
    performance_data.sort(key=lambda x: x['performance'], reverse=True)
    
    print(f"\n🏆 Model Performance Comparison ({metric}):")
    print("=" * 80)
    print(f"{'Rank':<5} {'Model Name':<20} {'Performance':<12} {'Timestamp':<15}")
    print("-" * 80)
    
    for i, model in enumerate(performance_data, 1):
        print(f"{i:<5} {model['model_name']:<20} {model['performance']:<12.4f} {model['timestamp']:<15}")
    
    print("=" * 80)
    
    return performance_data
