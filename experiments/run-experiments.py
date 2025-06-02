"""
Main experiment runner for Enhanced AMSDFF
"""

import os
import sys
import argparse
import yaml
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#import models accordingly
from data.dataset_loader import DatasetLoader
from models.amsdff import EnhancedAMSDFF
from models.base_models import TransformerModel
from training.trainer import AMSDFFTrainer
from evaluation.evaluator import ModelEvaluator
from utils.helpers import set_seed, create_results_table
from utils.visualization import plot_results


def load_config(config_path: str) -> dict:
    """Load experiment configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_single_experiment(
    model_name: str,
    dataset_name: str,
    config: dict,
    data_loader: DatasetLoader
) -> dict:
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"Running experiment: {model_name} on {dataset_name}")
    print(f"{'='*60}")
    
    # Load data
    dataloaders = data_loader.load_dataset(dataset_name)
    dataset_info = data_loader.get_dataset_info(dataset_name)
    
    # Create model
    if model_name == 'enhanced_amsdff':
        model = EnhancedAMSDFF(
            num_classes=dataset_info['num_classes'],
            hidden_size=config['model']['hidden_size'],
            pretrained_model=config['model']['pretrained_model']
        )
    elif model_name == 'lstm':
        from transformers import AutoModel
        base_model = AutoModel.from_pretrained(config['model']['pretrained_model'])
        model = LSTMModel(base_model, dataset_info['num_classes'])
    elif model_name == 'gru':
        from transformers import AutoModel
        base_model = AutoModel.from_pretrained(config['model']['pretrained_model'])
        model = GRUModel(base_model, dataset_info['num_classes'])
    elif model_name == 'transformer':
        from transformers import AutoModel
        base_model = AutoModel.from_pretrained(config['model']['pretrained_model'])
        model = TransformerModel(base_model, dataset_info['num_classes'])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create trainer
    trainer = AMSDFFTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        config=config['training']
    )
    
    # Train model
    print(f"Training {model_name}...")
    history = trainer.train(epochs=config['training']['epochs'])
    
    # Evaluate model
    print(f"Evaluating {model_name}...")
    evaluator = ModelEvaluator(model, trainer.device)
    results = evaluator.evaluate(dataloaders['test'])
    
    # Add training history to results
    results['history'] = history
    results['model_name'] = model_name
    results['dataset_name'] = dataset_name
    
    return results


def run_baseline_comparison(config: dict, data_loader: DatasetLoader) -> dict:
    """Run comparison with baseline models"""
    results = {}
    
    # Models to compare
    models = ['lstm', 'gru', 'transformer', 'enhanced_amsdff']
    datasets = config['datasets']
    
    for dataset_name in datasets:
        results[dataset_name] = {}
        
        for model_name in models:
            try:
                model_results = run_single_experiment(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    config=config,
                    data_loader=data_loader
                )
                results[dataset_name][model_name]
