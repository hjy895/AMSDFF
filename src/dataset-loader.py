"""
Dataset Loading Module for Enhanced AMSDFF
Handles multiple text classification datasets
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class TextClassificationDataset(Dataset):
    """
    PyTorch Dataset for text classification
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 256
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DatasetLoader:
    """
    Unified dataset loader for multiple text classification datasets
    """
    
    def __init__(
        self,
        tokenizer_name: str = 'distilbert-base-uncased',
        max_length: int = 256,
        val_split: float = 0.2,
        test_split: float = 0.1,
        random_state: int = 42
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        
        # Dataset configurations
        self.dataset_configs = {
            'news_categorization': {
                'num_classes': 4,
                'categories': ['World', 'Sports', 'Business', 'Technology']
            },
            'sentiment_analysis': {
                'num_classes': 2,
                'categories': ['Negative', 'Positive']
            },
            'topic_classification': {
                'num_classes': 5,
                'categories': ['Politics', 'Entertainment', 'Science', 'Health', 'Education']
            },
            'document_classification': {
                'num_classes': 8,
                'categories': ['Finance', 'Legal', 'Medical', 'Technical', 
                              'Marketing', 'Academic', 'News', 'Personal']
            }
        }
    
    def load_dataset(
        self,
        dataset_name: str,
        data_path: Optional[str] = None
    ) -> Dict[str, DataLoader]:
        """
        Load a specific dataset
        
        Args:
            dataset_name: Name of the dataset to load
            data_path: Optional path to dataset file
            
        Returns:
            Dictionary containing train, validation, and test dataloaders
        """
        # Load raw data
        if data_path and os.path.exists(data_path):
            texts, labels = self._load_from_file(data_path)
        else:
            texts, labels = self._load_synthetic_data(dataset_name)
        
        # Split data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, 
            test_size=self.test_split,
            random_state=self.random_state,
            stratify=labels
        )
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels,
            test_size=self.val_split,
            random_state=self.random_state,
            stratify=train_labels
        )
        
        # Create datasets
        train_dataset = TextClassificationDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        val_dataset = TextClassificationDataset(
            val_texts, val_labels, self.tokenizer, self.max_length
        )
        test_dataset = TextClassificationDataset(
            test_texts, test_labels, self.tokenizer, self.max_length
        )
        
        # Create dataloaders
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=16, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=16, shuffle=False),
            'test': DataLoader(test_dataset, batch_size=16, shuffle=False)
        }
        
        return dataloaders
    
    def _load_from_file(self, file_path: str) -> Tuple[List[str], List[int]]:
        """Load dataset from file"""
        # Support multiple file formats
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            # Assume 'text' and 'label' columns
            texts = df['text'].tolist()
            labels = df['label'].tolist()
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
            texts = df['text'].tolist()
            labels = df['label'].tolist()
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        return texts, labels
    
    def _load_synthetic_data(
        self,
        dataset_name: str,
        num_samples: int = 2000
    ) -> Tuple[List[str], List[int]]:
        """Generate synthetic data for testing"""
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        num_classes = config['num_classes']
        categories = config['categories']
        
        texts = []
        labels = []
        
        samples_per_class = num_samples // num_classes
        
        for class_idx, category in enumerate(categories):
            for i in range(samples_per_class):
                # Generate synthetic text
                text = self._generate_synthetic_text(category, i)
                texts.append(text)
                labels.append(class_idx)
        
        return texts, labels
    
    #def _generate_synthetic_text(self, category: str, index: int) -> str:
     #   """Generate synthetic text for a category"""
      #  templates = {
          #  'World': "International news about {topic} affecting global {aspect} with implications for {region}.",
          #  'Sports': "In today's {sport} match, the {team} showed remarkable {quality} during the {event}.",
         #   'Business': "The {company} announced {action} in the {sector} market, affecting {stakeholder} significantly.",
            'Technology': "New {tech} innovation in {field} promises to revolutionize how we {action} in the future.",
            'Politics': "Political developments in {country} regarding {policy} have sparked debate about {issue}.",
            'Entertainment': "The latest {medium} featuring {artist} has captivated audiences with its {quality}.",
            'Science': "Researchers discovered {finding} in {field} that could lead to breakthroughs in {application}.",
            'Health': "Medical experts recommend {advice} for {condition} to improve {outcome} in patients.",
            'Finance': "Financial analysis shows {trend} in {market} with potential impacts on {instrument}.",
            'Legal': "Legal proceedings regarding {case} have set precedent for {area} of law.",
            'Medical': "Clinical studies on {treatment} for {condition} show promising results in {metric}.",
            'Technical': "Technical documentation for {system} explains the {process} implementation details.",
            'Marketing': "Marketing campaign for {product} targets {audience} through innovative {strategy}.",
            'Academic': "Academic research in {field} explores the relationship between {concept1} and {concept2}.",
            'News': "Breaking news: {event} occurred in {location} with {impact} on local community.",
            'Personal': "Personal experience with {topic} taught valuable lessons about {insight}.",
            'Negative': "This {subject} was disappointing due to {reason} and failed to meet expectations.",
            'Positive': "Excellent {subject} that exceeded expectations with outstanding {quality} throughout.",
            'Education': "Educational program on {subject} helps students understand {concept} more effectively."
        }###
        
        # Get template for category
        template = templates.get(category, "Generic text about {topic} with details about {aspect}.")
        
        # Fill in template with pseudo-random words
        words = {
            'topic': ['economics', 'technology', 'society', 'environment', 'culture'][index % 5],
            'aspect': ['development', 'impact', 'trends', 'challenges', 'opportunities'][index % 5],
            'region': ['Asia', 'Europe', 'Americas', 'Africa', 'Pacific'][index % 5],
            'sport': ['football', 'basketball', 'tennis', 'baseball', 'soccer'][index % 5],
            'team': ['champions', 'underdogs', 'rookies', 'veterans', 'all-stars'][index % 5],
            'quality': ['skill', 'teamwork', 'strategy', 'endurance', 'technique'][index % 5],
            'event': ['championship', 'tournament', 'match', 'game', 'competition'][index % 5],
            'company': ['TechCorp', 'GlobalInc', 'StartupX', 'MegaCo', 'InnovateLtd'][index % 5],
            'action': ['expansion', 'merger', 'launch', 'restructuring', 'investment'][index % 5],
            'sector': ['technology', 'healthcare', 'finance', 'retail', 'energy'][index % 5],
            'stakeholder': ['investors', 'employees', 'customers', 'partners', 'shareholders'][index % 5],
            'tech': ['AI', 'blockchain', 'quantum', 'biotech', 'nanotech'][index % 5],
            'field': ['computing', 'medicine', 'engineering', 'research', 'industry'][index % 5],
            'country': ['nation', 'state', 'republic', 'kingdom', 'union'][index % 5],
            'policy': ['reform', 'regulation', 'initiative', 'legislation', 'agreement'][index % 5],
            'issue': ['equality', 'security', 'economy', 'healthcare', 'education'][index % 5],
            'medium': ['film', 'series', 'album', 'book', 'show'][index % 5],
            'artist': ['creator', 'performer', 'director', 'writer', 'producer'][index % 5],
            'finding': ['breakthrough', 'discovery', 'innovation', 'insight', 'solution'][index % 5],
            'application': ['treatment', 'technology', 'methodology', 'system', 'approach'][index % 5],
            'advice': ['exercise', 'nutrition', 'meditation', 'screening', 'prevention'][index % 5],
            'condition': ['wellness', 'disease', 'syndrome', 'disorder', 'health'][index % 5],
            'outcome': ['recovery', 'wellbeing', 'longevity', 'quality', 'health'][index % 5],
            'trend': ['growth', 'decline', 'volatility', 'stability', 'recovery'][index % 5],
            'market': ['equity', 'bond', 'commodity', 'forex', 'crypto'][index % 5],
            'instrument': ['portfolios', 'investments', 'assets', 'securities', 'funds'][index % 5],
            'case': ['precedent', 'dispute', 'lawsuit', 'appeal', 'ruling'][index % 5],
            'area': ['corporate', 'criminal', 'civil', 'international', 'constitutional'][index % 5],
            'treatment': ['therapy', 'medication', 'procedure', 'intervention', 'protocol'][index % 5],
            'metric': ['efficacy', 'safety', 'outcomes', 'survival', 'improvement'][index % 5],
            'system': ['software', 'hardware', 'network', 'database', 'platform'][index % 5],
            'process': ['deployment', 'configuration', 'optimization', 'integration', 'maintenance'][index % 5],
            'product': ['solution', 'service', 'platform', 'tool', 'application'][index % 5],
            'audience': ['millennials', 'professionals', 'enterprises', 'consumers', 'students'][index % 5],
            'strategy': ['digital', 'content', 'social', 'influencer', 'experiential'][index % 5],
            'concept1': ['theory', 'methodology', 'framework', 'paradigm', 'model'][index % 5],
            'concept2': ['practice', 'application', 'implementation', 'results', 'outcomes'][index % 5],
            'event': ['incident', 'announcement', 'development', 'occurrence', 'situation'][index % 5],
            'location': ['downtown', 'suburb', 'district', 'region', 'area'][index % 5],
            'impact': ['significant', 'minimal', 'positive', 'negative', 'mixed'][index % 5],
            'subject': ['product', 'service', 'experience', 'performance', 'quality'][index % 5],
            'reason': ['flaws', 'issues', 'problems', 'defects', 'shortcomings'][index % 5],
            'insight': ['perseverance', 'patience', 'dedication', 'growth', 'learning'][index % 5]
        }
        
        # Format template
        text = template.format(**words)
        
        # Add some variation
        if index % 3 == 0:
            text = f"Breaking: {text}"
        elif index % 3 == 1:
            text = f"{text} According to experts, this is significant."
        else:
            text = f"Update: {text} More details to follow."
        
        return text
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get information about a dataset"""
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return self.dataset_configs[dataset_name]


class DataCollator:
    """
    Custom data collator for batching
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features])
        }
        return batch
