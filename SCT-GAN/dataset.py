import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
import os
import pandas as pd
from typing import Dict, List, Tuple, Optional
import numpy as np
import re

# Set tokenizer parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_solidity_to_ast(code: str) -> Dict[str, Any]:
    """
    Parse Solidity code into a simplified AST structure
    """
    def extract_contract_info(code: str) -> Dict[str, Any]:
        # Extract contract name
        contract_match = re.search(r'contract\s+(\w+)', code)
        contract_name = contract_match.group(1) if contract_match else "Unknown"
        
        # Extract functions
        functions = []
        function_pattern = r'function\s+(\w+)\s*\(([^)]*)\)\s*(?:public|private|internal|external)?\s*(?:view|pure|payable)?\s*(?:returns\s*\(([^)]*)\))?\s*{'
        for match in re.finditer(function_pattern, code):
            func_name = match.group(1)
            params = match.group(2).split(',') if match.group(2) else []
            returns = match.group(3).split(',') if match.group(3) else []
            
            functions.append({
                'name': func_name,
                'parameters': [p.strip() for p in params],
                'returns': [r.strip() for r in returns]
            })
        
        # Extract state variables
        variables = []
        var_pattern = r'(?:uint|address|string|bool|mapping)\s+(?:\w+)\s+(\w+)'
        for match in re.finditer(var_pattern, code):
            variables.append(match.group(1))
        
        return {
            'type': 'Contract',
            'name': contract_name,
            'functions': functions,
            'variables': variables
        }
    
    try:
        # Clean the code
        code = re.sub(r'//.*?\n|/\*.*?\*/', '', code)  # Remove comments
        code = re.sub(r'\s+', ' ', code)  # Normalize whitespace
        
        # Parse the code
        ast = extract_contract_info(code)
        return ast
    except Exception as e:
        print(f"Error parsing code: {str(e)}")
        return None

def prepare_code2vec_input(ast: Dict[str, Any]) -> List[str]:
    """
    Convert AST to codeBert input format
    """
    paths = []
    
    def extract_paths(node: Dict[str, Any], current_path: List[str] = None):
        if current_path is None:
            current_path = []
            
        # Add current node to path
        if 'name' in node:
            current_path.append(node['name'])
            
        # Process functions
        if 'functions' in node:
            for func in node['functions']:
                func_path = current_path + [func['name']]
                paths.append(' '.join(func_path))
                
                # Add parameter paths
                for param in func['parameters']:
                    param_path = func_path + [param]
                    paths.append(' '.join(param_path))
                
                # Add return paths
                for ret in func['returns']:
                    ret_path = func_path + [ret]
                    paths.append(' '.join(ret_path))
        
        # Process variables
        if 'variables' in node:
            for var in node['variables']:
                var_path = current_path + [var]
                paths.append(' '.join(var_path))
    
    extract_paths(ast)
    return paths

class SmartContractVulnerabilityDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        split: str = "train",
        vulnerability_types: List[str] = None
    ):
        """
        Args:
            data_path: Path to the CSV file containing the dataset
            tokenizer: Tokenizer for encoding the source code
            max_length: Maximum sequence length
            split: "train" or "val" to specify which split to load
            vulnerability_types: List of vulnerability types to consider
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.vulnerability_types = vulnerability_types or [
            'ARTHM', 'DOS', 'LE', 'RENT', 'TimeM', 'TimeO', 'Tx-Origin', 'UE'
        ]
        
        # Load the dataset
        self.data = self._load_dataset(data_path)
        
    def _load_dataset(self, data_path: str) -> List[Dict]:
        """Load and preprocess the dataset from CSV"""
        dataset = []
        
        # Read the CSV file
        df = pd.read_csv(data_path)
        
        # Split into train/val if needed
        if self.split == "train":
            df = df.sample(frac=0.8, random_state=42)
        else:
            df = df.sample(frac=0.2, random_state=42)
        
        # Process each contract
        for _, row in df.iterrows():
            source_code = row['source_code']
            contract_name = row['contract_name']
            
            # Parse AST and get paths
            ast = parse_solidity_to_ast(source_code)
            ast_paths = prepare_code2vec_input(ast) if ast else []
            ast_path_text = ' '.join(ast_paths)
            
            # Split source code into lines
            lines = source_code.split('\n')
            
            # Create token-to-line mapping
            token_to_line = []
            current_line = 0
            
            # Tokenize each line separately to maintain mapping
            for line in lines:
                line_tokens = self.tokenizer.encode(line, add_special_tokens=False)
                token_to_line.extend([current_line] * len(line_tokens))
                current_line += 1
            
            # Add special tokens
            token_to_line = [0] + token_to_line + [0]  # [CLS] and [SEP] tokens
            
            # Truncate if too long
            if len(token_to_line) > self.max_length:
                token_to_line = token_to_line[:self.max_length]
            
            # Pad if too short
            if len(token_to_line) < self.max_length:
                token_to_line.extend([0] * (self.max_length - len(token_to_line)))
            
            # Create multi-label line labels for each vulnerability type
            line_labels = self._create_multi_label_line_labels(source_code, row)
            
            # Tokenize the source code
            encoding = self.tokenizer(
                source_code,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Tokenize AST paths
            ast_encoding = self.tokenizer(
                ast_path_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Convert line labels to tensor and ensure consistent shape
            vuln_tensor = torch.zeros((len(self.vulnerability_types), self.max_length), dtype=torch.long)
            for i, labels in enumerate(line_labels):
                if len(labels) > self.max_length:
                    labels = labels[:self.max_length]
                vuln_tensor[i, :len(labels)] = torch.tensor(labels, dtype=torch.long)
            
            # Convert token_to_line to tensor
            token_to_line_tensor = torch.tensor(token_to_line, dtype=torch.long)
            
            dataset.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'ast_input_ids': ast_encoding['input_ids'].squeeze(0),
                'ast_attention_mask': ast_encoding['attention_mask'].squeeze(0),
                'vulnerable_lines': vuln_tensor,
                'token_to_line': token_to_line_tensor,
                'source_code': source_code,
                'contract_name': contract_name
            })
        
        return dataset
    
    def _create_multi_label_line_labels(self, source_code: str, row: pd.Series) -> List[List[int]]:
        """Create multi-label line labels for each vulnerability type"""
        total_lines = len(source_code.split('\n'))
        line_labels = {vuln_type: [0] * total_lines for vuln_type in self.vulnerability_types}
        
        # Process each vulnerability type
        for vuln_type in self.vulnerability_types:
            vuln_lines = row[f'{vuln_type}_lines']
            if isinstance(vuln_lines, str):
                # Convert string representation of list to actual list
                try:
                    vuln_lines = eval(vuln_lines)
                except:
                    vuln_lines = []
            
            # Mark vulnerable lines for this type
            for line_num in vuln_lines:
                if isinstance(line_num, int) and 0 <= line_num < total_lines:
                    line_labels[vuln_type][line_num] = 1
        
        # Convert to list format
        return [line_labels[vuln_type] for vuln_type in self.vulnerability_types]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable length inputs
    """
    # Get the maximum length in this batch for each type of tensor
    max_input_len = max(item['input_ids'].size(0) for item in batch)
    
    # Pad all tensors to their respective maximum lengths
    padded_batch = {
        'input_ids': torch.stack([
            torch.nn.functional.pad(item['input_ids'], (0, max_input_len - item['input_ids'].size(0)))
            for item in batch
        ]),
        'attention_mask': torch.stack([
            torch.nn.functional.pad(item['attention_mask'], (0, max_input_len - item['attention_mask'].size(0)))
            for item in batch
        ]),
        'ast_input_ids': torch.stack([item['ast_input_ids'] for item in batch]),
        'ast_attention_mask': torch.stack([item['ast_attention_mask'] for item in batch]),
        'vulnerable_lines': torch.stack([item['vulnerable_lines'] for item in batch]),
        'token_to_line': torch.stack([item['token_to_line'] for item in batch]),
        'source_code': [item['source_code'] for item in batch],
        'contract_name': [item['contract_name'] for item in batch]
    }
    
    return padded_batch

def create_dataloaders(
    data_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_length: int = 1024,
    num_workers: int = 4,
    vulnerability_types: List[str] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        data_path: Path to the CSV file containing the dataset
        tokenizer: Tokenizer for encoding the source code
        batch_size: Batch size for training
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        vulnerability_types: List of vulnerability types to consider
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = SmartContractVulnerabilityDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        split="train",
        vulnerability_types=vulnerability_types
    )
    
    val_dataset = SmartContractVulnerabilityDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        split="val",
        vulnerability_types=vulnerability_types
    )
    
    # Create dataloaders with custom collate function
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return train_dataloader, val_dataloader

def inspect_dataloader(dataloader: torch.utils.data.DataLoader, num_batches: int = 1):
    """
    Inspect the contents of a dataloader
    
    Args:
        dataloader: The dataloader to inspect
        num_batches: Number of batches to inspect
    """
    print(f"\nDataloader has {len(dataloader)} batches")
    print(f"Batch size: {dataloader.batch_size}")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        print(f"\nBatch {batch_idx + 1}:")
        print(f"Number of samples in batch: {len(batch['input_ids'])}")
        print(f"Input shape: {batch['input_ids'].shape}")
        print(f"Vulnerability labels shape: {batch['vulnerable_lines'].shape}")
        
        # Print sample contract names
        print("\nSample contract names:")
        for name in batch['contract_name'][:2]:  # Show first 2 contracts
            print(f"- {name}")
            
        # Print vulnerability statistics
        vuln_labels = batch['vulnerable_lines']
        total_vulns = vuln_labels.sum().item()
        print(f"\nTotal vulnerable lines in batch: {total_vulns}")
        
        # Print sample of source code
        print("\nSample source code (first 200 chars):")
        print(batch['source_code'][0][:200] + "...")
        
        break  # Only show first batch

# Example usage:
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        data_path="path/to/your/vulnerabilities.csv",
        tokenizer=tokenizer,
        batch_size=8,
        max_length=1024
    )
    
    # Inspect train dataloader
    print("\nInspecting training dataloader:")
    inspect_dataloader(train_dataloader)
    
    # Inspect validation dataloader
    print("\nInspecting validation dataloader:")
    inspect_dataloader(val_dataloader) 