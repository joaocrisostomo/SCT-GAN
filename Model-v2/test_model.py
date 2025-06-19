#!/usr/bin/env python3
"""
Smart Contract Vulnerability Detection and Generation Script

This script loads a trained SmartContractTransformer model and provides functionality to:
1. Detect vulnerabilities in smart contracts
2. Generate synthetic smart contracts
3. Analyze contract-level and line-level vulnerabilities
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import json
import re
import argparse
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import sys
import os

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SmartContractTransformer
#from dataset import parse_solidity_to_ast, prepare_code2vec_input

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

class SmartContractAnalyzer:
    def __init__(self, model_path: str, tokenizer_name: str = "microsoft/codebert-base", device: str = None):
        """
        Initialize the Smart Contract Analyzer
        
        Args:
            model_path: Path to the trained model checkpoint
            tokenizer_name: Name of the tokenizer to use
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.model = SmartContractTransformer(
            d_model=768,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            max_length=1024,
            vocab_size=self.tokenizer.vocab_size,
            num_vulnerability_types=8
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Load trained model
        self._load_model(model_path)
        
        # Vulnerability types
        self.vulnerability_types = [
            'ARTHM', 'DOS', 'LE', 'RENT', 'TimeM', 'TimeO', 'Tx-Origin', 'UE'
        ]
        
        # Set model to evaluation mode
        self.model.eval()
        
    def _load_model(self, model_path: str):
        """Load the trained model from checkpoint"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded from {model_path}")
                print(f"Training epoch: {checkpoint.get('epoch', 'Unknown')}")
                print(f"Best validation loss: {checkpoint.get('val_loss', 'Unknown')}")
            else:
                # Direct state dict
                self.model.load_state_dict(checkpoint)
                print(f"Model loaded from {model_path}")
            
            # Ensure model is on the correct device
            self.model = self.model.to(self.device)
            
            # Verify all model parameters are on the correct device
            for name, param in self.model.named_parameters():
                if param.device != torch.device(self.device):
                    print(f"Warning: Parameter {name} is on {param.device}, moving to {self.device}")
                    param.data = param.data.to(self.device)
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_contract(self, contract_code: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess a smart contract for analysis
        
        Args:
            contract_code: The Solidity contract code
            
        Returns:
            Dictionary containing preprocessed tensors
        """
        # Parse AST and get paths
        ast = parse_solidity_to_ast(contract_code)
        ast_paths = prepare_code2vec_input(ast) if ast else []
        ast_path_text = ' '.join(ast_paths)
        
        # Tokenize contract code
        contract_encoding = self.tokenizer(
            contract_code,
            max_length=1024,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize AST paths
        ast_encoding = self.tokenizer(
            ast_path_text,
            max_length=1024,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create token-to-line mapping
        lines = contract_code.split('\n')
        token_to_line = []
        current_line = 0
        
        for line in lines:
            line_tokens = self.tokenizer.encode(line, add_special_tokens=False)
            token_to_line.extend([current_line] * len(line_tokens))
            current_line += 1
        
        # Add special tokens
        token_to_line = [0] + token_to_line + [0]
        
        # Truncate and pad
        if len(token_to_line) > 1024:
            token_to_line = token_to_line[:1024]
        else:
            token_to_line.extend([0] * (1024 - len(token_to_line)))
        
        # Create tensors and move to device
        token_to_line_tensor = torch.tensor(token_to_line, dtype=torch.long)
        
        # Ensure all tensors are on the correct device
        return {
            'input_ids': contract_encoding['input_ids'].to(self.device),
            'attention_mask': contract_encoding['attention_mask'].bool().to(self.device),
            'ast_input_ids': ast_encoding['input_ids'].to(self.device),
            'ast_attention_mask': ast_encoding['attention_mask'].bool().to(self.device),
            'token_to_line': token_to_line_tensor.to(self.device)
        }
    
    def detect_vulnerabilities(self, contract_code: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Detect vulnerabilities in a smart contract
        
        Args:
            contract_code: The Solidity contract code
            threshold: Probability threshold for vulnerability detection
            
        Returns:
            Dictionary containing vulnerability analysis results
        """
        with torch.no_grad():
            # Preprocess contract
            inputs = self.preprocess_contract(contract_code)
            
            # Ensure all inputs are on the correct device
            for key, tensor in inputs.items():
                if tensor.device != torch.device(self.device):
                    inputs[key] = tensor.to(self.device)
            
            # Run model
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                ast_input_ids=inputs['ast_input_ids'],
                ast_attention_mask=inputs['ast_attention_mask'],
                token_to_line=inputs['token_to_line']
            )
            
            # Get vulnerability predictions
            contract_vuln_logits = outputs['contract_vulnerability_logits']
            line_vuln_logits = outputs['line_vulnerability_logits']
            
            # Convert to probabilities
            contract_vuln_probs = torch.sigmoid(contract_vuln_logits)
            line_vuln_probs = torch.sigmoid(line_vuln_logits)
            
            # Process contract-level vulnerabilities
            contract_vulnerabilities = {}
            for i, vuln_type in enumerate(self.vulnerability_types):
                prob = contract_vuln_probs[0, i].item()
                contract_vulnerabilities[vuln_type] = {
                    'probability': prob,
                    'is_vulnerable': prob > threshold
                }
            
            # Process line-level vulnerabilities
            lines = contract_code.split('\n')
            line_vulnerabilities = {}
            
            for i, vuln_type in enumerate(self.vulnerability_types):
                line_vulnerabilities[vuln_type] = []
                for line_idx, line in enumerate(lines):
                    if line_idx < line_vuln_probs.size(1):
                        prob = line_vuln_probs[0, line_idx, i].item()
                        if prob > threshold:
                            line_vulnerabilities[vuln_type].append({
                                'line_number': line_idx + 1,
                                'line_content': line.strip(),
                                'probability': prob
                            })
            
            return {
                'contract_vulnerabilities': contract_vulnerabilities,
                'line_vulnerabilities': line_vulnerabilities,
                'summary': {
                    'total_vulnerable_lines': sum(len(vulns) for vulns in line_vulnerabilities.values()),
                    'vulnerability_types_found': [vuln_type for vuln_type, data in contract_vulnerabilities.items() if data['is_vulnerable']]
                }
            }
    
    def generate_synthetic_contract(self, contract_template: str, num_contracts: int = 1, 
                                  temperature: float = 0.8, max_length: int = 1024) -> List[str]:
        """
        Generate synthetic smart contracts based on a template
        
        Args:
            contract_template: Template contract code
            num_contracts: Number of contracts to generate
            temperature: Sampling temperature (higher = more random)
            max_length: Maximum length of generated contracts
            
        Returns:
            List of generated contract codes
        """
        generated_contracts = []
        
        with torch.no_grad():
            # Preprocess template
            inputs = self.preprocess_contract(contract_template)
            
            for _ in range(num_contracts):
                # Generate sequence
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    ast_input_ids=inputs['ast_input_ids'],
                    ast_attention_mask=inputs['ast_attention_mask'],
                    token_to_line=inputs['token_to_line']
                )
                
                generated_seq = outputs['generated_sequence']
                
                # Decode generated sequence
                generated_text = self.tokenizer.decode(generated_seq[0], skip_special_tokens=True)
                generated_contracts.append(generated_text)
        
        return generated_contracts
    
    def analyze_contract_file(self, file_path: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze a smart contract file
        
        Args:
            file_path: Path to the Solidity contract file
            threshold: Probability threshold for vulnerability detection
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                contract_code = f.read()
            
            return self.detect_vulnerabilities(contract_code, threshold)
            
        except Exception as e:
            return {'error': f"Error reading file {file_path}: {str(e)}"}
    
    def batch_analyze_contracts(self, contract_codes: List[str], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Analyze multiple contracts in batch
        
        Args:
            contract_codes: List of contract codes to analyze
            threshold: Probability threshold for vulnerability detection
            
        Returns:
            List of analysis results
        """
        results = []
        for i, contract_code in enumerate(contract_codes):
            print(f"Analyzing contract {i+1}/{len(contract_codes)}...")
            result = self.detect_vulnerabilities(contract_code, threshold)
            results.append(result)
        return results

def print_vulnerability_report(analysis_result: Dict[str, Any], contract_name: str = "Contract"):
    """Print a formatted vulnerability report"""
    print(f"\n{'='*60}")
    print(f"VULNERABILITY ANALYSIS REPORT: {contract_name}")
    print(f"{'='*60}")
    
    # Contract-level vulnerabilities
    print("\n📋 CONTRACT-LEVEL VULNERABILITIES:")
    print("-" * 40)
    for vuln_type, data in analysis_result['contract_vulnerabilities'].items():
        status = "🔴 VULNERABLE" if data['is_vulnerable'] else "🟢 SAFE"
        print(f"{vuln_type:12} | {status:15} | Probability: {data['probability']:.3f}")
    
    # Line-level vulnerabilities
    print("\n📄 LINE-LEVEL VULNERABILITIES:")
    print("-" * 40)
    for vuln_type, lines in analysis_result['line_vulnerabilities'].items():
        if lines:
            print(f"\n{vuln_type}:")
            for line_data in lines:
                print(f"  Line {line_data['line_number']:3d} | {line_data['probability']:.3f} | {line_data['line_content'][:50]}...")
    
    # Summary
    summary = analysis_result['summary']
    print(f"\n📊 SUMMARY:")
    print("-" * 40)
    print(f"Total vulnerable lines: {summary['total_vulnerable_lines']}")
    print(f"Vulnerability types found: {', '.join(summary['vulnerability_types_found']) if summary['vulnerability_types_found'] else 'None'}")

def main():
    parser = argparse.ArgumentParser(description='Smart Contract Vulnerability Detection and Generation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--mode', type=str, choices=['detect', 'generate', 'analyze_file'], required=True, help='Analysis mode')
    parser.add_argument('--input', type=str, help='Input contract code or file path')
    parser.add_argument('--output', type=str, help='Output file path for results')
    parser.add_argument('--threshold', type=float, default=0.5, help='Vulnerability detection threshold')
    parser.add_argument('--num_contracts', type=int, default=1, help='Number of contracts to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Generation temperature')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SmartContractAnalyzer(args.model_path, device=args.device)
    
    if args.mode == 'detect':
        if not args.input:
            print("Error: Input contract code required for detection mode")
            return
        
        # Analyze contract
        result = analyzer.detect_vulnerabilities(args.input, args.threshold)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print_vulnerability_report(result)
    
    elif args.mode == 'generate':
        if not args.input:
            print("Error: Input contract template required for generation mode")
            return
        
        # Generate synthetic contracts
        generated_contracts = analyzer.generate_synthetic_contract(
            args.input, 
            args.num_contracts, 
            args.temperature
        )
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({'generated_contracts': generated_contracts}, f, indent=2)
            print(f"Generated contracts saved to {args.output}")
        else:
            for i, contract in enumerate(generated_contracts):
                print(f"\n{'='*60}")
                print(f"GENERATED CONTRACT {i+1}")
                print(f"{'='*60}")
                print(contract)
    
    elif args.mode == 'analyze_file':
        if not args.input:
            print("Error: Input file path required for file analysis mode")
            return
        
        # Analyze contract file
        result = analyzer.analyze_contract_file(args.input, args.threshold)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print_vulnerability_report(result, Path(args.input).name)

if __name__ == "__main__":
    # Example usage if run directly
    if len(sys.argv) == 1:
        print("Smart Contract Vulnerability Detection and Generation Tool")
        print("\nExample usage:")
        print("python test_model.py --model_path checkpoints/best_model.pt --mode detect --input 'contract code here'")
        print("python test_model.py --model_path checkpoints/best_model.pt --mode analyze_file --input contract.sol")
        print("python test_model.py --model_path checkpoints/best_model.pt --mode generate --input template.sol --num_contracts 3")
    else:
        main()
