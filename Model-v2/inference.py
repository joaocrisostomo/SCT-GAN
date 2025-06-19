import torch
from transformers import AutoTokenizer
from model import SmartContractTransformer
import json
import re
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

class SmartContractAnalyzer:
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str = "microsoft/codebert-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the SmartContractAnalyzer with a trained model.
        
        Args:
            model_path: Path to the saved model checkpoint
            tokenizer_name: Name of the tokenizer to use
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Initialize model with same parameters as training
        self.model = SmartContractTransformer(
            d_model=768,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            max_length=1024,
            vocab_size=50265,
            num_vulnerability_types=8
        )
        
        # Load the trained model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Define vulnerability types
        self.vulnerability_types = [
            'ARTHM', 'DOS', 'LE', 'RENT', 'TimeM', 'TimeO', 'Tx-Origin', 'UE'
        ]
    
    def parse_solidity_to_ast(self, code: str) -> Dict[str, Any]:
        """Parse Solidity code into a simplified AST structure"""
        def extract_contract_info(code: str) -> Dict[str, Any]:
            contract_match = re.search(r'contract\s+(\w+)', code)
            contract_name = contract_match.group(1) if contract_match else "Unknown"
            
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
            code = re.sub(r'//.*?\n|/\*.*?\*/', '', code)
            code = re.sub(r'\s+', ' ', code)
            ast = extract_contract_info(code)
            return ast
        except Exception as e:
            print(f"Error parsing code: {str(e)}")
            return None

    def prepare_code2vec_input(self, ast: Dict[str, Any]) -> List[str]:
        """Convert AST to codeBert input format"""
        paths = []
        
        def extract_paths(node: Dict[str, Any], current_path: List[str] = None):
            if current_path is None:
                current_path = []
                
            if 'name' in node:
                current_path.append(node['name'])
                
            if 'functions' in node:
                for func in node['functions']:
                    func_path = current_path + [func['name']]
                    paths.append(' '.join(func_path))
                    
                    for param in func['parameters']:
                        param_path = func_path + [param]
                        paths.append(' '.join(param_path))
                    
                    for ret in func['returns']:
                        ret_path = func_path + [ret]
                        paths.append(' '.join(ret_path))
            
            if 'variables' in node:
                for var in node['variables']:
                    var_path = current_path + [var]
                    paths.append(' '.join(var_path))
        
        extract_paths(ast)
        return paths

    def detect_vulnerabilities(self, contract_code: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze a smart contract for vulnerabilities.
        
        Args:
            contract_code: The Solidity contract code to analyze
            threshold: Probability threshold for vulnerability detection
            
        Returns:
            Dictionary containing vulnerability analysis results
        """
        # Parse AST
        ast = self.parse_solidity_to_ast(contract_code)
        ast_paths = self.prepare_code2vec_input(ast) if ast else []
        ast_path_text = ' '.join(ast_paths)
        
        # Tokenize inputs
        contract_encoding = self.tokenizer(
            contract_code,
            max_length=1024,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        ast_encoding = self.tokenizer(
            ast_path_text,
            max_length=1024,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move tensors to device
        input_ids = contract_encoding['input_ids'].to(self.device)
        attention_mask = contract_encoding['attention_mask'].to(self.device)
        ast_input_ids = ast_encoding['input_ids'].to(self.device)
        ast_attention_mask = ast_encoding['attention_mask'].to(self.device)
        
        # Create token-to-line mapping
        lines = contract_code.split('\n')
        token_to_line = []
        current_line = 0
        
        for line in lines:
            line_tokens = self.tokenizer.encode(line, add_special_tokens=False)
            token_to_line.extend([current_line] * len(line_tokens))
            current_line += 1
        
        token_to_line = [0] + token_to_line + [0]
        if len(token_to_line) > 1024:
            token_to_line = token_to_line[:1024]
        if len(token_to_line) < 1024:
            token_to_line.extend([0] * (1024 - len(token_to_line)))
        
        token_to_line = torch.tensor(token_to_line, dtype=torch.long).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ast_input_ids=ast_input_ids,
                ast_attention_mask=ast_attention_mask,
                token_to_line=token_to_line
            )
            
            vuln_logits = outputs['vulnerability_logits']
            token_vuln_logits = outputs['token_vulnerability_logits']
            
            # Convert logits to probabilities
            vuln_probs = torch.sigmoid(vuln_logits)
            
            # Get predictions above threshold
            predictions = (vuln_probs > threshold).cpu().numpy()
            
            # Map predictions to lines
            line_predictions = {}
            for i, line in enumerate(lines):
                line_predictions[line] = {
                    vuln_type: bool(predictions[i, j])
                    for j, vuln_type in enumerate(self.vulnerability_types)
                }
        
        return {
            'vulnerabilities': line_predictions,
            'probabilities': vuln_probs.cpu().numpy().tolist()
        }

    def generate_synthetic_contract(
        self,
        contract_template: str,
        num_contracts: int = 1,
        temperature: float = 0.8,
        max_length: int = 1024
    ) -> List[str]:
        """
        Generate synthetic smart contracts based on a template.
        
        Args:
            contract_template: Template contract code to base generation on
            num_contracts: Number of contracts to generate
            temperature: Sampling temperature (higher = more random)
            max_length: Maximum length of generated contracts
            
        Returns:
            List of generated contract codes
        """
        # Parse template AST
        ast = self.parse_solidity_to_ast(contract_template)
        ast_paths = self.prepare_code2vec_input(ast) if ast else []
        ast_path_text = ' '.join(ast_paths)
        
        # Tokenize template
        template_encoding = self.tokenizer(
            contract_template,
            max_length=1024,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        ast_encoding = self.tokenizer(
            ast_path_text,
            max_length=1024,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move tensors to device
        input_ids = template_encoding['input_ids'].to(self.device)
        attention_mask = template_encoding['attention_mask'].to(self.device)
        ast_input_ids = ast_encoding['input_ids'].to(self.device)
        ast_attention_mask = ast_encoding['attention_mask'].to(self.device)
        
        generated_contracts = []
        
        for _ in range(num_contracts):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ast_input_ids=ast_input_ids,
                    ast_attention_mask=ast_attention_mask
                )
                
                generated_seq = outputs['generated_sequence']
                
                # Decode generated sequence
                generated_code = self.tokenizer.decode(
                    generated_seq[0].cpu().numpy(),
                    skip_special_tokens=True
                )
                
                generated_contracts.append(generated_code)
        
        return generated_contracts

def main():
    # Example usage
    analyzer = SmartContractAnalyzer(
        model_path="checkpoints/best_model.pt",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Example contract
    contract_code = """
    pragma solidity ^0.8.0;
    
    contract Example {
        uint256 public value;
        
        function setValue(uint256 _value) public {
            value = _value;
        }
        
        function getValue() public view returns (uint256) {
            return value;
        }
    }
    """
    
    # Detect vulnerabilities
    vulnerabilities = analyzer.detect_vulnerabilities(contract_code)
    print("\nVulnerability Analysis:")
    print(json.dumps(vulnerabilities, indent=2))
    
    # Generate synthetic contracts
    synthetic_contracts = analyzer.generate_synthetic_contract(
        contract_template=contract_code,
        num_contracts=2
    )
    
    print("\nGenerated Contracts:")
    for i, contract in enumerate(synthetic_contracts, 1):
        print(f"\nContract {i}:")
        print(contract)

if __name__ == "__main__":
    main() 