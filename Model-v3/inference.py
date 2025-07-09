import torch
from transformers import AutoTokenizer
from model import SmartContractTransformer
import json
import re
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import math
import random

class SmartContractAnalyzer:
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str = "microsoft/codebert-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_gan: bool = True
    ):
        """
        Initialize the SmartContractAnalyzer with a trained model.
        
        Args:
            model_path: Path to the saved model checkpoint
            tokenizer_name: Name of the tokenizer to use
            device: Device to run the model on ('cuda' or 'cpu')
            use_gan: Whether to enable GAN components
        """
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model with same parameters as training
        self.model = SmartContractTransformer(
            d_model=768,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            max_length=1024,
            vocab_size=self.tokenizer.vocab_size,
            num_vulnerability_types=8,
            use_gan=use_gan  # Only use_gan is a model parameter
        )
        
        # Load the trained model
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
            print(f"Training epoch: {checkpoint.get('epoch', 'Unknown')}")
            print(f"Best validation loss: {checkpoint.get('val_loss', 'Unknown')}")
            print(f"Training config: GAN={checkpoint.get('use_gan', 'Unknown')}")
        else:
            # Direct state dict
            self.model.load_state_dict(checkpoint)
            print(f"Model loaded from {model_path}")
        
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
        
        # Create token-to-line mapping that matches the actual tokenization
        lines = contract_code.split('\n')
        token_to_line = []
        current_line = 0
        
        # Add [CLS] token mapping (line 0)
        token_to_line.append(0)
        
        for line in lines:
            line_tokens = self.tokenizer.encode(line, add_special_tokens=False)
            # Map all tokens in this line to the same line number
            token_to_line.extend([current_line] * len(line_tokens))
            current_line += 1
        
        # Add [SEP] token mapping (line 0)
        token_to_line.append(0)
        
        # Ensure the mapping matches the tokenized length (1024)
        if len(token_to_line) > 1024:
            token_to_line = token_to_line[:1024]
        elif len(token_to_line) < 1024:
            token_to_line.extend([0] * (1024 - len(token_to_line)))
        
        token_to_line = torch.tensor(token_to_line, dtype=torch.long).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ast_input_ids=ast_input_ids,
                    ast_attention_mask=ast_attention_mask,
                    target_ids=input_ids,  # Use input_ids as target for inference
                    token_to_line=token_to_line
                )
            except Exception as e:
                print(f"Error during model forward pass: {str(e)}")
                print(f"Debug info:")
                print(f"  input_ids shape: {input_ids.shape}")
                print(f"  attention_mask shape: {attention_mask.shape}")
                print(f"  ast_input_ids shape: {ast_input_ids.shape}")
                print(f"  ast_attention_mask shape: {ast_attention_mask.shape}")
                print(f"  token_to_line shape: {token_to_line.shape}")
                print(f"  token_to_line max value: {token_to_line.max().item()}")
                print(f"  token_to_line min value: {token_to_line.min().item()}")
                
                # Try without target_ids for older model versions
                try:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        ast_input_ids=ast_input_ids,
                        ast_attention_mask=ast_attention_mask,
                        token_to_line=token_to_line
                    )
                except Exception as e2:
                    print(f"Error with fallback forward pass: {str(e2)}")
                    print("Model inference failed completely. Returning empty results.")
                    # Return empty results instead of raising
                    return {
                        'contract_vulnerabilities': {vuln_type: False for vuln_type in self.vulnerability_types},
                        'line_vulnerabilities': {},
                        'contract_probabilities': [[0.0] * len(self.vulnerability_types)],
                        'line_probabilities': []
                    }
            
            # Get contract-level and line-level vulnerability predictions
            contract_vuln_logits = outputs.get('contract_vulnerability_logits')
            line_vuln_logits = outputs.get('line_vulnerability_logits')
            
            if contract_vuln_logits is None or line_vuln_logits is None:
                print("Warning: Vulnerability detection outputs not found in model")
                print(f"Debug: contract_vuln_logits is None: {contract_vuln_logits is None}")
                print(f"Debug: line_vuln_logits is None: {line_vuln_logits is None}")
                print(f"Debug: Available outputs keys: {list(outputs.keys())}")
                # Return empty results
                return {
                    'contract_vulnerabilities': {vuln_type: False for vuln_type in self.vulnerability_types},
                    'line_vulnerabilities': {},
                    'contract_probabilities': [[0.0] * len(self.vulnerability_types)],
                    'line_probabilities': []
                }
            
            # Debug: Print shapes to understand the output
            print(f"Contract vuln logits shape: {contract_vuln_logits.shape}")
            print(f"Line vuln logits shape: {line_vuln_logits.shape}")
            print(f"Number of lines in contract: {len(lines)}")
            
            # Convert logits to probabilities
            contract_vuln_probs = torch.sigmoid(contract_vuln_logits)
            line_vuln_probs = torch.sigmoid(line_vuln_logits)
            
            # Get contract-level predictions above threshold
            contract_predictions = (contract_vuln_probs > threshold).cpu().numpy()
            
            # Get line-level predictions above threshold
            line_predictions = (line_vuln_probs > threshold).cpu().numpy()
            
            print(f"Contract predictions shape: {contract_predictions.shape}")
            print(f"Line predictions shape: {line_predictions.shape}")
            
            # Ensure we have the correct shape for line predictions
            # The model outputs [batch_size, seq_len, num_vulnerability_types]
            # We need to handle the case where seq_len might not match the number of lines
            if len(line_predictions.shape) == 3:
                line_predictions = line_predictions[0]  # Remove batch dimension

            num_lines = len(lines)
            num_preds = line_predictions.shape[0]
            # Truncate or pad line_predictions to match the number of lines
            if num_preds > num_lines:
                line_predictions = line_predictions[:num_lines]
            elif num_preds < num_lines:
                pad = np.zeros((num_lines - num_preds, line_predictions.shape[1]))
                line_predictions = np.concatenate([line_predictions, pad], axis=0)

            # Map line predictions to actual lines
            line_vulnerabilities = {}
            for i, line in enumerate(lines):
                line_vulnerabilities[i] = {}
                for j, vuln_type in enumerate(self.vulnerability_types):
                    pred_value = line_predictions[i, j]
                    if hasattr(pred_value, 'item'):
                        line_vulnerabilities[i][vuln_type] = bool(pred_value.item())
                    elif hasattr(pred_value, '__len__') and len(pred_value) == 1:
                        line_vulnerabilities[i][vuln_type] = bool(pred_value[0])
                    else:
                        line_vulnerabilities[i][vuln_type] = bool(pred_value)
            
            # Contract-level vulnerability summary
            # Ensure contract_predictions has the right shape
            if len(contract_predictions.shape) == 2:
                contract_predictions = contract_predictions[0]  # Remove batch dimension
            
            contract_vulnerabilities = {}
            for j, vuln_type in enumerate(self.vulnerability_types):
                # Explicitly convert to boolean, handling both scalar and array cases
                pred_value = contract_predictions[j]
                if hasattr(pred_value, 'item'):
                    contract_vulnerabilities[vuln_type] = bool(pred_value.item())
                elif hasattr(pred_value, '__len__') and len(pred_value) == 1:
                    contract_vulnerabilities[vuln_type] = bool(pred_value[0])
                else:
                    contract_vulnerabilities[vuln_type] = bool(pred_value)
        
        # Debug: Print the structure of what we're returning
        contract_probs_list = contract_vuln_probs.cpu().numpy().tolist()
        print(f"Debug: contract_vuln_probs shape: {contract_vuln_probs.shape}")
        print(f"Debug: contract_vuln_probs.cpu().numpy() shape: {contract_vuln_probs.cpu().numpy().shape}")
        print(f"Debug: contract_probs_list type: {type(contract_probs_list)}")
        print(f"Debug: contract_probs_list length: {len(contract_probs_list)}")
        if len(contract_probs_list) > 0:
            print(f"Debug: contract_probs_list[0] type: {type(contract_probs_list[0])}")
            print(f"Debug: contract_probs_list[0] length: {len(contract_probs_list[0])}")
        
        return {
            'contract_vulnerabilities': contract_vulnerabilities,
            'line_vulnerabilities': line_vulnerabilities,
            'contract_probabilities': contract_probs_list,
            'line_probabilities': line_vuln_probs.cpu().numpy().tolist()
        }

    def detect_vulnerabilities_with_fallback(self, contract_code: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze a smart contract for vulnerabilities with fallback for tensor stacking errors.
        
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
        
        # Get model predictions with fallback
        with torch.no_grad():
            try:
                # First attempt: normal forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ast_input_ids=ast_input_ids,
                    ast_attention_mask=ast_attention_mask,
                    target_ids=input_ids,  # Use input_ids as target for inference
                    token_to_line=token_to_line
                )
            except RuntimeError as e:
                if "stack expects each tensor to be equal size" in str(e):
                    print(f"⚠️  Tensor stacking error detected, using fallback method...")
                    # Fallback: Use a simplified forward pass without line-level detection
                    outputs = self._fallback_forward_pass(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        ast_input_ids=ast_input_ids,
                        ast_attention_mask=ast_attention_mask,
                        token_to_line=token_to_line
                    )
                else:
                    print(f"Error during model forward pass: {str(e)}")
                    # Return empty results
                    return {
                        'contract_vulnerabilities': {vuln_type: False for vuln_type in self.vulnerability_types},
                        'line_vulnerabilities': {},
                        'contract_probabilities': [[0.0] * len(self.vulnerability_types)],
                        'line_probabilities': []
                    }
            except Exception as e:
                print(f"Error during model forward pass: {str(e)}")
                # Try without target_ids for older model versions
                try:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        ast_input_ids=ast_input_ids,
                        ast_attention_mask=ast_attention_mask,
                        token_to_line=token_to_line
                    )
                except Exception as e2:
                    print(f"Error with fallback forward pass: {str(e2)}")
                    print("Model inference failed completely. Returning empty results.")
                    # Return empty results instead of raising
                    return {
                        'contract_vulnerabilities': {vuln_type: False for vuln_type in self.vulnerability_types},
                        'line_vulnerabilities': {},
                        'contract_probabilities': [[0.0] * len(self.vulnerability_types)],
                        'line_probabilities': []
                    }
            
            # Get contract-level and line-level vulnerability predictions
            contract_vuln_logits = outputs.get('contract_vulnerability_logits')
            line_vuln_logits = outputs.get('line_vulnerability_logits')
            
            if contract_vuln_logits is None or line_vuln_logits is None:
                print("Warning: Vulnerability detection outputs not found in model")
                print(f"Debug: contract_vuln_logits is None: {contract_vuln_logits is None}")
                print(f"Debug: line_vuln_logits is None: {line_vuln_logits is None}")
                print(f"Debug: Available outputs keys: {list(outputs.keys())}")
                # Return empty results
                return {
                    'contract_vulnerabilities': {vuln_type: False for vuln_type in self.vulnerability_types},
                    'line_vulnerabilities': {},
                    'contract_probabilities': [[0.0] * len(self.vulnerability_types)],
                    'line_probabilities': []
                }
            
            # Debug: Print shapes to understand the output
            print(f"Contract vuln logits shape: {contract_vuln_logits.shape}")
            print(f"Line vuln logits shape: {line_vuln_logits.shape}")
            print(f"Number of lines in contract: {len(lines)}")
            
            # Convert logits to probabilities
            contract_vuln_probs = torch.sigmoid(contract_vuln_logits)
            line_vuln_probs = torch.sigmoid(line_vuln_logits)
            
            # Get contract-level predictions above threshold
            contract_predictions = (contract_vuln_probs > threshold).cpu().numpy()
            
            # Get line-level predictions above threshold
            line_predictions = (line_vuln_probs > threshold).cpu().numpy()
            
            print(f"Contract predictions shape: {contract_predictions.shape}")
            print(f"Line predictions shape: {line_predictions.shape}")
            
            # Ensure we have the correct shape for line predictions
            # The model outputs [batch_size, seq_len, num_vulnerability_types]
            # We need to handle the case where seq_len might not match the number of lines
            if len(line_predictions.shape) == 3:
                line_predictions = line_predictions[0]  # Remove batch dimension

            num_lines = len(lines)
            num_preds = line_predictions.shape[0]
            # Truncate or pad line_predictions to match the number of lines
            if num_preds > num_lines:
                line_predictions = line_predictions[:num_lines]
            elif num_preds < num_lines:
                pad = np.zeros((num_lines - num_preds, line_predictions.shape[1]))
                line_predictions = np.concatenate([line_predictions, pad], axis=0)

            # Map line predictions to actual lines
            line_vulnerabilities = {}
            for i, line in enumerate(lines):
                line_vulnerabilities[i] = {}
                for j, vuln_type in enumerate(self.vulnerability_types):
                    pred_value = line_predictions[i, j]
                    if hasattr(pred_value, 'item'):
                        line_vulnerabilities[i][vuln_type] = bool(pred_value.item())
                    elif hasattr(pred_value, '__len__') and len(pred_value) == 1:
                        line_vulnerabilities[i][vuln_type] = bool(pred_value[0])
                    else:
                        line_vulnerabilities[i][vuln_type] = bool(pred_value)
            
            # Contract-level vulnerability summary
            # Ensure contract_predictions has the right shape
            if len(contract_predictions.shape) == 2:
                contract_predictions = contract_predictions[0]  # Remove batch dimension
            
            contract_vulnerabilities = {}
            for j, vuln_type in enumerate(self.vulnerability_types):
                # Explicitly convert to boolean, handling both scalar and array cases
                pred_value = contract_predictions[j]
                if hasattr(pred_value, 'item'):
                    contract_vulnerabilities[vuln_type] = bool(pred_value.item())
                elif hasattr(pred_value, '__len__') and len(pred_value) == 1:
                    contract_vulnerabilities[vuln_type] = bool(pred_value[0])
                else:
                    contract_vulnerabilities[vuln_type] = bool(pred_value)
        
        # Debug: Print the structure of what we're returning
        contract_probs_list = contract_vuln_probs.cpu().numpy().tolist()
        print(f"Debug: contract_vuln_probs shape: {contract_vuln_probs.shape}")
        print(f"Debug: contract_vuln_probs.cpu().numpy() shape: {contract_vuln_probs.cpu().numpy().shape}")
        print(f"Debug: contract_probs_list type: {type(contract_probs_list)}")
        print(f"Debug: contract_probs_list length: {len(contract_probs_list)}")
        if len(contract_probs_list) > 0:
            print(f"Debug: contract_probs_list[0] type: {type(contract_probs_list[0])}")
            print(f"Debug: contract_probs_list[0] length: {len(contract_probs_list[0])}")
        
        return {
            'contract_vulnerabilities': contract_vulnerabilities,
            'line_vulnerabilities': line_vulnerabilities,
            'contract_probabilities': contract_probs_list,
            'line_probabilities': line_vuln_probs.cpu().numpy().tolist()
        }

    def _fallback_forward_pass(self, input_ids, attention_mask, ast_input_ids, ast_attention_mask, token_to_line):
        """
        Fallback forward pass that bypasses the problematic line-level vulnerability detection.
        This method only computes contract-level vulnerabilities to avoid the tensor stacking error.
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        
        # Contract embeddings with improved processing
        contract_emb = self.model.embedding(input_ids) * math.sqrt(self.model.d_model)
        contract_emb = self.model.embedding_dropout(contract_emb)
        contract_emb = self.model.embedding_norm(contract_emb)
        contract_emb = self.model.pos_encoder(contract_emb.transpose(0, 1)).transpose(0, 1)
        
        # AST path embeddings with improved processing
        ast_emb = self.model.ast_embedding(ast_input_ids) * math.sqrt(self.model.d_model)
        ast_emb = self.model.ast_embedding_dropout(ast_emb)
        ast_emb = self.model.ast_embedding_norm(ast_emb)
        ast_emb = self.model.pos_encoder(ast_emb.transpose(0, 1)).transpose(0, 1)
        
        # Create source mask and ensure it's boolean
        src_mask = attention_mask if attention_mask is not None else torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        src_mask = src_mask.bool()  # Ensure boolean type
        
        # Encode contract with improved attention
        memory = self.model.encoder(contract_emb, src_key_padding_mask=~src_mask)
        
        # Apply AST path attention with residual connection - simplified
        if ast_attention_mask is not None:
            ast_attention_mask = ast_attention_mask.bool()
            ast_attn_output, _ = self.model.ast_attention(
                query=memory,
                key=ast_emb,
                value=ast_emb,
                key_padding_mask=~ast_attention_mask
            )
            memory = memory + 0.1 * ast_attn_output  # Reduced residual weight
        
        # Apply cross-attention between contract and AST - simplified
        if ast_attention_mask is not None:
            cross_attn_output, _ = self.model.cross_attention(
                query=memory,
                key=ast_emb,
                value=ast_emb,
                key_padding_mask=~ast_attention_mask
            )
            # Simplified feature fusion with reduced complexity
            fused_features = self.model.feature_fusion(torch.cat([memory, 0.1 * cross_attn_output], dim=-1))
            memory = memory + 0.1 * fused_features  # Reduced residual weight
        
        # Get contract-level vulnerability predictions with better feature aggregation
        # Apply contract-level attention to focus on vulnerability-relevant parts
        contract_attn_output, contract_attn_weights = self.model.contract_vuln_attention(
            query=memory.mean(dim=1, keepdim=True),  # Global query [batch_size, 1, d_model]
            key=memory,  # [batch_size, seq_len, d_model]
            value=memory,  # [batch_size, seq_len, d_model]
            attn_mask=None
        )  # Output: [batch_size, 1, d_model]
        
        # Aggregate contract features with attention
        global_avg = memory.mean(dim=1)  # [batch_size, d_model]
        attention_weighted = contract_attn_output.squeeze(1)  # [batch_size, d_model]
        
        # Concatenate global average and attention-weighted features
        contract_representation = torch.cat([
            global_avg,  # [batch_size, d_model]
            attention_weighted  # [batch_size, d_model]
        ], dim=-1)
        
        # Apply feature aggregation
        contract_features = self.model.contract_feature_aggregation(contract_representation)  # [batch_size, d_model]
        
        # Get contract-level vulnerability predictions
        contract_vuln_logits = self.model.contract_vulnerability_head(contract_features)  # [batch_size, num_vuln_types]
        
        # Create dummy line-level outputs to maintain compatibility
        # Use zeros for line-level predictions since we can't compute them properly
        line_vuln_logits = torch.zeros((batch_size, 1024, 8), device=device)  # [batch_size, max_lines, num_vuln_types]
        
        return {
            'contract_vulnerability_logits': contract_vuln_logits,
            'line_vulnerability_logits': line_vuln_logits
        }

    def detect_vulnerabilities_safe(self, contract_code: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Safe vulnerability detection that bypasses the problematic line-level detection.
        This method only computes contract-level vulnerabilities to avoid tensor stacking errors.
        
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
        
        # Get model predictions using safe forward pass
        with torch.no_grad():
            try:
                # Use the safe forward pass that bypasses line-level detection
                outputs = self._safe_forward_pass(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ast_input_ids=ast_input_ids,
                    ast_attention_mask=ast_attention_mask
                )
            except Exception as e:
                print(f"Error in safe forward pass: {str(e)}")
                # Return empty results
                return {
                    'contract_vulnerabilities': {vuln_type: False for vuln_type in self.vulnerability_types},
                    'line_vulnerabilities': {},
                    'contract_probabilities': [[0.0] * len(self.vulnerability_types)],
                    'line_probabilities': []
                }
            
            # Get contract-level vulnerability predictions
            contract_vuln_logits = outputs.get('contract_vulnerability_logits')
            
            if contract_vuln_logits is None:
                print("Warning: Contract vulnerability detection outputs not found")
                return {
                    'contract_vulnerabilities': {vuln_type: False for vuln_type in self.vulnerability_types},
                    'line_vulnerabilities': {},
                    'contract_probabilities': [[0.0] * len(self.vulnerability_types)],
                    'line_probabilities': []
                }
            
            # Convert logits to probabilities
            contract_vuln_probs = torch.sigmoid(contract_vuln_logits)
            
            # Get contract-level predictions above threshold
            contract_predictions = (contract_vuln_probs > threshold).cpu().numpy()
            
            # Ensure contract_predictions has the right shape
            if len(contract_predictions.shape) == 2:
                contract_predictions = contract_predictions[0]  # Remove batch dimension
            
            # Contract-level vulnerability summary
            contract_vulnerabilities = {}
            for j, vuln_type in enumerate(self.vulnerability_types):
                pred_value = contract_predictions[j]
                if hasattr(pred_value, 'item'):
                    contract_vulnerabilities[vuln_type] = bool(pred_value.item())
                elif hasattr(pred_value, '__len__') and len(pred_value) == 1:
                    contract_vulnerabilities[vuln_type] = bool(pred_value[0])
                else:
                    contract_vulnerabilities[vuln_type] = bool(pred_value)
            
            # Create empty line-level results since we can't compute them safely
            lines = contract_code.split('\n')
            line_vulnerabilities = {}
            for i, line in enumerate(lines):
                line_vulnerabilities[i] = {vuln_type: False for vuln_type in self.vulnerability_types}
            
            # Create dummy line probabilities (all zeros)
            line_vuln_probs = torch.zeros((1, len(lines), len(self.vulnerability_types)), device=self.device)
        
        return {
            'contract_vulnerabilities': contract_vulnerabilities,
            'line_vulnerabilities': line_vulnerabilities,
            'contract_probabilities': contract_vuln_probs.cpu().numpy().tolist(),
            'line_probabilities': line_vuln_probs.cpu().numpy().tolist()
        }

    def _safe_forward_pass(self, input_ids, attention_mask, ast_input_ids, ast_attention_mask):
        """
        Safe forward pass that completely bypasses the problematic line-level vulnerability detection.
        This method only computes contract-level vulnerabilities.
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        
        # Contract embeddings
        contract_emb = self.model.embedding(input_ids) * math.sqrt(self.model.d_model)
        contract_emb = self.model.embedding_dropout(contract_emb)
        contract_emb = self.model.embedding_norm(contract_emb)
        contract_emb = self.model.pos_encoder(contract_emb.transpose(0, 1)).transpose(0, 1)
        
        # AST path embeddings
        ast_emb = self.model.ast_embedding(ast_input_ids) * math.sqrt(self.model.d_model)
        ast_emb = self.model.ast_embedding_dropout(ast_emb)
        ast_emb = self.model.ast_embedding_norm(ast_emb)
        ast_emb = self.model.pos_encoder(ast_emb.transpose(0, 1)).transpose(0, 1)
        
        # Create source mask
        src_mask = attention_mask if attention_mask is not None else torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        src_mask = src_mask.bool()
        
        # Encode contract
        memory = self.model.encoder(contract_emb, src_key_padding_mask=~src_mask)
        
        # Apply AST path attention (simplified)
        if ast_attention_mask is not None:
            ast_attention_mask = ast_attention_mask.bool()
            ast_attn_output, _ = self.model.ast_attention(
                query=memory,
                key=ast_emb,
                value=ast_emb,
                key_padding_mask=~ast_attention_mask
            )
            memory = memory + 0.1 * ast_attn_output
        
        # Apply cross-attention (simplified)
        if ast_attention_mask is not None:
            cross_attn_output, _ = self.model.cross_attention(
                query=memory,
                key=ast_emb,
                value=ast_emb,
                key_padding_mask=~ast_attention_mask
            )
            fused_features = self.model.feature_fusion(torch.cat([memory, 0.1 * cross_attn_output], dim=-1))
            memory = memory + 0.1 * fused_features
        
        # Get contract-level vulnerability predictions
        # Apply contract-level attention
        contract_attn_output, _ = self.model.contract_vuln_attention(
            query=memory.mean(dim=1, keepdim=True),
            key=memory,
            value=memory,
            attn_mask=None
        )
        
        # Aggregate contract features
        global_avg = memory.mean(dim=1)
        attention_weighted = contract_attn_output.squeeze(1)
        
        # Concatenate features
        contract_representation = torch.cat([global_avg, attention_weighted], dim=-1)
        
        # Apply feature aggregation
        contract_features = self.model.contract_feature_aggregation(contract_representation)
        
        # Get contract-level vulnerability predictions
        contract_vuln_logits = self.model.contract_vulnerability_head(contract_features)
        
        return {
            'contract_vulnerability_logits': contract_vuln_logits
        }

    def generate_synthetic_contract(
        self,
        contract_template: str,
        num_contracts: int = 1,
        temperature: float = 0.8,
        max_length: int = 1024,
        use_beam_search: bool = False,
        beam_size: int = 3
    ) -> List[str]:
        """
        Generate synthetic smart contracts based on a template using the model's forward method.
        
        Args:
            contract_template: Template contract code to base generation on
            num_contracts: Number of contracts to generate
            temperature: Sampling temperature (higher = more random)
            max_length: Maximum length of generated contracts
            use_beam_search: Whether to use beam search for better quality
            beam_size: Beam size for beam search (if enabled)
            
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
        
        # Create token-to-line mapping for the template
        lines = contract_template.split('\n')
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
        
        generated_contracts = []
        
        for contract_idx in range(num_contracts):
            try:
                with torch.no_grad():
                    print(f"Generating contract {contract_idx + 1}/{num_contracts}...")
                    
                    if use_beam_search:
                        print("✓ Using beam search for better generation quality")
                        # Use beam search method
                        outputs = self.model.generate_with_beam_search(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            path_input_ids=ast_input_ids,
                            path_attention_mask=ast_attention_mask,
                            beam_size=beam_size,
                            max_length=max_length,
                            temperature=temperature
                        )
                        generated_sequence = outputs['generated_sequence']
                    else:
                        print("✓ Using syntax-aware generation with built-in constraints")
                        # Use model's forward method with syntax constraints
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            ast_input_ids=ast_input_ids,
                            ast_attention_mask=ast_attention_mask,
                            target_ids=None,  # None triggers generation mode
                            token_to_line=token_to_line,
                            apply_syntax_constraints=True  # Enable syntax constraints
                        )
                        generated_sequence = outputs['generated_sequence']
                    
                    print(f"Generated sequence shape: {generated_sequence.shape}")
                    
                    # Decode the generated sequence
                    generated_code = self.tokenizer.decode(
                        generated_sequence[0].cpu().numpy(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    print(f"Generated code length: {len(generated_code)}")
                    print(f"Generated code preview: {generated_code[:200]}...")
                    
                    # Clean up the generated code
                    generated_code = generated_code.strip()
                    
                    # Ensure we have some meaningful content
                    if len(generated_code) > 10 and not generated_code.isspace():
                        generated_contracts.append(generated_code)
                        print(f"Successfully generated contract {contract_idx + 1}")
                    else:
                        print(f"Generated contract {contract_idx} is too short or empty")
                        # Try template-based generation as fallback
                        try:
                            template_generated = self.generate_template_based_contract(contract_template)
                            generated_contracts.append(template_generated)
                            print(f"Successfully generated contract {contract_idx + 1} with template method")
                        except Exception as e:
                            print(f"Template generation also failed: {str(e)}")
                            # Final fallback: return a modified version of the template
                            generated_contracts.append(f"// Generated contract based on template\n{contract_template}")
                        
            except Exception as e:
                print(f"Error generating contract {contract_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Try template-based generation as fallback
                try:
                    template_generated = self.generate_template_based_contract(contract_template)
                    generated_contracts.append(template_generated)
                    print(f"Successfully generated contract {contract_idx + 1} with template method after error")
                except Exception as e2:
                    print(f"Template generation also failed: {str(e2)}")
                    # Final fallback: return a modified version of the template
                    generated_contracts.append(f"// Generated contract based on template\n{contract_template}")
        
        # Ensure we return at least one contract
        if not generated_contracts:
            try:
                template_generated = self.generate_template_based_contract(contract_template)
                generated_contracts.append(template_generated)
            except:
                generated_contracts.append(f"// Generated contract based on template\n{contract_template}")
        
        return generated_contracts

    def analyze_multiple_contracts(self, contract_codes: List[str], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Analyze multiple smart contracts for vulnerabilities.
        
        Args:
            contract_codes: List of Solidity contract codes to analyze
            threshold: Probability threshold for vulnerability detection
            
        Returns:
            List of dictionaries containing vulnerability analysis results for each contract
        """
        results = []
        
        for i, contract_code in enumerate(contract_codes):
            print(f"Analyzing contract {i+1}/{len(contract_codes)}...")
            result = self.detect_vulnerabilities(contract_code, threshold)
            result['contract_index'] = i
            result['contract_code'] = contract_code
            results.append(result)
        
        return results
    
    def get_vulnerability_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of vulnerability analysis across multiple contracts.
        
        Args:
            results: List of vulnerability analysis results from analyze_multiple_contracts
            
        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'total_contracts': len(results),
            'vulnerable_contracts': 0,
            'vulnerability_counts': {vuln_type: 0 for vuln_type in self.vulnerability_types},
            'most_common_vulnerabilities': [],
            'contracts_by_vulnerability': {vuln_type: [] for vuln_type in self.vulnerability_types}
        }
        
        for result in results:
            contract_vulns = result['contract_vulnerabilities']
            has_vulnerability = any(contract_vulns.values())
            
            if has_vulnerability:
                summary['vulnerable_contracts'] += 1
            
            for vuln_type, is_vulnerable in contract_vulns.items():
                if is_vulnerable:
                    summary['vulnerability_counts'][vuln_type] += 1
                    summary['contracts_by_vulnerability'][vuln_type].append(result['contract_index'])
        
        # Find most common vulnerabilities
        sorted_vulns = sorted(
            summary['vulnerability_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        summary['most_common_vulnerabilities'] = sorted_vulns
        
        return summary

    def test_generation_simple(self, contract_template: str) -> str:
        """
        Simple test function to debug generation issues.
        
        Args:
            contract_template: Template contract code to base generation on
            
        Returns:
            Generated contract code or error message
        """
        try:
            print("=== Testing Simple Generation ===")
            print(f"Template length: {len(contract_template)}")
            
            # Parse template AST
            ast = self.parse_solidity_to_ast(contract_template)
            ast_paths = self.prepare_code2vec_input(ast) if ast else []
            ast_path_text = ' '.join(ast_paths)
            print(f"AST paths: {len(ast_paths)}")
            
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
            
            print(f"Template tokens: {template_encoding['input_ids'].shape}")
            print(f"AST tokens: {ast_encoding['input_ids'].shape}")
            
            # Move tensors to device
            input_ids = template_encoding['input_ids'].to(self.device)
            attention_mask = template_encoding['attention_mask'].to(self.device)
            ast_input_ids = ast_encoding['input_ids'].to(self.device)
            ast_attention_mask = ast_encoding['attention_mask'].to(self.device)
            
            # Create simple token-to-line mapping
            token_to_line = torch.zeros((1024,), dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                print("Calling model...")
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ast_input_ids=ast_input_ids,
                    ast_attention_mask=ast_attention_mask,
                    target_ids=None,
                    token_to_line=token_to_line
                )
                
                print(f"Model returned keys: {list(outputs.keys())}")
                
                if 'generated_sequence' in outputs:
                    generated_seq = outputs['generated_sequence']
                    print(f"Generated sequence type: {type(generated_seq)}")
                    print(f"Generated sequence shape: {generated_seq.shape if hasattr(generated_seq, 'shape') else 'No shape'}")
                    
                    # Convert to numpy
                    if isinstance(generated_seq, torch.Tensor):
                        generated_seq = generated_seq.cpu().numpy()
                    
                    # Decode
                    generated_code = self.tokenizer.decode(
                        generated_seq,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    print(f"Generated code length: {len(generated_code)}")
                    print(f"Generated code: {generated_code}")
                    
                    return generated_code
                else:
                    return f"Error: No 'generated_sequence' in outputs. Available keys: {list(outputs.keys())}"
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

    def generate_simple_contract(self, input_ids, memory, src_mask, max_length=512):
        """
        Simple generation method as fallback.
        
        Args:
            input_ids: Input token IDs
            memory: Encoder memory
            src_mask: Source attention mask
            max_length: Maximum generation length
            
        Returns:
            Generated contract code
        """
        print("Using simple generation method...")
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Start with proper BOS token
        bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 0
        tgt = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        
        # Track repetition
        consecutive_same_tokens = 0
        last_token = None
        
        for i in range(max_length - 1):
            # Simple causal mask
            tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1)), diagonal=1).bool().to(device)
            tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))
            
            # Target embeddings
            tgt_emb = self.model.embedding(tgt) * math.sqrt(self.model.d_model)
            tgt_emb = self.model.embedding_dropout(tgt_emb)
            tgt_emb = self.model.embedding_norm(tgt_emb)
            tgt_emb = self.model.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)
            
            # Decode
            out = self.model.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=~src_mask
            )
            
            # Apply normalization and get logits
            out = self.model.output_norm(out)
            out = self.model.output_dropout(out)
            logits = self.model.output_layer(out[:, -1, :])
            
            # Apply temperature for diversity
            temperature = 0.8
            logits = logits / temperature
            
            # Apply top-k filtering
            top_k = 50
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits_mask = torch.full_like(logits, float('-inf'))
                logits_mask.scatter_(-1, top_k_indices, top_k_logits)
                logits = logits_mask
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Add noise if we're getting stuck
            if consecutive_same_tokens > 3:
                noise = torch.rand_like(probs) * 0.2
                probs = probs + noise
                probs = probs / probs.sum()
            
            # Sample from distribution instead of greedy decoding
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for repetition
            if last_token is not None and next_token.item() == last_token:
                consecutive_same_tokens += 1
            else:
                consecutive_same_tokens = 0
            last_token = next_token.item()
            
            # Append to sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if we hit EOS or PAD, or if too much repetition
            if (next_token == 2).any() or (next_token == 0).any():
                if i > 20:
                    break
            
            if consecutive_same_tokens > 8:
                break
        
        # Decode
        generated_code = self.tokenizer.decode(
            tgt[0].cpu().numpy(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return generated_code.strip()

    def diagnose_generation_model(self):
        """
        Diagnose the model's generation capabilities by examining the output layer.
        """
        print("=== Model Generation Diagnosis ===")
        
        # Check output layer weights
        output_layer = self.model.output_layer
        weights = output_layer.weight.data
        bias = output_layer.bias.data
        
        print(f"Output layer shape: {weights.shape}")
        print(f"Output layer weight stats:")
        print(f"  Mean: {weights.mean().item():.6f}")
        print(f"  Std: {weights.std().item():.6f}")
        print(f"  Min: {weights.min().item():.6f}")
        print(f"  Max: {weights.max().item():.6f}")
        
        print(f"Output layer bias stats:")
        print(f"  Mean: {bias.mean().item():.6f}")
        print(f"  Std: {bias.std().item():.6f}")
        print(f"  Min: {bias.min().item():.6f}")
        print(f"  Max: {bias.max().item():.6f}")
        
        # Check if weights are all the same (indicating poor training)
        weight_variance = weights.var().item()
        bias_variance = bias.var().item()
        
        print(f"Weight variance: {weight_variance:.6f}")
        print(f"Bias variance: {bias_variance:.6f}")
        
        if weight_variance < 1e-6:
            print("⚠️  WARNING: Output layer weights have very low variance - model may not be properly trained for generation")
        
        if bias_variance < 1e-6:
            print("⚠️  WARNING: Output layer bias has very low variance - model may not be properly trained for generation")
        
        # Test a simple forward pass
        print("\n=== Testing Simple Forward Pass ===")
        test_input = torch.randint(0, 1000, (1, 10)).to(self.device)
        test_attention = torch.ones(1, 10).to(self.device)
        
        try:
            with torch.no_grad():
                # Test encoder
                test_emb = self.model.embedding(test_input) * math.sqrt(self.model.d_model)
                test_emb = self.model.embedding_dropout(test_emb)
                test_emb = self.model.embedding_norm(test_emb)
                test_emb = self.model.pos_encoder(test_emb.transpose(0, 1)).transpose(0, 1)
                
                memory = self.model.encoder(test_emb, src_key_padding_mask=~test_attention.bool())
                
                # Test decoder with single token
                tgt = torch.full((1, 1), 0, dtype=torch.long, device=self.device)
                tgt_emb = self.model.embedding(tgt) * math.sqrt(self.model.d_model)
                tgt_emb = self.model.embedding_dropout(tgt_emb)
                tgt_emb = self.model.embedding_norm(tgt_emb)
                tgt_emb = self.model.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)
                
                out = self.model.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=None,
                    memory_key_padding_mask=~test_attention.bool()
                )
                
                out = self.model.output_norm(out)
                out = self.model.output_dropout(out)
                logits = self.model.output_layer(out[:, -1, :])
                
                print(f"Test logits shape: {logits.shape}")
                print(f"Test logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
                
                # Check if logits are reasonable
                probs = torch.softmax(logits, dim=-1)
                print(f"Test probs sum: {probs.sum().item():.4f}")
                print(f"Test probs max: {probs.max().item():.4f}")
                
                # Check top predictions
                top_probs, top_indices = torch.topk(probs, 5, dim=-1)
                print(f"Top 5 token probabilities: {top_probs[0].tolist()}")
                print(f"Top 5 token indices: {top_indices[0].tolist()}")
                
                # Check if token 1 is always the top prediction
                if top_indices[0, 0] == 1:
                    print("⚠️  WARNING: Token 1 is the top prediction - this explains the all-1 generation issue")
                else:
                    print("✅ Token 1 is not the top prediction - model should generate diverse tokens")
                    
        except Exception as e:
            print(f"Error during test forward pass: {str(e)}")
        
        print("=== Diagnosis Complete ===")

    def generate_template_based_contract(self, contract_template: str) -> str:
        """
        Generate a contract by modifying the template with simple transformations.
        This is a reliable fallback when model generation fails.
        
        Args:
            contract_template: Original contract template
            
        Returns:
            Modified contract code
        """
        print("Using template-based generation...")
        
        # Simple transformations to create variations
        import random
        import re
        
        # Make a copy of the template
        modified_contract = contract_template
        
        # Random transformations
        transformations = [
            # Change variable names
            lambda code: re.sub(r'\bvalue\b', random.choice(['data', 'state', 'storage', 'value']), code),
            lambda code: re.sub(r'\b_value\b', random.choice(['_data', '_state', '_storage', '_value']), code),
            # Change function names
            lambda code: re.sub(r'\bsetValue\b', random.choice(['setData', 'setState', 'setStorage', 'setValue']), code),
            lambda code: re.sub(r'\bgetValue\b', random.choice(['getData', 'getState', 'getStorage', 'getValue']), code),
            # Change visibility modifiers
            lambda code: re.sub(r'\bpublic\b', random.choice(['public', 'external']), code),
            lambda code: re.sub(r'\bprivate\b', random.choice(['private', 'internal']), code),
            # Add comments
            lambda code: code.replace('function', '// Modified function'),
            # Change pragma version
            lambda code: re.sub(r'pragma solidity \^?0\.\d+\.\d+', 
                              f'pragma solidity ^{random.randint(4, 8)}.{random.randint(0, 20)}.{random.randint(0, 20)}', code),
        ]
        
        # Apply 2-3 random transformations
        num_transformations = random.randint(2, 3)
        applied_transformations = random.sample(transformations, num_transformations)
        
        for transform in applied_transformations:
            try:
                modified_contract = transform(modified_contract)
            except:
                continue
        
        # Add a header comment
        header = f"// Generated contract based on template\n// Applied {num_transformations} transformations\n"
        modified_contract = header + modified_contract
        
        return modified_contract

    def generate_contract_with_beam_search(
        self,
        contract_template: str,
        num_contracts: int = 1,
        beam_size: int = 5,
        max_length: int = 1024,
        temperature: float = 0.8
    ) -> List[str]:
        """
        Generate synthetic smart contracts using beam search for better quality.
        
        Args:
            contract_template: Template contract code to base generation on
            num_contracts: Number of contracts to generate
            beam_size: Beam size for beam search (higher = better quality but slower)
            max_length: Maximum length of generated contracts
            temperature: Sampling temperature for beam search
            
        Returns:
            List of generated contract codes
        """
        print(f"🎯 Generating {num_contracts} contracts using beam search (beam_size={beam_size})")
        return self.generate_synthetic_contract(
            contract_template=contract_template,
            num_contracts=num_contracts,
            temperature=temperature,
            max_length=max_length,
            use_beam_search=True,
            beam_size=beam_size
        )

    def test_model_functionality(self) -> Dict[str, Any]:
        """
        Test the model's basic functionality to diagnose issues.
        
        Returns:
            Dictionary with test results
        """
        print("🧪 Testing model functionality...")
        
        test_results = {
            'model_loaded': True,
            'device': str(self.device),
            'vocab_size': self.tokenizer.vocab_size,
            'vulnerability_types': self.vulnerability_types,
            'basic_forward_pass': False,
            'error_message': None
        }
        
        try:
            # Create simple test inputs
            test_input_ids = torch.randint(0, 1000, (1, 10)).to(self.device)
            test_attention_mask = torch.ones(1, 10).to(self.device)
            test_ast_input_ids = torch.randint(0, 1000, (1, 10)).to(self.device)
            test_ast_attention_mask = torch.ones(1, 10).to(self.device)
            test_token_to_line = torch.zeros(1, 10, dtype=torch.long).to(self.device)
            
            print(f"  Test input shapes:")
            print(f"    input_ids: {test_input_ids.shape}")
            print(f"    attention_mask: {test_attention_mask.shape}")
            print(f"    ast_input_ids: {test_ast_input_ids.shape}")
            print(f"    ast_attention_mask: {test_ast_attention_mask.shape}")
            print(f"    token_to_line: {test_token_to_line.shape}")
            
            # Test basic forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=test_input_ids,
                    attention_mask=test_attention_mask,
                    ast_input_ids=test_ast_input_ids,
                    ast_attention_mask=test_ast_attention_mask,
                    token_to_line=test_token_to_line
                )
            
            print(f"  Model outputs keys: {list(outputs.keys())}")
            
            # Check if vulnerability outputs are present
            if 'contract_vulnerability_logits' in outputs:
                contract_shape = outputs['contract_vulnerability_logits'].shape
                print(f"  Contract vulnerability logits shape: {contract_shape}")
                test_results['contract_vuln_shape'] = list(contract_shape)
            
            if 'line_vulnerability_logits' in outputs:
                line_shape = outputs['line_vulnerability_logits'].shape
                print(f"  Line vulnerability logits shape: {line_shape}")
                test_results['line_vuln_shape'] = list(line_shape)
            
            test_results['basic_forward_pass'] = True
            print("  ✅ Basic forward pass successful")
            
        except Exception as e:
            test_results['basic_forward_pass'] = False
            test_results['error_message'] = str(e)
            print(f"  ❌ Basic forward pass failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return test_results
