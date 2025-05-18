import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import os
import time
from datetime import datetime
import json
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from model_code_loc import SmartContractVulnerabilityGAN, SmartContractTransformer, Generator, Discriminator, CodeDecoder, VulnerabilityClassifier
import re
from pattern_matcher import PatternMatcher
from typing import Dict, List, Tuple, Optional

class SolidityCodeLoss(nn.Module):
    def __init__(self, vocab_size=50000):
        super().__init__()
        self.vocab_size = vocab_size
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)
        
        # Initialize syntax rules with weights
        self.syntax_rules = {
            'pragma': 0.2,  # Increased weight for pragma
            'contract': 0.15,
            'function': 0.15,
            'returns': 0.1,
            'public': 0.1,
            'private': 0.1,
            'view': 0.1,
            'pure': 0.1
        }
        
        # Token mappings for Solidity syntax
        self.token_mappings = {
            'pragma': ['pragma', 'solidity'],
            'contract': ['contract'],
            'function': ['function'],
            'returns': ['returns'],
            'public': ['public'],
            'private': ['private'],
            'view': ['view'],
            'pure': ['pure']
        }
    
    def check_solidity_syntax(self, code):
        try:
            # Check for pragma solidity directive
            if not re.search(r'pragma\s+solidity\s+[\^]?[0-9]+\.[0-9]+(\.[0-9]+)?', code):
                print("Missing pragma solidity directive")
                return False
            
            # Check for contract declaration
            if not re.search(r'contract\s+\w+', code):
                print("Missing contract declaration")
                return False
            
            # Check for balanced braces
            brace_count = 0
            for char in code:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                if brace_count < 0:
                    print("Unbalanced braces")
                    return False
            if brace_count != 0:
                print("Unbalanced braces")
                return False
            
            # Check for balanced parentheses
            paren_count = 0
            for char in code:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                if paren_count < 0:
                    print("Unbalanced parentheses")
                    return False
            if paren_count != 0:
                print("Unbalanced parentheses")
                return False
            
            return True
        except Exception as e:
            print(f"Error checking syntax: {str(e)}")
            return False
    
    def forward(self, decoded_code, target_ids):
        # Ensure inputs have gradients
        if not decoded_code.requires_grad:
            decoded_code.requires_grad_(True)
        
        # Make tensors contiguous before reshaping
        decoded_code = decoded_code.contiguous()
        target_ids = target_ids.contiguous()
        
        # Get shapes
        batch_size, seq_length, vocab_size = decoded_code.shape
        
        # Reshape for cross entropy using reshape instead of view
        decoded_code = decoded_code.reshape(-1, vocab_size)
        target_ids = target_ids.reshape(-1)
        
        # Ensure target indices are within vocabulary size
        target_ids = torch.clamp(target_ids, 0, self.vocab_size - 1)
        
        # Compute cross entropy loss
        ce_loss = self.cross_entropy(decoded_code, target_ids)
        
        # Compute syntax-aware loss
        syntax_loss = self._compute_syntax_loss(decoded_code, target_ids)
        
        # Combine losses
        total_loss = ce_loss + syntax_loss
        
        return total_loss
    
    def _compute_syntax_loss(self, decoded_code, target_ids):
        syntax_loss = 0.0
        
        # Convert logits to probabilities
        probs = F.softmax(decoded_code, dim=-1)
        
        # Check for each syntax rule
        for rule, weight in self.syntax_rules.items():
            if rule in self.token_mappings:
                tokens = self.token_mappings[rule]
                for token in tokens:
                    # Get token index from target_ids
                    token_idx = target_ids[0]  # Use first token as reference
                    
                    # Ensure token index is within vocabulary size
                    token_idx = torch.clamp(token_idx, 0, self.vocab_size - 1)
                    
                    # Compute loss for this token
                    token_prob = probs[:, token_idx]
                    syntax_loss += weight * (1 - token_prob.mean())
        
        return syntax_loss

    def compute_semantic_similarity(self, generated_code, original_code):
        # Simple token-based similarity
        gen_tokens = set(re.findall(r'\w+', generated_code))
        orig_tokens = set(re.findall(r'\w+', original_code))
        
        if not gen_tokens or not orig_tokens:
            return 0.0
            
        intersection = len(gen_tokens.intersection(orig_tokens))
        union = len(gen_tokens.union(orig_tokens))
        
        return intersection / union if union > 0 else 0.0


class VulnerabilityDetectionTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, 
                 learning_rate=0.0002, beta1=0.5):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.pattern_matcher = PatternMatcher()
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.vulnerability_loss = nn.BCELoss()
        self.decoder_loss = SolidityCodeLoss(vocab_size=50000)
        self.classification_loss = nn.BCEWithLogitsLoss()
        
        # Pattern-specific loss functions
        self.pattern_losses = {
            'timestamp_dependence': nn.BCEWithLogitsLoss(),
            'reentrancy': nn.BCEWithLogitsLoss(),
            'integer_overflow': nn.BCEWithLogitsLoss(),
            'dangerous_delegatecall': nn.BCEWithLogitsLoss()
        }
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            self.model.generator.parameters(), 
            lr=learning_rate, 
            betas=(beta1, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.model.discriminator.parameters(), 
            lr=learning_rate, 
            betas=(beta1, 0.999)
        )
        self.optimizer_decoder = optim.Adam(
            self.model.decoder.parameters(),
            lr=learning_rate,
            betas=(beta1, 0.999)
        )
        self.optimizer_classifier = optim.Adam(
            self.model.vulnerability_classifier.parameters(),
            lr=learning_rate,
            betas=(beta1, 0.999)
        )
        
        # Initialize training metrics
        self.best_val_loss = float('inf')
        self.training_history = {
            'g_loss': [],
            'd_loss': [],
            'decoder_loss': [],
            'classifier_loss': [],
            'pattern_losses': {
                'timestamp_dependence': [],
                'reentrancy': [],
                'integer_overflow': [],
                'dangerous_delegatecall': []
            },
            'val_loss': [],
            'vulnerability_accuracy': [],
            'pattern_accuracy': {
                'timestamp_dependence': [],
                'reentrancy': [],
                'integer_overflow': [],
                'dangerous_delegatecall': []
            },
            'epoch_times': []
        }
        
    def _get_pattern_labels(self, code: str) -> Dict[str, torch.Tensor]:
        """
        Get pattern labels for a given code using the PatternMatcher.
        """
        pattern_tensors = self.pattern_matcher.get_pattern_tensor(code)
        return pattern_tensors
    
    def train_epoch(self):
        self.model.train()  # Set model to training mode
        self.model.codebert.eval()  # Keep CodeBERT in eval mode
        
        total_g_loss = 0
        total_d_loss = 0
        total_decoder_loss = 0
        total_classifier_loss = 0
        total_pattern_losses = {
            'timestamp_dependence': 0,
            'reentrancy': 0,
            'integer_overflow': 0,
            'dangerous_delegatecall': 0
        }
        batch_count = 0
        
        print("\nStarting training epoch...")
        for batch_idx, batch in enumerate(self.train_dataloader):
            try:
                # Get batch data and move to CUDA
                contract_ids = batch['input_ids'].cuda()
                contract_attention_mask = batch['attention_mask'].cuda()
                path_ids = batch['path_input_ids'].cuda()
                path_attention_mask = batch['path_attention_mask'].cuda()
                labels = batch['label'].cuda()
                batch_size = contract_ids.size(0)
                
                # Create vulnerability labels if not present
                if 'vulnerability_labels' not in batch:
                    vulnerability_labels = torch.zeros(batch_size, self.model.num_vulnerability_types).cuda()
                else:
                    vulnerability_labels = batch['vulnerability_labels'].cuda()
                
                
                # Get pattern labels using PatternMatcher
                pattern_labels = {}
                for i in range(batch_size):
                    code = self.model.tokenizer.decode(contract_ids[i])
                    batch_pattern_labels = self._get_pattern_labels(code)
                    for vuln_type, tensor in batch_pattern_labels.items():
                        if vuln_type not in pattern_labels:
                            pattern_labels[vuln_type] = []
                        pattern_labels[vuln_type].append(tensor)
                
                # Convert pattern labels to tensors
                pattern_labels = {
                    vuln_type: torch.stack(tensors).cuda()
                    for vuln_type, tensors in pattern_labels.items()
                }
                
                # Ensure all tensors have the same dtype
                contract_ids = contract_ids.long()
                path_ids = path_ids.long()
                labels = labels.float()
                vulnerability_labels = vulnerability_labels.float()
                
                # Reshape labels to match model output shape [batch_size, 1]
                labels = labels.view(-1, 1)
                
                # Real and fake labels
                real_label = torch.ones(batch_size, 1, dtype=torch.float32).cuda()
                fake_label = torch.zeros(batch_size, 1, dtype=torch.float32).cuda()
                
                # Get CodeBERT embeddings with no_grad
                with torch.no_grad():
                    contract_outputs = self.model.codebert(
                        input_ids=contract_ids,
                        attention_mask=contract_attention_mask
                    )
                    contract_embeddings = contract_outputs.last_hidden_state
                    
                    path_outputs = self.model.codebert(
                        input_ids=path_ids,
                        attention_mask=path_attention_mask
                    )
                    path_embeddings = path_outputs.last_hidden_state
                
                # Combine embeddings
                combined_embeddings = contract_embeddings + path_embeddings
                
                # Process through transformer
                transformed = self.model.transformer(combined_embeddings)
                mean_embeddings = transformed.mean(dim=1)
                
                # Train Generator
                self.optimizer_G.zero_grad()
                
                # Generate synthetic vulnerabilities
                synthetic = self.model.generator(mean_embeddings)
                
                # Get discriminator scores
                fake_scores = self.model.discriminator(synthetic)
                
                # Compute generator loss
                g_loss = self.adversarial_loss(fake_scores, real_label)
                g_loss.backward(retain_graph=True)
                self.optimizer_G.step()
                
                # Train Discriminator
                self.optimizer_D.zero_grad()
                
                # Get real scores (detach to prevent generator update)
                real_scores = self.model.discriminator(mean_embeddings.detach())
                fake_scores = self.model.discriminator(synthetic.detach())
                
                # Compute discriminator loss
                real_loss = self.adversarial_loss(real_scores, labels)
                fake_loss = self.adversarial_loss(fake_scores, fake_label)
                d_loss = (real_loss + fake_loss) / 2
                
                d_loss.backward(retain_graph=True)
                self.optimizer_D.step()
                
                # Train Vulnerability Classifier
                self.optimizer_classifier.zero_grad()
                
                # Get vulnerability predictions
                vulnerability_outputs = self.model.vulnerability_classifier(
                    mean_embeddings,
                    attention_mask=contract_attention_mask.bool()  # Convert to boolean
                )
                
                # Compute classification loss
                classifier_loss = self.classification_loss(
                    vulnerability_outputs['vulnerability_scores'],
                    vulnerability_labels
                )
                
                # Compute pattern-specific losses
                pattern_losses = {}
                for vuln_type, pattern_scores in vulnerability_outputs['pattern_scores'].items():
                    pattern_losses[vuln_type] = self.pattern_losses[vuln_type](
                        pattern_scores,
                        pattern_labels[vuln_type]
                    )
                    total_pattern_losses[vuln_type] += pattern_losses[vuln_type].item()
                
                # Combine losses
                total_loss = classifier_loss + sum(pattern_losses.values())
                total_loss.backward(retain_graph=True)
                self.optimizer_classifier.step()
                
                # Train Decoder
                self.optimizer_decoder.zero_grad()
                
                # Generate new synthetic embeddings for decoder
                synthetic_decoder = self.model.generator(mean_embeddings)
                
                # Prepare target sequence with pragma directive
                target_sequence = torch.full((batch_size, 1), self.model.decoder.special_tokens['pragma'], device='cuda')
                target_sequence = torch.cat([
                    target_sequence,
                    torch.full((batch_size, 1), self.model.decoder.special_tokens['solidity'], device='cuda')
                ], dim=1)
                
                # Decode synthetic embeddings
                decoded_code = self.model.decoder(synthetic_decoder, transformed, target_sequence)
                
                # Handle sequence length mismatches
                if decoded_code.dim() == 3:
                    current_batch_size, current_seq_length, current_vocab_size = decoded_code.shape
                    
                    if contract_ids.shape[1] != current_seq_length:
                        if contract_ids.shape[1] > current_seq_length:
                            contract_ids = contract_ids[:, :current_seq_length]
                        else:
                            padding = torch.zeros(batch_size, current_seq_length - contract_ids.shape[1], 
                                                dtype=contract_ids.dtype).cuda()
                            contract_ids = torch.cat([contract_ids, padding], dim=1)
                    
                    if decoded_code.shape[0] != contract_ids.shape[0]:
                        if decoded_code.shape[0] > contract_ids.shape[0]:
                            decoded_code = decoded_code[:contract_ids.shape[0]]
                        else:
                            padding = torch.zeros(contract_ids.shape[0] - decoded_code.shape[0], 
                                                current_seq_length, 
                                                current_vocab_size).cuda()
                            decoded_code = torch.cat([decoded_code, padding], dim=0)
                
                # Ensure decoded_code has gradients
                if not decoded_code.requires_grad:
                    decoded_code.requires_grad_(True)
                
                # Compute decoder loss
                decoder_loss = self.decoder_loss(decoded_code, contract_ids)
                
                # Check if loss is valid
                if torch.isnan(decoder_loss) or torch.isinf(decoder_loss):
                    print(f"Warning: Invalid decoder loss value: {decoder_loss.item()}")
                    continue
                
                # Backward pass for decoder
                decoder_loss.backward()
                self.optimizer_decoder.step()
                
                # Update metrics
                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
                total_decoder_loss += decoder_loss.item()
                total_classifier_loss += classifier_loss.item()
                batch_count += 1
                
                # Print batch progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"Batch {batch_idx + 1}/{len(self.train_dataloader)} - "
                          f"G Loss: {g_loss.item():.4f}, "
                          f"D Loss: {d_loss.item():.4f}, "
                          f"Decoder Loss: {decoder_loss.item():.4f}, "
                          f"Classifier Loss: {classifier_loss.item():.4f}")
                    for vuln_type, loss in pattern_losses.items():
                        print(f"{vuln_type} Loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                print(f"Error details: {type(e).__name__}")
                import traceback
                print(traceback.format_exc())
                continue
        
        # Check if any batches were processed
        if batch_count == 0:
            print("Warning: No batches were successfully processed in this epoch")
            return 0.0, 0.0, 0.0, 0.0
        
        # Calculate average losses
        avg_g_loss = total_g_loss / batch_count
        avg_d_loss = total_d_loss / batch_count
        avg_decoder_loss = total_decoder_loss / batch_count
        avg_classifier_loss = total_classifier_loss / batch_count
        avg_pattern_losses = {
            vuln_type: loss / batch_count
            for vuln_type, loss in total_pattern_losses.items()
        }
        
        return avg_g_loss, avg_d_loss, avg_decoder_loss, avg_classifier_loss, avg_pattern_losses


    def validate(self):
        self.model.eval()
        total_loss = 0
        total_vulnerability_correct = 0
        total_pattern_correct = {
            'timestamp_dependence': 0,
            'reentrancy': 0,
            'integer_overflow': 0,
            'dangerous_delegatecall': 0
        }
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                try:
                    # Get batch data
                    contract_ids = batch['input_ids'].cuda()
                    contract_attention_mask = batch['attention_mask'].cuda()
                    path_ids = batch['path_input_ids'].cuda()
                    path_attention_mask = batch['path_attention_mask'].cuda()
                    labels = batch['label'].cuda()
                    batch_size = contract_ids.size(0)
                    
                    # Create vulnerability labels if not present
                    if 'vulnerability_labels' not in batch:
                        vulnerability_labels = torch.zeros(batch_size, self.model.num_vulnerability_types).cuda()
                    else:
                        vulnerability_labels = batch['vulnerability_labels'].cuda()
                    
                    # Get pattern labels using PatternMatcher
                    pattern_labels = {}
                    for i in range(batch_size):
                        code = self.model.tokenizer.decode(contract_ids[i])
                        batch_pattern_labels = self._get_pattern_labels(code)
                        for vuln_type, tensor in batch_pattern_labels.items():
                            if vuln_type not in pattern_labels:
                                pattern_labels[vuln_type] = []
                            pattern_labels[vuln_type].append(tensor)
                    
                    # Convert pattern labels to tensors
                    pattern_labels = {
                        vuln_type: torch.stack(tensors).cuda()
                        for vuln_type, tensors in pattern_labels.items()
                    }
                    
                    # Get CodeBERT embeddings
                    contract_outputs = self.model.codebert(
                        input_ids=contract_ids,
                        attention_mask=contract_attention_mask
                    )
                    contract_embeddings = contract_outputs.last_hidden_state
                    
                    path_outputs = self.model.codebert(
                        input_ids=path_ids,
                        attention_mask=path_attention_mask
                    )
                    path_embeddings = path_outputs.last_hidden_state
                    
                    # Combine embeddings
                    combined_embeddings = contract_embeddings + path_embeddings
                    
                    # Process through transformer
                    transformed = self.model.transformer(combined_embeddings)
                    mean_embeddings = transformed.mean(dim=1)
                    
                    # Get vulnerability predictions
                    vulnerability_outputs = self.model.vulnerability_classifier(
                        mean_embeddings,
                        attention_mask=contract_attention_mask.bool()  # Convert to boolean
                    )
                    
                    # Compute classification loss
                    classifier_loss = self.classification_loss(
                        vulnerability_outputs['vulnerability_scores'],
                        vulnerability_labels
                    )
                    
                    # Compute pattern-specific losses
                    pattern_losses = {}
                    for vuln_type, pattern_scores in vulnerability_outputs['pattern_scores'].items():
                        pattern_losses[vuln_type] = self.pattern_losses[vuln_type](
                            pattern_scores,
                            pattern_labels[vuln_type]
                        )
                    
                    # Combine losses
                    total_loss += classifier_loss.item() + sum(loss.item() for loss in pattern_losses.values())
                    
                    # Compute vulnerability accuracy
                    vulnerability_preds = (torch.sigmoid(vulnerability_outputs['vulnerability_scores']) > 0.5).float()
                    total_vulnerability_correct += (vulnerability_preds == vulnerability_labels).all(dim=1).sum().item()
                    
                    # Compute pattern-specific accuracies
                    for vuln_type, pattern_scores in vulnerability_outputs['pattern_scores'].items():
                        pattern_preds = (torch.sigmoid(pattern_scores) > 0.5).float()
                        total_pattern_correct[vuln_type] += (pattern_preds == pattern_labels[vuln_type]).all(dim=1).sum().item()
                    
                    total_samples += batch_size
                    
                except Exception as e:
                    print(f"Error in validation batch: {str(e)}")
                    continue
        
        # Calculate average metrics
        avg_loss = total_loss / len(self.val_dataloader)
        vulnerability_accuracy = total_vulnerability_correct / total_samples if total_samples > 0 else 0
        pattern_accuracy = {
            vuln_type: correct / total_samples if total_samples > 0 else 0
            for vuln_type, correct in total_pattern_correct.items()
        }
        
        # Print validation results
        print("\nValidation Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Vulnerability Accuracy: {vulnerability_accuracy:.4f}")
        print("\nPattern-specific Accuracies:")
        for vuln_type, accuracy in pattern_accuracy.items():
            print(f"{vuln_type}: {accuracy:.4f}")
        
        return avg_loss, vulnerability_accuracy, pattern_accuracy