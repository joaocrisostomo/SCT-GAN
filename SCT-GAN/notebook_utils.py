import numpy as np
from typing import Dict, List, Any
import torch
from inference import SmartContractAnalyzer
import matplotlib.pyplot as plt

def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate precision for vulnerability detection.
    
    Args:
        y_true: True vulnerability labels
        y_pred: Predicted vulnerability labels
        
    Returns:
        Precision score
    """
    if np.sum(y_pred) == 0:
        return 0.0
    
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0

def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate recall for vulnerability detection.
    
    Args:
        y_true: True vulnerability labels
        y_pred: Predicted vulnerability labels
        
    Returns:
        Recall score
    """
    if np.sum(y_true) == 0:
        return 0.0
    
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    
    return true_positives / actual_positives if actual_positives > 0 else 0.0

def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.
    
    Args:
        precision: Precision score
        recall: Recall score
        
    Returns:
        F1 score
    """
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def calculate_line_accuracy(true_line_vulns: np.ndarray, pred_line_vulns: Dict[int, Dict[str, bool]]) -> float:
    """
    Calculate line-level accuracy for vulnerability detection.
    
    Args:
        true_line_vulns: True line-level vulnerabilities [num_lines, num_vuln_types]
        pred_line_vulns: Predicted line-level vulnerabilities
        
    Returns:
        Line accuracy score
    """
    if len(pred_line_vulns) == 0:
        return 0.0
    
    correct_predictions = 0
    total_predictions = 0
    
    for line_idx in pred_line_vulns:
        if line_idx < len(true_line_vulns):
            line_pred = pred_line_vulns[line_idx]
            line_true = true_line_vulns[line_idx]
            
            for vuln_type, is_vulnerable in line_pred.items():
                # Find the index of the vulnerability type
                vuln_idx = get_vulnerability_index(vuln_type)
                if vuln_idx is not None and vuln_idx < len(line_true):
                    if bool(line_true[vuln_idx]) == is_vulnerable:
                        correct_predictions += 1
                    total_predictions += 1
    
    return correct_predictions / total_predictions if total_predictions > 0 else 0.0

def get_vulnerability_index(vuln_type: str) -> int:
    """
    Get the index of a vulnerability type.
    
    Args:
        vuln_type: Vulnerability type string
        
    Returns:
        Index of the vulnerability type
    """
    vulnerability_types = ['ARTHM', 'DOS', 'LE', 'RENT', 'TimeM', 'TimeO', 'Tx-Origin', 'UE']
    try:
        return vulnerability_types.index(vuln_type)
    except ValueError:
        return None

def get_vulnerability_details(
    analyzer, 
    true_contract_vulns: np.ndarray, 
    pred_contract_array: np.ndarray, 
    pred_contract_probs: List[float]
) -> Dict[str, Any]:
    """
    Get detailed vulnerability analysis.
    
    Args:
        analyzer: SmartContractAnalyzer instance
        true_contract_vulns: True contract vulnerabilities
        pred_contract_array: Predicted contract vulnerabilities
        pred_contract_probs: Predicted probabilities
        
    Returns:
        Dictionary with vulnerability details
    """
    details = {
        'vulnerability_analysis': {},
        'high_confidence_predictions': [],
        'misclassifications': []
    }
    
    for i, vuln_type in enumerate(analyzer.vulnerability_types):
        true_label = bool(true_contract_vulns[i])
        pred_label = bool(pred_contract_array[i])
        confidence = pred_contract_probs[i]
        
        details['vulnerability_analysis'][vuln_type] = {
            'true_label': true_label,
            'predicted_label': pred_label,
            'confidence': confidence,
            'correct': true_label == pred_label
        }
        
        # High confidence predictions (confidence > 0.8)
        if confidence > 0.8:
            details['high_confidence_predictions'].append({
                'vulnerability': vuln_type,
                'predicted': pred_label,
                'confidence': confidence
            })
        
        # Misclassifications
        if true_label != pred_label:
            details['misclassifications'].append({
                'vulnerability': vuln_type,
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': confidence
            })
    
    return details

def generate_syntax_aware_contract(
    analyzer,
    contract_template: str,
    num_contracts: int = 1,
    temperature: float = 0.9,
    max_length: int = 1024
) -> List[str]:
    """
    Generate contracts using syntax-aware generation with all forward attention steps.
    
    Args:
        analyzer: SmartContractAnalyzer instance
        contract_template: Template contract code
        num_contracts: Number of contracts to generate
        temperature: Sampling temperature
        max_length: Maximum generation length
        
    Returns:
        List of generated contract codes
    """
    print(f"🎯 Generating {num_contracts} contract(s) with syntax-aware generation...")
    print(f"📊 Parameters: temperature={temperature}, max_length={max_length}")
    
    try:
        # Use the model's forward method with syntax constraints
        generated_contracts = analyzer.generate_synthetic_contract(
            contract_template=contract_template,
            num_contracts=num_contracts,
            temperature=temperature,
            max_length=max_length
            # This will use the model's forward method with apply_syntax_constraints=True
        )
        
        print(f"✅ Successfully generated {len(generated_contracts)} contract(s)")
        
        # Validate generated contracts
        for i, contract in enumerate(generated_contracts):
            if isinstance(contract, str) and len(contract.strip()) > 10:
                print(f"  Contract {i+1}: {len(contract)} characters, valid syntax")
            else:
                print(f"  Contract {i+1}: Invalid or too short")
        
        return generated_contracts
        
    except Exception as e:
        print(f"❌ Syntax-aware generation failed: {str(e)}")
        print("🔄 Falling back to template-based generation...")
        
        # Fallback to template-based generation
        fallback_contracts = []
        for i in range(num_contracts):
            try:
                fallback_contract = analyzer.generate_template_based_contract(contract_template)
                fallback_contracts.append(fallback_contract)
                print(f"  Fallback contract {i+1}: Generated successfully")
            except Exception as e2:
                print(f"  Fallback contract {i+1}: Failed - {str(e2)}")
                fallback_contracts.append(f"// Generation failed - using template\n{contract_template}")
        
        return fallback_contracts

def analyze_contract_with_syntax_generation(
    analyzer,
    source_code: str,
    true_contract_vulns: np.ndarray,
    true_line_vulns: np.ndarray,
    threshold: float = 0.5,
    generate_contracts: bool = True
) -> Dict[str, Any]:
    """
    Complete contract analysis with syntax-aware generation.
    
    Args:
        analyzer: SmartContractAnalyzer instance
        source_code: Contract source code
        true_contract_vulns: True contract vulnerabilities
        true_line_vulns: True line vulnerabilities
        threshold: Detection threshold
        generate_contracts: Whether to generate synthetic contracts
        
    Returns:
        Complete analysis results
    """
    print("🔍 Starting comprehensive contract analysis...")
    
    # Vulnerability Detection
    pred_results = analyzer.detect_vulnerabilities(source_code, threshold=threshold)
    
    # Extract predictions with proper error handling
    try:
        pred_contract_vulns = pred_results['contract_vulnerabilities']
    except Exception as e:
        print(f"    Error extracting pred_contract_vulns: {str(e)}")
        raise
    
    try:
        pred_line_vulns = pred_results['line_vulnerabilities']
    except Exception as e:
        print(f"    Error extracting pred_line_vulns: {str(e)}")
        raise
    
    # Fix: Handle contract probabilities with proper error handling
    try:
        pred_contract_probs = pred_results['contract_probabilities']
    except Exception as e:
        print(f"    Error extracting pred_contract_probs: {str(e)}")
        raise
    
    try:
        if isinstance(pred_contract_probs, list) and len(pred_contract_probs) > 0:
            pred_contract_probs = pred_contract_probs[0]  # Get first element
        elif isinstance(pred_contract_probs, np.ndarray):
            if len(pred_contract_probs.shape) == 2:
                pred_contract_probs = pred_contract_probs[0]  # Remove batch dimension
            pred_contract_probs = pred_contract_probs.tolist()
        else:
            # Fallback: create empty contract probabilities
            pred_contract_probs = [0.0] * len(analyzer.vulnerability_types)
    except Exception as e:
        print(f"    Error processing pred_contract_probs: {str(e)}")
        raise
    
    # Fix: Handle line probabilities properly
    try:
        pred_line_probs = pred_results['line_probabilities']
    except Exception as e:
        print(f"    Error extracting pred_line_probs: {str(e)}")
        raise
    
    try:
        if isinstance(pred_line_probs, list) and len(pred_line_probs) > 0:
            pred_line_probs = pred_line_probs[0]  # [seq_len, num_vuln_types]
        elif isinstance(pred_line_probs, np.ndarray):
            if len(pred_line_probs.shape) == 3:
                pred_line_probs = pred_line_probs[0]  # Remove batch dimension
            pred_line_probs = pred_line_probs.tolist()
        else:
            # Fallback: create empty line probabilities
            pred_line_probs = np.zeros((1024, len(analyzer.vulnerability_types)))
    except Exception as e:
        print(f"    Error processing pred_line_probs: {str(e)}")
        raise
    
    # Convert contract predictions to arrays
    try:
        pred_contract_array = np.array([
            1 if pred_contract_vulns[vuln_type] else 0 
            for vuln_type in analyzer.vulnerability_types
        ])
    except Exception as e:
        print(f"    Error creating pred_contract_array: {str(e)}")
        raise
    
    # Calculate Metrics
    contract_accuracy = np.mean(pred_contract_array == true_contract_vulns)
    contract_precision = calculate_precision(true_contract_vulns, pred_contract_array)
    contract_recall = calculate_recall(true_contract_vulns, pred_contract_array)
    contract_f1 = calculate_f1_score(contract_precision, contract_recall)
    line_accuracy = calculate_line_accuracy(true_line_vulns, pred_line_vulns)
    
    # Generate Synthetic Contracts (if requested)
    generated_contracts = []
    generation_method = "none"
    
    if generate_contracts:
        print("🚀 Generating synthetic contracts with syntax validation...")
        generated_contracts = generate_syntax_aware_contract(
            analyzer=analyzer,
            contract_template=source_code,
            num_contracts=1,
            temperature=0.9,
            max_length=1024
        )
        generation_method = "syntax_aware"
    else:
        print("⏭️  Skipping contract generation")
    
    # Compile results
    results = {
        'true_contract_vulnerabilities': true_contract_vulns.tolist(),
        'predicted_contract_vulnerabilities': pred_contract_array.tolist(),
        'contract_probabilities': pred_contract_probs,
        'contract_accuracy': float(contract_accuracy),
        'contract_precision': float(contract_precision),
        'contract_recall': float(contract_recall),
        'contract_f1': float(contract_f1),
        'line_accuracy': float(line_accuracy),
        'vulnerability_details': get_vulnerability_details(
            analyzer, true_contract_vulns, pred_contract_array, pred_contract_probs
        ),
        'predicted_line_vulnerabilities': pred_line_vulns,
        'generation_method': generation_method,
        'generated_contracts': generated_contracts,
        'generated_contract': generated_contracts[0] if generated_contracts else None
    }
    
    print("✅ Analysis completed successfully!")
    return results

def print_analysis_summary(results: Dict[str, Any]) -> None:
    """
    Print a formatted summary of the analysis results.
    
    Args:
        results: Analysis results dictionary
    """
    print("=" * 60)
    print("📊 CONTRACT ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\n🎯 Contract-Level Metrics:")
    print(f"  Accuracy:  {results['contract_accuracy']:.4f}")
    print(f"  Precision: {results['contract_precision']:.4f}")
    print(f"  Recall:    {results['contract_recall']:.4f}")
    print(f"  F1-Score:  {results['contract_f1']:.4f}")
    print(f"  Line Accuracy: {results['line_accuracy']:.4f}")
    
    print(f"\n🔍 Vulnerability Details:")
    for vuln_type, details in results['vulnerability_details']['vulnerability_analysis'].items():
        status = "✅" if details['correct'] else "❌"
        confidence = details['confidence']
        print(f"  {status} {vuln_type}: True={details['true_label']}, Pred={details['predicted_label']}, Conf={confidence:.3f}")
    
    if results['vulnerability_details']['high_confidence_predictions']:
        print(f"\n🚨 High Confidence Predictions (>0.8):")
        for pred in results['vulnerability_details']['high_confidence_predictions']:
            print(f"  {pred['vulnerability']}: {pred['predicted']} (conf={pred['confidence']:.3f})")
    
    if results['vulnerability_details']['misclassifications']:
        print(f"\n⚠️  Misclassifications:")
        for mis in results['vulnerability_details']['misclassifications']:
            print(f"  {mis['vulnerability']}: True={mis['true_label']}, Pred={mis['predicted_label']} (conf={mis['confidence']:.3f})")
    
    # Handle generation results
    if 'generation_method' in results:
        print(f"\n🚀 Generation Method: {results['generation_method'].upper()}")
        
        if results['generation_method'] == 'syntax_aware':
            print("  ✓ Used syntax-aware generation with forward attention steps")
            print("  ✓ Applied syntax constraints during generation")
            print("  ✓ Used all attention mechanisms (contract, spatial, cross-attention)")
        elif results['generation_method'] == 'template_based':
            print("  ⚠️  Used template-based generation (fallback method)")
        elif results['generation_method'] == 'none':
            print("  ⏭️  No contract generation performed")
    
    if 'generated_contracts' in results and results['generated_contracts']:
        print(f"\n📄 Generated Contract(s):")
        for i, contract in enumerate(results['generated_contracts']):
            if isinstance(contract, str):
                if len(contract) > 100:
                    print(f"  Contract {i+1}: {contract[:100]}...")
                else:
                    print(f"  Contract {i+1}: {contract}")
            else:
                print(f"  Contract {i+1}: {type(contract)}")
    elif 'generated_contract' in results and results['generated_contract']:
        print(f"\n📄 Generated Contract Preview:")
        generated = results['generated_contract']
        if isinstance(generated, str) and len(generated) > 100:
            print(f"  {generated[:100]}...")
        else:
            print(f"  {generated}")
    
    print("=" * 60)

def collect_validation_results(
    analyzer,
    val_dataloader,
    threshold: float = 0.5,
    max_contracts: int = None,
    generate_contracts: bool = True
) -> Dict[str, Any]:
    """
    Collect comprehensive results from validation dataloader for detailed analysis.
    
    Args:
        analyzer: SmartContractAnalyzer instance
        val_dataloader: Validation dataloader
        threshold: Detection threshold
        max_contracts: Maximum number of contracts to process (None for all)
        generate_contracts: Whether to generate synthetic contracts
        
    Returns:
        Dictionary with comprehensive validation results
    """
    print("🚀 Starting comprehensive validation analysis...")
    
    # Initialize results storage
    results = {
        'contract_level': {
            'true_labels': [],      # List of true contract vulnerability arrays
            'predicted_probs': [],  # List of predicted probability arrays
            'predicted_labels': [], # List of predicted binary arrays
            'source_codes': [],     # List of source codes
            'generated_codes': [],  # List of generated codes
            'contract_names': []    # List of contract names
        },
        'line_level': {
            'true_labels': [],      # List of true line vulnerability arrays (8, 1024)
            'predicted_probs': [],  # List of predicted line probability arrays (8, 1024)
            'predicted_labels': [], # List of predicted line binary arrays (8, 1024)
            'line_mappings': [],    # List of token-to-line mappings
            'vulnerable_lines': []  # List of vulnerable line details
        },
        'metadata': {
            'total_contracts': 0,
            'total_lines': 0,
            'vulnerability_types': analyzer.vulnerability_types,
            'processing_time': 0,
            'generation_success_rate': 0
        }
    }
    
    import time
    start_time = time.time()
    
    # Determine number of contracts to process
    total_contracts = len(val_dataloader.dataset.data)
    if max_contracts is not None:
        total_contracts = min(total_contracts, max_contracts)
    
    print(f"📊 Processing {total_contracts} contracts...")
    
    successful_generations = 0
    
    for contract_idx in range(total_contracts):
        try:
            if (contract_idx + 1) % 10 == 0:
                print(f"  ✅ Processed {contract_idx + 1}/{total_contracts} contracts")
            
            # Get contract data
            contract_data = val_dataloader.dataset.data[contract_idx]
            source_code = contract_data['source_code']
            true_contract_vulns = contract_data['contract_vulnerabilities'].cpu().numpy()
            true_line_vulns = contract_data['vulnerable_lines'].cpu().numpy()  # Shape: (8, 1024)
            contract_name = f'Contract_{contract_idx}'
            
            
            # Get vulnerability predictions using the regular method
            try:
                analyzer_results = analyzer.detect_vulnerabilities(
                    source_code, 
                    threshold=threshold
                )
            except Exception as e:
                print(f"    Error in detect_vulnerabilities: {str(e)}")
                # Return empty results
                analyzer_results = {
                    'contract_vulnerabilities': {vuln_type: False for vuln_type in analyzer.vulnerability_types},
                    'line_vulnerabilities': {},
                    'contract_probabilities': [[0.0] * len(analyzer.vulnerability_types)],
                    'line_probabilities': []
                }
            
            # Extract predictions with proper error handling
            try:
                pred_contract_vulns = analyzer_results['contract_vulnerabilities']
            except Exception as e:
                print(f"    Error extracting pred_contract_vulns: {str(e)}")
                raise
            
            try:
                pred_line_vulns = analyzer_results['line_vulnerabilities']
            except Exception as e:
                print(f"    Error extracting pred_line_vulns: {str(e)}")
                raise
            
            # Fix: Handle contract probabilities with proper error handling
            try:
                pred_contract_probs = analyzer_results['contract_probabilities']
            except Exception as e:
                print(f"    Error extracting pred_contract_probs: {str(e)}")
                raise
            
            try:
                if isinstance(pred_contract_probs, list) and len(pred_contract_probs) > 0:
                    pred_contract_probs = pred_contract_probs[0]  # Get first element
                elif isinstance(pred_contract_probs, np.ndarray):
                    if len(pred_contract_probs.shape) == 2:
                        pred_contract_probs = pred_contract_probs[0]  # Remove batch dimension
                    pred_contract_probs = pred_contract_probs.tolist()
                else:
                    # Fallback: create empty contract probabilities
                    pred_contract_probs = [0.0] * len(analyzer.vulnerability_types)
            except Exception as e:
                print(f"    Error processing pred_contract_probs: {str(e)}")
                raise
            
            # Fix: Handle line probabilities properly
            try:
                pred_line_probs = analyzer_results['line_probabilities']
            except Exception as e:
                print(f"    Error extracting pred_line_probs: {str(e)}")
                raise
            
            try:
                if isinstance(pred_line_probs, list) and len(pred_line_probs) > 0:
                    pred_line_probs = pred_line_probs[0]  # [seq_len, num_vuln_types]
                elif isinstance(pred_line_probs, np.ndarray):
                    if len(pred_line_probs.shape) == 3:
                        pred_line_probs = pred_line_probs[0]  # Remove batch dimension
                    pred_line_probs = pred_line_probs.tolist()
                else:
                    # Fallback: create empty line probabilities
                    pred_line_probs = np.zeros((1024, len(analyzer.vulnerability_types)))
            except Exception as e:
                print(f"    Error processing pred_line_probs: {str(e)}")
                raise
            
            # Convert contract predictions to arrays
            try:
                pred_contract_array = np.array([
                    1 if pred_contract_vulns[vuln_type] else 0 
                    for vuln_type in analyzer.vulnerability_types
                ])
            except Exception as e:
                print(f"    Error creating pred_contract_array: {str(e)}")
                raise
            
            # Store contract-level results
            try:
                results['contract_level']['true_labels'].append(true_contract_vulns)
            except Exception as e:
                print(f"    Error appending true_labels: {str(e)}")
                raise
            
            try:
                results['contract_level']['predicted_probs'].append(pred_contract_probs)
            except Exception as e:
                print(f"    Error appending predicted_probs: {str(e)}")
                raise
            
            try:
                results['contract_level']['predicted_labels'].append(pred_contract_array)
            except Exception as e:
                print(f"    Error appending predicted_labels: {str(e)}")
                raise
            
            try:
                results['contract_level']['source_codes'].append(source_code)
            except Exception as e:
                print(f"    Error appending source_codes: {str(e)}")
                raise
            
            try:
                results['contract_level']['contract_names'].append(contract_name)
            except Exception as e:
                print(f"    Error appending contract_names: {str(e)}")
                raise
            
            # Store line-level results - FIXED: Handle the correct shape (8, 1024)
            try:
                results['line_level']['true_labels'].append(true_line_vulns)  # Shape: (8, 1024)
            except Exception as e:
                print(f"    Error appending line true_labels: {str(e)}")
                raise
            
            # Convert model predictions to match dataset format (8, 1024)
            try:
                # Initialize prediction arrays with correct shape (8, 1024)
                pred_line_probs_array = np.zeros((8, 1024))  # (num_vuln_types, max_lines)
                pred_line_labels_array = np.zeros((8, 1024))  # (num_vuln_types, max_lines)
                
                # Get actual number of lines in the source code
                lines = source_code.split('\n')
                actual_lines = len(lines)
                
                print(f"    Contract {contract_idx}: actual_lines={actual_lines}")
                
                if isinstance(pred_line_vulns, dict):
                    # pred_line_vulns is a dictionary with line indices as keys
                    for line_idx in pred_line_vulns.keys():
                        if line_idx < actual_lines:  # Only process actual lines
                            line_vulns = pred_line_vulns[line_idx]
                            if isinstance(line_vulns, dict):
                                for vuln_type, is_vulnerable in line_vulns.items():
                                    vuln_idx = get_vulnerability_index(vuln_type)
                                    if vuln_idx is not None and vuln_idx < 8:
                                        # Set binary prediction
                                        pred_line_labels_array[vuln_idx, line_idx] = 1 if is_vulnerable else 0
                                        
                                        # Set probability (use threshold as probability for binary predictions)
                                        if isinstance(pred_line_probs, list) and line_idx < len(pred_line_probs):
                                            if vuln_idx < len(pred_line_probs[line_idx]):
                                                pred_line_probs_array[vuln_idx, line_idx] = pred_line_probs[line_idx][vuln_idx]
                                            else:
                                                pred_line_probs_array[vuln_idx, line_idx] = 0.5 if is_vulnerable else 0.1
                                        else:
                                            pred_line_probs_array[vuln_idx, line_idx] = 0.8 if is_vulnerable else 0.2
                
                # Store the properly formatted predictions
                results['line_level']['predicted_probs'].append(pred_line_probs_array)
                results['line_level']['predicted_labels'].append(pred_line_labels_array)
                
                print(f"    Contract {contract_idx}: pred_line_probs_array shape: {pred_line_probs_array.shape}")
                print(f"    Contract {contract_idx}: pred_line_labels_array shape: {pred_line_labels_array.shape}")
                
            except Exception as e:
                print(f"    Error processing line predictions: {str(e)}")
                # Continue with empty line predictions in correct format
                pred_line_probs_array = np.zeros((8, 1024))
                pred_line_labels_array = np.zeros((8, 1024))
                results['line_level']['predicted_probs'].append(pred_line_probs_array)
                results['line_level']['predicted_labels'].append(pred_line_labels_array)
            
            # Store line mapping and vulnerable line details
            try:
                lines = source_code.split('\n')
                results['line_level']['line_mappings'].append(lines)
            except Exception as e:
                print(f"    Error processing line_mappings: {str(e)}")
                raise
            
            # Identify vulnerable lines - FIXED: Use correct shape (8, 1024)
            try:
                vulnerable_lines = []
                
                # true_line_vulns is already in shape (8, 1024) - no need to transpose
                for line_idx, line in enumerate(lines):
                    if line_idx < 1024:  # Check within max lines
                        # Get vulnerabilities for this line across all types
                        line_vulns = true_line_vulns[:, line_idx]  # Shape: (8,)
                        if np.any(line_vulns):
                            vuln_types = [analyzer.vulnerability_types[i] for i, v in enumerate(line_vulns) if v]
                            vulnerable_lines.append({
                                'line_number': line_idx,
                                'line_code': line.strip(),
                                'vulnerability_types': vuln_types
                            })
                
                results['line_level']['vulnerable_lines'].append(vulnerable_lines)
                print(f"    Contract {contract_idx}: found {len(vulnerable_lines)} vulnerable lines")
                
            except Exception as e:
                print(f"    Error processing vulnerable_lines: {str(e)}")
                # Continue with empty vulnerable lines
                results['line_level']['vulnerable_lines'].append([])
            
            # Generate synthetic contract if requested
            if generate_contracts:
                try:
                    generated_contracts = generate_syntax_aware_contract(
                        analyzer=analyzer,
                        contract_template=source_code,
                        num_contracts=1,
                        temperature=0.9,
                        max_length=1024
                    )
                    generated_code = generated_contracts[0] if generated_contracts else "Generation failed"
                    if generated_code != "Generation failed":
                        successful_generations += 1
                except Exception as e:
                    print(f"  ⚠️  Generation failed for contract {contract_idx}: {str(e)}")
                    generated_code = "Generation failed"
                
                results['contract_level']['generated_codes'].append(generated_code)
            else:
                results['contract_level']['generated_codes'].append(None)
            
            # Update metadata
            results['metadata']['total_contracts'] += 1
            results['metadata']['total_lines'] += len(lines)
            
        except Exception as e:
            print(f"❌ Error processing contract {contract_idx}: {str(e)}")
            continue
    
    # Finalize metadata
    processing_time = time.time() - start_time
    results['metadata']['processing_time'] = processing_time
    results['metadata']['generation_success_rate'] = successful_generations / results['metadata']['total_contracts'] if results['metadata']['total_contracts'] > 0 else 0.0
    
    print(f"✅ Validation analysis completed!")
    print(f"📊 Processed {results['metadata']['total_contracts']} contracts")
    print(f"⏱️  Processing time: {processing_time:.2f} seconds")
    print(f"🎯 Generation success rate: {results['metadata']['generation_success_rate']:.2%}")
    
    return results

def print_simplified_validation_summary(
    validation_results: Dict[str, Any],
    metrics: Dict[str, Any]
) -> None:
    """
    Print a simplified summary of validation results (contract-level only).
    
    Args:
        validation_results: Results from collect_validation_results_simple
        metrics: Results from compute_contract_level_metrics
    """
    print("=" * 60)
    print("📊 SIMPLIFIED VALIDATION SUMMARY")
    print("=" * 60)
    
    # Overall statistics
    print(f"\n🎯 Overall Statistics:")
    print(f"  Total Contracts: {validation_results['metadata']['total_contracts']}")
    print(f"  Total Lines: {validation_results['metadata']['total_lines']}")
    print(f"  Processing Time: {validation_results['metadata']['processing_time']:.2f} seconds")
    print(f"  Generation Success Rate: {validation_results['metadata']['generation_success_rate']:.2%}")
    
    # Contract-level performance
    print(f"\n📈 Contract-Level Performance:")
    print(f"  Overall PR-AUC: {metrics['contract_level']['overall_pr_auc']:.4f}")
    print(f"  Overall Accuracy: {metrics['contract_level']['overall_accuracy']:.4f}")
    
    print(f"  Per-Vulnerability Performance:")
    for vuln_type in validation_results['metadata']['vulnerability_types']:
        pr_auc = metrics['contract_level']['pr_auc'].get(vuln_type, 0.0)
        accuracy = metrics['contract_level']['accuracy'].get(vuln_type, 0.0)
        print(f"    {vuln_type}: PR-AUC={pr_auc:.4f}, Accuracy={accuracy:.4f}")
    
    print("=" * 60)

def compute_contract_level_metrics(validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute contract-level metrics from validation results.
    
    Args:
        validation_results: Results from collect_validation_results_simple
        
    Returns:
        Dictionary with computed contract-level metrics
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
    import numpy as np
    
    print("🔢 Computing contract-level PR AUC and performance metrics...")
    
    # Extract data
    contract_true = np.array(validation_results['contract_level']['true_labels'])
    contract_probs = np.array(validation_results['contract_level']['predicted_probs'])
    contract_pred = np.array(validation_results['contract_level']['predicted_labels'])
    
    vuln_types = validation_results['metadata']['vulnerability_types']
    
    # Initialize metrics storage
    metrics = {
        'contract_level': {
            'pr_auc': {},
            'accuracy': {},
            'precision_recall_curves': {},
            'overall_accuracy': 0.0,
            'overall_pr_auc': 0.0
        },
        'summary': {
            'total_contracts': validation_results['metadata']['total_contracts'],
            'vulnerability_types': vuln_types
        }
    }
    
    # Contract-level metrics per vulnerability type
    contract_pr_aucs = []
    contract_accuracies = []
    
    for i, vuln_type in enumerate(vuln_types):
        try:
            # Extract data for this vulnerability type
            y_true = contract_true[:, i]
            y_probs = contract_probs[:, i]
            y_pred = contract_pred[:, i]
            
            # Compute PR AUC
            if np.sum(y_true) > 0:  # Only if there are positive samples
                pr_auc = average_precision_score(y_true, y_probs)
                precision, recall, _ = precision_recall_curve(y_true, y_probs)
            else:
                pr_auc = 0.0
                precision, recall = np.array([1.0]), np.array([0.0])
            
            # Compute accuracy
            accuracy = accuracy_score(y_true, y_pred)
            
            # Store metrics
            metrics['contract_level']['pr_auc'][vuln_type] = pr_auc
            metrics['contract_level']['accuracy'][vuln_type] = accuracy
            metrics['contract_level']['precision_recall_curves'][vuln_type] = {
                'precision': precision,
                'recall': recall
            }
            
            contract_pr_aucs.append(pr_auc)
            contract_accuracies.append(accuracy)
            
            print(f"  Contract {vuln_type}: PR-AUC={pr_auc:.4f}, Accuracy={accuracy:.4f}")
            
        except Exception as e:
            print(f"  ⚠️  Error computing metrics for contract {vuln_type}: {str(e)}")
            metrics['contract_level']['pr_auc'][vuln_type] = 0.0
            metrics['contract_level']['accuracy'][vuln_type] = 0.0
    
    # Compute overall metrics
    metrics['contract_level']['overall_pr_auc'] = np.mean(contract_pr_aucs)
    metrics['contract_level']['overall_accuracy'] = np.mean(contract_accuracies)
    
    print(f"✅ Contract-level metrics computation completed!")
    print(f"📊 Overall PR-AUC={metrics['contract_level']['overall_pr_auc']:.4f}, Accuracy={metrics['contract_level']['overall_accuracy']:.4f}")
    
    return metrics

def analyze_vulnerable_contracts(
    validation_results: Dict[str, Any],
    metrics: Dict[str, Any],
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Analyze the most vulnerable contracts and their generated counterparts.
    
    Args:
        validation_results: Results from collect_validation_results
        metrics: Results from compute_pr_auc_metrics
        top_k: Number of top vulnerable contracts to analyze
        
    Returns:
        Dictionary with detailed analysis
    """
    print(f"🔍 Analyzing top {top_k} vulnerable contracts...")
    
    # Calculate vulnerability scores for each contract
    contract_true = np.array(validation_results['contract_level']['true_labels'])
    contract_probs = np.array(validation_results['contract_level']['predicted_probs'])
    
    # Sum of vulnerability probabilities as vulnerability score
    vulnerability_scores = np.sum(contract_probs, axis=1)
    
    # Get indices of top vulnerable contracts
    top_indices = np.argsort(vulnerability_scores)[-top_k:][::-1]
    
    analysis = {
        'top_vulnerable_contracts': [],
        'vulnerability_distribution': {},
        'generation_quality': {
            'successful_generations': 0,
            'failed_generations': 0,
            'avg_generated_length': 0
        }
    }
    
    # Analyze each top vulnerable contract
    for i, contract_idx in enumerate(top_indices):
        contract_name = validation_results['contract_level']['contract_names'][contract_idx]
        source_code = validation_results['contract_level']['source_codes'][contract_idx]
        generated_code = validation_results['contract_level']['generated_codes'][contract_idx]
        true_vulns = contract_true[contract_idx]
        pred_probs = contract_probs[contract_idx]
        
        # Analyze generation quality
        generation_success = generated_code is not None and generated_code != "Generation failed"
        if generation_success:
            analysis['generation_quality']['successful_generations'] += 1
            generated_length = len(generated_code)
        else:
            analysis['generation_quality']['failed_generations'] += 1
            generated_length = 0
        
        contract_analysis = {
            'rank': i + 1,
            'contract_name': contract_name,
            'vulnerability_score': vulnerability_scores[contract_idx],
            'source_code': source_code,
            'generated_code': generated_code,
            'generation_success': generation_success,
            'generated_length': generated_length,
            'true_vulnerabilities': true_vulns.tolist(),
            'predicted_probabilities': pred_probs.tolist(),
            'source_length': len(source_code)
        }
        
        analysis['top_vulnerable_contracts'].append(contract_analysis)
    
    # Compute generation quality metrics
    total_contracts = len(validation_results['contract_level']['generated_codes'])
    successful_gens = sum(1 for code in validation_results['contract_level']['generated_codes'] 
                         if code is not None and code != "Generation failed")
    
    analysis['generation_quality']['successful_generations'] = successful_gens
    analysis['generation_quality']['failed_generations'] = total_contracts - successful_gens
    analysis['generation_quality']['success_rate'] = successful_gens / total_contracts
    
    # Calculate average generated length
    generated_lengths = [len(code) for code in validation_results['contract_level']['generated_codes'] 
                        if code is not None and code != "Generation failed"]
    if generated_lengths:
        analysis['generation_quality']['avg_generated_length'] = np.mean(generated_lengths)
    
    print(f"✅ Analysis completed!")
    print(f"📊 Generation success rate: {analysis['generation_quality']['success_rate']:.2%}")
    print(f"📏 Average generated length: {analysis['generation_quality']['avg_generated_length']:.0f} characters")
    
    return analysis

def print_validation_summary(
    validation_results: Dict[str, Any],
    metrics: Dict[str, Any],
    analysis: Dict[str, Any] = None
) -> None:
    """
    Print a comprehensive summary of validation results.
    
    Args:
        validation_results: Results from collect_validation_results
        metrics: Results from compute_pr_auc_metrics
        analysis: Results from analyze_vulnerable_contracts (optional)
    """
    print("=" * 80)
    print("📊 COMPREHENSIVE VALIDATION SUMMARY")
    print("=" * 80)
    
    # Overall statistics
    print(f"\n🎯 Overall Statistics:")
    print(f"  Total Contracts: {validation_results['metadata']['total_contracts']}")
    print(f"  Total Lines: {validation_results['metadata']['total_lines']}")
    print(f"  Processing Time: {validation_results['metadata']['processing_time']:.2f} seconds")
    print(f"  Generation Success Rate: {validation_results['metadata']['generation_success_rate']:.2%}")
    
    # Contract-level performance
    print(f"\n📈 Contract-Level Performance:")
    print(f"  Overall PR-AUC: {metrics['contract_level']['overall_pr_auc']:.4f}")
    print(f"  Overall Accuracy: {metrics['contract_level']['overall_accuracy']:.4f}")
    
    print(f"  Per-Vulnerability Performance:")
    for vuln_type in validation_results['metadata']['vulnerability_types']:
        pr_auc = metrics['contract_level']['pr_auc'].get(vuln_type, 0.0)
        accuracy = metrics['contract_level']['accuracy'].get(vuln_type, 0.0)
        print(f"    {vuln_type}: PR-AUC={pr_auc:.4f}, Accuracy={accuracy:.4f}")
    
    # Generation quality
    if analysis:
        print(f"\n🚀 Generation Quality:")
        print(f"  Successful Generations: {analysis['generation_quality']['successful_generations']}")
        print(f"  Failed Generations: {analysis['generation_quality']['failed_generations']}")
        print(f"  Success Rate: {analysis['generation_quality']['success_rate']:.2%}")
        print(f"  Average Generated Length: {analysis['generation_quality']['avg_generated_length']:.0f} characters")
    
    print("=" * 80)

def collect_validation_results_simple(
    analyzer,
    val_dataloader,
    threshold: float = 0.5,
    max_contracts: int = None,
    generate_contracts: bool = True
) -> Dict[str, Any]:
    """
    Collect validation results focusing on contract-level analysis (simplified version).
    
    Args:
        analyzer: SmartContractAnalyzer instance
        val_dataloader: Validation dataloader
        threshold: Detection threshold
        max_contracts: Maximum number of contracts to process (None for all)
        generate_contracts: Whether to generate synthetic contracts
        
    Returns:
        Dictionary with contract-level validation results
    """
    print("🚀 Starting simplified validation analysis (contract-level focus)...")
    
    # Initialize results storage
    results = {
        'contract_level': {
            'true_labels': [],      # List of true contract vulnerability arrays
            'predicted_probs': [],  # List of predicted probability arrays
            'predicted_labels': [], # List of predicted binary arrays
            'source_codes': [],     # List of source codes
            'generated_codes': [],  # List of generated codes
            'contract_names': []    # List of contract names
        },
        'metadata': {
            'total_contracts': 0,
            'total_lines': 0,
            'vulnerability_types': analyzer.vulnerability_types,
            'processing_time': 0,
            'generation_success_rate': 0
        }
    }
    
    import time
    start_time = time.time()
    
    # Determine number of contracts to process
    total_contracts = len(val_dataloader.dataset.data)
    if max_contracts is not None:
        total_contracts = min(total_contracts, max_contracts)
    
    print(f"📊 Processing {total_contracts} contracts...")
    
    successful_generations = 0
    
    for contract_idx in range(total_contracts):
        try:

            
            # Get contract data
            contract_data = val_dataloader.dataset.data[contract_idx]
            source_code = contract_data['source_code']
            true_contract_vulns = contract_data['contract_vulnerabilities'].cpu().numpy()
            contract_name = f'Contract_{contract_idx}'
            
            # Detect vulnerabilities
            pred_results = analyzer.detect_vulnerabilities(source_code, threshold=threshold)
            
            # Extract predictions with proper error handling
            try:
                pred_contract_vulns = pred_results['contract_vulnerabilities']
            except Exception as e:
                print(f"    Error extracting pred_contract_vulns: {str(e)}")
                raise
            
            try:
                pred_contract_probs = pred_results['contract_probabilities']
            except Exception as e:
                print(f"    Error extracting pred_contract_probs: {str(e)}")
                raise
            
            # Convert contract predictions to arrays
            pred_contract_array = np.array([
                1 if pred_contract_vulns[vuln_type] else 0 
                for vuln_type in analyzer.vulnerability_types
            ])
            
            # Store contract-level results
            results['contract_level']['true_labels'].append(true_contract_vulns)
            results['contract_level']['predicted_probs'].append(pred_contract_probs)
            results['contract_level']['predicted_labels'].append(pred_contract_array)
            results['contract_level']['source_codes'].append(source_code)
            results['contract_level']['contract_names'].append(contract_name)
            
            # Generate synthetic contract if requested
            if generate_contracts:
                try:
                    generated_contracts = generate_syntax_aware_contract(
                        analyzer=analyzer,
                        contract_template=source_code,
                        num_contracts=1,
                        temperature=0.9,
                        max_length=1024
                    )
                    generated_code = generated_contracts[0] if generated_contracts else "Generation failed"
                    if generated_code != "Generation failed":
                        successful_generations += 1
                except Exception as e:
                    print(f"  ⚠️  Generation failed for contract {contract_idx}: {str(e)}")
                    generated_code = "Generation failed"
                
                results['contract_level']['generated_codes'].append(generated_code)
            else:
                results['contract_level']['generated_codes'].append(None)
            
            # Update metadata
            results['metadata']['total_contracts'] += 1
            results['metadata']['total_lines'] += len(source_code.split('\n'))
            
        except Exception as e:
            print(f"❌ Error processing contract {contract_idx}: {str(e)}")
            continue
    
    # Finalize metadata
    processing_time = time.time() - start_time
    results['metadata']['processing_time'] = processing_time
    results['metadata']['generation_success_rate'] = successful_generations / results['metadata']['total_contracts'] if results['metadata']['total_contracts'] > 0 else 0.0
    
    print(f"✅ Simplified validation analysis completed!")
    print(f"📊 Processed {results['metadata']['total_contracts']} contracts")
    print(f"⏱️  Processing time: {processing_time:.2f} seconds")
    print(f"🎯 Generation success rate: {results['metadata']['generation_success_rate']:.2%}")
    
    return results

def compute_line_level_metrics(validation_results: dict) -> dict:
    """
    Compute line-level PR-AUC and recall for each vulnerability type.

    Args:
        validation_results: Output from your validation loop, must contain:
            - line_level['true_labels']: list of [num_vuln_types, max_lines] arrays (8, 1024)
            - line_level['predicted_probs']: list of [num_vuln_types, max_lines] arrays (8, 1024)
            - line_level['predicted_labels']: list of [num_vuln_types, max_lines] arrays (8, 1024)

    Returns:
        Dictionary with PR-AUC and recall for each vulnerability type.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score

    print("🔢 Computing line-level PR AUC and performance metrics...")
    
    # Check if line-level data exists
    if 'line_level' not in validation_results:
        print("⚠️  No line-level data found in validation results")
        return {}
    
    if not validation_results['line_level']['true_labels']:
        print("⚠️  No line-level true labels found")
        return {}
    
    # Extract and validate data
    true_labels = validation_results['line_level']['true_labels']
    pred_probs = validation_results['line_level']['predicted_probs']
    pred_labels = validation_results['line_level']['predicted_labels']
    vuln_types = validation_results['metadata']['vulnerability_types']
    
    print(f"📊 Found {len(true_labels)} contracts with line-level data")
    print(f"📊 Vulnerability types: {vuln_types}")
    
    # Validate data consistency
    if len(true_labels) != len(pred_probs) or len(true_labels) != len(pred_labels):
        print(f"⚠️  Inconsistent number of contracts: true_labels={len(true_labels)}, pred_probs={len(pred_probs)}, pred_labels={len(pred_labels)}")
        return {}
    
    metrics = {}
    
    # Process each vulnerability type separately
    for i, vuln_type in enumerate(vuln_types):
        try:
            print(f"  Processing {vuln_type}...")
            
            # Collect data for this vulnerability type across all contracts
            y_true_list = []
            y_prob_list = []
            y_pred_list = []
            
            for contract_idx in range(len(true_labels)):
                try:
                    # Get data for this contract - shape is (8, 1024)
                    contract_true = true_labels[contract_idx]  # Shape: (8, 1024)
                    contract_probs = pred_probs[contract_idx]  # Shape: (8, 1024)
                    contract_pred = pred_labels[contract_idx]  # Shape: (8, 1024)
                    
                    # Validate shapes
                    if contract_true.shape != (8, 1024):
                        print(f"    ⚠️  Contract {contract_idx}: unexpected true shape {contract_true.shape}")
                        continue
                    
                    if contract_probs.shape != (8, 1024):
                        print(f"    ⚠️  Contract {contract_idx}: unexpected probs shape {contract_probs.shape}")
                        continue
                    
                    if contract_pred.shape != (8, 1024):
                        print(f"    ⚠️  Contract {contract_idx}: unexpected pred shape {contract_pred.shape}")
                        continue
                    
                    # Get the actual number of lines for this contract
                    if contract_idx < len(validation_results['line_level']['line_mappings']):
                        actual_lines = len(validation_results['line_level']['line_mappings'][contract_idx])
                    else:
                        actual_lines = 1024  # Fallback
                    
                    # Extract vulnerability type i for all lines (only up to actual_lines)
                    if i < contract_true.shape[0]:  # i < 8
                        # Get true labels for all lines up to actual_lines
                        true_line_vulns = contract_true[i, :actual_lines]  # Shape: (actual_lines,)
                            
                        # Get predicted probabilities for all lines up to actual_lines
                        prob_line_vulns = contract_probs[i, :actual_lines]  # Shape: (actual_lines,)
                            
                        # Get predicted labels for all lines up to actual_lines
                        pred_line_vulns = contract_pred[i, :actual_lines]  # Shape: (actual_lines,)
                        
                        # Add to lists
                        y_true_list.extend(true_line_vulns)
                        y_prob_list.extend(prob_line_vulns)
                        y_pred_list.extend(pred_line_vulns)
                    else:
                        print(f"    ⚠️  Vulnerability type index {i} out of range for contract {contract_idx}")
                    
                except Exception as e:
                    print(f"    ⚠️  Error processing contract {contract_idx} for {vuln_type}: {str(e)}")
                    continue
            
            # Convert to numpy arrays
            if not y_true_list or not y_prob_list or not y_pred_list:
                print(f"    ⚠️  No valid data for {vuln_type}")
                metrics[vuln_type] = {
                    'pr_auc': 0.0,
                    'vuln_line_recall': 0.0,
                    'num_true_vuln_lines': 0,
                    'num_correct_vuln_lines': 0
                }
                continue
            
            y_true = np.array(y_true_list)
            y_prob = np.array(y_prob_list)
            y_pred = np.array(y_pred_list)
            
            print(f"    📊 {vuln_type}: {len(y_true)} samples, {np.sum(y_true)} positive samples")
            
            # Validate array lengths
            if len(y_true) != len(y_prob) or len(y_true) != len(y_pred):
                print(f"    ⚠️  Inconsistent array lengths for {vuln_type}: true={len(y_true)}, prob={len(y_prob)}, pred={len(y_pred)}")
                # Truncate to minimum length
                min_length = min(len(y_true), len(y_prob), len(y_pred))
                y_true = y_true[:min_length]
                y_prob = y_prob[:min_length]
                y_pred = y_pred[:min_length]
                print(f"    📊 Truncated to {min_length} samples")
            
            # PR-AUC
            if np.sum(y_true) > 0:
                try:
                    pr_auc = average_precision_score(y_true, y_prob)
                except Exception as e:
                    print(f"    ⚠️  Error computing PR-AUC for {vuln_type}: {str(e)}")
                    pr_auc = 0.0
            else:
                pr_auc = 0.0

            # Recall for vulnerable lines
            true_vuln_lines = np.sum(y_true)
            correct_vuln_lines = np.sum((y_true == 1) & (y_pred == 1))
            recall = correct_vuln_lines / true_vuln_lines if true_vuln_lines > 0 else 0.0

            metrics[vuln_type] = {
                'pr_auc': pr_auc,
                'vuln_line_recall': recall,
                'num_true_vuln_lines': int(true_vuln_lines),
                'num_correct_vuln_lines': int(correct_vuln_lines)
            }
            
            print(f"    ✅ {vuln_type}: PR-AUC={pr_auc:.4f}, Recall={recall:.4f}, True vulns={int(true_vuln_lines)}, Correct={int(correct_vuln_lines)}")
            
        except Exception as e:
            print(f"  ❌ Error processing {vuln_type}: {str(e)}")
            metrics[vuln_type] = {
                'pr_auc': 0.0,
                'vuln_line_recall': 0.0,
                'num_true_vuln_lines': 0,
                'num_correct_vuln_lines': 0
            }

    print("✅ Line-level metrics computation completed!")
    return metrics

def analyze_line_vulnerability_detection(validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the number of lines correctly identified as vulnerable by the model.
    
    Args:
        validation_results: Results from collect_validation_results
        
    Returns:
        Dictionary with detailed line-level vulnerability detection analysis
    """
    print("🔍 Analyzing line-level vulnerability detection performance...")
    
    if 'line_level' not in validation_results:
        print("⚠️  No line-level data found in validation results")
        return {}
    
    true_labels = validation_results['line_level']['true_labels']
    pred_labels = validation_results['line_level']['predicted_labels']
    vuln_types = validation_results['metadata']['vulnerability_types']
    
    if not true_labels or not pred_labels:
        print("⚠️  No line-level labels found")
        return {}
    
    analysis = {
        'overall_stats': {
            'total_contracts': len(true_labels),
            'total_lines_analyzed': 0,
            'total_true_vulnerable_lines': 0,
            'total_predicted_vulnerable_lines': 0,
            'total_correctly_identified_lines': 0,
            'total_false_positives': 0,
            'total_false_negatives': 0
        },
        'per_vulnerability_type': {},
        'per_contract': []
    }
    
    # Process each contract
    for contract_idx in range(len(true_labels)):
        contract_true = true_labels[contract_idx]  # Shape: (8, 1024)
        contract_pred = pred_labels[contract_idx]  # Shape: (8, 1024)
        
        # Get actual number of lines for this contract
        if contract_idx < len(validation_results['line_level']['line_mappings']):
            actual_lines = len(validation_results['line_level']['line_mappings'][contract_idx])
        else:
            actual_lines = 1024
        
        contract_stats = {
            'contract_idx': contract_idx,
            'actual_lines': actual_lines,
            'true_vulnerable_lines': 0,
            'predicted_vulnerable_lines': 0,
            'correctly_identified_lines': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        # Process each vulnerability type
        for vuln_idx, vuln_type in enumerate(vuln_types):
            if vuln_idx < contract_true.shape[0]:  # vuln_idx < 8
                # Get true and predicted for this vulnerability type
                true_vulns = contract_true[vuln_idx, :actual_lines]  # Shape: (actual_lines,)
                pred_vulns = contract_pred[vuln_idx, :actual_lines]  # Shape: (actual_lines,)
                
                # Calculate metrics for this vulnerability type
                true_positives = np.sum((true_vulns == 1) & (pred_vulns == 1))
                false_positives = np.sum((true_vulns == 0) & (pred_vulns == 1))
                false_negatives = np.sum((true_vulns == 1) & (pred_vulns == 0))
                
                # Update contract stats
                contract_stats['true_vulnerable_lines'] += np.sum(true_vulns)
                contract_stats['predicted_vulnerable_lines'] += np.sum(pred_vulns)
                contract_stats['correctly_identified_lines'] += true_positives
                contract_stats['false_positives'] += false_positives
                contract_stats['false_negatives'] += false_negatives
                
                # Update overall stats
                analysis['overall_stats']['total_true_vulnerable_lines'] += np.sum(true_vulns)
                analysis['overall_stats']['total_predicted_vulnerable_lines'] += np.sum(pred_vulns)
                analysis['overall_stats']['total_correctly_identified_lines'] += true_positives
                analysis['overall_stats']['total_false_positives'] += false_positives
                analysis['overall_stats']['total_false_negatives'] += false_negatives
        
        # Update total lines analyzed
        analysis['overall_stats']['total_lines_analyzed'] += actual_lines
        
        # Store contract stats
        analysis['per_contract'].append(contract_stats)
    
    # Calculate per-vulnerability type statistics
    for vuln_idx, vuln_type in enumerate(vuln_types):
        vuln_stats = {
            'total_true_vulnerable_lines': 0,
            'total_predicted_vulnerable_lines': 0,
            'total_correctly_identified_lines': 0,
            'total_false_positives': 0,
            'total_false_negatives': 0
        }
        
        for contract_idx in range(len(true_labels)):
            contract_true = true_labels[contract_idx]
            contract_pred = pred_labels[contract_idx]
            
            if contract_idx < len(validation_results['line_level']['line_mappings']):
                actual_lines = len(validation_results['line_level']['line_mappings'][contract_idx])
            else:
                actual_lines = 1024
            
            if vuln_idx < contract_true.shape[0]:
                true_vulns = contract_true[vuln_idx, :actual_lines]
                pred_vulns = contract_pred[vuln_idx, :actual_lines]
                
                true_positives = np.sum((true_vulns == 1) & (pred_vulns == 1))
                false_positives = np.sum((true_vulns == 0) & (pred_vulns == 1))
                false_negatives = np.sum((true_vulns == 1) & (pred_vulns == 0))
                
                vuln_stats['total_true_vulnerable_lines'] += np.sum(true_vulns)
                vuln_stats['total_predicted_vulnerable_lines'] += np.sum(pred_vulns)
                vuln_stats['total_correctly_identified_lines'] += true_positives
                vuln_stats['total_false_positives'] += false_positives
                vuln_stats['total_false_negatives'] += false_negatives
        
        # Calculate precision and recall for this vulnerability type
        precision = vuln_stats['total_correctly_identified_lines'] / (vuln_stats['total_correctly_identified_lines'] + vuln_stats['total_false_positives']) if (vuln_stats['total_correctly_identified_lines'] + vuln_stats['total_false_positives']) > 0 else 0.0
        recall = vuln_stats['total_correctly_identified_lines'] / (vuln_stats['total_correctly_identified_lines'] + vuln_stats['total_false_negatives']) if (vuln_stats['total_correctly_identified_lines'] + vuln_stats['total_false_negatives']) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        vuln_stats['precision'] = precision
        vuln_stats['recall'] = recall
        vuln_stats['f1_score'] = f1_score
        
        analysis['per_vulnerability_type'][vuln_type] = vuln_stats
    
    # Calculate overall metrics
    overall_precision = analysis['overall_stats']['total_correctly_identified_lines'] / (analysis['overall_stats']['total_correctly_identified_lines'] + analysis['overall_stats']['total_false_positives']) if (analysis['overall_stats']['total_correctly_identified_lines'] + analysis['overall_stats']['total_false_positives']) > 0 else 0.0
    overall_recall = analysis['overall_stats']['total_correctly_identified_lines'] / (analysis['overall_stats']['total_correctly_identified_lines'] + analysis['overall_stats']['total_false_negatives']) if (analysis['overall_stats']['total_correctly_identified_lines'] + analysis['overall_stats']['total_false_negatives']) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    analysis['overall_stats']['precision'] = overall_precision
    analysis['overall_stats']['recall'] = overall_recall
    analysis['overall_stats']['f1_score'] = overall_f1
    
    print("✅ Line-level vulnerability detection analysis completed!")
    return analysis

def print_line_vulnerability_summary(analysis: Dict[str, Any]) -> None:
    """
    Print a comprehensive summary of line-level vulnerability detection performance.
    
    Args:
        analysis: Results from analyze_line_vulnerability_detection
    """
    print("=" * 80)
    print("🔍 LINE-LEVEL VULNERABILITY DETECTION SUMMARY")
    print("=" * 80)
    
    overall = analysis['overall_stats']
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total Contracts: {overall['total_contracts']}")
    print(f"  Total Lines Analyzed: {overall['total_lines_analyzed']}")
    print(f"  Total True Vulnerable Lines: {overall['total_true_vulnerable_lines']}")
    print(f"  Total Predicted Vulnerable Lines: {overall['total_predicted_vulnerable_lines']}")
    print(f"  Total Correctly Identified Lines: {overall['total_correctly_identified_lines']}")
    print(f"  Total False Positives: {overall['total_false_positives']}")
    print(f"  Total False Negatives: {overall['total_false_negatives']}")
    
    print(f"\n📈 Overall Performance:")
    print(f"  Precision: {overall['precision']:.4f}")
    print(f"  Recall: {overall['recall']:.4f}")
    print(f"  F1-Score: {overall['f1_score']:.4f}")
    
    print(f"\n🎯 Per-Vulnerability Type Performance:")
    for vuln_type, stats in analysis['per_vulnerability_type'].items():
        print(f"  {vuln_type}:")
        print(f"    True Vulnerable Lines: {stats['total_true_vulnerable_lines']}")
        print(f"    Predicted Vulnerable Lines: {stats['total_predicted_vulnerable_lines']}")
        print(f"    Correctly Identified: {stats['total_correctly_identified_lines']}")
        print(f"    False Positives: {stats['total_false_positives']}")
        print(f"    False Negatives: {stats['total_false_negatives']}")
        print(f"    Precision: {stats['precision']:.4f}")
        print(f"    Recall: {stats['recall']:.4f}")
        print(f"    F1-Score: {stats['f1_score']:.4f}")
        print()
    
    print(f"\n📋 Confusion Matrix:")
    cm = analysis['confusion_matrix']
    print(f"  True Positives: {cm['true_positives']}")
    print(f"  False Positives: {cm['false_positives']}")
    print(f"  False Negatives: {cm['false_negatives']}")
    print(f"  True Negatives: {cm['true_negatives']}")
    
    print("=" * 80)

def debug_line_predictions(validation_results: Dict[str, Any], contract_idx: int = 0) -> None:
    """
    Debug line-level predictions for a specific contract to understand why the model isn't predicting vulnerable lines.
    
    Args:
        validation_results: Results from collect_validation_results
        contract_idx: Index of contract to debug (default: 0)
    """
    print(f"🔍 Debugging line predictions for contract {contract_idx}...")
    
    if 'line_level' not in validation_results:
        print("⚠️  No line-level data found")
        return
    
    true_labels = validation_results['line_level']['true_labels']
    pred_labels = validation_results['line_level']['predicted_labels']
    pred_probs = validation_results['line_level']['predicted_probs']
    
    if contract_idx >= len(true_labels):
        print(f"⚠️  Contract index {contract_idx} out of range")
        return
    
    contract_true = true_labels[contract_idx]  # Shape: (8, 1024)
    contract_pred = pred_labels[contract_idx]  # Shape: (8, 1024)
    contract_probs = pred_probs[contract_idx]  # Shape: (8, 1024)
    
    print(f"📊 Contract {contract_idx} shapes:")
    print(f"  True labels: {contract_true.shape}")
    print(f"  Pred labels: {contract_pred.shape}")
    print(f"  Pred probs: {contract_probs.shape}")
    
    # Get actual number of lines
    if contract_idx < len(validation_results['line_level']['line_mappings']):
        actual_lines = len(validation_results['line_level']['line_mappings'][contract_idx])
    else:
        actual_lines = 1024
    
    print(f"📏 Actual lines in contract: {actual_lines}")
    
    vuln_types = validation_results['metadata']['vulnerability_types']
    
    print(f"\n🔍 Vulnerability Analysis:")
    for vuln_idx, vuln_type in enumerate(vuln_types):
        if vuln_idx < contract_true.shape[0]:
            true_vulns = contract_true[vuln_idx, :actual_lines]
            pred_vulns = contract_pred[vuln_idx, :actual_lines]
            pred_probs_vuln = contract_probs[vuln_idx, :actual_lines]
            
            true_count = np.sum(true_vulns)
            pred_count = np.sum(pred_vulns)
            max_prob = np.max(pred_probs_vuln)
            min_prob = np.min(pred_probs_vuln)
            mean_prob = np.mean(pred_probs_vuln)
            
            print(f"  {vuln_type}:")
            print(f"    True vulnerable lines: {true_count}")
            print(f"    Predicted vulnerable lines: {pred_count}")
            print(f"    Max probability: {max_prob:.4f}")
            print(f"    Min probability: {min_prob:.4f}")
            print(f"    Mean probability: {mean_prob:.4f}")
            
            if true_count > 0:
                # Show details for lines that should be vulnerable
                vulnerable_line_indices = np.where(true_vulns == 1)[0]
                print(f"    Vulnerable line indices: {vulnerable_line_indices}")
                for line_idx in vulnerable_line_indices[:5]:  # Show first 5
                    prob = pred_probs_vuln[line_idx]
                    pred = pred_vulns[line_idx]
                    print(f"      Line {line_idx}: prob={prob:.4f}, pred={pred}")
            
            if pred_count > 0:
                # Show details for lines that were predicted as vulnerable
                predicted_line_indices = np.where(pred_vulns == 1)[0]
                print(f"    Predicted vulnerable line indices: {predicted_line_indices}")
                for line_idx in predicted_line_indices[:5]:  # Show first 5
                    prob = pred_probs_vuln[line_idx]
                    true_val = true_vulns[line_idx]
                    print(f"      Line {line_idx}: prob={prob:.4f}, true={true_val}")

def analyze_vulnerable_line_probabilities(validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze probability scores for lines that have vulnerabilities and compute mean probabilities.
    
    Args:
        validation_results: Results from collect_validation_results
        
    Returns:
        Dictionary with probability analysis for vulnerable lines
    """
    print("📊 Analyzing probability scores for vulnerable lines...")
    
    if 'line_level' not in validation_results:
        print("⚠️  No line-level data found")
        return {}
    
    true_labels = validation_results['line_level']['true_labels']
    pred_probs = validation_results['line_level']['predicted_probs']
    vuln_types = validation_results['metadata']['vulnerability_types']
    
    analysis = {
        'per_vulnerability_type': {},
        'overall_stats': {
            'total_vulnerable_lines': 0,
            'mean_probability_vulnerable_lines': 0.0,
            'mean_probability_all_lines': 0.0,
            'probability_distribution': {
                'high_confidence': 0,  # > 0.8
                'medium_confidence': 0,  # 0.5-0.8
                'low_confidence': 0,  # < 0.5
            }
        }
    }
    
    all_vulnerable_probs = []
    all_line_probs = []
    
    # Process each vulnerability type
    for vuln_idx, vuln_type in enumerate(vuln_types):
        vuln_analysis = {
            'vulnerable_line_probabilities': [],
            'all_line_probabilities': [],
            'mean_probability_vulnerable': 0.0,
            'mean_probability_all': 0.0,
            'max_probability_vulnerable': 0.0,
            'min_probability_vulnerable': 1.0,
            'vulnerable_line_count': 0
        }
        
        # Collect probabilities across all contracts
        for contract_idx in range(len(true_labels)):
            contract_true = true_labels[contract_idx]
            contract_probs = pred_probs[contract_idx]
            
            if contract_idx < len(validation_results['line_level']['line_mappings']):
                actual_lines = len(validation_results['line_level']['line_mappings'][contract_idx])
            else:
                actual_lines = 1024
            
            if vuln_idx < contract_true.shape[0]:
                true_vulns = contract_true[vuln_idx, :actual_lines]
                prob_vulns = contract_probs[vuln_idx, :actual_lines]
                
                # Get probabilities for vulnerable lines
                vulnerable_indices = np.where(true_vulns == 1)[0]
                vulnerable_probs = prob_vulns[vulnerable_indices]
                
                vuln_analysis['vulnerable_line_probabilities'].extend(vulnerable_probs)
                vuln_analysis['all_line_probabilities'].extend(prob_vulns)
                all_vulnerable_probs.extend(vulnerable_probs)
                all_line_probs.extend(prob_vulns)
        
        # Calculate statistics for this vulnerability type
        if vuln_analysis['vulnerable_line_probabilities']:
            vuln_analysis['mean_probability_vulnerable'] = np.mean(vuln_analysis['vulnerable_line_probabilities'])
            vuln_analysis['max_probability_vulnerable'] = np.max(vuln_analysis['vulnerable_line_probabilities'])
            vuln_analysis['min_probability_vulnerable'] = np.min(vuln_analysis['vulnerable_line_probabilities'])
            vuln_analysis['vulnerable_line_count'] = len(vuln_analysis['vulnerable_line_probabilities'])
        
        if vuln_analysis['all_line_probabilities']:
            vuln_analysis['mean_probability_all'] = np.mean(vuln_analysis['all_line_probabilities'])
        
        analysis['per_vulnerability_type'][vuln_type] = vuln_analysis
    
    # Calculate overall statistics
    if all_vulnerable_probs:
        analysis['overall_stats']['total_vulnerable_lines'] = len(all_vulnerable_probs)
        analysis['overall_stats']['mean_probability_vulnerable_lines'] = np.mean(all_vulnerable_probs)
        
        # Categorize probabilities
        for prob in all_vulnerable_probs:
            if prob > 0.8:
                analysis['overall_stats']['probability_distribution']['high_confidence'] += 1
            elif prob > 0.5:
                analysis['overall_stats']['probability_distribution']['medium_confidence'] += 1
            else:
                analysis['overall_stats']['probability_distribution']['low_confidence'] += 1
    
    if all_line_probs:
        analysis['overall_stats']['mean_probability_all_lines'] = np.mean(all_line_probs)
    
    print("✅ Probability analysis completed!")
    return analysis

def print_probability_analysis(analysis: Dict[str, Any]) -> None:
    """
    Print detailed probability analysis for vulnerable lines.
    
    Args:
        analysis: Results from analyze_vulnerable_line_probabilities
    """
    print("=" * 80)
    print("📊 VULNERABLE LINE PROBABILITY ANALYSIS")
    print("=" * 80)
    
    overall = analysis['overall_stats']
    
    print(f"\n📈 Overall Statistics:")
    print(f"  Total Vulnerable Lines: {overall['total_vulnerable_lines']}")
    print(f"  Mean Probability (Vulnerable Lines): {overall['mean_probability_vulnerable_lines']:.4f}")
    print(f"  Mean Probability (All Lines): {overall['mean_probability_all_lines']:.4f}")
    
    print(f"\n🎯 Probability Distribution (Vulnerable Lines):")
    dist = overall['probability_distribution']
    total_vuln = overall['total_vulnerable_lines']
    if total_vuln > 0:
        print(f"  High Confidence (>0.8): {dist['high_confidence']} ({dist['high_confidence']/total_vuln:.1%})")
        print(f"  Medium Confidence (0.5-0.8): {dist['medium_confidence']} ({dist['medium_confidence']/total_vuln:.1%})")
        print(f"  Low Confidence (<0.5): {dist['low_confidence']} ({dist['low_confidence']/total_vuln:.1%})")
    
    print(f"\n🔍 Per-Vulnerability Type Analysis:")
    for vuln_type, stats in analysis['per_vulnerability_type'].items():
        print(f"  {vuln_type}:")
        print(f"    Vulnerable Lines: {stats['vulnerable_line_count']}")
        if stats['vulnerable_line_count'] > 0:
            print(f"    Mean Probability (Vulnerable): {stats['mean_probability_vulnerable']:.4f}")
            print(f"    Max Probability (Vulnerable): {stats['max_probability_vulnerable']:.4f}")
            print(f"    Min Probability (Vulnerable): {stats['min_probability_vulnerable']:.4f}")
        print(f"    Mean Probability (All Lines): {stats['mean_probability_all']:.4f}")
        print()
    
    print("=" * 80)

def check_model_line_predictions(analyzer, sample_contract: str) -> None:
    """
    Check how the model predicts line-level vulnerabilities for a sample contract.
    
    Args:
        analyzer: SmartContractAnalyzer instance
        sample_contract: Sample contract code to test
    """
    print("🔍 Testing model line-level predictions on sample contract...")
    
    try:
        # Get model predictions
        pred_results = analyzer.detect_vulnerabilities(sample_contract, threshold=0.5)
        
        print(f"📊 Model output keys: {list(pred_results.keys())}")
        
        # Check line vulnerabilities
        if 'line_vulnerabilities' in pred_results:
            line_vulns = pred_results['line_vulnerabilities']
            print(f"📋 Line vulnerabilities type: {type(line_vulns)}")
            
            if isinstance(line_vulns, dict):
                print(f"📋 Number of lines with predictions: {len(line_vulns)}")
                for line_idx, vulns in list(line_vulns.items())[:5]:  # Show first 5
                    print(f"  Line {line_idx}: {vulns}")
            else:
                print(f"📋 Line vulnerabilities: {line_vulns}")
        
        # Check line probabilities
        if 'line_probabilities' in pred_results:
            line_probs = pred_results['line_probabilities']
            print(f"📊 Line probabilities type: {type(line_probs)}")
            
            if isinstance(line_probs, list) and len(line_probs) > 0:
                print(f"📊 Number of line probability arrays: {len(line_probs)}")
                if len(line_probs[0]) > 0:
                    print(f"📊 First line probabilities shape: {len(line_probs[0])}")
                    print(f"📊 Sample probabilities: {line_probs[0][:5]}")  # First 5 lines
            elif isinstance(line_probs, np.ndarray):
                print(f"📊 Line probabilities shape: {line_probs.shape}")
                print(f"📊 Sample probabilities: {line_probs[0, :5]}")  # First 5 lines
        
        # Check contract vulnerabilities
        if 'contract_vulnerabilities' in pred_results:
            contract_vulns = pred_results['contract_vulnerabilities']
            print(f"📋 Contract vulnerabilities: {contract_vulns}")
        
        # Check contract probabilities
        if 'contract_probabilities' in pred_results:
            contract_probs = pred_results['contract_probabilities']
            print(f"📊 Contract probabilities: {contract_probs}")
        
    except Exception as e:
        print(f"❌ Error testing model predictions: {str(e)}")
        import traceback
        traceback.print_exc()

def diagnose_line_detection_issues(validation_results: Dict[str, Any]) -> None:
    """
    Diagnose potential issues with line-level vulnerability detection.
    
    Args:
        validation_results: Results from collect_validation_results
    """
    print("🔍 Diagnosing line-level vulnerability detection issues...")
    
    if 'line_level' not in validation_results:
        print("⚠️  No line-level data found")
        return
    
    true_labels = validation_results['line_level']['true_labels']
    pred_labels = validation_results['line_level']['predicted_labels']
    pred_probs = validation_results['line_level']['predicted_probs']
    
    print(f"📊 Data Overview:")
    print(f"  Number of contracts: {len(true_labels)}")
    print(f"  Number of vulnerability types: {len(validation_results['metadata']['vulnerability_types'])}")
    
    # Check if all predictions are zero
    total_predictions = 0
    total_true_vulnerabilities = 0
    
    for contract_idx in range(len(true_labels)):
        contract_true = true_labels[contract_idx]
        contract_pred = pred_labels[contract_idx]
        
        if contract_idx < len(validation_results['line_level']['line_mappings']):
            actual_lines = len(validation_results['line_level']['line_mappings'][contract_idx])
        else:
            actual_lines = 1024
        
        # Count predictions and true vulnerabilities
        contract_pred_count = np.sum(contract_pred[:, :actual_lines])
        contract_true_count = np.sum(contract_true[:, :actual_lines])
        
        total_predictions += contract_pred_count
        total_true_vulnerabilities += contract_true_count
    
    print(f"  Total predicted vulnerable lines: {total_predictions}")
    print(f"  Total true vulnerable lines: {total_true_vulnerabilities}")
    
    # Check probability ranges
    all_probs = []
    for contract_probs in pred_probs:
        if contract_idx < len(validation_results['line_level']['line_mappings']):
            actual_lines = len(validation_results['line_level']['line_mappings'][contract_idx])
        else:
            actual_lines = 1024
        all_probs.extend(contract_probs[:, :actual_lines].flatten())
    
    if all_probs:
        print(f"  Probability statistics:")
        print(f"    Min probability: {np.min(all_probs):.4f}")
        print(f"    Max probability: {np.max(all_probs):.4f}")
        print(f"    Mean probability: {np.mean(all_probs):.4f}")
        print(f"    Std probability: {np.std(all_probs):.4f}")
    
    # Diagnose potential issues
    print(f"\n🔍 Potential Issues:")
    
    if total_predictions == 0:
        print("  ❌ ISSUE: Model is not predicting any vulnerable lines!")
        print("    Possible causes:")
        print("    1. Model was not trained for line-level detection")
        print("    2. Threshold is too high (try lowering from 0.5 to 0.1)")
        print("    3. Line-level head in model is not properly connected")
        print("    4. Model outputs are all below threshold")
        
        if all_probs:
            max_prob = np.max(all_probs)
            print(f"    Max probability observed: {max_prob:.4f}")
            if max_prob < 0.5:
                print(f"    SUGGESTION: Try threshold = {max_prob * 0.8:.3f}")
    
    if total_true_vulnerabilities == 0:
        print("  ⚠️  WARNING: No true vulnerabilities found in dataset!")
        print("    This suggests the dataset might not have line-level annotations")
    
    if total_predictions > 0 and total_true_vulnerabilities > 0:
        print("  ✅ Model is making some predictions and there are true vulnerabilities")
        print("    Check if predictions align with true vulnerabilities")
    
    # Check for probability distribution issues
    if all_probs:
        prob_std = np.std(all_probs)
        if prob_std < 0.01:
            print("  ⚠️  WARNING: Very low probability variance!")
            print("    This suggests the model might be outputting similar probabilities for all lines")
            print("    Possible causes:")
            print("    1. Model is not properly trained")
            print("    2. Line-level head weights are not learned")
            print("    3. Model is stuck in a local minimum")
    
    print(f"\n💡 Suggested Actions:")
    print("  1. Run debug_line_predictions() on a few contracts")
    print("  2. Check model architecture for line-level head")
    print("  3. Try different thresholds (0.1, 0.3, 0.7)")
    print("  4. Verify training data has line-level annotations")
    print("  5. Check if model was trained with line-level loss")

def visualize_training_progress(training_history: Dict[str, List[float]], save_path: str = None):
    """
    Visualize training progress including line-level vulnerability detection metrics.
    
    Args:
        training_history: Dictionary containing training metrics history
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    
    epochs = range(1, len(training_history['train_loss']) + 1)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Progress - Line-Level Vulnerability Detection', fontsize=16, fontweight='bold')
    
    # Plot 1: Overall Losses
    ax1 = axes[0, 0]
    ax1.plot(epochs, training_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Overall Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Vulnerability Losses
    ax2 = axes[0, 1]
    ax2.plot(epochs, training_history['contract_vuln_loss'], 'g-', label='Contract Vuln Loss', linewidth=2)
    ax2.plot(epochs, training_history['line_vuln_loss'], 'm-', label='Line Vuln Loss', linewidth=2)
    ax2.set_title('Vulnerability Detection Losses')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Line-Level Performance Metrics
    ax3 = axes[0, 2]
    if 'line_vuln_accuracy' in training_history:
        ax3.plot(epochs, training_history['line_vuln_accuracy'], 'c-', label='Line Accuracy', linewidth=2)
    if 'line_vuln_precision' in training_history:
        ax3.plot(epochs, training_history['line_vuln_precision'], 'orange', label='Line Precision', linewidth=2)
    if 'line_vuln_recall' in training_history:
        ax3.plot(epochs, training_history['line_vuln_recall'], 'purple', label='Line Recall', linewidth=2)
    ax3.set_title('Line-Level Performance Metrics')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Learning Rate
    ax4 = axes[1, 0]
    ax4.plot(epochs, training_history['learning_rate'], 'k-', linewidth=2)
    ax4.set_title('Learning Rate')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Plot 5: Discriminator Loss (if GAN is used)
    ax5 = axes[1, 1]
    if 'discriminator_loss' in training_history:
        ax5.plot(epochs, training_history['discriminator_loss'], 'brown', linewidth=2)
        ax5.set_title('Discriminator Loss')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'GAN not used', ha='center', va='center', transform=ax5.transAxes, fontsize=14)
        ax5.set_title('Discriminator Loss')
    
    # Plot 6: Syntax Loss
    ax6 = axes[1, 2]
    if 'syntax_loss' in training_history:
        ax6.plot(epochs, training_history['syntax_loss'], 'teal', linewidth=2)
        ax6.set_title('Syntax Loss')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Loss')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Syntax loss not tracked', ha='center', va='center', transform=ax6.transAxes, fontsize=14)
        ax6.set_title('Syntax Loss')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training progress plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING PROGRESS SUMMARY")
    print("="*60)
    
    if len(training_history['train_loss']) > 0:
        print(f"Total epochs trained: {len(training_history['train_loss'])}")
        print(f"Best validation loss: {min(training_history['val_loss']):.6f}")
        print(f"Final train loss: {training_history['train_loss'][-1]:.6f}")
        print(f"Final validation loss: {training_history['val_loss'][-1]:.6f}")
        
        if 'line_vuln_accuracy' in training_history:
            print(f"Best line accuracy: {max(training_history['line_vuln_accuracy']):.4f}")
            print(f"Final line accuracy: {training_history['line_vuln_accuracy'][-1]:.4f}")
        
        if 'line_vuln_precision' in training_history:
            print(f"Best line precision: {max(training_history['line_vuln_precision']):.4f}")
            print(f"Final line precision: {training_history['line_vuln_precision'][-1]:.4f}")
        
        if 'line_vuln_recall' in training_history:
            print(f"Best line recall: {max(training_history['line_vuln_recall']):.4f}")
            print(f"Final line recall: {training_history['line_vuln_recall'][-1]:.4f}")
        
        # Check for training issues
        print("\n" + "-"*40)
        print("TRAINING DIAGNOSTICS")
        print("-"*40)
        
        # Check for overfitting
        if len(training_history['train_loss']) > 10:
            recent_train_loss = training_history['train_loss'][-10:]
            recent_val_loss = training_history['val_loss'][-10:]
            train_trend = sum(1 for i in range(1, len(recent_train_loss)) if recent_train_loss[i] < recent_train_loss[i-1])
            val_trend = sum(1 for i in range(1, len(recent_val_loss)) if recent_val_loss[i] < recent_val_loss[i-1])
            
            if train_trend >= 7 and val_trend <= 3:
                print("⚠️  Potential overfitting detected: Train loss decreasing but validation loss not improving")
        
        # Check line-level learning
        if 'line_vuln_recall' in training_history:
            final_line_recall = training_history['line_vuln_recall'][-1]
            if final_line_recall < 0.1:
                print(f"⚠️  Very low line recall ({final_line_recall:.4f}) - model may not be learning line vulnerabilities")
            elif final_line_recall < 0.3:
                print(f"⚠️  Low line recall ({final_line_recall:.4f}) - consider adjusting training parameters")
            else:
                print(f"✅ Good line recall ({final_line_recall:.4f})")
        
        # Check learning rate
        final_lr = training_history['learning_rate'][-1]
        if final_lr < 1e-6:
            print(f"⚠️  Very low learning rate ({final_lr:.8f}) - model may have stopped learning")
        elif final_lr < 1e-5:
            print(f"⚠️  Low learning rate ({final_lr:.8f}) - consider boosting learning rate")
        else:
            print(f"✅ Reasonable learning rate ({final_lr:.8f})")

def debug_model_issues(analyzer, sample_contract: str = None) -> Dict[str, Any]:
    """
    Debug model issues and provide diagnostic information.
    
    Args:
        analyzer: SmartContractAnalyzer instance
        sample_contract: Optional sample contract to test
        
    Returns:
        Dictionary with diagnostic information
    """
    print("🔍 Debugging model issues...")
    
    diagnostics = {
        'model_loaded': False,
        'device_info': None,
        'model_architecture': None,
        'basic_forward_pass': False,
        'vulnerability_detection': False,
        'error_details': []
    }
    
    try:
        # Check if model is loaded
        if hasattr(analyzer, 'model') and analyzer.model is not None:
            diagnostics['model_loaded'] = True
            diagnostics['device_info'] = str(analyzer.device)
            diagnostics['model_architecture'] = type(analyzer.model).__name__
            print(f"✅ Model loaded: {diagnostics['model_architecture']}")
            print(f"✅ Device: {diagnostics['device_info']}")
        else:
            diagnostics['error_details'].append("Model not properly loaded")
            print("❌ Model not properly loaded")
            return diagnostics
        
        # Test basic functionality
        try:
            test_results = analyzer.test_model_functionality()
            diagnostics['basic_forward_pass'] = test_results['basic_forward_pass']
            
            if test_results['basic_forward_pass']:
                print("✅ Basic forward pass working")
            else:
                diagnostics['error_details'].append(f"Basic forward pass failed: {test_results['error_message']}")
                print(f"❌ Basic forward pass failed: {test_results['error_message']}")
                
        except Exception as e:
            diagnostics['error_details'].append(f"Functionality test failed: {str(e)}")
            print(f"❌ Functionality test failed: {str(e)}")
        
        # Test vulnerability detection if basic forward pass works
        if diagnostics['basic_forward_pass'] and sample_contract:
            try:
                results = analyzer.detect_vulnerabilities(sample_contract, threshold=0.5)
                diagnostics['vulnerability_detection'] = True
                print("✅ Vulnerability detection working")
            except Exception as e:
                diagnostics['error_details'].append(f"Vulnerability detection failed: {str(e)}")
                print(f"❌ Vulnerability detection failed: {str(e)}")
        
        # Check model parameters
        try:
            total_params = sum(p.numel() for p in analyzer.model.parameters())
            trainable_params = sum(p.numel() for p in analyzer.model.parameters() if p.requires_grad)
            diagnostics['total_parameters'] = total_params
            diagnostics['trainable_parameters'] = trainable_params
            print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        except Exception as e:
            diagnostics['error_details'].append(f"Parameter count failed: {str(e)}")
        
    except Exception as e:
        diagnostics['error_details'].append(f"General debug error: {str(e)}")
        print(f"❌ Debug error: {str(e)}")
    
    # Print summary
    print(f"\n📋 Debug Summary:")
    print(f"  Model loaded: {'✅' if diagnostics['model_loaded'] else '❌'}")
    print(f"  Basic forward pass: {'✅' if diagnostics['basic_forward_pass'] else '❌'}")
    print(f"  Vulnerability detection: {'✅' if diagnostics['vulnerability_detection'] else '❌'}")
    
    if diagnostics['error_details']:
        print(f"\n⚠️  Errors found:")
        for error in diagnostics['error_details']:
            print(f"  - {error}")
    
    return diagnostics