#!/usr/bin/env python3
"""
Smart Contract Analysis and Generation Script

This script demonstrates how to use the trained model to:
1. Detect vulnerabilities in smart contracts
2. Generate synthetic smart contracts
"""

import torch
from transformers import AutoTokenizer
from model import SmartContractTransformer
import json
import re
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from inference import SmartContractAnalyzer

def main():
    # Initialize the analyzer with your trained model
    print("Initializing SmartContractAnalyzer...")
    analyzer = SmartContractAnalyzer(
        model_path="checkpoints/best_model.pt",  # Path to your saved model
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Analyzer initialized successfully!")

    # Example contract to analyze
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

    # 1. Detect vulnerabilities
    print("\n=== Vulnerability Detection ===")
    print("Analyzing contract for vulnerabilities...")
    vulnerabilities = analyzer.detect_vulnerabilities(contract_code)
    print("\nVulnerability Analysis Results:")
    print(json.dumps(vulnerabilities, indent=2))

    # 2. Generate synthetic contracts
    print("\n=== Synthetic Contract Generation ===")
    print("Generating synthetic contracts...")
    synthetic_contracts = analyzer.generate_synthetic_contract(
        contract_template=contract_code,
        num_contracts=2,  # Generate 2 variations
        temperature=0.8   # Control randomness (0.0 to 1.0)
    )

    print("\nGenerated Contracts:")
    for i, contract in enumerate(synthetic_contracts, 1):
        print(f"\nContract {i}:")
        print(contract)

    # 3. Analyze generated contracts
    print("\n=== Analysis of Generated Contracts ===")
    for i, contract in enumerate(synthetic_contracts, 1):
        print(f"\nAnalyzing Generated Contract {i}:")
        vulnerabilities = analyzer.detect_vulnerabilities(contract)
        print(json.dumps(vulnerabilities, indent=2))

def analyze_custom_contract(contract_code: str, model_path: str = "checkpoints/best_model.pt"):
    """
    Analyze a custom smart contract for vulnerabilities.
    
    Args:
        contract_code: The Solidity contract code to analyze
        model_path: Path to the trained model checkpoint
    """
    print("Initializing SmartContractAnalyzer...")
    analyzer = SmartContractAnalyzer(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Analyzer initialized successfully!")

    print("\n=== Vulnerability Detection ===")
    print("Analyzing contract for vulnerabilities...")
    vulnerabilities = analyzer.detect_vulnerabilities(contract_code)
    print("\nVulnerability Analysis Results:")
    print(json.dumps(vulnerabilities, indent=2))

def generate_from_template(template_code: str, num_contracts: int = 2, temperature: float = 0.8,
                         model_path: str = "checkpoints/best_model.pt"):
    """
    Generate synthetic contracts from a template.
    
    Args:
        template_code: The template contract code
        num_contracts: Number of contracts to generate
        temperature: Sampling temperature (0.0 to 1.0)
        model_path: Path to the trained model checkpoint
    """
    print("Initializing SmartContractAnalyzer...")
    analyzer = SmartContractAnalyzer(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Analyzer initialized successfully!")

    print("\n=== Synthetic Contract Generation ===")
    print(f"Generating {num_contracts} synthetic contracts...")
    synthetic_contracts = analyzer.generate_synthetic_contract(
        contract_template=template_code,
        num_contracts=num_contracts,
        temperature=temperature
    )

    print("\nGenerated Contracts:")
    for i, contract in enumerate(synthetic_contracts, 1):
        print(f"\nContract {i}:")
        print(contract)

    return synthetic_contracts

if __name__ == "__main__":
    # Run the main example
    main()

    # Example of how to use the utility functions
    print("\n=== Example of Using Utility Functions ===")
    
    # Example custom contract
    custom_contract = """
    pragma solidity ^0.8.0;
    
    contract CustomContract {
        mapping(address => uint256) public balances;
        
        function deposit() public payable {
            balances[msg.sender] += msg.value;
        }
        
        function withdraw(uint256 amount) public {
            require(balances[msg.sender] >= amount, "Insufficient balance");
            balances[msg.sender] -= amount;
            payable(msg.sender).transfer(amount);
        }
    }
    """
    
    # Analyze custom contract
    print("\nAnalyzing custom contract...")
    analyze_custom_contract(custom_contract)
    
    # Generate from template
    print("\nGenerating from custom template...")
    generated = generate_from_template(
        template_code=custom_contract,
        num_contracts=2,
        temperature=0.8
    ) 