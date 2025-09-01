import re
import random
import ast
from typing import List, Tuple, Dict, Any
import torch
from transformers import AutoTokenizer

class SmartContractAugmenter:
    def __init__(self, tokenizer_name: str = "microsoft/codebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Common Solidity transformations
        self.transformations = [
            self._change_variable_names,
            self._change_function_names,
            self._change_visibility_modifiers,
            self._change_pragma_version,
            self._add_comments,
            self._change_data_types,
            self._reorder_functions,
            self._add_modifiers,
            self._change_parameter_names,
            self._add_events
        ]
        
        # Variable name mappings
        self.variable_mappings = {
            'value': ['data', 'state', 'storage', 'amount', 'balance', 'total'],
            'balance': ['amount', 'value', 'total', 'sum', 'funds'],
            'owner': ['admin', 'manager', 'controller', 'authority'],
            'user': ['account', 'address', 'participant', 'member'],
            'token': ['coin', 'asset', 'currency', 'unit'],
            'price': ['cost', 'rate', 'fee', 'amount'],
            'time': ['duration', 'period', 'interval', 'deadline']
        }
        
        # Function name mappings
        self.function_mappings = {
            'setValue': ['setData', 'setState', 'setStorage', 'setAmount', 'setBalance'],
            'getValue': ['getData', 'getState', 'getStorage', 'getAmount', 'getBalance'],
            'transfer': ['send', 'move', 'dispatch', 'forward'],
            'withdraw': ['extract', 'pull', 'remove', 'claim'],
            'deposit': ['add', 'put', 'store', 'save'],
            'mint': ['create', 'generate', 'produce', 'issue'],
            'burn': ['destroy', 'remove', 'eliminate', 'consume']
        }
        
        # Visibility modifier mappings
        self.visibility_mappings = {
            'public': ['external', 'public'],
            'private': ['internal', 'private'],
            'internal': ['private', 'internal'],
            'external': ['public', 'external']
        }
        
        # Data type mappings
        self.datatype_mappings = {
            'uint256': ['uint128', 'uint64', 'uint32'],
            'uint128': ['uint256', 'uint64', 'uint32'],
            'address': ['address payable', 'address'],
            'string': ['bytes', 'string'],
            'bool': ['uint8', 'bool']
        }
    
    def _change_variable_names(self, code: str) -> str:
        """Change variable names in the contract"""
        modified_code = code
        
        # Apply variable name changes
        for old_name, new_names in self.variable_mappings.items():
            if random.random() < 0.3:  # 30% chance to change each variable type
                new_name = random.choice(new_names)
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(old_name) + r'\b'
                modified_code = re.sub(pattern, new_name, modified_code)
        
        return modified_code
    
    def _change_function_names(self, code: str) -> str:
        """Change function names in the contract"""
        modified_code = code
        
        for old_name, new_names in self.function_mappings.items():
            if random.random() < 0.4:  # 40% chance to change each function type
                new_name = random.choice(new_names)
                pattern = r'\b' + re.escape(old_name) + r'\b'
                modified_code = re.sub(pattern, new_name, modified_code)
        
        return modified_code
    
    def _change_visibility_modifiers(self, code: str) -> str:
        """Change visibility modifiers"""
        modified_code = code
        
        for old_mod, new_mods in self.visibility_mappings.items():
            if random.random() < 0.2:  # 20% chance to change each modifier
                new_mod = random.choice(new_mods)
                pattern = r'\b' + re.escape(old_mod) + r'\b'
                modified_code = re.sub(pattern, new_mod, modified_code)
        
        return modified_code
    
    def _change_pragma_version(self, code: str) -> str:
        """Change Solidity version"""
        # Generate random version
        major = random.randint(4, 8)
        minor = random.randint(0, 20)
        patch = random.randint(0, 20)
        
        new_version = f"^{major}.{minor}.{patch}"
        modified_code = re.sub(r'pragma solidity \^?0\.\d+\.\d+', f'pragma solidity {new_version}', code)
        
        return modified_code
    
    def _add_comments(self, code: str) -> str:
        """Add random comments to the code"""
        lines = code.split('\n')
        modified_lines = []
        
        for line in lines:
            modified_lines.append(line)
            # 10% chance to add a comment after a line
            if random.random() < 0.1 and line.strip() and not line.strip().startswith('//'):
                comments = [
                    '// This function handles the main logic',
                    '// Ensure proper validation',
                    '// Update state variables',
                    '// Check access control',
                    '// Emit events for transparency',
                    '// Handle edge cases',
                    '// Optimize gas usage'
                ]
                comment = random.choice(comments)
                modified_lines.append(comment)
        
        return '\n'.join(modified_lines)
    
    def _change_data_types(self, code: str) -> str:
        """Change data types in variable declarations"""
        modified_code = code
        
        for old_type, new_types in self.datatype_mappings.items():
            if random.random() < 0.15:  # 15% chance to change each type
                new_type = random.choice(new_types)
                pattern = r'\b' + re.escape(old_type) + r'\b'
                modified_code = re.sub(pattern, new_type, modified_code)
        
        return modified_code
    
    def _reorder_functions(self, code: str) -> str:
        """Reorder functions in the contract"""
        # Extract contract structure
        contract_match = re.search(r'(contract\s+\w+\s*\{)(.*?)(\})', code, re.DOTALL)
        if not contract_match:
            return code
        
        contract_start = contract_match.group(1)
        contract_body = contract_match.group(2)
        contract_end = contract_match.group(3)
        
        # Split into functions and other elements
        lines = contract_body.split('\n')
        functions = []
        other_lines = []
        current_function = []
        in_function = False
        
        for line in lines:
            if re.match(r'\s*function\s+', line):
                if current_function:
                    functions.append('\n'.join(current_function))
                current_function = [line]
                in_function = True
            elif in_function:
                current_function.append(line)
                if line.strip() == '}':
                    in_function = False
            else:
                other_lines.append(line)
        
        if current_function:
            functions.append('\n'.join(current_function))
        
        # Randomly reorder functions
        if len(functions) > 1 and random.random() < 0.3:
            random.shuffle(functions)
        
        # Reconstruct contract
        new_body = '\n'.join(other_lines) + '\n' + '\n'.join(functions)
        modified_code = contract_start + new_body + contract_end
        
        return modified_code
    
    def _add_modifiers(self, code: str) -> str:
        """Add modifiers to functions"""
        modifiers = [
            'onlyOwner',
            'whenNotPaused',
            'nonReentrant',
            'validAddress',
            'positiveAmount'
        ]
        
        modified_code = code
        
        # Find function declarations and add modifiers
        function_pattern = r'(function\s+\w+\s*\([^)]*\)\s*)(public|private|internal|external)?'
        
        def add_modifier(match):
            func_decl = match.group(1)
            visibility = match.group(2) or ''
            
            if random.random() < 0.2:  # 20% chance to add modifier
                modifier = random.choice(modifiers)
                return func_decl + modifier + ' ' + visibility
            
            return match.group(0)
        
        modified_code = re.sub(function_pattern, add_modifier, modified_code)
        
        return modified_code
    
    def _change_parameter_names(self, code: str) -> str:
        """Change parameter names in functions"""
        modified_code = code
        
        # Common parameter patterns
        param_patterns = [
            (r'_value', ['_data', '_amount', '_input', '_param']),
            (r'_address', ['_account', '_user', '_target', '_recipient']),
            (r'_amount', ['_value', '_quantity', '_sum', '_total']),
            (r'_owner', ['_admin', '_manager', '_controller'])
        ]
        
        for old_param, new_params in param_patterns:
            if random.random() < 0.25:  # 25% chance to change each parameter type
                new_param = random.choice(new_params)
                pattern = r'\b' + re.escape(old_param) + r'\b'
                modified_code = re.sub(pattern, new_param, modified_code)
        
        return modified_code
    
    def _add_events(self, code: str) -> str:
        """Add events to the contract"""
        events = [
            'event ValueSet(address indexed user, uint256 value);',
            'event Transfer(address indexed from, address indexed to, uint256 amount);',
            'event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);',
            'event Paused(address indexed account);',
            'event Unpaused(address indexed account);'
        ]
        
        # Find contract opening and add events
        contract_match = re.search(r'(contract\s+\w+\s*\{)', code)
        if contract_match and random.random() < 0.3:  # 30% chance to add events
            event = random.choice(events)
            modified_code = code.replace(contract_match.group(1), 
                                       contract_match.group(1) + '\n    ' + event)
        else:
            modified_code = code
        
        return modified_code
    
    def augment_contract(self, original_contract: str, num_variants: int = 3) -> List[Tuple[str, str]]:
        """
        Create augmented training pairs from an original contract.
        
        Args:
            original_contract: The original smart contract code
            num_variants: Number of variants to create
            
        Returns:
            List of (input_contract, target_contract) pairs
        """
        training_pairs = []
        
        for i in range(num_variants):
            # Apply 2-4 random transformations
            num_transforms = random.randint(2, 4)
            selected_transforms = random.sample(self.transformations, num_transforms)
            
            # Create variant
            variant = original_contract
            for transform in selected_transforms:
                variant = transform(variant)
            
            # Ensure the variant is different from original
            if variant.strip() != original_contract.strip():
                training_pairs.append((original_contract, variant))
        
        return training_pairs
    
    def create_training_batch(self, contracts: List[str], batch_size: int = 4) -> Dict[str, torch.Tensor]:
        """
        Create a training batch with augmented data.
        
        Args:
            contracts: List of original contracts
            batch_size: Size of the batch
            
        Returns:
            Dictionary containing tokenized inputs and targets
        """
        all_pairs = []
        
        # Create augmented pairs for each contract
        for contract in contracts:
            pairs = self.augment_contract(contract, num_variants=2)
            all_pairs.extend(pairs)
        
        # Randomly sample pairs for the batch
        if len(all_pairs) > batch_size:
            selected_pairs = random.sample(all_pairs, batch_size)
        else:
            selected_pairs = all_pairs
        
        # Tokenize inputs and targets
        input_texts = [pair[0] for pair in selected_pairs]
        target_texts = [pair[1] for pair in selected_pairs]
        
        # Tokenize inputs
        input_encodings = self.tokenizer(
            input_texts,
            max_length=1024,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize targets
        target_encodings = self.tokenizer(
            target_texts,
            max_length=1024,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'target_ids': target_encodings['input_ids'],  # Use input_ids as target_ids
            'target_attention_mask': target_encodings['attention_mask']
        }

def test_augmentation():
    """Test the augmentation functionality"""
    augmenter = SmartContractAugmenter()
    
    test_contract = """
    pragma solidity ^0.8.0;
    
    contract Example {
        uint256 public value;
        address public owner;
        
        event ValueSet(address indexed user, uint256 value);
        
        constructor() {
            owner = msg.sender;
        }
        
        function setValue(uint256 _value) public {
            require(msg.sender == owner, "Not authorized");
            value = _value;
            emit ValueSet(msg.sender, _value);
        }
        
        function getValue() public view returns (uint256) {
            return value;
        }
        
        function transferOwnership(address _newOwner) public {
            require(msg.sender == owner, "Not authorized");
            owner = _newOwner;
        }
    }
    """
    
    print("Original Contract:")
    print(test_contract)
    print("\n" + "="*50 + "\n")
    
    # Create augmented variants
    pairs = augmenter.augment_contract(test_contract, num_variants=3)
    
    for i, (input_contract, target_contract) in enumerate(pairs):
        print(f"Training Pair {i+1}:")
        print("Input (Original):")
        print(input_contract[:200] + "..." if len(input_contract) > 200 else input_contract)
        print("\nTarget (Augmented):")
        print(target_contract[:200] + "..." if len(target_contract) > 200 else target_contract)
        print("\n" + "-"*30 + "\n")

if __name__ == "__main__":
    test_augmentation() 