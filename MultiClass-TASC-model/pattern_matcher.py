import re
from typing import Dict, List, Tuple, Optional
import torch

class PatternMatcher:
    def __init__(self):
        # Define vulnerability patterns
        self.patterns = {
            'timestamp_dependence': {
                'TDInvocation': [
                    r'block\.timestamp',
                    r'now\s*[=<>]',
                    r'block\.timestamp\s*[=<>]'
                ],
                'TDAssign': [
                    r'block\.timestamp\s*=\s*[^;]+',
                    r'now\s*=\s*[^;]+'
                ],
                'TDContaminate': [
                    r'block\.timestamp\s*[+\-*/]\s*[^;]+',
                    r'now\s*[+\-*/]\s*[^;]+'
                ]
            },
            'reentrancy': {
                'callValueInvocation': [
                    r'\.call\s*\(\s*[^)]*value\s*:',
                    r'\.send\s*\(',
                    r'\.transfer\s*\('
                ],
                'balanceDeduction': [
                    r'balance\s*-=\s*[^;]+',
                    r'balance\s*=\s*balance\s*-\s*[^;]+'
                ],
                'zeroParameter': [
                    r'require\s*\(\s*[^)]*==\s*0\s*\)',
                    r'if\s*\(\s*[^)]*==\s*0\s*\)'
                ],
                'ModifierConstrain': [
                    r'modifier\s+\w+\s*{[^}]*require\s*\([^)]*\)[^}]*}'
                ]
            },
            'integer_overflow': {
                'arithmeticOperation': [
                    r'[a-zA-Z_]\w*\s*[+\-*/]\s*[^;]+',
                    r'[0-9]+\s*[+\-*/]\s*[^;]+'
                ],
                'safeLibraryInvocation': [
                    r'SafeMath\.(add|sub|mul|div)',
                    r'using\s+SafeMath\s+for\s+uint'
                ],
                'conditionDeclaration': [
                    r'require\s*\(\s*[^)]*[<>]=?\s*[^)]*\)',
                    r'if\s*\(\s*[^)]*[<>]=?\s*[^)]*\)'
                ]
            },
            'dangerous_delegatecall': {
                'delegateInvocation': [
                    r'\.delegatecall\s*\(',
                    r'\.call\s*\(\s*[^)]*delegate\s*:'
                ],
                'ownerInvocation': [
                    r'require\s*\(\s*msg\.sender\s*==\s*owner\s*\)',
                    r'modifier\s+\w+\s*{[^}]*require\s*\(\s*msg\.sender\s*==\s*owner\s*\)[^}]*}'
                ]
            }
        }
        
        # Compile regex patterns
        self.compiled_patterns = {
            vuln_type: {
                pattern_name: [re.compile(pattern) for pattern in patterns]
                for pattern_name, patterns in pattern_dict.items()
            }
            for vuln_type, pattern_dict in self.patterns.items()
        }
    
    def match_patterns(self, code: str) -> Dict[str, Dict[str, List[Tuple[int, str]]]]:
        """
        Match vulnerability patterns in the given code.
        Returns a dictionary of matched patterns with their line numbers and matched text.
        """
        matches = {
            vuln_type: {
                pattern_name: []
                for pattern_name in pattern_dict.keys()
            }
            for vuln_type, pattern_dict in self.patterns.items()
        }
        
        # Split code into lines
        lines = code.split('\n')
        
        # Match patterns for each line
        for line_num, line in enumerate(lines, 1):
            for vuln_type, pattern_dict in self.compiled_patterns.items():
                for pattern_name, patterns in pattern_dict.items():
                    for pattern in patterns:
                        for match in pattern.finditer(line):
                            matches[vuln_type][pattern_name].append(
                                (line_num, match.group())
                            )
        
        return matches
    
    def get_pattern_scores(self, code: str) -> Dict[str, Dict[str, float]]:
        """
        Get pattern scores for the given code.
        Returns a dictionary of pattern scores for each vulnerability type.
        """
        matches = self.match_patterns(code)
        
        scores = {
            vuln_type: {
                pattern_name: min(1.0, len(pattern_matches) * 0.5)
                for pattern_name, pattern_matches in pattern_dict.items()
            }
            for vuln_type, pattern_dict in matches.items()
        }
        
        return scores
    
    def get_vulnerability_explanation(self, code: str) -> List[str]:
        """
        Get explanations for detected vulnerabilities in the code.
        """
        matches = self.match_patterns(code)
        explanations = []
        
        # Timestamp Dependence
        if matches['timestamp_dependence']['TDInvocation']:
            if matches['timestamp_dependence']['TDAssign'] or matches['timestamp_dependence']['TDContaminate']:
                explanations.append(
                    "Timestamp Dependence: Block timestamp is used in critical operations. "
                    "Found at lines: " + 
                    ", ".join(str(line) for line, _ in matches['timestamp_dependence']['TDInvocation'])
                )
        
        # Reentrancy
        if matches['reentrancy']['callValueInvocation']:
            if (matches['reentrancy']['balanceDeduction'] and 
                matches['reentrancy']['zeroParameter'] and 
                not matches['reentrancy']['ModifierConstrain']):
                explanations.append(
                    "Reentrancy: Unsafe call.value usage without proper balance deduction. "
                    "Found at lines: " + 
                    ", ".join(str(line) for line, _ in matches['reentrancy']['callValueInvocation'])
                )
        
        # Integer Overflow
        if matches['integer_overflow']['arithmeticOperation']:
            if (not matches['integer_overflow']['safeLibraryInvocation'] and 
                not matches['integer_overflow']['conditionDeclaration']):
                explanations.append(
                    "Integer Overflow: Arithmetic operations without safety checks. "
                    "Found at lines: " + 
                    ", ".join(str(line) for line, _ in matches['integer_overflow']['arithmeticOperation'])
                )
        
        # Dangerous Delegatecall
        if matches['dangerous_delegatecall']['delegateInvocation']:
            if not matches['dangerous_delegatecall']['ownerInvocation']:
                explanations.append(
                    "Dangerous Delegatecall: Unauthorized delegatecall usage. "
                    "Found at lines: " + 
                    ", ".join(str(line) for line, _ in matches['dangerous_delegatecall']['delegateInvocation'])
                )
        
        return explanations
    
    def get_pattern_tensor(self, code: str) -> Dict[str, torch.Tensor]:
        """
        Convert pattern matches to tensor format for model training.
        """
        scores = self.get_pattern_scores(code)
        
        # Convert scores to tensors
        pattern_tensors = {
            'timestamp_dependence': torch.tensor([
                scores['timestamp_dependence']['TDInvocation'],
                scores['timestamp_dependence']['TDAssign'],
                scores['timestamp_dependence']['TDContaminate']
            ]),
            'reentrancy': torch.tensor([
                scores['reentrancy']['callValueInvocation'],
                scores['reentrancy']['balanceDeduction'],
                scores['reentrancy']['zeroParameter'],
                scores['reentrancy']['ModifierConstrain']
            ]),
            'integer_overflow': torch.tensor([
                scores['integer_overflow']['arithmeticOperation'],
                scores['integer_overflow']['safeLibraryInvocation'],
                scores['integer_overflow']['conditionDeclaration']
            ]),
            'dangerous_delegatecall': torch.tensor([
                scores['dangerous_delegatecall']['delegateInvocation'],
                scores['dangerous_delegatecall']['ownerInvocation']
            ])
        }
        
        return pattern_tensors