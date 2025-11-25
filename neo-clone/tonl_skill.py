"""
TONL Skill for Neo-Clone
Provides TONL (Token-Optimized Notation Language) encoding and decoding capabilities
"""

import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from tonl_neo_clone import TONLNeoClone, TONLEncodeOptions, TONLDelimiter

# Import BaseSkill from skills framework
try:
    from skills import BaseSkill, SkillMetadata, SkillCategory, SkillParameter, SkillParameterType
    BASESKILL_AVAILABLE = True
except ImportError:
    # Fallback definitions if skills module not available
    BASESKILL_AVAILABLE = False
    class BaseSkill(ABC):
        def __init__(self):
            pass
    
    class SkillMetadata:
        def __init__(self, name, category, description, capabilities=None):
            self.name = name
            self.category = category
            self.description = description
            self.capabilities = capabilities or []
    
    class SkillCategory:
        GENERAL = "general"
        DATA_PROCESSING = "data_processing"
    
    class SkillParameter:
        pass
    
    class SkillParameterType:
        STRING = "string"
        INTEGER = "integer"
        BOOLEAN = "boolean"
        OBJECT = "object"
        ARRAY = "array"


@dataclass
class TONLSkillResult:
    """Result of TONL skill operation"""
    success: bool
    data: Any
    stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tonl_output: Optional[str] = None


class TONLSkill(BaseSkill):
    """
    TONL Skill for Neo-Clone
    Provides efficient data encoding/decoding for LLM token optimization
    """
    """
    TONL Skill for Neo-Clone
    Provides efficient data encoding/decoding for LLM token optimization
    """
    
    def __init__(self):
        if BASESKILL_AVAILABLE:
            super().__init__()
        self.tonl = TONLNeoClone()
        self._name = "TONL Encoder/Decoder"
        self._version = "1.0.0"
        
        # Set up metadata for BaseSkill compatibility
        if BASESKILL_AVAILABLE and hasattr(self, 'metadata'):
            # Note: metadata is set up in BaseSkill.__init__, we don't need to modify it
            pass
    
    @property
    def name(self) -> str:
        """Get skill name"""
        return self._name if hasattr(self, '_name') else "TONL Encoder/Decoder"
    
    @property
    def version(self) -> str:
        """Get skill version"""
        return self._version if hasattr(self, '_version') else "1.0.0"
    
    def encode_data(self, data: Any, options: Optional[Dict[str, Any]] = None) -> TONLSkillResult:
        """
        Encode data to TONL format
        
        Args:
            data: Python data to encode (dict, list, etc.)
            options: Encoding options
                - delimiter: ',' | '|' | ';' | '\\t' (default: ',')
                - include_types: bool (default: False)
                - compact: bool (default: True)
        
        Returns:
            TONLSkillResult with encoded data and statistics
        """
        try:
            # Parse options
            encode_options = self._parse_encode_options(options)
            
            # Encode data
            tonl_output, stats = self.tonl.encode(data, encode_options)
            
            # Prepare result
            result_stats = {
                'original_bytes': stats.original_size,
                'tonl_bytes': stats.compressed_size,
                'compression_ratio': stats.compression_ratio,
                'space_saved_percent': (1 - stats.compression_ratio) * 100,
                'estimated_tokens': stats.token_estimate,
                'processing_time_ms': stats.processing_time * 1000
            }
            
            return TONLSkillResult(
                success=True,
                data=data,
                stats=result_stats,
                tonl_output=tonl_output
            )
            
        except Exception as e:
            return TONLSkillResult(
                success=False,
                data=None,
                error=f"Encoding failed: {str(e)}"
            )
    
    def decode_data(self, tonl_text: str, strict: bool = False) -> TONLSkillResult:
        """
        Decode TONL format to Python data
        
        Args:
            tonl_text: TONL formatted string
            strict: Enable strict parsing mode
        
        Returns:
            TONLSkillResult with decoded data
        """
        try:
            decoded_data = self.tonl.decode(tonl_text, strict)
            
            return TONLSkillResult(
                success=True,
                data=decoded_data
            )
            
        except Exception as e:
            return TONLSkillResult(
                success=False,
                data=None,
                error=f"Decoding failed: {str(e)}"
            )
    
    def compress_json(self, json_str: str, options: Optional[Dict[str, Any]] = None) -> TONLSkillResult:
        """
        Compress JSON string to TONL format
        
        Args:
            json_str: JSON string to compress
            options: Encoding options
        
        Returns:
            TONLSkillResult with compressed TONL and statistics
        """
        try:
            # Parse JSON
            data = json.loads(json_str)
            
            # Encode to TONL
            result = self.encode_data(data, options)
            
            if result.success:
                result.stats['original_json_length'] = len(json_str)
                result.stats['compression_from_json'] = len(result.tonl_output) / len(json_str)
            
            return result
            
        except json.JSONDecodeError as e:
            return TONLSkillResult(
                success=False,
                data=None,
                error=f"Invalid JSON: {str(e)}"
            )
        except Exception as e:
            return TONLSkillResult(
                success=False,
                data=None,
                error=f"Compression failed: {str(e)}"
            )
    
    def analyze_compression(self, data: Any) -> TONLSkillResult:
        """
        Analyze compression potential for given data
        
        Args:
            data: Python data to analyze
        
        Returns:
            TONLSkillResult with detailed compression analysis
        """
        try:
            # Get basic stats
            stats = self.tonl.get_stats(data)
            
            # Analyze data structure
            analysis = self._analyze_data_structure(data)
            
            # Combine results
            combined_stats = {
                **stats,
                'structure_analysis': analysis,
                'recommendations': self._get_compression_recommendations(data, stats)
            }
            
            return TONLSkillResult(
                success=True,
                data=data,
                stats=combined_stats
            )
            
        except Exception as e:
            return TONLSkillResult(
                success=False,
                data=None,
                error=f"Analysis failed: {str(e)}"
            )
    
    def validate_tonl(self, tonl_text: str) -> TONLSkillResult:
        """
        Validate TONL format
        
        Args:
            tonl_text: TONL text to validate
        
        Returns:
            TONLSkillResult with validation results
        """
        try:
            validation = self.tonl.validate(tonl_text)
            
            return TONLSkillResult(
                success=validation['valid'],
                data=validation,
                error=None if validation['valid'] else f"Validation failed: {', '.join(validation['errors'])}"
            )
            
        except Exception as e:
            return TONLSkillResult(
                success=False,
                data=None,
                error=f"Validation error: {str(e)}"
            )
    
    def convert_format(self, input_data: Union[str, Dict, List], 
                      input_format: str = 'auto',
                      output_format: str = 'tonl',
                      options: Optional[Dict[str, Any]] = None) -> TONLSkillResult:
        """
        Convert between different data formats
        
        Args:
            input_data: Input data (string, dict, or list)
            input_format: Input format ('json', 'tonl', 'auto')
            output_format: Output format ('json', 'tonl', 'python')
            options: Conversion options
        
        Returns:
            TONLSkillResult with converted data
        """
        try:
            # Auto-detect input format
            if input_format == 'auto':
                if isinstance(input_data, str):
                    if input_data.strip().startswith('#version') or ':' in input_data:
                        input_format = 'tonl'
                    else:
                        input_format = 'json'
                else:
                    input_format = 'python'
            
            # Parse input
            if input_format == 'json':
                if isinstance(input_data, str):
                    parsed_data = json.loads(input_data)
                else:
                    parsed_data = input_data
            elif input_format == 'tonl':
                if isinstance(input_data, str):
                    parsed_data = self.tonl.decode(input_data)
                else:
                    raise ValueError("TONL input must be a string")
            else:  # python
                parsed_data = input_data
            
            # Convert to output format
            if output_format == 'json':
                output = json.dumps(parsed_data, indent=2)
            elif output_format == 'tonl':
                encode_options = self._parse_encode_options(options)
                output, _ = self.tonl.encode(parsed_data, encode_options)
            else:  # python
                output = parsed_data
            
            return TONLSkillResult(
                success=True,
                data=output
            )
            
        except Exception as e:
            return TONLSkillResult(
                success=False,
                data=None,
                error=f"Conversion failed: {str(e)}"
            )
    
    def _parse_encode_options(self, options: Optional[Dict[str, Any]]) -> TONLEncodeOptions:
        """Parse encoding options from dict"""
        if not options:
            return TONLEncodeOptions()
        
        # Parse delimiter
        delimiter = TONLDelimiter.COMMA
        if 'delimiter' in options:
            delim_str = options['delimiter']
            if delim_str == '|':
                delimiter = TONLDelimiter.PIPE
            elif delim_str == ';':
                delimiter = TONLDelimiter.SEMICOLON
            elif delim_str == '\\t' or delim_str == '\t':
                delimiter = TONLDelimiter.TAB
        
        return TONLEncodeOptions(
            delimiter=delimiter,
            include_types=options.get('include_types', False),
            compact=options.get('compact', True),
            version=options.get('version', '1.0')
        )
    
    def _analyze_data_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze data structure for compression potential"""
        analysis = {
            'data_type': type(data).__name__,
            'total_elements': 0,
            'nested_depth': 0,
            'array_count': 0,
            'object_count': 0,
            'uniform_arrays': 0,
            'string_fields': [],
            'numeric_fields': []
        }
        
        def analyze_recursive(obj, depth=0):
            analysis['nested_depth'] = max(analysis['nested_depth'], depth)
            
            if isinstance(obj, dict):
                analysis['object_count'] += 1
                analysis['total_elements'] += len(obj)
                
                for key, value in obj.items():
                    if isinstance(value, str):
                        analysis['string_fields'].append(key)
                    elif isinstance(value, (int, float)):
                        analysis['numeric_fields'].append(key)
                    elif isinstance(value, list):
                        analysis['array_count'] += 1
                        if value and all(isinstance(item, dict) for item in value):
                            analysis['uniform_arrays'] += 1
                    analyze_recursive(value, depth + 1)
                    
            elif isinstance(obj, list):
                analysis['array_count'] += 1
                analysis['total_elements'] += len(obj)
                
                if obj and all(isinstance(item, dict) for item in obj):
                    analysis['uniform_arrays'] += 1
                
                for item in obj:
                    analyze_recursive(item, depth + 1)
        
        analyze_recursive(data)
        
        return analysis
    
    def _get_compression_recommendations(self, data: Any, stats: Dict[str, Any]) -> List[str]:
        """Get compression recommendations based on data analysis"""
        recommendations = []
        
        if stats['compression_ratio'] > 0.9:
            recommendations.append("Low compression achieved - consider using tabular format for uniform object arrays")
        
        if stats['compression_ratio'] < 0.5:
            recommendations.append("Excellent compression achieved - TONL is highly effective for this data")
        
        # Check for potential optimizations
        analysis = self._analyze_data_structure(data)
        
        if analysis['uniform_arrays'] > 0:
            recommendations.append(f"Found {analysis['uniform_arrays']} uniform object arrays - perfect for tabular TONL format")
        
        if analysis['nested_depth'] > 5:
            recommendations.append("Deep nesting detected - consider flattening structure for better compression")
        
        if len(analysis['string_fields']) > len(analysis['numeric_fields']):
            recommendations.append("String-heavy data - TONL will provide good token savings")
        
        return recommendations
    
    def get_skill_info(self) -> Dict[str, Any]:
        """Get skill information"""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'capabilities': [
                'encode_data',
                'decode_data', 
                'compress_json',
                'analyze_compression',
                'validate_tonl',
                'convert_format'
            ],
            'supported_delimiters': [d.value for d in TONLDelimiter],
            'benefits': [
                'Reduced token usage for LLM processing',
                'Human-readable format',
                'Tabular array optimization',
                'Type preservation',
                'Round-trip compatibility'
            ]
        }
    
    def get_parameters(self) -> List[SkillParameter]:
        """Get skill parameters for BaseSkill compatibility"""
        if not BASESKILL_AVAILABLE:
            return []
        
        return [
            SkillParameter(
                name="data",
                param_type=SkillParameterType.OBJECT,
                required=True,
                description="Data to encode/decode"
            ),
            SkillParameter(
                name="options",
                param_type=SkillParameterType.OBJECT,
                required=False,
                description="Encoding options (delimiter, include_types, etc.)"
            ),
            SkillParameter(
                name="operation",
                param_type=SkillParameterType.STRING,
                required=False,
                default="encode",
                choices=["encode", "decode", "compress", "analyze", "validate"],
                description="Operation to perform"
            )
        ]
    
    async def _execute_async(self, **kwargs) -> Dict[str, Any]:
        """Execute skill asynchronously for BaseSkill compatibility"""
        operation = kwargs.get('operation', 'encode')
        data = kwargs.get('data')
        options = kwargs.get('options', {})
        
        try:
            if operation == 'encode':
                result = self.encode_data(data, options)
                return {
                    'success': result.success,
                    'result': result.tonl_output if result.success else None,
                    'stats': result.stats,
                    'error': result.error
                }
            elif operation == 'decode':
                tonl_text = kwargs.get('tonl_text', data)
                result = self.decode_data(tonl_text)
                return {
                    'success': result.success,
                    'result': result.data if result.success else None,
                    'error': result.error
                }
            elif operation == 'compress':
                json_str = kwargs.get('json_str', data)
                result = self.compress_json(json_str, options)
                return {
                    'success': result.success,
                    'result': result.tonl_output if result.success else None,
                    'stats': result.stats,
                    'error': result.error
                }
            elif operation == 'analyze':
                result = self.analyze_compression(data)
                return {
                    'success': result.success,
                    'result': result.stats if result.success else None,
                    'error': result.error
                }
            elif operation == 'validate':
                tonl_text = kwargs.get('tonl_text', data)
                result = self.validate_tonl(tonl_text)
                return {
                    'success': result.success,
                    'result': result.data if result.success else None,
                    'error': result.error
                }
            else:
                return {
                    'success': False,
                    'error': f"Unknown operation: {operation}"
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


# Skill interface for Neo-Clone integration
def tonl_encode(data: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Encode data to TONL format"""
    skill = TONLSkill()
    result = skill.encode_data(data, options)
    
    if result.success:
        return {
            'success': True,
            'tonl': result.tonl_output,
            'stats': result.stats
        }
    else:
        return {
            'success': False,
            'error': result.error
        }


def tonl_decode(tonl_text: str, strict: bool = False) -> Dict[str, Any]:
    """Decode TONL format to data"""
    skill = TONLSkill()
    result = skill.decode_data(tonl_text, strict)
    
    if result.success:
        return {
            'success': True,
            'data': result.data
        }
    else:
        return {
            'success': False,
            'error': result.error
        }


def tonl_compress(json_str: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compress JSON to TONL"""
    skill = TONLSkill()
    result = skill.compress_json(json_str, options)
    
    if result.success:
        return {
            'success': True,
            'tonl': result.tonl_output,
            'stats': result.stats
        }
    else:
        return {
            'success': False,
            'error': result.error
        }


def tonl_analyze(data: Any) -> Dict[str, Any]:
    """Analyze compression potential"""
    skill = TONLSkill()
    result = skill.analyze_compression(data)
    
    if result.success:
        return {
            'success': True,
            'analysis': result.stats
        }
    else:
        return {
            'success': False,
            'error': result.error
        }


# Alias for compatibility with import statements expecting 'TonlSkill'
TonlSkill = TONLSkill