"""
Data Inspector Skill for Neo-Clone
Advanced data inspection with statistical analysis, data quality assessment, and insights generation for CSV, JSON, and text files.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'skills'))
from base_skill import BaseSkill, SkillResult
import os
import json
import csv
import time
from collections import Counter, defaultdict
from datetime import datetime
import statistics
from functools import lru_cache
from typing import Dict, Any, Optional, List, Union
import logging
import hashlib

logger = logging.getLogger(__name__)

class DataInspectorSkill(BaseSkill):

    def __init__(self):
        super().__init__(
            name='data_inspector',
            description='Advanced data inspection with statistical analysis, data quality assessment, and insights generation for CSV, JSON, and text files.',
            example='Analyze this CSV file and provide insights about the data quality and patterns'
        )
        self._cache = {}
        self._max_cache_size = 50

    @property
    def parameters(self) -> Dict[str, str]:
        return {
            'file_path': 'string - Path to data file to analyze (required)',
            'data_type': 'string - Type of data file (csv, json, txt). Auto-detected if not specified',
            'analysis_depth': 'string - Depth of analysis (basic, detailed, comprehensive). Default: detailed',
            'sample_size': 'integer - Number of rows to sample for large files (default: 1000)'
        }

    def _execute(self, params: Dict[str, Any]) -> SkillResult:
        """Execute data inspection with given parameters"""
        try:
            file_path = params.get('file_path', '')
            data_type = params.get('data_type', '')
            analysis_depth = params.get('analysis_depth', 'detailed')
            sample_size = params.get('sample_size', 1000)

            if not file_path:
                return SkillResult(False, "No file path provided", {"error": "Missing file_path parameter"})

            if not os.path.exists(file_path):
                return SkillResult(False, f"File not found: {file_path}", {"error": "File not found"})

            # Auto-detect data type if not specified
            if not data_type:
                data_type = self._detect_file_type(file_path)

            # Perform analysis based on data type
            if data_type == 'csv':
                result = self._analyze_csv(file_path, analysis_depth, sample_size)
            elif data_type == 'json':
                result = self._analyze_json(file_path, analysis_depth)
            elif data_type == 'txt':
                result = self._analyze_text(file_path, analysis_depth)
            else:
                return SkillResult(False, f"Unsupported data type: {data_type}", {"error": "Unsupported data type"})

            return SkillResult(True, f"Data inspection completed for {os.path.basename(file_path)}", result)

        except Exception as e:
            logger.error(f"Data inspection failed: {str(e)}")
            return SkillResult(False, f"Data inspection failed: {str(e)}", {"error": str(e)})

    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type based on extension"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            return 'csv'
        elif ext == '.json':
            return 'json'
        elif ext in ['.txt', '.log', '.md']:
            return 'txt'
        else:
            return 'unknown'

    def _analyze_csv(self, file_path: str, analysis_depth: str = 'detailed', sample_size: int = 1000) -> Dict[str, Any]:
        """Analyze CSV file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                
                if not rows:
                    return {"error": "CSV file is empty"}

                # Basic statistics
                total_rows = len(rows)
                columns = list(rows[0].keys()) if rows else []
                
                analysis = {
                    "file_type": "CSV",
                    "total_rows": total_rows,
                    "total_columns": len(columns),
                    "columns": columns,
                    "sample_data": rows[:5] if rows else []
                }

                if analysis_depth in ['detailed', 'comprehensive']:
                    # Column analysis
                    column_stats = {}
                    for col in columns:
                        values = [row.get(col, '') for row in rows]
                        non_null_values = [v for v in values if v and v.strip()]
                        
                        column_stats[col] = {
                            "total_values": len(values),
                            "non_null_values": len(non_null_values),
                            "null_values": len(values) - len(non_null_values),
                            "unique_values": len(set(non_null_values)),
                            "sample_values": list(set(non_null_values))[:10]
                        }
                        
                        # Numeric analysis
                        try:
                            numeric_values = [float(v) for v in non_null_values if v.replace('.', '').replace('-', '').isdigit()]
                            if numeric_values:
                                column_stats[col].update({
                                    "min": min(numeric_values),
                                    "max": max(numeric_values),
                                    "mean": statistics.mean(numeric_values),
                                    "median": statistics.median(numeric_values)
                                })
                        except:
                            pass
                    
                    analysis["column_statistics"] = column_stats

                return analysis

        except Exception as e:
            return {"error": f"CSV analysis failed: {str(e)}"}

    def _analyze_json(self, file_path: str, analysis_depth: str = 'detailed') -> Dict[str, Any]:
        """Analyze JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            analysis = {
                "file_type": "JSON",
                "data_structure": type(data).__name__
            }

            if isinstance(data, dict):
                analysis.update({
                    "total_keys": len(data),
                    "keys": list(data.keys())[:20],  # Show first 20 keys
                    "sample_data": {k: data[k] for k in list(data.keys())[:5]}
                })
            elif isinstance(data, list):
                analysis.update({
                    "total_items": len(data),
                    "item_type": type(data[0]).__name__ if data else "empty",
                    "sample_items": data[:5] if data else []
                })

            return analysis

        except Exception as e:
            return {"error": f"JSON analysis failed: {str(e)}"}

    def _analyze_text(self, file_path: str, analysis_depth: str = 'detailed') -> Dict[str, Any]:
        """Analyze text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            lines = content.split('\n')
            words = content.split()
            
            analysis = {
                "file_type": "Text",
                "total_characters": len(content),
                "total_lines": len(lines),
                "total_words": len(words),
                "average_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0,
                "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
            }

            if analysis_depth in ['detailed', 'comprehensive']:
                # Word frequency
                word_freq = Counter(word.lower().strip('.,!?;:"()[]') for word in words if word.strip())
                analysis.update({
                    "most_common_words": word_freq.most_common(10),
                    "unique_words": len(word_freq),
                    "vocabulary_richness": len(word_freq) / len(words) if words else 0
                })

            return analysis

        except Exception as e:
            return {"error": f"Text analysis failed: {str(e)}"}

# Test the skill
if __name__ == "__main__":
    skill = DataInspectorSkill()
    
    # Test with a simple CSV
    result = skill.execute({
        "file_path": "test.csv",
        "analysis_depth": "detailed"
    })
    
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    if result.data:
        print(f"Data: {result.data}")