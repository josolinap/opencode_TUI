"""
Evidence collection and preservation for Neo-OSINT
"""

import asyncio
import json
import hashlib
import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import zipfile
import base64

from ..core.config import NeoOSINTConfig


class EvidenceCollector:
    """Evidence collection and preservation system"""
    
    def __init__(self, config: NeoOSINTConfig):
        self.config = config
        self.logger = logging.getLogger("neo_osint.evidence")
        
        # Ensure evidence directory exists
        self.evidence_dir = Path(config.workspace_dir) / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
    
    async def collect_evidence(
        self,
        investigation_id: str,
        query: str,
        search_results: List[Dict[str, Any]],
        scraped_content: Dict[str, str],
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Collect and preserve evidence from investigation"""
        self.logger.info(f"Collecting evidence for investigation {investigation_id}")
        
        evidence_files = []
        
        # Create investigation-specific directory
        investigation_dir = self.evidence_dir / investigation_id
        investigation_dir.mkdir(exist_ok=True)
        
        try:
            # 1. Save raw search results
            search_file = await self._save_search_results(
                investigation_dir, search_results
            )
            evidence_files.append(search_file)
            
            # 2. Save scraped content
            content_file = await self._save_scraped_content(
                investigation_dir, scraped_content
            )
            evidence_files.append(content_file)
            
            # 3. Save analysis results
            analysis_file = await self._save_analysis_results(
                investigation_dir, analysis
            )
            evidence_files.append(analysis_file)
            
            # 4. Save metadata
            metadata_file = await self._save_metadata(
                investigation_dir, investigation_id, query, search_results, analysis
            )
            evidence_files.append(metadata_file)
            
            # 5. Generate hash manifest
            manifest_file = await self._generate_hash_manifest(
                investigation_dir, evidence_files
            )
            evidence_files.append(manifest_file)
            
            # 6. Create evidence package
            if self.config.evidence.encryption_enabled:
                package_file = await self._create_encrypted_package(
                    investigation_dir, investigation_id
                )
                evidence_files.append(package_file)
            
            self.logger.info(f"Evidence collection completed: {len(evidence_files)} files")
            return evidence_files
            
        except Exception as e:
            self.logger.error(f"Evidence collection failed: {str(e)}")
            raise
    
    async def _save_search_results(
        self,
        investigation_dir: Path,
        search_results: List[Dict[str, Any]]
    ) -> str:
        """Save search results to file"""
        filename = "search_results.json"
        filepath = investigation_dir / filename
        
        # Add timestamps and metadata
        enhanced_results = []
        for result in search_results:
            enhanced_result = result.copy()
            enhanced_result["collected_at"] = datetime.now().isoformat()
            enhanced_result["hash"] = self._calculate_hash(str(result))
            enhanced_results.append(enhanced_result)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(enhanced_results, f, indent=2, ensure_ascii=False)
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(filepath)
        self.logger.info(f"Saved search results: {filename} (SHA256: {file_hash})")
        
        return str(filepath)
    
    async def _save_scraped_content(
        self,
        investigation_dir: Path,
        scraped_content: Dict[str, str]
    ) -> str:
        """Save scraped content to file"""
        filename = "scraped_content.json"
        filepath = investigation_dir / filename
        
        # Enhance content with metadata
        enhanced_content = {}
        for url, content in scraped_content.items():
            enhanced_content[url] = {
                "content": content,
                "length": len(content),
                "collected_at": datetime.now().isoformat(),
                "hash": self._calculate_hash(content)
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(enhanced_content, f, indent=2, ensure_ascii=False)
        
        file_hash = self._calculate_file_hash(filepath)
        self.logger.info(f"Saved scraped content: {filename} (SHA256: {file_hash})")
        
        return str(filepath)
    
    async def _save_analysis_results(
        self,
        investigation_dir: Path,
        analysis: Dict[str, Any]
    ) -> str:
        """Save analysis results to file"""
        filename = "analysis_results.json"
        filepath = investigation_dir / filename
        
        # Add metadata
        enhanced_analysis = analysis.copy()
        enhanced_analysis["generated_at"] = datetime.now().isoformat()
        enhanced_analysis["hash"] = self._calculate_hash(str(analysis))
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(enhanced_analysis, f, indent=2, ensure_ascii=False)
        
        file_hash = self._calculate_file_hash(filepath)
        self.logger.info(f"Saved analysis results: {filename} (SHA256: {file_hash})")
        
        return str(filepath)
    
    async def _save_metadata(
        self,
        investigation_dir: Path,
        investigation_id: str,
        query: str,
        search_results: List[Dict[str, Any]],
        analysis: Dict[str, Any]
    ) -> str:
        """Save investigation metadata"""
        filename = "investigation_metadata.json"
        filepath = investigation_dir / filename
        
        metadata = {
            "investigation_id": investigation_id,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_search_results": len(search_results),
                "total_artifacts": len(analysis.get("artifacts", [])),
                "threat_level": analysis.get("threat_level", "unknown"),
                "confidence_score": analysis.get("confidence_score", 0.0)
            },
            "collection_info": {
                "tool": "Neo-OSINT",
                "version": "1.0.0",
                "evidence_version": "1.0",
                "hash_algorithms": self.config.evidence.hash_algorithms
            },
            "chain_of_custody": {
                "created_at": datetime.now().isoformat(),
                "created_by": "Neo-OSINT",
                "preservation_method": "digital"
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        file_hash = self._calculate_file_hash(filepath)
        self.logger.info(f"Saved metadata: {filename} (SHA256: {file_hash})")
        
        return str(filepath)
    
    async def _generate_hash_manifest(
        self,
        investigation_dir: Path,
        evidence_files: List[str]
    ) -> str:
        """Generate hash manifest for all evidence files"""
        filename = "hash_manifest.json"
        filepath = investigation_dir / filename
        
        manifest = {
            "generated_at": datetime.now().isoformat(),
            "algorithm": "sha256",
            "files": {}
        }
        
        for file_path in evidence_files:
            if os.path.exists(file_path):
                file_hash = self._calculate_file_hash(file_path)
                relative_path = os.path.relpath(file_path, investigation_dir)
                manifest["files"][relative_path] = {
                    "sha256": file_hash,
                    "size": os.path.getsize(file_path),
                    "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        file_hash = self._calculate_file_hash(filepath)
        self.logger.info(f"Generated hash manifest: {filename} (SHA256: {file_hash})")
        
        return str(filepath)
    
    async def _create_encrypted_package(
        self,
        investigation_dir: Path,
        investigation_id: str
    ) -> str:
        """Create encrypted evidence package"""
        # For now, create a simple ZIP package
        # In a real implementation, you would add encryption
        package_name = f"{investigation_id}_evidence.zip"
        package_path = self.evidence_dir / package_name
        
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in investigation_dir.rglob('*'):
                if file_path.is_file() and not file_path.name.endswith('.zip'):
                    arcname = file_path.relative_to(investigation_dir)
                    zipf.write(file_path, arcname)
        
        file_hash = self._calculate_file_hash(package_path)
        self.logger.info(f"Created evidence package: {package_name} (SHA256: {file_hash})")
        
        return str(package_path)
    
    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA256 hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    async def verify_evidence_integrity(self, investigation_id: str) -> Dict[str, Any]:
        """Verify integrity of collected evidence"""
        investigation_dir = self.evidence_dir / investigation_id
        manifest_file = investigation_dir / "hash_manifest.json"
        
        if not manifest_file.exists():
            return {"valid": False, "error": "Hash manifest not found"}
        
        try:
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            
            verification_results = {
                "valid": True,
                "verified_files": 0,
                "failed_files": [],
                "missing_files": []
            }
            
            for file_path, file_info in manifest["files"].items():
                full_path = investigation_dir / file_path
                
                if not full_path.exists():
                    verification_results["missing_files"].append(file_path)
                    verification_results["valid"] = False
                    continue
                
                current_hash = self._calculate_file_hash(full_path)
                expected_hash = file_info["sha256"]
                
                if current_hash == expected_hash:
                    verification_results["verified_files"] += 1
                else:
                    verification_results["failed_files"].append({
                        "file": file_path,
                        "expected": expected_hash,
                        "actual": current_hash
                    })
                    verification_results["valid"] = False
            
            return verification_results
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        pass