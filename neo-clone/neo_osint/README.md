# Neo-OSINT: Enhanced AI-Powered OSINT Tool

An advanced threat intelligence and investigation system that expands upon Robin's capabilities with integrated Neo-Clone AI brain, advanced analytics, and modular extensibility.

## 🚀 Key Features

### 🧠 **Neo-Clone Integration**

- Seamless integration with Neo-Clone's AI brain for enhanced analysis
- Advanced reasoning traces and decision-making transparency
- Dynamic skill generation for specialized investigations
- Memory system for context-aware investigations

### 🔍 **Enhanced Search Capabilities**

- **15+ Dark Web Search Engines**: Ahmia, OnionLand, DarkRunt, Torgle, Amnesia, Kaizer, Anima, Tornado, TorNet, Torland, and more
- **Clear Web Integration**: Optional clear web search engines for context
- **Intelligent Query Refinement**: AI-powered query optimization
- **Advanced Result Filtering**: Machine learning-based relevance scoring
- **Concurrent Processing**: Asynchronous multi-engine searches

### 🤖 **Advanced AI Analysis**

- **Multi-Model Support**: OpenAI, Anthropic, Google, Ollama, and local models
- **Threat Intelligence Artifacts**: Automatic extraction of IOCs, emails, crypto addresses, domains, etc.
- **Contextual Analysis**: Deep understanding of threat actor TTPs
- **Confidence Scoring**: Reliability assessment for all findings
- **Executive Summaries**: High-level actionable intelligence

### 🔒 **Security & Anonymization**

- **Tor Integration**: Full Tor network support with identity rotation
- **Rate Limiting**: Configurable request throttling
- **User Agent Rotation**: Anti-detection measures
- **Secure Evidence Handling**: Hash verification and chain of custody

### 📁 **Evidence Collection**

- **Comprehensive Preservation**: Screenshots, content, metadata
- **Hash Verification**: SHA256, MD5, SHA1 hashing
- **Chain of Custody**: Complete audit trail
- **Encryption Support**: Optional evidence encryption
- **Integrity Verification**: Automated evidence validation

### 🔌 **Plugin System**

- **Modular Architecture**: Easy extensibility
- **Built-in Plugins**: VirusTotal, Shodan, IOC Extractor
- **Custom Plugins**: Simple plugin development
- **Async Execution**: Concurrent plugin processing

## 📋 Comparison with Robin

| Feature             | Robin                     | Neo-OSINT                     |
| ------------------- | ------------------------- | ----------------------------- |
| Search Engines      | 15                        | 15+ Dark Web + Clear Web      |
| AI Integration      | Basic LLM calls           | Neo-Clone Brain + Multi-Model |
| Evidence Collection | Basic file saving         | Comprehensive with hashing    |
| Plugin System       | ❌                        | ✅ Built-in                   |
| Security Features   | Basic Tor                 | Advanced anonymization        |
| Reporting           | Simple markdown           | Multiple formats + metadata   |
| Threat Intelligence | Basic artifact extraction | Advanced IOC analysis         |
| Memory/Context      | ❌                        | ✅ Neo-Clone memory           |
| Reasoning Traces    | ❌                        | ✅ Full transparency          |

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Tor (for dark web access)
- Neo-Clone (for AI integration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Tor

```bash
# Ubuntu/Debian
sudo apt install tor

# macOS
brew install tor

# Windows (WSL)
sudo apt install tor
```

### Configure Tor

```bash
# Start Tor service
sudo systemctl start tor
sudo systemctl enable tor

# Or run manually
tor &
```

## 🚀 Quick Start

### 1. Initialize Configuration

```bash
python -m neo_osint.cli init-config
```

### 2. Configure API Keys

Edit `neo_osint_config.json`:

```json
{
  "ai_models": [
    {
      "provider": "openai",
      "model_name": "gpt-4",
      "api_key": "your-openai-key"
    },
    {
      "provider": "anthropic",
      "model_name": "claude-3-5-sonnet-20241022",
      "api_key": "your-anthropic-key"
    }
  ]
}
```

### 3. Run Investigation

```bash
# Basic investigation
python -m neo_osint.cli investigate -q "ransomware payments"

# Advanced investigation with options
python -m neo_osint.cli investigate \
  -q "sensitive credentials exposure" \
  --max-results 100 \
  --include-clear-web \
  --format json \
  --output investigation_report.json \
  --include-raw-data
```

## 📊 Usage Examples

### Basic OSINT Investigation

```bash
# Search for ransomware activities
python -m neo_osint.cli investigate -q "ransomware groups"

# Investigate data breaches
python -m neo_osint.cli investigate -q "data breach database leak"

# Monitor threat actor communications
python -m neo_osint.cli investigate -q "cybercrime forum discussions"
```

### Advanced Investigations

```bash
# Comprehensive investigation with all features
python -m neo_osint.cli investigate \
  -q "advanced persistent threats" \
  --max-results 200 \
  --include-clear-web \
  --save-evidence \
  --use-plugins \
  --format html \
  --output comprehensive_report.html \
  --include-raw-data

# Quick intelligence gathering
python -m neo_osint.cli investigate \
  -q "zero-day exploits" \
  --max-results 50 \
  --format markdown
```

### Evidence Management

```bash
# Verify evidence integrity
python -m neo_osint.cli verify-evidence --investigation-id <uuid>

# List available plugins
python -m neo_osint.cli list-plugins

# Validate configuration
python -m neo_osint.cli verify-config --config custom_config.json
```

## 🔧 Configuration

### Search Engine Configuration

```json
{
  "search_engines": [
    {
      "name": "Ahmia",
      "url": "http://juhanurmihxlp77nkq76byazcldy2hlmovfu2epvl5ankdibsot4csyd.onion/search/?q={query}",
      "enabled": true,
      "priority": 1,
      "timeout": 30
    }
  ]
}
```

### AI Model Configuration

```json
{
  "ai_models": [
    {
      "provider": "openai",
      "model_name": "gpt-4",
      "api_key": "your-api-key",
      "max_tokens": 4000,
      "temperature": 0.3,
      "enabled": true
    }
  ]
}
```

### Security Configuration

```json
{
  "security": {
    "use_tor": true,
    "tor_socks_port": 9050,
    "tor_control_port": 9051,
    "rotate_identity": true,
    "rotation_interval": 5,
    "max_request_rate": 1.0,
    "user_agent_rotation": true
  }
}
```

### Evidence Configuration

```json
{
  "evidence": {
    "evidence_dir": "evidence",
    "hash_algorithms": ["sha256", "md5"],
    "screenshot_enabled": true,
    "metadata_collection": true,
    "encryption_enabled": false,
    "retention_days": 365
  }
}
```

## 🔌 Plugin Development

### Creating Custom Plugins

```python
from neo_osint.plugins.manager import OSINTPlugin
from neo_osint.core.config import NeoOSINTConfig

class CustomPlugin(OSINTPlugin):
    @property
    def name(self) -> str:
        return "custom_plugin"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Custom OSINT plugin"

    async def execute(self, query, search_results, scraped_content, analysis):
        # Your plugin logic here
        return {
            "custom_data": "analysis results",
            "indicators": []
        }
```

### Built-in Plugins

#### VirusTotal Plugin

- Checks IOCs against VirusTotal database
- Provides malware analysis and reputation data

#### Shodan Plugin

- Searches for infrastructure information
- Identifies exposed services and vulnerabilities

#### IOC Extractor Plugin

- Enhanced indicator extraction
- CVE, malware, and threat actor identification

## 📈 Reports

### Markdown Report

```markdown
# Neo-OSINT Investigation Report

## Investigation Details

- **Investigation ID:** 12345678-1234-1234-1234-123456789012
- **Query:** ransomware payments
- **Threat Level:** HIGH

## Executive Summary

Investigation revealed active ransomware operations with multiple payment addresses and infrastructure...

## Key Findings

- 15 Bitcoin addresses identified
- 3 ransomware families detected
- Multiple onion services hosting payment portals

## Threat Intelligence Artifacts

- **Bitcoin:** 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa (confidence: 0.95)
- **Email:** support@ransomware.onion (confidence: 0.87)
- **Domain:** payment-ransomware.onion (confidence: 0.92)
```

### JSON Report

```json
{
  "investigation_id": "12345678-1234-1234-1234-123456789012",
  "query": "ransomware payments",
  "analysis": {
    "executive_summary": "...",
    "threat_level": "HIGH",
    "artifacts": [...],
    "confidence_score": 0.87
  },
  "metadata": {
    "total_search_results": 45,
    "filtered_results": 20,
    "evidence_files": 5
  }
}
```

## 🔍 Investigation Workflow

1. **Query Input**: User provides investigation query
2. **AI Refinement**: Neo-Clone optimizes the query for dark web search
3. **Multi-Engine Search**: Concurrent search across 15+ engines
4. **Result Filtering**: AI-powered relevance filtering
5. **Content Scraping**: Extract content from filtered results
6. **Threat Analysis**: Neo-Clone performs advanced analysis
7. **Plugin Enhancement**: Additional analysis via plugins
8. **Evidence Collection**: Preserve all findings with hashing
9. **Report Generation**: Create comprehensive investigation report

## 🛡️ Security Considerations

### Legal & Ethical Use

- ⚠️ **Educational and lawful investigative purposes only**
- ⚠️ **Comply with all relevant laws and institutional policies**
- ⚠️ **Respect privacy and avoid unauthorized access**
- ⚠️ **Use responsibly and at your own risk**

### Operational Security (OPSEC)

- All traffic routed through Tor network
- Identity rotation to prevent tracking
- User agent rotation for fingerprint avoidance
- Rate limiting to avoid detection
- Secure evidence handling with chain of custody

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Robin**: Original inspiration for dark web OSINT automation
- **Neo-Clone**: AI brain integration and advanced reasoning
- **Thomas Roccia**: OSINT methodology inspiration
- **Open Source Community**: Various tools and libraries

## 📞 Support

- 📧 **Issues**: Report bugs via GitHub Issues
- 📖 **Documentation**: Check the `/docs` directory
- 🔧 **Configuration**: Use `init-config` command for setup
- 🧪 **Testing**: Run validation scripts before use

---

**Neo-OSINT** - Advanced OSINT for the Modern Threat Intelligence Landscape

_Built with ❤️ by the Neo-Clone AI Community_
