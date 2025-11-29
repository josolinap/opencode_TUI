# OpenCode TUI + Neo-Clone AI Agent 🧠

**Advanced AI-Powered Terminal Development Environment with Enhanced Brain System**

![TUI with Neo-Clone Agent](assets/TUI%20with%20neo-clone%20agent.png)

---

## 🌟 **Overview**

OpenCode TUI with Neo-Clone Agent is a revolutionary terminal-based development environment that combines the power of OpenCode's TUI interface with an advanced AI brain system. This enhanced version provides intelligent code assistance, automated workflows, and multi-model AI capabilities - all running locally in your terminal.

### 🎯 **Key Features**

- **🤖 Neo-Clone AI Agent**: Advanced brain system with 12+ specialized skills
- **🧠 MiniMax Agent**: Sophisticated reasoning and intent classification
- **⚡ TUI Interface**: Lightning-fast terminal user interface
- **🔧 Multi-Model Support**: 36+ free AI models integrated
- **📚 Smart Code Analysis**: Intelligent code understanding and generation
- **🔄 Automated Workflows**: Streamlined development processes
- **🎯 Multi-Skill Orchestration**: Coordinates multiple AI capabilities
- **💾 Persistent Memory**: Learns from your coding patterns

---

## 🚀 **Quick Start**

### **Prerequisites**

- **Bun** (JavaScript runtime and package manager)
- **Python 3.8+** (for Neo-Clone brain system)
- **Git** (for version control)

### **Installation**

1. **Clone the Repository**

   ```bash
   git clone https://github.com/josolinap/opencode_TUI.git
   cd opencode_TUI
   ```

2. **Install Dependencies**

   ```bash
   # Install OpenCode dependencies
   bun install

   # Install Neo-Clone Python dependencies
   cd neo-clone
   pip install -r requirements.txt
   cd ..
   ```

3. **Configure the System**

   ```bash
   # Set up the default free model
   opencode config set model "opencode/big-pickle"

   # Or configure your preferred model
   opencode config set model "anthropic/claude-3-sonnet"
   ```

4. **Launch the TUI**
   ```bash
   # Start the OpenCode TUI with Neo-Clone Agent
   bun run tui
   ```

**That's it! 🎉** Your enhanced AI development environment is ready!

---

## 🧠 **How the Brain System Works**

### **Neo-Clone Brain Architecture**

The Neo-Clone brain is a sophisticated AI system designed to enhance your coding experience through intelligent assistance and automation.

#### **Core Components**

1. **🧩 Base Brain** (`neo-clone/brain/base_brain.py`)
   - Foundation of the AI system
   - Manages basic operations and skill coordination
   - Handles memory and context management

2. **🧠 Enhanced Brain** (`neo-clone/brain/enhanced_brain.py`)
   - Advanced reasoning capabilities
   - Multi-skill orchestration
   - Dynamic learning and adaptation

3. **💾 Memory Systems**
   - **Persistent Memory**: Long-term storage of preferences and patterns
   - **Vector Memory**: Semantic search and context retrieval
   - **Cache System**: Fast access to frequently used data

#### **Skill Registry**

The brain includes 12+ specialized skills that work together:

1. **💻 Code Generation** (`skills/code_generation.py`)
   - Generate and explain Python ML code
   - Create algorithms and implementations
   - Optimize existing code

2. **📝 Text Analysis** (`skills/text_analysis.py`)
   - Sentiment analysis and content moderation
   - Text processing and summarization
   - Content classification

3. **📊 Data Inspector** (`skills/data_inspector.py`)
   - Analyze CSV/JSON data
   - Provide insights and summaries
   - Data visualization suggestions

4. **🤖 ML Training** (`skills/ml_training.py`)
   - ML model training guidance
   - Best practices and recommendations
   - Model optimization tips

5. **📁 File Manager** (`skills/file_manager.py`)
   - Read files and analyze content
   - Manage directories and operations
   - File organization and cleanup

6. **🔍 Web Search** (`skills/web_search.py`)
   - Search the web for information
   - Fact-check and verify resources
   - Find documentation and examples

7. **🧠 MiniMax Agent** (`skills/minimax_agent.py`)
   - Advanced reasoning and decision-making
   - Intent classification and understanding
   - Dynamic skill creation and management

8. **🎯 Multi-Skill Orchestrator** (`skills/multi_skill_orchestrator.py`)
   - Coordinates multiple skills for complex tasks
   - Task decomposition and parallel processing
   - Result synthesis and workflow optimization

9. **📋 Planning Skill** (`skills/additional_skills.py`)
   - Project planning and roadmaps
   - Task breakdown and timeline estimation
   - Resource planning and milestones

10. **📈 Advanced File Manager** (`skills/more_skills.py`)
    - Enhanced file operations
    - Directory management and analysis
    - File system organization

11. **🔬 Advanced Text Analysis** (`skills/more_skills.py`)
    - Entity extraction and keyword analysis
    - Text summarization and complexity assessment
    - Content type detection

12. **📚 Additional Skills** (Public APIs, Programming Books)
    - External API integrations
    - Educational resource access
    - Extended functionality modules

---

## 🤖 **How the Neo-Clone Agent Works**

### **Agent Lifecycle**

1. **🔍 Intent Analysis**
   - Analyzes user requests with confidence scoring
   - Determines the best approach for each task
   - Selects appropriate skills and models

2. **🎯 Skill Selection**
   - Dynamically chooses the right skill for the job
   - Can combine multiple skills for complex tasks
   - Creates custom skills on-demand when needed

3. **🧠 Reasoning Process**
   - Uses advanced reasoning traces for transparency
   - Provides detailed decision-making process
   - Learns from previous interactions

4. **💡 Response Generation**
   - Generates structured, helpful responses
   - Includes performance metrics and confidence scores
   - Provides explanations and next steps

### **Example Interaction**

```
User: "Help me optimize this neural network code"

Neo-Clone Agent Response:
[Neo Reasoning] Analyzing request for neural network optimization...
[Skill Used] code_generation + ml_training
[Skill Output] Here are 3 optimization strategies for your neural network:
1. Batch normalization implementation
2. Learning rate scheduling
3. Dropout layer optimization

Performance: 95% confidence | Processing time: 1.2s
```

---

## 🤖 **Neo-Clone Agent - Full Brain System Access**

The **Neo-Clone Agent** is a specialized AI agent configuration that provides **complete access to the Neo-Clone brain system** with all advanced capabilities unlocked.

### **🚀 Key Features**

- **🧠 Full Brain System Access**: Complete integration with Neo-Clone's advanced AI brain
- **🎯 Multi-Skill Orchestration**: Coordinates all 7+ specialized skills simultaneously
- **🧠 Advanced Reasoning**: Powered by MiniMax agent for complex problem-solving
- **🔄 Intelligent Routing**: Automatically selects optimal skills for each task
- **📈 Continuous Learning**: Adapts and improves from every interaction
- **⚡ High Performance**: Optimized for complex multi-step tasks

### **🛠️ Agent Configuration**

```json
{
  "neo-clone": {
    "name": "neo-clone",
    "description": "Neo-Clone AI Agent with full brain system access",
    "model": "opencode/big-pickle",
    "mode": "primary",
    "tools": {
      "neo-clone": true,
      "todoread": true,
      "todowrite": true,
      "bash": true,
      "edit": true,
      "glob": true,
      "grep": true,
      "list": true,
      "read": true,
      "write": true,
      "websearch": true
    },
    "options": {
      "enable_brain_system": true,
      "use_advanced_reasoning": true,
      "multi_skill_orchestration": true,
      "intelligent_routing": true
    }
  }
}
```

### **🎯 When to Use Neo-Clone Agent**

**Perfect For:**
- **Complex Multi-Step Tasks** - Combines multiple AI skills
- **Advanced Code Analysis** - Full brain intelligence for coding
- **Data Science Workflows** - Code generation + data analysis
- **Research Projects** - Web search + reasoning + analysis
- **Enterprise-Level Tasks** - Maximum AI capabilities

**Example Usage:**
```bash
# Start TUI with Neo-Clone Agent
bun run tui

# Select Agent: neo-clone
# Select Model: opencode/big-pickle

# Now you have full Neo-Clone brain system access!
```

### **🧠 What Makes Neo-Clone Agent Special**

| Feature | Neo-Clone Agent | Standard Agents |
|---------|----------------|-----------------|
| **Brain Access** | ✅ Complete Neo-Clone brain | ❌ Limited access |
| **Skill Coordination** | ✅ Multi-skill orchestration | ❌ Single skill focus |
| **Advanced Reasoning** | ✅ MiniMax integration | ❌ Basic reasoning |
| **Learning Capability** | ✅ Continuous adaptation | ❌ Static behavior |
| **Intelligence Level** | ✅ Maximum AI power | ❌ Standard AI |

### **⚙️ Advanced Configuration**

To customize the Neo-Clone Agent, edit `opencode.json`:

```json
"agent": {
  "neo-clone": {
    "temperature": 0.7,
    "topP": 0.9,
    "prompt": "Custom system prompt for Neo-Clone behavior",
    "permission": {
      "edit": "allow",
      "bash": {"*": "allow"},
      "webfetch": "allow"
    }
  }
}
```

**The Neo-Clone Agent represents the pinnacle of AI assistance - complete access to the Neo-Clone brain system for maximum intelligence and capability! 🧠✨**

---

## 🎯 **MiniMax Agent Integration**

The MiniMax Agent is the core reasoning engine that powers Neo-Clone's advanced capabilities:

### **Key Features**

- **🧠 Advanced Reasoning**: Complex problem-solving and decision-making
- **🎯 Intent Classification**: Understands user intent with high accuracy
- **🔄 Dynamic Skill Creation**: Generates new skills based on requirements
- **📊 Performance Monitoring**: Tracks and optimizes agent performance
- **🎓 Continuous Learning**: Improves from each interaction

### **How It Works**

1. **Input Processing**: Analyzes user input and context
2. **Intent Classification**: Determines the user's goal and requirements
3. **Skill Selection**: Chooses or creates the appropriate skill
4. **Execution**: Performs the task with detailed reasoning
5. **Learning**: Updates knowledge base for future interactions

---

## ⚡ **Benefits of TUI vs Basic OpenCode**

### **🚀 Performance Advantages**

| Feature            | Basic OpenCode | OpenCode TUI + Neo-Clone               |
| ------------------ | -------------- | -------------------------------------- |
| **Speed**          | Standard       | ⚡ Lightning fast terminal interface   |
| **Memory Usage**   | Higher         | 💾 Optimized for terminal environments |
| **AI Integration** | Basic          | 🧠 Advanced Neo-Clone brain system     |
| **Skills**         | Limited        | 🔧 7+ specialized AI skills            |
| **Learning**       | None           | 🎓 Continuous adaptation and learning  |
| **Automation**     | Manual         | 🤖 Intelligent workflow automation     |

### **🎯 Enhanced Capabilities**

1. **🧠 Intelligent Assistance**
   - Context-aware code suggestions
   - Automated refactoring recommendations
   - Smart error detection and fixes

2. **⚡ Workflow Automation**
   - Automated testing and deployment
   - Intelligent code reviews
   - Streamlined development processes

3. **📚 Advanced Learning**
   - Learns your coding patterns
   - Adapts to your preferences
   - Provides personalized suggestions

4. **🔍 Deep Analysis**
   - Comprehensive code analysis
   - Performance optimization suggestions
   - Security vulnerability detection

5. **💬 Natural Communication**
   - Chat-like interface for complex tasks
   - Natural language processing
   - Context-aware conversations

### **🎨 User Experience**

- **🖥️ Terminal Native**: Perfect for developers who love the command line
- **⌨️ Keyboard-Driven**: Efficient keyboard shortcuts and navigation
- **🎯 Focused Interface**: Minimal distractions, maximum productivity
- **📱 Remote Friendly**: Works seamlessly over SSH connections

---

## 🔧 **Advanced Configuration**

### **Custom Models**

```bash
# List available models
opencode models list

# Set custom model
opencode config set model "your-preferred-model"

# Configure model parameters
opencode config set temperature 0.7
opencode config set max_tokens 2048
```

### **Neo-Clone Configuration**

Edit `neo-clone/config.py` to customize:

```python
# Brain configuration
BRAIN_CONFIG = {
    "memory_size": 1000,
    "learning_rate": 0.01,
    "skill_timeout": 30,
    "confidence_threshold": 0.8
}

# Model preferences
MODEL_PREFERENCES = {
    "code_generation": "anthropic/claude-3-sonnet",
    "text_analysis": "openai/gpt-4",
    "data_inspector": "google/gemini-pro"
}
```

### **Skill Development**

Create custom skills in `skills/` directory:

```python
from base_skill import BaseSkill, SkillCategory, SkillResult
from typing import Dict, Any

class CustomSkill(BaseSkill):
    """Custom skill for specialized functionality"""

    def __init__(self):
        super().__init__(
            name="custom_skill",
            description="Custom functionality description",
            category=SkillCategory.GENERAL,
            capabilities=["custom_processing", "specialized_analysis"]
        )

    def execute(self, params: Dict[str, Any]) -> SkillResult:
        """Execute the custom skill"""
        # Your custom logic here
        result = "Custom processing completed"

        return SkillResult(
            success=True,
            output=result,
            skill_name=self.name,
            execution_time=0.1,
            metadata={"processed": True}
        )
```

**Skills are automatically registered by the skills manager when placed in the `skills/` directory.**

---

## 📁 **Project Structure**

```
opencode_TUI/
├── 📁 neo-clone/                 # AI brain system
│   ├── 📁 brain/                 # Core brain components
│   │   ├── base_brain.py         # Foundation system
│   │   ├── enhanced_brain.py     # Advanced capabilities
│   │   ├── unified_memory.py     # Memory management
│   │   ├── persistent_memory.py  # Long-term storage
│   │   ├── vector_memory.py      # Semantic search
│   │   └── data_models.py        # Data structures
│   ├── 📁 monitoring/            # Performance monitoring
│   ├── 📁 data/                  # Data storage
│   ├── 📁 scripts/               # Utility scripts
│   ├── minimax_agent.py          # Advanced reasoning engine
│   ├── enhanced_llm_client.py    # LLM client with routing
│   ├── main.py                   # Entry point
│   ├── config.py                 # Configuration
│   └── requirements.txt          # Python dependencies
├── 📁 skills/                    # Centralized skills directory
│   ├── opencode_skills_manager.py # Skills manager
│   ├── base_skill.py             # Base skill framework
│   ├── code_generation.py        # Code generation skill
│   ├── text_analysis.py          # Text analysis skill
│   ├── data_inspector.py         # Data analysis skill
│   ├── web_search.py             # Web search skill
│   ├── file_manager.py           # File management skill
│   ├── ml_training.py            # ML training guidance
│   ├── minimax_agent.py          # Advanced reasoning skill
│   ├── multi_skill_orchestrator.py # Skill coordination
│   ├── additional_skills.py      # Planning & organization
│   ├── more_skills.py            # Advanced file/text skills
│   └── [additional utility skills]
├── 📁 packages/                  # OpenCode packages
│   ├── 📁 opencode/              # Core TUI application
│   ├── 📁 app/                   # Web interface
│   └── 📁 console/               # Console components
├── 📁 docs/                      # Documentation
├── 📁 examples/                  # Example code
├── 📁 config/                    # Configuration files
├── 📁 core/                      # Core functionality
├── 📁 monitoring/                # Performance monitoring
├── 📁 tests/                     # Test files
├── 📄 opencode.json              # Agent configurations
├── 📄 package.json               # Node.js dependencies
├── 📄 README.md                  # This file
└── 📄 start-web.bat              # Web development launcher
```

---

## 🚀 **Usage Examples**

### **Code Generation**

```bash
# Start TUI
bun run tui

# In the TUI, type:
"Create a Python function to analyze sentiment in text"

# Neo-Clone will:
# 1. Analyze your request
# 2. Select the code_generation skill
# 3. Generate optimized Python code
# 4. Provide explanations and examples
```

### **Data Analysis**

```bash
# In the TUI:
"Analyze this CSV file and provide insights"

# Neo-Clone will:
# 1. Use the data_inspector skill
# 2. Parse and analyze the data
# 3. Generate visualizations
# 4. Provide detailed insights
```

### **Web Development**

```bash
# Start web development server
start-web.bat

# Or manually:
bun run packages/opencode/src/index.ts serve --port 4096
bun run --cwd packages/app dev
```

---

## 🔍 **Troubleshooting**

### **Common Issues**

1. **Model Connection Issues**

   ```bash
   # Check model status
   opencode models status

   # Reset configuration
   opencode config reset
   ```

2. **Python Dependencies**

   ```bash
   # Reinstall dependencies
   cd neo-clone
   pip install -r requirements.txt --force-reinstall
   ```

3. **Memory Issues**
   ```bash
   # Clear Neo-Clone memory
   cd neo-clone
   python -c "from brain.memory import clear_memory; clear_memory()"
   ```

### **Performance Optimization**

```bash
# Increase memory limit
export NODE_OPTIONS="--max-old-space-size=4096"

# Optimize Python performance
export PYTHONOPTIMIZE=2

# Clear caches
bun run clean
cd neo-clone && python clear_cache.py
```

---

## 🤝 **Contributing**

We welcome contributions! Here's how you can help:

### **Development Setup**

```bash
# Clone your fork
git clone https://github.com/your-username/opencode_TUI.git
cd opencode_TUI

# Install dependencies
bun install
cd neo-clone && pip install -r requirements.txt

# Create a feature branch
git checkout -b feature-name

# Make your changes
# ...

# Test your changes
bun run test
cd neo-clone && python -m pytest

# Submit a pull request
git push origin feature-name
```

### **Areas for Contribution**

- 🧠 **Brain System**: Improve reasoning and learning capabilities
- 🔧 **Skills**: Create new specialized skills
- 📚 **Documentation**: Improve guides and examples
- 🐛 **Bug Fixes**: Help us squash bugs
- ⚡ **Performance**: Optimize speed and memory usage

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 **Contact & Support**

- **Email**: [mail@josolinap.dedyn.io](mailto:mail@josolinap.dedyn.io)
- **GitHub Issues**: [Create an Issue](https://github.com/josolinap/opencode_TUI/issues)
- **Discord Community**: [Join our Discord](https://discord.gg/opencode)

---

## 🙏 **Acknowledgments**

- **OpenCode Team**: For the amazing TUI foundation
- **Neo-Clone Contributors**: For the advanced brain system
- **AI Community**: For inspiration and feedback
- **Open Source Community**: For making this possible

---

## 🚀 **Roadmap**

### **Upcoming Features**

- 🌐 **Web Interface**: Enhanced web-based IDE
- 📱 **Mobile App**: Remote development on mobile devices
- 🔄 **Real-time Collaboration**: Pair programming in the TUI
- 🧠 **Advanced Learning**: ML-powered personalization
- 🔌 **Plugin System**: Extensible architecture
- 📊 **Analytics Dashboard**: Development insights and metrics

### **Version History**

- **v2.0**: Neo-Clone Agent integration
- **v1.5**: Enhanced brain system with MiniMax agent
- **v1.0**: Initial TUI release

---

**⭐ Star this repository if you find it helpful!**

**🔄 Fork and contribute to make it even better!**

---

_Built with ❤️ by the OpenCode TUI + Neo-Clone team_
