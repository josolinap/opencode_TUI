# 🗺️ **Workspace Analysis & Cleanup Plan**

## 📊 **Current Directory Structure**

```
opencode_TUI/
├── .husky/                    # Git hooks
├── assets/                    # Images and assets
├── backups/                   # 94+ backup files (massive duplication)
├── core/                      # Core functionality modules
├── data/                      # Data storage (logs, memory, cache)
├── docs/                      # Documentation files
├── examples/                  # Example code and demos
├── github/                    # GitHub Actions and scripts
├── infra/                     # Infrastructure configuration
├── llm_integrations/          # LLM provider docs
├── local_llm_setup/           # Local LLM setup docs (not installed)
├── monitoring/                # Performance monitoring
├── neo-clone/                # Main Neo-Clone agent (BUT with duplicates)
├── packages/                 # Node.js packages
├── patches/                  # Patch files
├── repository_explorations/  # External repo info
├── script/                   # Build and utility scripts
├── scripts/                  # Python scripts (duplicates with script/)
├── sdks/                     # VSCode SDK
├── skills/                   # Skills modules (duplicates with neo-clone/)
├── specs/                    # Project specifications
├── tests/                    # Test files
└── [30+ config/root files]   # Massive root clutter
```

## 🚨 **Major Issues Identified**

### **1. Massive Duplication**

- **`neo-clone/`** has full skills copy
- **`skills/`** has another skills copy
- **`script/`** and **`scripts/`** duplicate
- **`backups/`** has 94+ redundant backup files
- **Multiple `local_llm_setup/`** copies

### **2. Root Directory Clutter**

- 30+ files in root directory
- Mixed configuration files
- No clear organization

### **3. Naming Convention Issues**

- Inconsistent: `script/` vs `scripts/`
- Unclear: `core/` vs `neo-clone/`
- Redundant: Multiple documentation locations

### **4. Broken Import Paths**

- Skills importing from wrong locations
- Circular dependencies
- Mixed relative/absolute imports

## 🎯 **Proposed Clean Structure**

```
opencode_TUI/
├── src/                          # All source code
│   ├── neo-clone/               # Main agent
│   │   ├── brain/               # Brain systems
│   │   ├── skills/              # All skills (single source)
│   │   ├── integrations/        # LLM integrations
│   │   └── main.py             # Entry point
│   ├── opencode/               # OpenCode TUI specific
│   └── shared/                 # Shared utilities
├── config/                      # Configuration files
├── docs/                        # Documentation (merged)
├── tests/                       # All tests
├── scripts/                     # Build and utility scripts
├── assets/                      # Static assets
├── data/                        # Runtime data
└── tools/                       # Setup and local tools
```

## 📋 **Cleanup Actions Required**

### **Phase 1: Consolidation**

1. **Merge duplicate skills** into single location
2. **Consolidate documentation** from multiple dirs
3. **Merge script directories**
4. **Clean up root files**

### **Phase 2: Reorganization**

1. **Create proper src/ structure**
2. **Move files logically**
3. **Update all import paths**
4. **Fix configuration references**

### **Phase 3: Cleanup**

1. **Remove backup directory** (94+ files)
2. **Delete empty directories**
3. **Update .gitignore**
4. **Fix all broken references**

## ⚠️ **Risks & Mitigations**

### **High Risk**

- **Breaking import paths** - Will fix systematically
- **Breaking Neo-Clone functionality** - Will test after each move
- **Losing important files** - Will backup before deletion

### **Medium Risk**

- **Git history issues** - Will use proper git moves
- **Configuration file references** - Will update all configs

## 🔧 **Tools Needed for Cleanup**

1. **File mapping script** - Track all moves
2. **Import path finder** - Locate all imports
3. **Test runner** - Verify functionality after changes
4. **Backup script** - Safety before major changes

## 📈 **Expected Benefits**

- **Reduced complexity** - Clear structure
- **Better maintainability** - Single source of truth
- **Faster development** - No more confusion
- **Cleaner imports** - Consistent paths
- **Easier testing** - Organized test structure

## 🎯 **Success Criteria**

1. ✅ No duplicate files/directories
2. ✅ Clear naming conventions
3. ✅ All imports working
4. ✅ All tests passing
5. ✅ Neo-Clone functional
6. ✅ Documentation updated
7. ✅ < 20 root files (down from 30+)
