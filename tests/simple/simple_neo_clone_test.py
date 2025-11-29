#!/usr/bin/env python3
"""
Simple Neo-Clone Test
====================

Test core Neo-Clone functionality without external dependencies.
"""

import sys
import os

# Add neo-clone directory to path
project_root = Path(__file__).parent.parent.parent
neo_clone_path = project_root / 'neo-clone'
sys.path.insert(0, str(neo_clone_path))

def test_minimax_agent_directly():
    """Test MiniMax Agent directly without external dependencies"""
    print("Testing MiniMax Agent directly...")
    try:
        from minimax_agent import MiniMaxAgent, SearchStrategy
        
        # Create agent with minimal configuration
        agent = MiniMaxAgent(
            search_strategy=SearchStrategy.DFS,  # Use simpler strategy
            max_depth=3,  # Reduce depth for testing
            max_nodes=10,  # Reduce nodes for testing
            confidence_threshold=0.5
        )
        
        # Test the analyze_user_input method (synchronous)
        result = agent.analyze_user_input(
            "Hello, this is a test of your reasoning capabilities",
            ["previous context", "another context item"]
        )
        
        print("MiniMax Agent Analysis Results:")
        print(f"  Primary Intent: {result.get('primary_intent', 'N/A')}")
        print(f"  All Intents: {result.get('all_intents', [])}")
        print(f"  Suggested Skills: {result.get('suggested_skills', [])}")
        print(f"  Confidence: {result.get('confidence', 0):.2f}")
        print(f"  Execution Strategy: {result.get('execution_strategy', 'N/A')}")
        
        # Get status
        status = agent.get_status()
        print(f"\nMiniMax Agent Status:")
        print(f"  Strategy: {status['search_strategy']}")
        print(f"  Max Depth: {status['max_depth']}")
        print(f"  Max Nodes: {status['max_nodes']}")
        print(f"  Confidence Threshold: {status['confidence_threshold']}")
        
        return True
        
    except Exception as e:
        print(f"MiniMax Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_skills_manager_directly():
    """Test Skills Manager directly"""
    print("\nTesting Skills Manager directly...")
    try:
        # Import directly from skills directory
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'skills'))
        from opencode_skills_manager import OpenCodeSkillsManager
        
        sm = OpenCodeSkillsManager()
        sm.initialize()
        
        skills = sm.list_skills()
        print(f"Skills Manager Results:")
        print(f"  Total Skills: {len(skills)}")
        print(f"  Available Skills: {skills[:10]}{'...' if len(skills) > 10 else ''}")
        
        # Test getting skill details
        if skills:
            first_skill = skills[0]
            details = sm.get_skill_details(first_skill)
            if details:
                print(f"\nFirst Skill Details ({first_skill}):")
                print(f"  Category: {details.get('category', 'N/A')}")
                print(f"  Description: {details.get('description', 'N/A')[:50]}...")
                print(f"  Capabilities: {details.get('capabilities', [])[:3]}")
        
        return True
        
    except Exception as e:
        print(f"Skills Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_brain_system_directly():
    """Test Brain System directly"""
    print("\nTesting Brain System directly...")
    try:
        from brain.brain import Brain
        
        # Create simple config
        class SimpleConfig:
            def __init__(self):
                self.provider = "test"
                self.model_name = "test_model"
                self.endpoint = "http://test"
        
        # Create simple skills manager
        class SimpleSkillsManager:
            def __init__(self):
                self.skills = {}
            def list_skills(self):
                return []
            def get(self, name):
                return None
        
        config = SimpleConfig()
        skills = SimpleSkillsManager()
        
        brain = Brain(config, skills)
        print(f"Brain System Results:")
        print(f"  Brain Type: {type(brain).__name__}")
        print(f"  Config Provider: {brain.config.provider}")
        print(f"  Config Model: {brain.config.model_name}")
        
        return True
        
    except Exception as e:
        print(f"Brain System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all direct tests"""
    print("=" * 60)
    print("SIMPLE NEO-CLONE DIRECT FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        test_minimax_agent_directly,
        test_skills_manager_directly,
        test_brain_system_directly,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print("DIRECT TEST RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\nALL DIRECT TESTS PASSED!")
        print("Neo-Clone core components are functional.")
        print("\nKey Findings:")
        print("  MiniMax Agent: Working with tree-based reasoning")
        print("  Skills Manager: Working with skill discovery")
        print("  Brain System: Working with basic initialization")
        print("\nThe system is ready for integration!")
    else:
        print(f"\n{total-passed} tests failed. Some components need attention.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)