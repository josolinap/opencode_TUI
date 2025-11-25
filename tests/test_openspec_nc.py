"""
OpenSpec-NC Test Suite
Production tests for OpenSpec-NC implementation
"""

import asyncio
import sys
import os
from pathlib import Path

# Add neo-clone to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'neo-clone'))

from openspec_neo_clone import OpenSpecEngine, Specification, SpecRequirement
from openspec_skill import OpenSpecSkill


async def test_openspec_functionality():
    """Test OpenSpec-NC core functionality"""
    print("Testing OpenSpec-NC functionality...")
    
    # Test engine initialization
    engine = OpenSpecEngine("test_workspace")
    assert engine.workspace_path.exists(), "Workspace should be created"
    
    # Test specification creation
    spec = Specification(
        id="test-spec",
        title="Test Specification",
        description="A test specification",
        author="Test",
        requirements=[
            SpecRequirement(
                id="test-spec-req-1",
                title="Test Requirement",
                description="A test requirement",
                acceptance_criteria=["Test passes"],
                priority="high"
            )
        ]
    )
    
    spec_file = engine.create_specification(spec)
    assert Path(spec_file).exists(), "Specification file should be created"
    
    # Test specification loading
    loaded_spec = engine.load_specification("test-spec")
    assert loaded_spec.id == spec.id, "Loaded spec should match original"
    assert len(loaded_spec.requirements) == 1, "Should have one requirement"
    
    # Test task generation
    tasks = engine.generate_tasks_from_spec("test-spec")
    assert len(tasks) == 1, "Should generate one task"
    
    print("✓ OpenSpec-NC functionality tests passed")
    return True


async def test_openspec_skill():
    """Test OpenSpec-NC skill interface"""
    print("Testing OpenSpec-NC skill interface...")
    
    skill = OpenSpecSkill()
    assert skill.name == "OpenSpec Skill", "Skill should have correct name"
    
    # Test specification creation via skill
    result = await skill.create_specification(
        title="Skill Test Spec",
        description="Testing skill interface",
        author="Test",
        requirements=[
            {
                "title": "Skill Test Requirement",
                "description": "Testing skill functionality",
                "acceptance_criteria": ["Skill works correctly"],
                "priority": "medium"
            }
        ]
    )
    
    assert result["success"], "Specification creation should succeed"
    assert "specification_id" in result, "Should return specification ID"
    
    # Test task generation via skill
    tasks_result = await skill.generate_tasks(result["specification_id"])
    assert tasks_result["success"], "Task generation should succeed"
    assert tasks_result["count"] == 1, "Should generate one task"
    
    print("✓ OpenSpec-NC skill interface tests passed")
    return True


async def main():
    """Run all OpenSpec-NC tests"""
    print("OpenSpec-NC Test Suite")
    print("=" * 40)
    
    try:
        await test_openspec_functionality()
        await test_openspec_skill()
        
        print("\n🎉 All OpenSpec-NC tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
    finally:
        # Cleanup
        import shutil
        test_workspace = Path("test_workspace")
        if test_workspace.exists():
            shutil.rmtree(test_workspace)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)