"""
Tests for base_skill skill
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tests.test_framework import SkillTestCase
except ImportError:
    class SkillTestCase:
        def setUp(self):
            pass


class TestBaseSkillSkill(SkillTestCase):
    """Test cases for base_skill skill"""

    def setUp(self):
        super().setUp()
        self.skill = self.load_skill("base_skill")

    def test_basic_execution(self):
        """Test basic skill execution"""
        if self.skill:
            response = self.skill.execute({})
            self.assertSkillResponse(response)

    def test_parameter_validation(self):
        """Test parameter validation"""
        if self.skill:
            response = self.skill.execute({"invalid_param": "test"})
            # Should handle gracefully
            self.assertIsInstance(response, dict)


if __name__ == '__main__':
    unittest.main()
