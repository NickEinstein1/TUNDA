import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.response.generator import EmpathicResponseGenerator, ResponseContext


def test_golden_response_template():
    random.seed(1)
    generator = EmpathicResponseGenerator()
    context = ResponseContext(
        user_text="I am feeling really sad today.",
        emotion="sad",
        confidence=0.8,
        conversation_history=[],
        empathy_style="supportive",
        user_preferences={"user_name": "Alex"},
        relevant_memories=[]
    )
    response = generator._generate_template_response(context, generator.empathy_patterns.get_pattern("sad", "supportive"))
    assert isinstance(response, str)
    assert len(response) > 10
