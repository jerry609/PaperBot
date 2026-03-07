from __future__ import annotations

from types import SimpleNamespace

from paperbot.repro.models import Blueprint, ImplementationSpec, PaperContext, ReproductionPlan
from paperbot.repro.nodes.generation_node import GenerationNode


class StubLLM:
    def __init__(self):
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return 'print("ok")\n'


def test_generation_node_uses_project_llm_with_memory_context():
    llm = StubLLM()
    node = GenerationNode(llm_client=llm, use_rag=False)
    node.memory.get_relevant_context = (
        lambda **_: "Prior experience: Successfully generated trainer.py"
    )
    paper_context = PaperContext(
        title="Transformer Repro",
        abstract="Self-attention model",
        method_section="Use encoder-decoder layers.",
    )
    plan = ReproductionPlan(
        project_name="Transformer Repro",
        description="demo",
        file_structure={"model.py": "Model implementation"},
        key_components=["Model", "Attention"],
        dependencies=["torch"],
    )
    spec = ImplementationSpec()

    result = __import__("asyncio").run(
        node._generate_file_enhanced(
            filepath="model.py",
            purpose="Model implementation",
            paper_context=paper_context,
            plan=plan,
            spec=spec,
            blueprint=None,
        )
    )

    assert result.strip() == 'print("ok")'
    assert llm.calls
    user_prompt = llm.calls[0]["user"]
    assert "Prior experience: Successfully generated trainer.py" in user_prompt
    assert "Relevant Code Patterns" in user_prompt
