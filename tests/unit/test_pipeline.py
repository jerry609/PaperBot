"""
Pipeline 单元测试
"""

import pytest
import asyncio

try:
    from src.paperbot.core.pipeline import (
        Pipeline,
        PipelineStage,
        PipelineResult,
        StageResult,
    )
    from src.paperbot.core.abstractions import ExecutionResult
except ImportError:
    from core.pipeline import (
        Pipeline,
        PipelineStage,
        PipelineResult,
        StageResult,
    )
    from core.abstractions import ExecutionResult


class TestPipelineStage:
    """PipelineStage 测试"""
    
    @pytest.mark.asyncio
    async def test_stage_runs_function(self):
        async def my_fn(ctx):
            return ExecutionResult.ok({"result": ctx["input"] + 1})
        
        stage = PipelineStage(name="test", run_fn=my_fn)
        result = await stage.run({"input": 5})
        
        assert result.status == "success"
        assert result.output["result"] == 6
    
    @pytest.mark.asyncio
    async def test_stage_skips_when_condition_met(self):
        async def my_fn(ctx):
            return ExecutionResult.ok({"ran": True})
        
        stage = PipelineStage(
            name="test",
            run_fn=my_fn,
            skip_if=lambda ctx: ctx.get("skip", False),
        )
        
        result = await stage.run({"skip": True})
        assert result.status == "skipped"
    
    @pytest.mark.asyncio
    async def test_stage_catches_exception(self):
        async def failing_fn(ctx):
            raise ValueError("Stage failed")
        
        stage = PipelineStage(name="failing", run_fn=failing_fn)
        result = await stage.run({})
        
        assert result.status == "error"
        assert "Stage failed" in result.error


class TestPipeline:
    """Pipeline 测试"""
    
    @pytest.mark.asyncio
    async def test_pipeline_runs_all_stages(self):
        results = []
        
        async def stage1(ctx):
            results.append("stage1")
            return ExecutionResult.ok({"s1": True})
        
        async def stage2(ctx):
            results.append("stage2")
            return ExecutionResult.ok({"s2": True})
        
        pipeline = Pipeline("test")
        pipeline.add_stage(PipelineStage("s1", stage1))
        pipeline.add_stage(PipelineStage("s2", stage2))
        
        result = await pipeline.run({})
        
        assert result.status == "success"
        assert len(result.stages) == 2
        assert results == ["stage1", "stage2"]
    
    @pytest.mark.asyncio
    async def test_pipeline_stops_on_critical_failure(self):
        results = []
        
        async def stage1(ctx):
            results.append("stage1")
            raise ValueError("Critical failure")
        
        async def stage2(ctx):
            results.append("stage2")
            return ExecutionResult.ok({})
        
        pipeline = Pipeline("test")
        pipeline.add_stage(PipelineStage("s1", stage1, is_critical=True))
        pipeline.add_stage(PipelineStage("s2", stage2))
        
        result = await pipeline.run({})
        
        assert result.status == "failed"
        assert results == ["stage1"]  # stage2 不应该运行
    
    @pytest.mark.asyncio
    async def test_pipeline_continues_on_non_critical_failure(self):
        results = []
        
        async def stage1(ctx):
            results.append("stage1")
            raise ValueError("Non-critical failure")
        
        async def stage2(ctx):
            results.append("stage2")
            return ExecutionResult.ok({})
        
        pipeline = Pipeline("test")
        pipeline.add_stage(PipelineStage("s1", stage1, is_critical=False))
        pipeline.add_stage(PipelineStage("s2", stage2))
        
        result = await pipeline.run({})
        
        assert result.status == "success"
        assert results == ["stage1", "stage2"]
    
    @pytest.mark.asyncio
    async def test_pipeline_updates_context(self):
        async def stage1(ctx):
            ctx["value"] = 10
            return ExecutionResult.ok({"added": 10})
        
        async def stage2(ctx):
            return ExecutionResult.ok({"value": ctx.get("value", 0)})
        
        pipeline = Pipeline("test")
        pipeline.add_stage(PipelineStage(
            "s1", stage1,
            update_context=lambda ctx, r: ctx,
        ))
        pipeline.add_stage(PipelineStage("s2", stage2))
        
        ctx = {}
        result = await pipeline.run(ctx)
        
        assert result.status == "success"

