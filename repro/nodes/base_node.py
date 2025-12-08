# repro/nodes/base_node.py
"""
Base node class for ReproAgent pipeline.
Provides common functionality for all processing nodes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Callable, TypeVar, Generic
import logging

logger = logging.getLogger(__name__)


@dataclass
class NodeResult:
    """Result of a node execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def ok(cls, data: Any, **metadata) -> "NodeResult":
        """Create a successful result."""
        return cls(success=True, data=data, metadata=metadata)
    
    @classmethod
    def fail(cls, error: str, **metadata) -> "NodeResult":
        """Create a failed result."""
        return cls(success=False, error=error, metadata=metadata)


T = TypeVar('T')


class BaseNode(ABC, Generic[T]):
    """
    Base class for all pipeline nodes.
    
    Features:
    - Input validation
    - Execution timing
    - Error handling with optional retry
    - Success/failure hooks
    - Logging
    
    Usage:
        class MyNode(BaseNode[MyOutputType]):
            async def _execute(self, input_data: Any, **kwargs) -> MyOutputType:
                # Node logic here
                return result
    """
    
    def __init__(
        self,
        node_name: str = None,
        max_retries: int = 0,
        on_success: Optional[Callable[[NodeResult], None]] = None,
        on_failure: Optional[Callable[[NodeResult], None]] = None,
    ):
        self.node_name = node_name or self.__class__.__name__
        self.max_retries = max_retries
        self._on_success = on_success
        self._on_failure = on_failure
        self.logger = logging.getLogger(f"node.{self.node_name}")
    
    # ==================== Main Entry Point ====================
    
    async def run(self, input_data: Any, **kwargs) -> NodeResult:
        """
        Execute the node with input validation, timing, and error handling.
        
        Args:
            input_data: Input data for the node
            **kwargs: Additional arguments passed to _execute
        
        Returns:
            NodeResult with success status and output data
        """
        start_time = datetime.now()
        
        # Validate input
        validation_error = self._validate_input(input_data, **kwargs)
        if validation_error:
            result = NodeResult.fail(f"Validation failed: {validation_error}")
            self._handle_failure(result)
            return result
        
        # Execute with retry
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.info(f"Starting {self.node_name} (attempt {attempt + 1})")
                output = await self._execute(input_data, **kwargs)
                
                duration = (datetime.now() - start_time).total_seconds()
                result = NodeResult.ok(
                    output,
                    attempts=attempt + 1,
                    duration_seconds=duration,
                )
                result.duration_seconds = duration
                
                self._handle_success(result)
                return result
                
            except Exception as e:
                last_error = str(e)
                self.logger.warning(
                    f"{self.node_name} attempt {attempt + 1} failed: {last_error}"
                )
                if attempt < self.max_retries:
                    self.logger.info(f"Retrying {self.node_name}...")
        
        # All attempts failed
        duration = (datetime.now() - start_time).total_seconds()
        result = NodeResult.fail(
            last_error or "Unknown error",
            attempts=self.max_retries + 1,
        )
        result.duration_seconds = duration
        self._handle_failure(result)
        return result
    
    # ==================== Abstract Methods ====================
    
    @abstractmethod
    async def _execute(self, input_data: Any, **kwargs) -> T:
        """
        Core node logic. Override in subclasses.
        
        Args:
            input_data: Validated input data
            **kwargs: Additional arguments
        
        Returns:
            Output data of type T
        
        Raises:
            Exception: On failure (will be caught and converted to NodeResult)
        """
        pass
    
    # ==================== Hooks ====================
    
    def _validate_input(self, input_data: Any, **kwargs) -> Optional[str]:
        """
        Validate input before execution. Override to add validation.
        
        Args:
            input_data: Input data to validate
        
        Returns:
            Error message if invalid, None if valid
        """
        return None  # Default: no validation
    
    def _handle_success(self, result: NodeResult) -> None:
        """Called after successful execution."""
        self.logger.info(
            f"{self.node_name} completed in {result.duration_seconds:.2f}s"
        )
        if self._on_success:
            self._on_success(result)
    
    def _handle_failure(self, result: NodeResult) -> None:
        """Called after failed execution."""
        self.logger.error(
            f"{self.node_name} failed: {result.error}"
        )
        if self._on_failure:
            self._on_failure(result)
    
    # ==================== Utilities ====================
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.node_name}, retries={self.max_retries})"


class StatefulNode(BaseNode[T]):
    """
    Node that can mutate a shared state object.
    
    Usage:
        class MyStatefulNode(StatefulNode[MyOutputType]):
            async def _execute(self, input_data: Any, state: MyState, **kwargs) -> MyOutputType:
                # Modify state
                state.some_field = "new value"
                return result
    """
    
    async def run_with_state(self, input_data: Any, state: Any, **kwargs) -> NodeResult:
        """Execute node with state object."""
        return await self.run(input_data, state=state, **kwargs)
