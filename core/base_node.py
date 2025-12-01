"""
节点基类定义

来源: BettaFish/QueryEngine/nodes/base_node.py
适配: PaperBot 学者追踪系统

定义所有处理节点的基础接口，支持:
- 输入验证
- 输出处理
- 日志记录
- 状态变更
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar, Generic
from loguru import logger

# 泛型类型变量
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')
StateT = TypeVar('StateT')


class BaseNode(ABC, Generic[InputT, OutputT]):
    """
    节点基类
    
    所有处理节点都应继承此类，实现 run() 方法
    """

    def __init__(self, node_name: str = "", llm_client: Any = None):
        """
        初始化节点
        
        Args:
            node_name: 节点名称，用于日志
            llm_client: 可选的 LLM 客户端
        """
        self.node_name = node_name or self.__class__.__name__
        self.llm_client = llm_client

    @abstractmethod
    def run(self, input_data: InputT, **kwargs) -> OutputT:
        """
        执行节点处理逻辑
        
        Args:
            input_data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            处理结果
        """
        pass

    def validate_input(self, input_data: InputT) -> bool:
        """
        验证输入数据
        
        Args:
            input_data: 输入数据
            
        Returns:
            验证是否通过
        """
        return True

    def process_output(self, output: OutputT) -> OutputT:
        """
        处理/后处理输出数据
        
        Args:
            output: 原始输出
            
        Returns:
            处理后的输出
        """
        return output

    def execute(self, input_data: InputT, **kwargs) -> OutputT:
        """
        完整的执行流程：验证 → 运行 → 后处理
        
        Args:
            input_data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            处理结果
        """
        self.log_info(f"开始执行")
        
        # 验证输入
        if not self.validate_input(input_data):
            raise ValueError(f"节点 {self.node_name} 输入验证失败")
        
        # 执行处理
        try:
            result = self.run(input_data, **kwargs)
            result = self.process_output(result)
            self.log_info(f"执行完成")
            return result
        except Exception as e:
            self.log_error(f"执行失败: {str(e)}")
            raise

    # ============ 日志方法 ============
    
    def log_info(self, message: str):
        """记录信息日志"""
        logger.info(f"[{self.node_name}] {message}")
    
    def log_debug(self, message: str):
        """记录调试日志"""
        logger.debug(f"[{self.node_name}] {message}")
    
    def log_warning(self, message: str):
        """记录警告日志"""
        logger.warning(f"[{self.node_name}] ⚠️ {message}")

    def log_error(self, message: str):
        """记录错误日志"""
        logger.error(f"[{self.node_name}] ❌ {message}")
    
    def log_success(self, message: str):
        """记录成功日志"""
        logger.info(f"[{self.node_name}] ✅ {message}")


class StateMutationNode(BaseNode[InputT, OutputT], Generic[InputT, OutputT, StateT]):
    """
    带状态变更功能的节点基类
    
    除了处理输入输出，还可以修改全局状态
    """
    
    @abstractmethod
    def mutate_state(self, input_data: InputT, state: StateT, **kwargs) -> StateT:
        """
        修改状态
        
        Args:
            input_data: 输入数据
            state: 当前状态
            **kwargs: 额外参数
            
        Returns:
            修改后的状态
        """
        pass

    def execute_with_state(
        self, 
        input_data: InputT, 
        state: StateT, 
        **kwargs
    ) -> tuple[OutputT, StateT]:
        """
        带状态变更的完整执行流程
        
        Args:
            input_data: 输入数据
            state: 当前状态
            **kwargs: 额外参数
            
        Returns:
            (处理结果, 新状态) 元组
        """
        result = self.execute(input_data, **kwargs)
        new_state = self.mutate_state(input_data, state, **kwargs)
        return result, new_state


class LLMNode(BaseNode[str, str]):
    """
    LLM 调用节点
    
    专门用于 LLM 交互的节点，提供便捷的 prompt 调用方法
    """
    
    def __init__(self, llm_client: Any, node_name: str = ""):
        super().__init__(node_name=node_name, llm_client=llm_client)
        if llm_client is None:
            raise ValueError("LLMNode 需要 llm_client")

    def call_llm(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        **kwargs
    ) -> str:
        """
        调用 LLM
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            **kwargs: 额外参数
            
        Returns:
            LLM 响应
        """
        self.log_debug(f"调用 LLM (prompt长度: {len(user_prompt)})")
        response = self.llm_client.invoke(system_prompt, user_prompt, **kwargs)
        self.log_debug(f"LLM 响应长度: {len(response)}")
        return response

    def call_llm_stream(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        **kwargs
    ):
        """
        流式调用 LLM
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            **kwargs: 额外参数
            
        Yields:
            LLM 响应块
        """
        self.log_debug(f"流式调用 LLM")
        yield from self.llm_client.stream_invoke(system_prompt, user_prompt, **kwargs)
