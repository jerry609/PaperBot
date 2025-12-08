"""
轻量依赖注入容器，用于解耦 Agent/执行器与具体实现。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Type, TypeVar

T = TypeVar("T")


class Container:
    _instance: "Container" | None = None

    def __init__(self) -> None:
        self._factories: Dict[Type[Any], tuple[Callable[[], Any], bool]] = {}
        self._singletons: Dict[Type[Any], Any] = {}

    @classmethod
    def instance(cls) -> "Container":
        if cls._instance is None:
            cls._instance = Container()
        return cls._instance

    def register(self, interface: Type[T], factory: Callable[[], T], singleton: bool = False) -> None:
        """注册依赖工厂。"""
        self._factories[interface] = (factory, singleton)

    def resolve(self, interface: Type[T]) -> T:
        """获取依赖实例。"""
        if interface in self._singletons:
            return self._singletons[interface]

        factory_tuple = self._factories.get(interface)
        if not factory_tuple:
            raise ValueError(f"No factory registered for {interface}")

        factory, as_singleton = factory_tuple
        instance = factory()
        if as_singleton:
            self._singletons[interface] = instance
        return instance


def inject(*dependencies: Type[Any]):
    """
    类装饰器：为 __init__ 注入依赖。
    仅当调用方未显式传递对应 kwarg 时才注入，避免覆盖。
    """

    def decorator(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            container = Container.instance()
            for dep in dependencies:
                key = dep.__name__
                # 使用类名的小写作为默认 kw 名称，避免覆盖显式传入
                kw_key = key[0].lower() + key[1:]
                if kw_key not in kwargs:
                    try:
                        kwargs[kw_key] = container.resolve(dep)
                    except ValueError:
                        pass  # 未注册则跳过
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls

    return decorator

