"""
依赖注入容器单元测试
"""

import pytest

try:
    from src.paperbot.core.di import Container, inject
except ImportError:
    from core.di import Container, inject


class TestContainer:
    """Container 测试"""
    
    def setup_method(self):
        """每个测试前重置容器"""
        Container._instance = None
    
    def test_singleton_instance(self):
        c1 = Container.instance()
        c2 = Container.instance()
        assert c1 is c2
    
    def test_register_and_resolve(self):
        container = Container.instance()
        
        class MyService:
            pass
        
        container.register(MyService, lambda: MyService())
        
        instance = container.resolve(MyService)
        assert isinstance(instance, MyService)
    
    def test_singleton_returns_same_instance(self):
        container = Container.instance()
        
        class MySingleton:
            pass
        
        container.register(MySingleton, lambda: MySingleton(), singleton=True)
        
        i1 = container.resolve(MySingleton)
        i2 = container.resolve(MySingleton)
        assert i1 is i2
    
    def test_non_singleton_returns_new_instance(self):
        container = Container.instance()
        
        class MyTransient:
            pass
        
        container.register(MyTransient, lambda: MyTransient(), singleton=False)
        
        i1 = container.resolve(MyTransient)
        i2 = container.resolve(MyTransient)
        assert i1 is not i2
    
    def test_resolve_unregistered_raises(self):
        container = Container.instance()
        
        class Unregistered:
            pass
        
        with pytest.raises(ValueError):
            container.resolve(Unregistered)


class TestInjectDecorator:
    """inject 装饰器测试"""
    
    def setup_method(self):
        Container._instance = None
    
    def test_inject_provides_dependency(self):
        container = Container.instance()
        
        class Logger:
            def log(self, msg):
                return f"LOG: {msg}"
        
        container.register(Logger, lambda: Logger(), singleton=True)
        
        @inject(Logger)
        class MyClass:
            def __init__(self, logger: Logger = None):
                self.logger = logger
            
            def do_something(self):
                return self.logger.log("test")
        
        obj = MyClass()
        assert obj.do_something() == "LOG: test"
    
    def test_inject_does_not_override_explicit(self):
        container = Container.instance()
        
        class Logger:
            def __init__(self, name="default"):
                self.name = name
        
        container.register(Logger, lambda: Logger("injected"), singleton=True)
        
        @inject(Logger)
        class MyClass:
            def __init__(self, logger: Logger = None):
                self.logger = logger
        
        custom_logger = Logger("custom")
        obj = MyClass(logger=custom_logger)
        assert obj.logger.name == "custom"

