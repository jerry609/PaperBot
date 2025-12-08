import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import docker
    from docker.errors import DockerException, APIError
    HAS_DOCKER = True
except ImportError:
    docker = None
    DockerException = Exception
    APIError = Exception
    HAS_DOCKER = False

from .execution_result import ExecutionResult

logger = logging.getLogger(__name__)


class DockerExecutor:
    """
    轻量 Docker 执行器
    - 挂载代码目录只读
    - 可选独立 cache 目录（可写）
    - 限制 CPU/内存/超时
    - 可禁用网络
    """

    def __init__(self, image: str, cpu_shares: int = 1, mem_limit: str = "1g", network: bool = False):
        self.image = image
        self.cpu_shares = cpu_shares
        self.mem_limit = mem_limit
        self.network_disabled = not network
        self.client = None
        
        if HAS_DOCKER and docker is not None:
            try:
                self.client = docker.from_env()
            except DockerException as e:
                logger.warning(f"Docker not available: {e}")

    def available(self) -> bool:
        return self.client is not None

    def run(
        self,
        workdir: Path,
        commands: List[str],
        timeout_sec: int = 300,
        cache_dir: Optional[Path] = None,
        record_meta: bool = True,
    ) -> ExecutionResult:
        if not self.client:
            return ExecutionResult(status="error", exit_code=1, error="Docker client unavailable")

        container = None
        start = time.time()
        try:
            binds = {
                str(workdir): {"bind": "/workspace", "mode": "ro"},
            }
            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
                binds[str(cache_dir)] = {"bind": "/cache", "mode": "rw"}

            container = self.client.containers.run(
                self.image,
                command=["/bin/sh", "-c", " && ".join(commands)],
                working_dir="/workspace",
                detach=True,
                cpu_shares=self.cpu_shares,
                mem_limit=self.mem_limit,
                network_disabled=self.network_disabled,
                volumes=binds,
                environment={"PIP_CACHE_DIR": "/cache/pip"} if cache_dir else {},
            )

            exit_code = container.wait(timeout=timeout_sec)
            logs = container.logs(stdout=True, stderr=True).decode(errors="ignore")
            duration = time.time() - start
            status = "success" if exit_code.get("StatusCode", 1) == 0 else "failed"
            result = ExecutionResult(
                status=status,
                exit_code=exit_code.get("StatusCode", 1),
                logs=logs[-8000:],
                duration_sec=duration,
            )
            if record_meta:
                result.runtime_meta = {
                    "image": self.image,
                    "cpu_shares": self.cpu_shares,
                    "mem_limit": self.mem_limit,
                    "network_enabled": not self.network_disabled,
                    "timeout_sec": timeout_sec,
                }
            return result
        except APIError as e:
            logger.error(f"Docker API error: {e}")
            return ExecutionResult(status="error", exit_code=1, error=str(e))
        except Exception as e:
            logger.error(f"Docker exec error: {e}")
            return ExecutionResult(status="error", exit_code=1, error=str(e))
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

