# utils/smart_downloader.py

import asyncio
import time
from typing import Dict, List, Any, Optional
from collections import deque
from dataclasses import dataclass
import logging
from .downloader import PaperDownloader

@dataclass
class DownloadStats:
    """下载统计信息"""
    total_attempts: int = 0
    successful_downloads: int = 0
    failed_downloads: int = 0
    cached_hits: int = 0
    avg_download_time: float = 0.0
    current_success_rate: float = 1.0
    consecutive_failures: int = 0

class SmartDownloadManager:
    """智能下载管理器 - 动态调整并发数"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 创建基础下载器
        self.downloader = PaperDownloader(config)
        
        # 并发控制参数
        self.min_concurrent = 1      # 最小并发数
        self.max_concurrent = 4      # 最大并发数  
        self.current_concurrent = 2  # 当前并发数
        
        # 性能监控参数
        self.stats = DownloadStats()
        self.recent_times = deque(maxlen=10)  # 最近10次下载时间
        self.adjustment_threshold = 5         # 调整并发数的评估周期
        
        # 安全参数
        self.failure_threshold = 0.3  # 失败率阈值 (30%)
        self.slow_threshold = 10.0    # 慢下载阈值 (10秒)
        self.rest_interval = 1.0      # 请求间隔
        
        # 创建信号量
        self.semaphore = asyncio.Semaphore(self.current_concurrent)
        
        self.logger.info(f"智能下载管理器初始化 - 并发范围: {self.min_concurrent}-{self.max_concurrent}")

    async def download_papers_smart(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """智能批量下载论文"""
        if not papers:
            return []
        
        valid_papers = [p for p in papers if p.get('url') and p.get('url').strip()]
        if not valid_papers:
            self.logger.warning("没有找到有效的PDF下载链接")
            return []
        
        self.logger.info(f"🚀 开始智能下载 {len(valid_papers)} 篇论文")
        self.logger.info(f"📊 初始并发数: {self.current_concurrent}")
        
        start_time = time.time()
        results = []
        
        # 分批处理，每批动态调整并发数
        batch_size = max(8, self.current_concurrent * 2)  # 批次大小
        
        for i in range(0, len(valid_papers), batch_size):
            batch = valid_papers[i:i + batch_size]
            batch_results = await self._process_batch(batch, i, len(valid_papers))
            results.extend(batch_results)
            
            # 动态调整并发数
            await self._adjust_concurrency()
            
            # 批次间休息
            if i + batch_size < len(valid_papers):
                await asyncio.sleep(self.rest_interval)
        
        # 最终统计
        total_time = time.time() - start_time
        self._print_final_stats(results, total_time)
        
        return results

    async def _process_batch(self, batch: List[Dict[str, Any]], start_idx: int, total: int) -> List[Dict[str, Any]]:
        """处理一个批次的下载"""
        self.logger.info(f"\n📦 处理批次 [{start_idx+1}-{min(start_idx+len(batch), total)}/{total}] - 并发数: {self.current_concurrent}")
        
        # 创建下载任务
        tasks = []
        for i, paper in enumerate(batch):
            task = self._download_with_monitoring(paper, start_idx + i + 1, total)
            tasks.append(task)
        
        # 执行批次下载
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"批次下载异常: {str(result)}")
                processed_results.append({'success': False, 'error': str(result)})
            else:
                processed_results.append(result)
        
        return processed_results

    async def _download_with_monitoring(self, paper: Dict[str, Any], index: int, total: int) -> Dict[str, Any]:
        """带监控的单篇论文下载"""
        async with self.semaphore:
            start_time = time.time()
            
            try:
                print(f"🔄 [{index}/{total}] 下载: {paper['title'][:50]}...")
                
                # 执行下载
                result = await self.downloader.download_paper(paper['url'], paper['title'])
                download_time = time.time() - start_time
                
                # 更新统计信息
                self._update_stats(result, download_time)
                
                # 显示结果
                if result and result.get('success'):
                    if result.get('cached'):
                        print(f"📋 缓存命中 (耗时: {download_time:.1f}s)")
                    else:
                        size_kb = result.get('size', 0) / 1024
                        print(f"✅ 下载成功 (耗时: {download_time:.1f}s, 大小: {size_kb:.1f}KB)")
                else:
                    error_msg = result.get('error', '未知错误') if result else '下载失败'
                    print(f"❌ 下载失败: {error_msg}")
                
                return result or {'success': False, 'error': '下载失败'}
                
            except Exception as e:
                download_time = time.time() - start_time
                self._update_stats({'success': False}, download_time)
                self.logger.error(f"下载异常 [{index}/{total}]: {str(e)}")
                print(f"❌ 下载异常: {str(e)}")
                return {'success': False, 'error': str(e)}

    def _update_stats(self, result: Dict[str, Any], download_time: float):
        """更新下载统计信息"""
        self.stats.total_attempts += 1
        
        if result and result.get('success'):
            if result.get('cached'):
                self.stats.cached_hits += 1
            else:
                self.stats.successful_downloads += 1
                self.recent_times.append(download_time)
            self.stats.consecutive_failures = 0
        else:
            self.stats.failed_downloads += 1
            self.stats.consecutive_failures += 1
        
        # 计算成功率
        if self.stats.total_attempts > 0:
            self.stats.current_success_rate = (
                self.stats.successful_downloads + self.stats.cached_hits
            ) / self.stats.total_attempts
        
        # 计算平均下载时间
        if self.recent_times:
            self.stats.avg_download_time = sum(self.recent_times) / len(self.recent_times)

    async def _adjust_concurrency(self):
        """动态调整并发数"""
        if self.stats.total_attempts < self.adjustment_threshold:
            return  # 样本太少，不调整
        
        old_concurrent = self.current_concurrent
        
        # 决策逻辑
        if self.stats.consecutive_failures >= 3:
            # 连续失败，降低并发
            self.current_concurrent = max(self.min_concurrent, self.current_concurrent - 1)
            reason = f"连续失败{self.stats.consecutive_failures}次"
            
        elif self.stats.current_success_rate < self.failure_threshold:
            # 成功率太低，降低并发
            self.current_concurrent = max(self.min_concurrent, self.current_concurrent - 1)
            reason = f"成功率过低({self.stats.current_success_rate:.1%})"
            
        elif self.stats.avg_download_time > self.slow_threshold:
            # 平均速度太慢，降低并发
            self.current_concurrent = max(self.min_concurrent, self.current_concurrent - 1)
            reason = f"平均速度过慢({self.stats.avg_download_time:.1f}s)"
            
        elif (self.stats.current_success_rate > 0.8 and 
              self.stats.avg_download_time < 5.0 and 
              self.stats.consecutive_failures == 0):
            # 表现良好，增加并发
            self.current_concurrent = min(self.max_concurrent, self.current_concurrent + 1)
            reason = f"性能良好(成功率{self.stats.current_success_rate:.1%})"
        else:
            return  # 保持当前并发数
        
        # 如果并发数发生变化，更新信号量
        if old_concurrent != self.current_concurrent:
            self.logger.info(f"🔧 调整并发数: {old_concurrent} → {self.current_concurrent} ({reason})")
            
            # 创建新的信号量
            self.semaphore = asyncio.Semaphore(self.current_concurrent)
            
            # 调整后稍作休息
            await asyncio.sleep(2.0)

    def _print_final_stats(self, results: List[Dict[str, Any]], total_time: float):
        """打印最终统计信息"""
        success_count = sum(1 for r in results if r.get('success'))
        cached_count = sum(1 for r in results if r.get('success') and r.get('cached'))
        download_count = success_count - cached_count
        
        print(f"\n🎉 智能下载完成统计:")
        print(f"✅ 成功下载: {success_count}/{len(results)} 篇论文")
        print(f"📋 缓存命中: {cached_count} 篇")
        print(f"⬇️  实际下载: {download_count} 篇")
        print(f"⏱️  总耗时: {total_time:.1f} 秒")
        print(f"🔧 最终并发数: {self.current_concurrent}")
        
        if download_count > 0:
            avg_time = total_time / download_count
            print(f"📊 平均速度: {avg_time:.2f} 秒/篇")
        
        success_rate = success_count / len(results) if results else 0
        print(f"📈 成功率: {success_rate:.1%}")
        
        # 性能总结
        if success_rate >= 0.9:
            print(f"🏆 下载性能: 优秀")
        elif success_rate >= 0.8:
            print(f"👍 下载性能: 良好")
        elif success_rate >= 0.6:
            print(f"⚠️  下载性能: 一般")
        else:
            print(f"❌ 下载性能: 需要优化")