"""
Performance monitoring and profiling for the streaming pipeline.
"""
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""
    name: str
    total_time: float = 0.0
    call_count: int = 0
    error_count: int = 0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=10))
    
    @property
    def avg_time(self) -> float:
        """Average processing time."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0
    
    @property
    def recent_avg(self) -> float:
        """Recent average (last 10 calls)."""
        return sum(self.recent_times) / len(self.recent_times) if self.recent_times else 0.0
    
    @property
    def error_rate(self) -> float:
        """Error rate as percentage."""
        return (self.error_count / self.call_count * 100) if self.call_count > 0 else 0.0


class PerformanceMonitor:
    """
    Monitor and profile the streaming pipeline performance.
    
    Tracks:
    - Per-stage latency
    - Throughput
    - Error rates
    - Buffer health
    - End-to-end latency
    """
    
    def __init__(self):
        self.stages: Dict[str, StageMetrics] = defaultdict(
            lambda: StageMetrics(name="unknown")
        )
        self.session_start = time.time()
        self.total_sentences = 0
        self.total_frames = 0
        self.total_audio_duration = 0.0
        
        # Latency tracking
        self.e2e_latencies = deque(maxlen=50)
        self.buffer_health_history = deque(maxlen=100)
    
    def start_stage(self, stage_name: str) -> float:
        """Start timing a stage. Returns start timestamp."""
        return time.time()
    
    def end_stage(self, stage_name: str, start_time: float, error: bool = False):
        """End timing a stage and record metrics."""
        elapsed = time.time() - start_time
        
        stage = self.stages[stage_name]
        stage.name = stage_name
        stage.total_time += elapsed
        stage.call_count += 1
        stage.recent_times.append(elapsed)
        
        if error:
            stage.error_count += 1
    
    def record_sentence(self, audio_duration: float):
        """Record a processed sentence."""
        self.total_sentences += 1
        self.total_audio_duration += audio_duration
    
    def record_frames(self, num_frames: int):
        """Record generated frames."""
        self.total_frames += num_frames
    
    def record_e2e_latency(self, latency: float):
        """Record end-to-end latency."""
        self.e2e_latencies.append(latency)
    
    def record_buffer_health(self, health: float):
        """Record buffer health (0.0 to 1.0)."""
        self.buffer_health_history.append(health)
    
    def get_summary(self) -> Dict:
        """Get performance summary."""
        elapsed = time.time() - self.session_start
        
        summary = {
            "session_duration": round(elapsed, 2),
            "total_sentences": self.total_sentences,
            "total_frames": self.total_frames,
            "total_audio_duration": round(self.total_audio_duration, 2),
            "sentences_per_second": round(self.total_sentences / elapsed, 2) if elapsed > 0 else 0,
            "frames_per_second": round(self.total_frames / elapsed, 2) if elapsed > 0 else 0,
            "realtime_factor": round(self.total_audio_duration / elapsed, 2) if elapsed > 0 else 0,
            "stages": {},
            "e2e_latency": {
                "avg": round(sum(self.e2e_latencies) / len(self.e2e_latencies), 3) if self.e2e_latencies else 0,
                "min": round(min(self.e2e_latencies), 3) if self.e2e_latencies else 0,
                "max": round(max(self.e2e_latencies), 3) if self.e2e_latencies else 0,
            },
            "buffer_health": {
                "avg": round(sum(self.buffer_health_history) / len(self.buffer_health_history), 2) if self.buffer_health_history else 0,
                "min": round(min(self.buffer_health_history), 2) if self.buffer_health_history else 0,
            }
        }
        
        # Add per-stage metrics
        for stage_name, metrics in self.stages.items():
            summary["stages"][stage_name] = {
                "avg_time": round(metrics.avg_time, 3),
                "recent_avg": round(metrics.recent_avg, 3),
                "call_count": metrics.call_count,
                "error_count": metrics.error_count,
                "error_rate": round(metrics.error_rate, 1),
            }
        
        return summary
    
    def print_summary(self):
        """Print formatted performance summary."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Session Duration: {summary['session_duration']}s")
        print(f"Total Sentences: {summary['total_sentences']}")
        print(f"Total Frames: {summary['total_frames']}")
        print(f"Audio Duration: {summary['total_audio_duration']}s")
        print(f"Realtime Factor: {summary['realtime_factor']}x")
        print(f"\nThroughput:")
        print(f"  Sentences/sec: {summary['sentences_per_second']}")
        print(f"  Frames/sec: {summary['frames_per_second']}")
        print(f"\nEnd-to-End Latency:")
        print(f"  Average: {summary['e2e_latency']['avg']}s")
        print(f"  Min: {summary['e2e_latency']['min']}s")
        print(f"  Max: {summary['e2e_latency']['max']}s")
        print(f"\nBuffer Health:")
        print(f"  Average: {summary['buffer_health']['avg']}")
        print(f"  Min: {summary['buffer_health']['min']}")
        print(f"\nStage Performance:")
        
        for stage_name, metrics in summary['stages'].items():
            print(f"  {stage_name}:")
            print(f"    Avg Time: {metrics['avg_time']}s")
            print(f"    Recent Avg: {metrics['recent_avg']}s")
            print(f"    Calls: {metrics['call_count']}")
            print(f"    Errors: {metrics['error_count']} ({metrics['error_rate']}%)")
        
        print("="*60 + "\n")
    
    def reset(self):
        """Reset all metrics."""
        self.stages.clear()
        self.session_start = time.time()
        self.total_sentences = 0
        self.total_frames = 0
        self.total_audio_duration = 0.0
        self.e2e_latencies.clear()
        self.buffer_health_history.clear()

