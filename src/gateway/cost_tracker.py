import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
from src.config import settings

class CostTracker:
    """
    Tracks and persists token usage, cost, and latency metrics per model and task type.
    """
    def __init__(self, stats_file: Optional[Path] = None):
        self.stats_file = stats_file or (settings.outputs_dir / "gateway_costs.json")
        self.stats = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.stats_file.exists():
            try:
                return json.loads(self.stats_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to load gateway cost stats: {e}")
        return {
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_requests": 0,
            "total_latency": 0.0,
            "by_task": {},
            "by_model": {}
        }

    def save(self):
        try:
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            self.stats_file.write_text(
                json.dumps(self.stats, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Failed to save gateway cost stats: {e}")

    def track_request(
        self,
        task: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        latency: float
    ):
        total = prompt_tokens + completion_tokens
        
        self.stats["total_tokens"] += total
        self.stats["total_cost"] += round(cost, 6)
        self.stats["total_requests"] += 1
        self.stats["total_latency"] += round(latency, 4)

        # Task breakdown
        if task not in self.stats["by_task"]:
            self.stats["by_task"][task] = {"requests": 0, "tokens": 0, "cost": 0.0, "latency": 0.0}
        t_stats = self.stats["by_task"][task]
        t_stats["requests"] += 1
        t_stats["tokens"] += total
        t_stats["cost"] = round(t_stats["cost"] + cost, 6)
        t_stats["latency"] = round(t_stats["latency"] + latency, 4)

        # Model breakdown
        if model not in self.stats["by_model"]:
            self.stats["by_model"][model] = {"requests": 0, "tokens": 0, "cost": 0.0, "latency": 0.0}
        m_stats = self.stats["by_model"][model]
        m_stats["requests"] += 1
        m_stats["tokens"] += total
        m_stats["cost"] = round(m_stats["cost"] + cost, 6)
        m_stats["latency"] = round(m_stats["latency"] + latency, 4)

        self.save()

cost_tracker = CostTracker()
