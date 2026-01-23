"""Visualization utilities for FieldOS Engine."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

from ..rl.evaluate import EvaluationReport

logger = logging.getLogger("fieldos_engine.viz")


def plot_policy_performance(report: EvaluationReport, output_path: str) -> str:
    """
    Generate a bar chart of play performance per bucket.
    
    Args:
        report: Evaluation report from evaluate_policy_detailed
        output_path: Path to save the PNG file
        
    Returns:
        Absolute path to the saved file
    """
    
    # Extract data
    buckets = [bp.bucket for bp in report.bucket_performance]
    
    # Collect all plays seen across all buckets
    all_plays = set()
    for bp in report.bucket_performance:
        all_plays.update(bp.play_performance.keys())
    
    sorted_plays = sorted(list(all_plays))
    
    # Prepare data for grouped bar chart
    x = np.arange(len(buckets))
    width = 0.8 / len(sorted_plays)  # Distribute width among plays
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, play_id in enumerate(sorted_plays):
        # Get mean reward for this play in each bucket
        scores = []
        for bp in report.bucket_performance:
            scores.append(bp.play_performance.get(play_id, 0.0))
            
        offset = width * i
        rects = ax.bar(x + offset, scores, width, label=play_id)
        
    # Add labels
    ax.set_ylabel('Mean Reward (Yards + Bonuses)')
    ax.set_title(f'Play Performance by Situation - {report.policy_name}')
    ax.set_xticks(x + width * (len(sorted_plays) - 1) / 2)
    ax.set_xticklabels(buckets, rotation=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    # Save
    p = Path(output_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(p))
    plt.close()
    
    logger.info(f"Saved performance plot to {p}")
    return str(p)
