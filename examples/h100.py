import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import amp  # Updated import
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanogcg.gcg import GCGResult
import nanogcg
from tqdm import tqdm

# Set CUDA deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gcg_optimization.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationMetrics:
    """Tracks optimization metrics during GCG runs"""
    total_time: float
    peak_memory: float
    iterations: int
    best_loss: float
    found_adversarial: bool
    average_iteration_time: float

class EnhancedGCGConfig(nanogcg.GCGConfig):
    """Extended configuration for GCG optimization"""
    def __init__(
        self,
        num_steps: int = 250,
        search_width: int = 512,
        batch_size: Optional[int] = None,
        success_threshold: float = 0.1,
        max_memory_usage: float = 0.9,
        use_flash_attention: bool = True,
        use_amp: bool = True,
        **kwargs
    ):
        super().__init__(num_steps=num_steps, search_width=search_width, batch_size=batch_size, **kwargs)
        self.success_threshold = success_threshold
        self.max_memory_usage = max_memory_usage
        self.use_flash_attention = use_flash_attention
        self.use_amp = use_amp

def setup_h100_optimizations(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """Configure model for optimal H100 performance"""
    # Enable tensor cores
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable flash attention if available
    if hasattr(model.config, 'use_flash_attention'):
        model.config.use_flash_attention = True
    
    # Set optimal memory formats
    model = model.to(memory_format=torch.channels_last)
    
    return model

def monitor_gpu_metrics() -> dict:
    """Monitor GPU metrics during optimization"""
    if torch.cuda.is_available():
        return {
            'memory_allocated': torch.cuda.memory_allocated() / 1024**3,
            'memory_reserved': torch.cuda.memory_reserved() / 1024**3,
            'max_memory_allocated': torch.cuda.max_memory_allocated() / 1024**3,
            'device_utilization': torch.cuda.utilization()
        }
    return {}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Enhanced GCG Optimization for H100')
    parser.add_argument("--model", type=str, default="Qwen/Qwen-8B-Chat")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_steps", type=int, default=250)
    parser.add_argument("--search_width", type=int, default=512)
    parser.add_argument("--success_threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def run_optimized_gcg(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list,
    target: str,
    config: EnhancedGCGConfig
) -> Tuple[GCGResult, OptimizationMetrics]:
    """Run GCG with enhanced monitoring and optimization"""
    start_time = time.time()
    
    # Initialize batch size if None
    if config.batch_size is None:
        config.batch_size = config.search_width
    
    # Store original batch size for reference
    original_batch_size = config.batch_size
    
    scaler = amp.GradScaler('cuda') if config.use_amp else None
    metrics = []
    found_adversarial = False
    result = None
    
    with tqdm(total=config.num_steps, desc="GCG Optimization") as pbar:
        for step in range(config.num_steps):
            step_start = time.time()
            
            with amp.autocast('cuda', enabled=config.use_amp):
                result = nanogcg.run(model, tokenizer, messages, target, config)
            
            current_metrics = monitor_gpu_metrics()
            metrics.append(current_metrics)
            
            if result.best_loss < config.success_threshold:
                found_adversarial = True
                logger.info(f"Found successful adversarial example at step {step}")
                break
            
            if current_metrics.get('memory_allocated', 0) > config.max_memory_usage:
                logger.warning("Memory usage exceeded threshold, reducing batch size")
                config.batch_size = max(1, config.batch_size // 2)
                logger.info(f"New batch size: {config.batch_size} (original was {original_batch_size})")
            
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{result.best_loss:.4f}",
                'memory': f"{current_metrics.get('memory_allocated', 0):.2f}GB",
                'batch_size': config.batch_size
            })
    
    # Reset batch size to original value
    config.batch_size = original_batch_size
    
    end_time = time.time()
    optimization_metrics = OptimizationMetrics(
        total_time=end_time - start_time,
        peak_memory=max(m.get('memory_allocated', 0) for m in metrics),
        iterations=len(metrics),
        best_loss=result.best_loss if result else float('inf'),
        found_adversarial=found_adversarial,
        average_iteration_time=(end_time - start_time) / len(metrics)
    )
    
    return result, optimization_metrics

def main():
    args = parse_args()
    logger.info("Starting GCG optimization with enhanced configuration")
    
    # Set deterministic behavior
    torch.use_deterministic_algorithms(True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        device_map="auto",
    )
    model = setup_h100_optimizations(model)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    config = EnhancedGCGConfig(
        num_steps=args.num_steps,
        search_width=args.search_width,
        success_threshold=args.success_threshold,
        seed=args.seed,
        early_stop=True
    )
    
    messages = [{"role": "user", "content": args.prompt}]
    
    result, metrics = run_optimized_gcg(model, tokenizer, messages, args.target, config)
    
    logger.info("\nOptimization Results:")
    logger.info(f"Total time: {metrics.total_time:.2f} seconds")
    logger.info(f"Peak memory usage: {metrics.peak_memory:.2f} GB")
    logger.info(f"Average iteration time: {metrics.average_iteration_time:.3f} seconds")
    logger.info(f"Found adversarial example: {metrics.found_adversarial}")
    logger.info(f"Best loss achieved: {metrics.best_loss:.4f}")
    
    messages[-1]["content"] = messages[-1]["content"] + " " + result.best_string
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id
        )
    
    print("\nFinal Results:")
    print(f"Prompt:\n{messages[-1]['content']}\n")
    print(f"Generation:\n{tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]}")

if __name__ == "__main__":
    main()