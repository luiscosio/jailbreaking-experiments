import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import amp
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanogcg.gcg import GCGResult
import nanogcg

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
    best_loss: float
    found_adversarial: bool

class EnhancedGCGConfig(nanogcg.GCGConfig):
    """Extended configuration for GCG optimization"""
    def __init__(
        self,
        num_steps: int = 1000,  # Increased default steps
        search_width: int = 512,
        batch_size: Optional[int] = None,
        success_threshold: float = 0.1,
        max_memory_usage: float = 0.9,
        use_flash_attention: bool = True,
        use_amp: bool = True,
        **kwargs
    ):
        kwargs['early_stop'] = False  # Disable built-in early stopping
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
            'max_memory_allocated': torch.cuda.max_memory_allocated() / 1024**3
        }
    return {}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Enhanced GCG Optimization for H100')
    parser.add_argument("--model", type=str, default="Qwen/Qwen-8B-Chat")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_steps", type=int, default=1000)  # Increased default steps
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
    """Run GCG with enhanced monitoring"""
    start_time = time.time()
    initial_metrics = monitor_gpu_metrics()
    
    # Single run of GCG optimization
    with amp.autocast('cuda', enabled=config.use_amp):
        result = nanogcg.run(model, tokenizer, messages, target, config)
    
    final_metrics = monitor_gpu_metrics()
    end_time = time.time()
    
    found_adversarial = result.best_loss < config.success_threshold
    if found_adversarial:
        logger.info(f"Successfully found adversarial example with loss: {result.best_loss}")
    else:
        logger.info(f"Completed optimization. Best loss: {result.best_loss}")
    
    optimization_metrics = OptimizationMetrics(
        total_time=end_time - start_time,
        peak_memory=final_metrics.get('max_memory_allocated', 0),
        best_loss=result.best_loss,
        found_adversarial=found_adversarial
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
        batch_size=args.search_width  # Initialize batch size to search width
    )
    
    messages = [{"role": "user", "content": args.prompt}]
    
    # Single optimization run
    result, metrics = run_optimized_gcg(model, tokenizer, messages, args.target, config)
    
    logger.info("\nOptimization Results:")
    logger.info(f"Total time: {metrics.total_time:.2f} seconds")
    logger.info(f"Peak memory usage: {metrics.peak_memory:.2f} GB")
    logger.info(f"Best loss achieved: {metrics.best_loss:.4f}")
    logger.info(f"Found adversarial example: {metrics.found_adversarial}")
    
    messages[-1]["content"] = messages[-1]["content"] + " " + result.best_string
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            do_sample=True,  # Enable sampling
            temperature=0.6,
            top_p=0.9,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=torch.ones_like(input_ids)
        )
    
    print("\nFinal Results:")
    print(f"Prompt:\n{messages[-1]['content']}\n")
    print(f"Generation:\n{tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]}")

if __name__ == "__main__":
    main()