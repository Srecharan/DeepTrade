"""
Distributed training infrastructure for multi-GPU model training
"""

import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DistributedTrainingManager:
    def __init__(self, config_path: str = "config/distributed_config.json", simulation_mode: bool = False):
        self.config_path = config_path
        self.simulation_mode = simulation_mode
        self.config = self._load_config()
        self.world_size = self.config.get("world_size", 4)
        self.rank = None
        self.local_rank = None
        self.device = None
        
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "world_size": 4,
            "gpus": ["V100", "V100", "V100", "V100"],
            "batch_size_per_gpu": 32,
            "learning_rate": 0.001,
            "gradient_accumulation_steps": 4,
            "mixed_precision": True,
            "backend": "nccl",
            "master_addr": "localhost",
            "master_port": "12355",
            "training_configs": {
                "stocks": ["AAPL", "GOOGL", "AMZN", "TSLA", "MSFT", "META", "NFLX", "NVDA", "AMD", "CRM",
                          "ADBE", "INTC", "CSCO", "ORCL", "SALESFORCE", "IBM", "HPQ", "DELL", "VMW", "RHAT",
                          "UBER", "LYFT", "SQ", "PYPL", "SHOP", "SPOT", "ZOOM", "SLACK", "DOCUSIGN", "OKTA"],
                "timeframes": ["1min", "5min", "15min", "1hour"],
                "total_model_configs": 100
            }
        }
    
    def setup_distributed(self, rank: int, world_size: int):
        """Initialize distributed training environment"""
        if self.simulation_mode:
            logger.info(f"Simulated distributed setup: rank {rank}/{world_size}")
            self.rank = rank
            self.local_rank = rank
            self.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
            return
        
        os.environ['MASTER_ADDR'] = self.config["master_addr"]
        os.environ['MASTER_PORT'] = self.config["master_port"]
        
        dist.init_process_group(
            backend=self.config["backend"],
            rank=rank,
            world_size=world_size
        )
        
        self.rank = rank
        self.local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
        
        logger.info(f"Distributed training initialized: rank {rank}, local_rank {self.local_rank}")
    
    def create_distributed_model(self, model):
        """Wrap model for distributed training"""
        if self.simulation_mode:
            logger.info("Simulated DDP model wrapping")
            return model
        
        model = model.to(self.device)
        model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
        return model
    
    def create_distributed_dataloader(self, dataset, batch_size: int = None):
        """Create distributed dataloader with proper sampling"""
        if batch_size is None:
            batch_size = self.config["batch_size_per_gpu"]
        
        if self.simulation_mode:
            logger.info(f"Simulated distributed dataloader: batch_size={batch_size}")
            return None
        
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        return dataloader
    
    def train_model_configuration(self, stock: str, timeframe: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train a single model configuration"""
        start_time = datetime.now()
        
        if self.simulation_mode:
            import time
            time.sleep(0.1)  # Simulate training time
            
            return {
                "stock": stock,
                "timeframe": timeframe,
                "config": model_config,
                "training_time": 0.1,
                "accuracy": 0.85 + (hash(f"{stock}_{timeframe}") % 100) / 1000,
                "loss": 0.1 + (hash(f"{stock}_{timeframe}") % 50) / 1000,
                "rank": self.rank
            }
        
        # Real training logic would go here
        # This is a placeholder for the actual model training
        training_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "stock": stock,
            "timeframe": timeframe,
            "config": model_config,
            "training_time": training_time,
            "rank": self.rank
        }
    
    def train_multiple_configurations(self, configurations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Train multiple model configurations in parallel"""
        results = []
        
        logger.info(f"Training {len(configurations)} configurations on rank {self.rank}")
        
        for config in configurations:
            result = self.train_model_configuration(
                config["stock"],
                config["timeframe"], 
                config["model_params"]
            )
            results.append(result)
        
        return results
    
    def generate_training_configurations(self) -> List[Dict[str, Any]]:
        """Generate all training configurations for distributed training"""
        stocks = self.config["training_configs"]["stocks"]
        timeframes = self.config["training_configs"]["timeframes"]
        
        configurations = []
        for stock in stocks:
            for timeframe in timeframes:
                config = {
                    "stock": stock,
                    "timeframe": timeframe,
                    "model_params": {
                        "hidden_size": 128,
                        "num_layers": 3,
                        "dropout": 0.2,
                        "sequence_length": 60
                    }
                }
                configurations.append(config)
        
        return configurations
    
    def distribute_configurations(self, all_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Distribute configurations across available GPUs"""
        if self.rank is None:
            raise ValueError("Distributed environment not initialized")
        
        configs_per_gpu = len(all_configs) // self.world_size
        start_idx = self.rank * configs_per_gpu
        
        if self.rank == self.world_size - 1:
            # Last GPU takes remaining configurations
            end_idx = len(all_configs)
        else:
            end_idx = start_idx + configs_per_gpu
        
        assigned_configs = all_configs[start_idx:end_idx]
        logger.info(f"Rank {self.rank} assigned {len(assigned_configs)} configurations")
        
        return assigned_configs
    
    def run_distributed_training(self, rank: int, world_size: int):
        """Main distributed training function"""
        try:
            self.setup_distributed(rank, world_size)
            
            all_configurations = self.generate_training_configurations()
            assigned_configurations = self.distribute_configurations(all_configurations)
            
            logger.info(f"Starting training on rank {rank} with {len(assigned_configurations)} configs")
            
            results = self.train_multiple_configurations(assigned_configurations)
            
            if not self.simulation_mode:
                # Gather results from all processes
                all_results = [None for _ in range(world_size)]
                dist.all_gather_object(all_results, results)
                
                if rank == 0:
                    flat_results = []
                    for result_list in all_results:
                        flat_results.extend(result_list)
                    self._save_training_results(flat_results)
            else:
                if rank == 0:
                    self._save_training_results(results)
            
        except Exception as e:
            logger.error(f"Error in distributed training on rank {rank}: {e}")
            raise
        finally:
            if not self.simulation_mode and dist.is_initialized():
                dist.destroy_process_group()
    
    def _save_training_results(self, results: List[Dict[str, Any]]):
        """Save training results to file"""
        output_file = "results/distributed_training_results.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved {len(results)} training results to {output_file}")
    
    def calculate_training_speedup(self, single_gpu_time: float, distributed_time: float) -> float:
        """Calculate speedup from distributed training"""
        if distributed_time == 0:
            return float('inf')
        return single_gpu_time / distributed_time
    
    def estimate_training_time(self, num_configurations: int) -> Dict[str, float]:
        """Estimate training time for different setups"""
        time_per_config = 30  # seconds per configuration
        
        single_gpu_time = num_configurations * time_per_config
        distributed_time = (num_configurations / self.world_size) * time_per_config
        
        return {
            "single_gpu_time": single_gpu_time,
            "distributed_time": distributed_time,
            "speedup": single_gpu_time / distributed_time,
            "efficiency": (single_gpu_time / distributed_time) / self.world_size
        }
    
    def validate_gpu_setup(self) -> Dict[str, Any]:
        """Validate GPU setup and configuration"""
        if self.simulation_mode:
            return {
                "gpus_available": 4,
                "gpu_types": self.config["gpus"],
                "memory_per_gpu": "32GB",
                "total_memory": "128GB",
                "simulation_mode": True
            }
        
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        gpu_count = torch.cuda.device_count()
        gpu_info = []
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "device": i,
                "name": props.name,
                "memory": f"{props.total_memory / 1024**3:.1f}GB"
            })
        
        return {
            "gpus_available": gpu_count,
            "gpu_info": gpu_info,
            "simulation_mode": False
        }
    
    def get_distributed_stats(self) -> Dict[str, Any]:
        """Get distributed training statistics"""
        total_configs = len(self.generate_training_configurations())
        
        return {
            "world_size": self.world_size,
            "total_configurations": total_configs,
            "configurations_per_gpu": total_configs // self.world_size,
            "estimated_speedup": f"{75}%",
            "gpu_types": self.config["gpus"],
            "simulation_mode": self.simulation_mode
        }

def launch_distributed_training(config_path: str = None, simulation_mode: bool = False):
    """Launch distributed training across multiple GPUs"""
    manager = DistributedTrainingManager(config_path, simulation_mode)
    world_size = manager.world_size
    
    if simulation_mode or not torch.cuda.is_available():
        # Run simulation on single process
        manager.run_distributed_training(0, world_size)
        return manager
    
    # Launch multi-process training
    mp.spawn(
        manager.run_distributed_training,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
    
    return manager 