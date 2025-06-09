#!/usr/bin/env python3
"""
DeepTrade AI - Distributed Training Runner
Easy-to-use script for launching distributed training infrastructure
"""

import os
import sys
import argparse
import json
import subprocess
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from infrastructure.distributed_training import main as distributed_main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check GPU availability and configuration"""
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Distributed training requires GPUs.")
            return False
        
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPU(s) available")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        return True
        
    except ImportError:
        logger.error("PyTorch not installed. Please install with CUDA support.")
        return False

def check_distributed_dependencies():
    """Check if distributed training dependencies are available"""
    required_packages = [
        'torch',
        'torchvision', 
        'numpy',
        'pandas',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.info("Install with: pip install torch torchvision numpy pandas scikit-learn")
        return False
    
    return True

def setup_aws_environment():
    """Setup AWS environment for distributed training"""
    
    # Check AWS credentials
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    
    if not aws_access_key or not aws_secret_key:
        logger.warning("AWS credentials not found in environment variables")
        logger.info("For AWS deployment, set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return False
    
    logger.info(f"AWS credentials configured for region: {aws_region}")
    return True

def create_training_environment():
    """Create necessary directories and environment setup"""
    
    directories = [
        'models/distributed_checkpoints',
        'results/distributed_training',
        'logs/distributed_training',
        'data/distributed_cache'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def run_local_distributed_training(args):
    """Run distributed training on local GPUs"""
    
    logger.info("Starting local distributed training...")
    
    # Set environment variables for local distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Run distributed training
    try:
        if args.simulate:
            # Run in simulation mode
            subprocess.run([
                sys.executable, 
                'infrastructure/distributed_training.py',
                '--simulate'
            ], check=True)
        else:
            # Run actual distributed training
            subprocess.run([
                sys.executable,
                'infrastructure/distributed_training.py'
            ], check=True)
            
        logger.info("Distributed training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Distributed training failed: {e}")
        return False
    
    return True

def launch_aws_distributed_training(args):
    """Launch distributed training on AWS EC2 instances"""
    
    logger.info("Launching distributed training on AWS...")
    
    # AWS configuration
    config = {
        'instance_type': 'p3.8xlarge',  # 4x V100 GPUs
        'ami_id': 'ami-0c02fb55956c7d316',  # Deep Learning AMI
        'key_name': args.aws_key_name,
        'security_group': args.aws_security_group,
        'subnet_id': args.aws_subnet_id,
        'iam_role': 'DeepTradeDistributedTraining'
    }
    
    # Create EC2 launch script
    launch_script = f"""#!/bin/bash
# DeepTrade AI Distributed Training Launch Script

# Update system
sudo apt-get update
sudo apt-get install -y git python3-pip

# Clone repository
git clone https://github.com/your-username/DeepTrade.git
cd DeepTrade

# Install dependencies
pip3 install -r requirements.txt

# Set up environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export MASTER_ADDR={config['instance_type']}
export MASTER_PORT=12355

# Run distributed training
python3 infrastructure/distributed_training.py --world-size 4

# Upload results to S3
aws s3 cp results/ s3://deeptrade-training-results/ --recursive
"""
    
    # Save launch script
    with open('scripts/aws_launch.sh', 'w') as f:
        f.write(launch_script)
    
    logger.info("AWS launch script created: scripts/aws_launch.sh")
    logger.info("To launch AWS training:")
    logger.info(f"1. Create EC2 instance: {config['instance_type']}")
    logger.info("2. Upload and run aws_launch.sh")
    logger.info("3. Monitor training logs")
    
    return True

def monitor_training_progress():
    """Monitor distributed training progress"""
    
    results_dir = Path('results/distributed_training')
    
    if not results_dir.exists():
        logger.info("No training results found yet...")
        return
    
    # Check for simulation results
    sim_file = results_dir / 'simulation_results.json'
    if sim_file.exists():
        with open(sim_file, 'r') as f:
            results = json.load(f)
        
        logger.info("Simulation Results:")
        for gpu_id, gpu_data in results.items():
            logger.info(f"  {gpu_id}: {gpu_data['assigned_models']} models, "
                       f"~{gpu_data['estimated_training_time']} estimated time")
    
    # Check for actual training results
    result_files = list(results_dir.glob('gpu_*_results.json'))
    if result_files:
        logger.info(f"Found {len(result_files)} GPU result files")
        
        total_models = 0
        avg_loss = 0
        
        for result_file in result_files:
            with open(result_file, 'r') as f:
                gpu_results = json.load(f)
            
            total_models += len(gpu_results)
            if gpu_results:
                gpu_avg_loss = sum(r['final_loss'] for r in gpu_results.values()) / len(gpu_results)
                avg_loss += gpu_avg_loss
        
        if result_files:
            avg_loss /= len(result_files)
            logger.info(f"Training Progress: {total_models} models completed, avg loss: {avg_loss:.4f}")

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='DeepTrade AI Distributed Training Runner')
    parser.add_argument('--mode', choices=['local', 'aws', 'simulation'], default='simulation',
                       help='Training mode: local GPUs, AWS EC2, or simulation')
    parser.add_argument('--simulate', action='store_true',
                       help='Run in simulation mode (no actual training)')
    parser.add_argument('--monitor', action='store_true',
                       help='Monitor training progress')
    parser.add_argument('--setup-only', action='store_true',
                       help='Only setup environment, do not start training')
    
    # AWS-specific arguments
    parser.add_argument('--aws-key-name', default='deeptrade-key',
                       help='AWS key pair name')
    parser.add_argument('--aws-security-group', default='deeptrade-sg',
                       help='AWS security group')
    parser.add_argument('--aws-subnet-id', 
                       help='AWS subnet ID')
    
    args = parser.parse_args()
    
    logger.info("🚀 DeepTrade AI Distributed Training Runner")
    logger.info("=" * 50)
    
    # Monitor mode
    if args.monitor:
        monitor_training_progress()
        return
    
    # Check dependencies
    if not check_distributed_dependencies():
        sys.exit(1)
    
    # Setup environment
    create_training_environment()
    
    if args.setup_only:
        logger.info("Environment setup completed.")
        return
    
    # Run based on mode
    if args.mode == 'simulation':
        logger.info("Running in simulation mode...")
        success = run_local_distributed_training(argparse.Namespace(simulate=True))
        
    elif args.mode == 'local':
        if not check_gpu_availability():
            logger.error("GPU check failed. Cannot run local distributed training.")
            sys.exit(1)
        
        success = run_local_distributed_training(args)
        
    elif args.mode == 'aws':
        if not setup_aws_environment():
            logger.error("AWS setup failed. Cannot run AWS distributed training.")
            sys.exit(1)
        
        success = launch_aws_distributed_training(args)
    
    if success:
        logger.info("✅ Distributed training setup completed successfully!")
        logger.info("Check results in: results/distributed_training/")
        logger.info("Monitor with: python scripts/run_distributed_training.py --monitor")
    else:
        logger.error("❌ Distributed training setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 