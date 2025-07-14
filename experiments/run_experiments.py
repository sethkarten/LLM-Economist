"""
Script to run the main experiments from the LLM Economist paper.
"""

import os
import sys
import subprocess
import argparse
from typing import List, Dict, Any


def run_command(cmd: List[str], description: str = ""):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Success: {description}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running {description}: {e}")
        print(f"stderr: {e.stderr}")
        return None


def rational_agents_experiment(args):
    """Run rational agents experiment."""
    base_cmd = [
        sys.executable, "-m", "llm_economist.main",
        "--scenario", "rational",
        "--num-agents", str(args.num_agents),
        "--worker-type", "LLM",
        "--planner-type", "LLM",
        "--max-timesteps", str(args.max_timesteps),
        "--history-len", str(args.history_len),
        "--two-timescale", str(args.two_timescale),
        "--prompt-algo", args.prompt_algo,
        "--llm", args.llm
    ]
    
    if args.wandb:
        base_cmd.append("--wandb")
    
    if args.port:
        base_cmd.extend(["--port", str(args.port)])
    
    if args.service:
        base_cmd.extend(["--service", args.service])
    
    return run_command(base_cmd, "Rational Agents Experiment")


def bounded_rational_experiment(args):
    """Run bounded rational agents experiment."""
    base_cmd = [
        sys.executable, "-m", "llm_economist.main",
        "--scenario", "bounded",
        "--num-agents", str(args.num_agents),
        "--worker-type", "LLM",
        "--planner-type", "LLM",
        "--max-timesteps", str(args.max_timesteps),
        "--history-len", str(args.history_len),
        "--two-timescale", str(args.two_timescale),
        "--prompt-algo", args.prompt_algo,
        "--llm", args.llm,
        "--percent-ego", str(args.percent_ego),
        "--percent-alt", str(args.percent_alt),
        "--percent-adv", str(args.percent_adv)
    ]
    
    if args.wandb:
        base_cmd.append("--wandb")
    
    if args.port:
        base_cmd.extend(["--port", str(args.port)])
    
    if args.service:
        base_cmd.extend(["--service", args.service])
    
    return run_command(base_cmd, "Bounded Rational Agents Experiment")


def democratic_voting_experiment(args):
    """Run democratic voting experiment."""
    base_cmd = [
        sys.executable, "-m", "llm_economist.main",
        "--scenario", "democratic",
        "--num-agents", str(args.num_agents),
        "--worker-type", "LLM",
        "--planner-type", "LLM",
        "--max-timesteps", str(args.max_timesteps),
        "--history-len", str(args.history_len),
        "--two-timescale", str(args.two_timescale),
        "--prompt-algo", args.prompt_algo,
        "--llm", args.llm
    ]
    
    if args.wandb:
        base_cmd.append("--wandb")
    
    if args.port:
        base_cmd.extend(["--port", str(args.port)])
    
    if args.service:
        base_cmd.extend(["--service", args.service])
    
    return run_command(base_cmd, "Democratic Voting Experiment")


def llm_comparison_experiment(args):
    """Run LLM comparison experiment."""
    models = ["gpt-4o-mini", "llama3:8b", "meta-llama/llama-3.1-8b-instruct"]
    
    for model in models:
        base_cmd = [
            sys.executable, "-m", "llm_economist.main",
            "--scenario", "rational",
            "--num-agents", str(args.num_agents),
            "--worker-type", "LLM",
            "--planner-type", "LLM",
            "--max-timesteps", str(args.max_timesteps),
            "--history-len", str(args.history_len),
            "--two-timescale", str(args.two_timescale),
            "--prompt-algo", args.prompt_algo,
            "--llm", model
        ]
        
        if args.wandb:
            base_cmd.append("--wandb")
        
        if args.port and "llama" in model:
            base_cmd.extend(["--port", str(args.port)])
        
        if args.service and "llama" in model:
            base_cmd.extend(["--service", args.service])
        
        run_command(base_cmd, f"LLM Comparison - {model}")


def scalability_experiment(args):
    """Run scalability experiment with different numbers of agents."""
    agent_counts = [5, 10, 25, 50, 100]
    
    for num_agents in agent_counts:
        base_cmd = [
            sys.executable, "-m", "llm_economist.main",
            "--scenario", "rational",
            "--num-agents", str(num_agents),
            "--worker-type", "LLM",
            "--planner-type", "LLM",
            "--max-timesteps", str(args.max_timesteps),
            "--history-len", str(args.history_len),
            "--two-timescale", str(args.two_timescale),
            "--prompt-algo", args.prompt_algo,
            "--llm", args.llm
        ]
        
        if args.wandb:
            base_cmd.append("--wandb")
        
        if args.port:
            base_cmd.extend(["--port", str(args.port)])
        
        if args.service:
            base_cmd.extend(["--service", args.service])
        
        run_command(base_cmd, f"Scalability Test - {num_agents} agents")



def tax_year_ablation_experiment(args):
    """Run tax year length ablation experiment."""
    timescales = [5, 10, 25, 50, 100]
    
    for timescale in timescales:
        # Adjust max_timesteps to have similar number of tax years
        max_timesteps = timescale * 20  # 20 tax years
        
        base_cmd = [
            sys.executable, "-m", "llm_economist.main",
            "--scenario", "rational",
            "--num-agents", str(args.num_agents),
            "--worker-type", "LLM",
            "--planner-type", "LLM",
            "--max-timesteps", str(max_timesteps),
            "--history-len", str(args.history_len),
            "--two-timescale", str(timescale),
            "--prompt-algo", args.prompt_algo,
            "--llm", args.llm
        ]
        
        if args.wandb:
            base_cmd.append("--wandb")
        
        if args.port:
            base_cmd.extend(["--port", str(args.port)])
        
        if args.service:
            base_cmd.extend(["--service", args.service])
        
        run_command(base_cmd, f"Tax Year Ablation - {timescale} steps")


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Run LLM Economist experiments")
    
    # Experiment selection
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["rational", "bounded", "democratic", 
                                "llm_comparison", "scalability", 
                                "tax_year_ablation", "all"],
                        help="Which experiment to run")
    
    # Common parameters
    parser.add_argument("--num-agents", type=int, default=5,
                        help="Number of agents")
    parser.add_argument("--max-timesteps", type=int, default=2500,
                        help="Maximum timesteps")
    parser.add_argument("--history-len", type=int, default=50,
                        help="History length")
    parser.add_argument("--two-timescale", type=int, default=25,
                        help="Two timescale parameter")
    parser.add_argument("--prompt-algo", type=str, default="io",
                        choices=["io", "cot"],
                        help="Prompting algorithm")
    parser.add_argument("--llm", type=str, default="gpt-4o-mini",
                        help="LLM model to use")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for local LLM server")
    parser.add_argument("--service", type=str, default="vllm",
                        choices=["vllm", "ollama"],
                        help="Local LLM service")
    
    # Bounded rationality parameters
    parser.add_argument("--percent-ego", type=int, default=100,
                        help="Percentage of egotistical agents")
    parser.add_argument("--percent-alt", type=int, default=0,
                        help="Percentage of altruistic agents")
    parser.add_argument("--percent-adv", type=int, default=0,
                        help="Percentage of adversarial agents")
    
    # Logging
    parser.add_argument("--wandb", action="store_true",
                        help="Enable WandB logging")
    
    args = parser.parse_args()
    
    # Run selected experiment(s)
    if args.experiment == "rational" or args.experiment == "all":
        rational_agents_experiment(args)
    
    if args.experiment == "bounded" or args.experiment == "all":
        bounded_rational_experiment(args)
    
    if args.experiment == "democratic" or args.experiment == "all":
        democratic_voting_experiment(args)
    
    if args.experiment == "llm_comparison" or args.experiment == "all":
        llm_comparison_experiment(args)
    
    if args.experiment == "scalability" or args.experiment == "all":
        scalability_experiment(args)
    

    
    if args.experiment == "tax_year_ablation" or args.experiment == "all":
        tax_year_ablation_experiment(args)
    
    print("Experiments completed!")


if __name__ == "__main__":
    main() 