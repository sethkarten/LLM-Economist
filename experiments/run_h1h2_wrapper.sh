#!/bin/bash
# Wrapper to install package and run REINFORCE++ experiments

set -e

echo "Installing llm_economist package..."
pip install -e .

echo "Running REINFORCE++ experiment..."
python experiments/run_reinforce_h1h2.py "$@"
