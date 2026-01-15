from setuptools import setup, find_packages

setup(
    name="llm_economist",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "vllm>=0.6.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.25.0",
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
        "wandb>=0.15.0",
    ],
    python_requires=">=3.8",
)
