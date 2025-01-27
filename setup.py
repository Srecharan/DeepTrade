from setuptools import setup, find_packages

setup(
    name="deeptrade",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    author="Rex",
    description="DeepTrade: Advanced Stock Prediction with NLP & Automated Trading",
    python_requires=">=3.9",
)
