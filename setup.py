from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gemma-novelty-framework",
    version="0.1.0",
    author="Roberto Jimenez and collaborators",
    author_email="",
    description="A recursive, self-optimizing framework for generating and evaluating true novelty in AI systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gemma-novelty-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.0",
        ],
        "visualization": [
            "plotly>=5.0",
            "networkx>=2.6",
            "matplotlib>=3.4",
            "seaborn>=0.11",
        ],
    },
    package_data={
        "gemma_novelty": [
            "config/*.yaml",
            "config/tasks/*.yaml",
            "config/environments/*.yaml",
        ],
    },
    entry_points={
        "console_scripts": [
            "gemma-train=scripts.train\:main",
            "gemma-evaluate=scripts.evaluate\:main",
            "gemma-simulate=scripts.simulate\:main",
        ],
    },
)
