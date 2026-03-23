from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mnemon-ai",
    version="1.0.2",
    author="Mahika Jadhav",
    author_email="mahikajadhav22@gmail.com",
    description="The intelligence layer between your agents and oblivion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smartass-4ever/Mnemon",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "sentence-transformers>=2.2.0",  # real semantic embeddings — auto-activated
        "cryptography>=41.0.0",          # Fernet AES encryption — auto-activated
    ],
    extras_require={
        # LLM providers — install the one you use
        # Mnemon auto-detects whichever key is set in your environment
        "anthropic": ["anthropic>=0.20.0"],          # ANTHROPIC_API_KEY
        "openai":    ["openai>=1.0.0"],               # OPENAI_API_KEY
        "google":    ["google-generativeai>=0.7.0"],  # GOOGLE_API_KEY
        "groq":      ["groq>=0.9.0"],                 # GROQ_API_KEY (free tier)
        # Install all providers
        "all-llm": [
            "anthropic>=0.20.0",
            "openai>=1.0.0",
            "google-generativeai>=0.7.0",
            "groq>=0.9.0",
        ],
        # Scale
        "redis": ["redis>=5.0.0"],
        # Dev
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mnemon=mnemon.cli.main:main",
        ],
    },
)
