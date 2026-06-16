from setuptools import setup, find_packages

# Minimal setuptools entry for editable installs when Pixi is not used.
# Training, explanation, and factory code under ``src/`` use ``beartype`` / ``jaxtyping`` at
# runtime; those belong here. Code under ``src/models/hf/aloe/`` is Hub-facing and must stay
# free of those imports so ``trust_remote_code`` users only need ``transformers`` + ``torch``.
# Do not list ``beartype`` in any Hugging Face model-card ``requirements.txt`` for end users.
setup(
    name="interpretability-thesis",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "torch",
        "torchvision",
        "beartype",
        "tqdm",
        "pyyaml",
    ],
)
