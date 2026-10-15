from setuptools import setup, find_packages

setup(
    name="hausafs2",
    version="1.0.0",
    description="Hausa FastSpeech 2 – Linguistically grounded TTS for Hausa",
    author="Ismail Suleman",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1",
        "numpy",
        "scipy",
        "librosa>=0.10",
        "soundfile",
        "pyyaml",
        "tgt",
        "tqdm",
        "matplotlib",
    ],
    extras_require={
        "world": ["pyworld"],
        "test": ["pytest"],
    },
)
