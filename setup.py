from setuptools import setup, find_packages

setup(
    name="audiotrain",
    version="1.0.0",
    description="Modern Python AI Neural Amp Trainer replicating AudioTrainHHAG.exe and SincoANN",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "audiotrain": ["assets/*.wav", "configs/*.yaml"],
    },
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.22.0",
        "scipy>=1.9.0",
        "soundfile>=0.12.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "audiotrain=audiotrain.cli:main",
        ],
    },
)
