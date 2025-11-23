from setuptools import setup, find_packages
import pathlib

def read_requirements():
    req_file = pathlib.Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        return []
    with open(req_file, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="audio-augmentation",
    version="0.1.0",
    description="Audio augmentation library for PyTorch with codec degradation, noise addition, RIR simulation, and more",
    long_description=open("README.md", encoding="utf-8").read() if pathlib.Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=read_requirements(),
    python_requires=">=3.10",
    include_package_data=True,
    package_data={
        "audio_augmentation": ["config.yaml"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio",
    ],
)

