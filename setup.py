from setuptools import setup, find_packages

setup(
    name="coretempai",
    version="0.1.0",
    description="CoreTempAI package for computational fluid dynamics temperature prediction",
    author="",
    author_email="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
)