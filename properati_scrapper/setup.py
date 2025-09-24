"""
Setup script para el scraper de Properati
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="properati-scraper",
    version="1.0.0",
    author="CDA Team",
    author_email="team@cda.com",
    description="Web scraper para propiedades de Properati.com.ar",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cda/properati-scraper",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "seleniumbase>=4.20.0",
        "selenium>=4.15.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "pandas>=2.0.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "properati-scraper=src.properati_full_scraper:main",
        ],
    },
)
