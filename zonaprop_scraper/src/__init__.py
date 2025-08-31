"""
ZonaProp Scraper Package
Scraper para extraer datos de propiedades de ZonaProp.com.ar
"""

from .scraper import ZonaPropScraper
from .data_extractor import PropertyDataExtractor
from .config import SCRAPER_CONFIG, SELENIUM_CONFIG, DATA_FIELDS

__version__ = "1.0.0"
__author__ = "Ian Bernasconi, Jeremías Feferovich"
__email__ = "your.email@example.com"

__all__ = [
    "ZonaPropScraper",
    "PropertyDataExtractor", 
    "SCRAPER_CONFIG",
    "SELENIUM_CONFIG",
    "DATA_FIELDS"
]
