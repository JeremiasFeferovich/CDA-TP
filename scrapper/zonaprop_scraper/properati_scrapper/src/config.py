"""
Configuración para el scraper de Properati
"""
import os
from pathlib import Path

# Configuración de URLs
BASE_URL = "https://www.properati.com.ar"
SEARCH_URL = "https://www.properati.com.ar/s/capital-federal/venta/{}?propertyType=apartment%2Chouse"

# Configuración de directorios
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_DIR = DATA_DIR / "logs"

# Configuración del scraper
SCRAPER_CONFIG = {
    "delay_between_requests": (1, 3),      # Delay aleatorio entre requests (min, max) segundos
    "delay_between_pages": (5, 10),       # Delay entre páginas
    "max_retries": 3,                      # Máximo número de reintentos
    "timeout": 15,                         # Timeout por página en segundos
    "max_pages": 100,                      # Máximo número de páginas a scrapear
    "max_properties_per_session": 1000,    # Límite de propiedades por sesión
}

# Configuración de Selenium
SELENIUM_CONFIG = {
    "headless": False,                # Ejecutar en modo visible para evitar detección
    "window_size": (1920, 1080),     # Tamaño de ventana
    "implicit_wait": 1,               # Espera implícita
    "page_load_timeout": 15,          # Timeout de carga de página
}

# User Agents para rotación
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
]

# Selectores CSS para extracción de datos de Properati
SELECTORS = {
    # Contenedor principal de cada propiedad
    "property_container": '.snippet',
    
    # Información básica dentro de information2
    "title": '.title',
    "price": '.price',
    "location": '.location',
    
    # Características de la propiedad
    "properties_container": '.properties',
    "bedrooms": '.properties__bedrooms',
    "bathrooms": '.properties__bathrooms',
    "area": '.properties__area',
    "amenities": '[class*="properties__amenity"]',
    
    # Información adicional
    "description": '.description',
    "publisher": '.publisher',
    
    # Atributos del contenedor
    "property_id": 'data-idanuncio',
    "detail_url": 'data-url',
    
    # Contenedores principales
    "content_container": '.snippet__content',
    "information2_container": '.information2',
    
    # Paginación (si existe)
    "next_page": '.pagination__next',
    "current_page": '.pagination__current',
}

# Campos de datos a extraer
DATA_FIELDS = [
    "property_id",
    "title",
    "price",
    "currency", 
    "location",
    "neighborhood",
    "area",
    "bedrooms", 
    "bathrooms",
    "amenities",
    "property_type",
    "description",
    "publisher",
    "detail_url",
    "scraping_date",
    "page_number"
]

# Configuración de logging
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    "rotation": "1 day",
    "retention": "7 days",
}
