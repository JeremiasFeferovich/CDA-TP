"""
Configuración para el scraper de ZonaProp
"""
import os
from pathlib import Path

# Configuración de URLs
BASE_URL = "https://www.zonaprop.com.ar"
SEARCH_URL = "https://www.zonaprop.com.ar/departamentos-venta-pagina-{}.html"

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
    "headless": False,                # Ejecutar en modo visible para evitar Cloudflare
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

# Selectores CSS para extracción de datos
SELECTORS = {
    # Contenedor principal de cada propiedad
    "property_container": '[data-qa="posting PROPERTY"]',
    
    # Información básica
    "price": '.postingPrices-module__price',
    "expenses": '.postingPrices-module__expenses',
    "property_id": 'data-id',
    
    # Ubicación
    "address": '.postingLocations-module__location-address',
    "location": '.postingLocations-module__location-text',
    
    # Características principales
    "main_features": '.postingMainFeatures-module__posting-main-features-span',
    
    # Amenities y características adicionales
    "pills": '.pills-module__trigger-pill-item-span',
    "description": '.postingDescription-module__description',
    
    # Información de contacto
    "publisher": '.postingPublisher-module__publisher-name',
    
    # URL de detalle
    "detail_url": 'data-to-posting',
    
    # Número de fotos
    "photo_count": '.postingMultimediaTags-module__tag-info',
    
    # Paginación
    "next_page": '.pagination-module__next-page',
    "current_page": '.pagination-module__current-page',
}

# Campos de datos a extraer
DATA_FIELDS = [
    "property_id",
    "price",
    "currency", 
    "expenses",
    "address",
    "neighborhood",
    "zone",
    "total_surface",
    "rooms",
    "bedrooms", 
    "bathrooms",
    "parking_spaces",
    "amenities",
    "property_status",
    "property_type",
    "orientation",
    "floor",
    "publisher",
    "phone",
    "email",
    "detail_url",
    "photo_count",
    "scraping_date",
    "description"
]

# Configuración de logging
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    "rotation": "1 day",
    "retention": "7 days",
}
