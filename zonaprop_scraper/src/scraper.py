"""
Scraper principal para ZonaProp
"""
import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException


# WebDriver Manager
from webdriver_manager.chrome import ChromeDriverManager

# Logging
from loguru import logger

# Imports locales
from .config import (
    SEARCH_URL, SCRAPER_CONFIG, SELENIUM_CONFIG, USER_AGENTS, 
    DATA_FIELDS, LOGS_DIR, RAW_DATA_DIR, LOGGING_CONFIG
)
from .data_extractor import PropertyDataExtractor
from .utils import random_delay, validate_property_data

class ZonaPropScraper:
    """
    Scraper principal para extraer datos de propiedades de ZonaProp
    """
    
    def __init__(self, headless: bool = True, max_pages: int = None):
        """
        Inicializa el scraper
        
        Args:
            headless: Si ejecutar en modo headless
            max_pages: Máximo número de páginas a scrapear
        """
        self.headless = headless if headless is not None else SELENIUM_CONFIG["headless"]
        self.max_pages = max_pages if max_pages is not None else SCRAPER_CONFIG["max_pages"]
        self.driver = None
        self.extractor = None
        self.scraped_properties = []
        
        # Configurar logging
        self._setup_logging()
        
        logger.info("ZonaPropScraper inicializado")
    
    def _setup_logging(self):
        """Configura el sistema de logging"""
        # Crear directorio de logs si no existe
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Configurar loguru
        log_file = LOGS_DIR / f"scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logger.add(
            log_file,
            level=LOGGING_CONFIG["level"],
            format=LOGGING_CONFIG["format"],
            rotation=LOGGING_CONFIG["rotation"],
            retention=LOGGING_CONFIG["retention"]
        )
    
    def _setup_driver(self):
        """Configura y inicializa el WebDriver de Chrome"""
        try:
            # Configurar opciones de Chrome
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument("--headless")
            
            # Opciones para evitar detección
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # User Agent aleatorio
            user_agent = random.choice(USER_AGENTS)
            chrome_options.add_argument(f"--user-agent={user_agent}")
            
            # Tamaño de ventana
            chrome_options.add_argument(f"--window-size={SELENIUM_CONFIG['window_size'][0]},{SELENIUM_CONFIG['window_size'][1]}")
            
            # Configurar el servicio - usar chromedriver del sistema si está disponible
            try:
                # Intentar usar chromedriver del sistema primero
                service = Service("/usr/bin/chromedriver")
            except:
                try:
                    # Si no está disponible, usar webdriver-manager
                    chromedriver_path = ChromeDriverManager().install()
                    # Verificar que sea el archivo correcto
                    if chromedriver_path.endswith('THIRD_PARTY_NOTICES.chromedriver'):
                        # Buscar el archivo chromedriver real en el mismo directorio
                        import os
                        driver_dir = os.path.dirname(chromedriver_path)
                        real_driver = os.path.join(driver_dir, 'chromedriver')
                        if os.path.exists(real_driver):
                            chromedriver_path = real_driver
                    service = Service(chromedriver_path)
                except Exception as e:
                    logger.error(f"No se pudo configurar ChromeDriver: {e}")
                    raise
            
            # Crear el driver
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Configuraciones adicionales
            self.driver.implicitly_wait(SELENIUM_CONFIG["implicit_wait"])
            self.driver.set_page_load_timeout(SELENIUM_CONFIG["page_load_timeout"])
            
            # Ejecutar script para evitar detección
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Inicializar el extractor
            self.extractor = PropertyDataExtractor(self.driver)
            
            logger.info("WebDriver configurado exitosamente")
            
        except Exception as e:
            logger.error(f"Error configurando WebDriver: {e}")
            raise
    
    def _navigate_to_page(self, page_number: int) -> bool:
        """
        Navega a una página específica
        
        Args:
            page_number: Número de página a navegar
            
        Returns:
            True si la navegación fue exitosa, False en caso contrario
        """
        try:
            # Construir URL
            if page_number == 1:
                url = "https://www.zonaprop.com.ar/departamentos-venta.html"
            else:
                url = SEARCH_URL.format(page_number)
            
            logger.info(f"Navegando a página {page_number}: {url}")
            
            # Navegar a la URL
            self.driver.get(url)
            
            # Esperar a que se cargue la página inicial
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Verificar si la página se cargó correctamente
            # Intentar encontrar elementos de propiedades o paginación
            try:
                WebDriverWait(self.driver, 10).until(
                    lambda driver: (
                        len(driver.find_elements(By.CSS_SELECTOR, '[data-qa="posting PROPERTY"]')) > 0 or
                        "departamentos" in driver.title.lower()
                    )
                )
            except TimeoutException:
                logger.warning(f"No se detectaron propiedades en página {page_number}")
                return False
            
            # Delay para simular comportamiento humano
            random_delay(*SCRAPER_CONFIG["delay_between_requests"])
            
            logger.info(f"Navegación exitosa a página {page_number}")
            return True
            
        except TimeoutException:
            logger.error(f"Timeout navegando a página {page_number}")
            return False
        except Exception as e:
            logger.error(f"Error navegando a página {page_number}: {e}")
            return False
    
    def _check_for_blocking(self) -> bool:
        """
        Verifica si la página está siendo bloqueada
        
        Returns:
            True si está bloqueada, False en caso contrario
        """
        try:
            page_source = self.driver.page_source.lower()
            blocking_indicators = [
                "checking your browser", "verify you are human",
                "access denied", "blocked", "captcha", "rate limit"
            ]
            
            for indicator in blocking_indicators:
                if indicator in page_source:
                    logger.warning(f"Posible bloqueo detectado: {indicator}")
                    return True
            
            return False
        except Exception as e:
            logger.debug(f"Error verificando bloqueo: {e}")
            return False
    
    def _scrape_current_page(self) -> List[Dict[str, Any]]:
        """
        Extrae datos de la página actual
        
        Returns:
            Lista de propiedades extraídas
        """
        try:
            properties = self.extractor.extract_properties_from_page()
            
            # Validar datos
            valid_properties = []
            for prop in properties:
                if validate_property_data(prop):
                    valid_properties.append(prop)
                else:
                    logger.warning(f"Propiedad inválida descartada: {prop.get('property_id', 'Unknown')}")
            
            logger.info(f"Extraídas {len(valid_properties)} propiedades válidas de {len(properties)} totales")
            
            return valid_properties
            
        except Exception as e:
            logger.error(f"Error extrayendo datos de página actual: {e}")
            return []
    
    def scrape(self, start_page: int = 1) -> List[Dict[str, Any]]:
        """
        Ejecuta el scraping completo
        
        Args:
            start_page: Página desde la cual comenzar
            
        Returns:
            Lista de todas las propiedades extraídas
        """
        try:
            # Configurar driver
            self._setup_driver()
            
            logger.info(f"Iniciando scraping desde página {start_page} hasta página {self.max_pages}")
            
            current_page = start_page
            total_properties = 0
            
            while current_page <= self.max_pages:
                try:
                    # Verificar límite de propiedades por sesión
                    if total_properties >= SCRAPER_CONFIG["max_properties_per_session"]:
                        logger.info(f"Alcanzado límite de {SCRAPER_CONFIG['max_properties_per_session']} propiedades por sesión")
                        break
                    
                    # Navegar a la página con reintentos
                    navigation_success = False
                    for attempt in range(SCRAPER_CONFIG["max_retries"]):
                        if self._navigate_to_page(current_page):
                            navigation_success = True
                            break
                        else:
                            logger.warning(f"Intento {attempt + 1} de navegación a página {current_page} falló")
                            if attempt < SCRAPER_CONFIG["max_retries"] - 1:
                                random_delay(2, 5)  # Delay más largo entre reintentos
                    
                    if not navigation_success:
                        logger.error(f"No se pudo navegar a página {current_page} después de {SCRAPER_CONFIG['max_retries']} intentos")
                        current_page += 1
                        continue
                    
                    # Verificar si estamos siendo bloqueados
                    if self._check_for_blocking():
                        logger.error(f"Página {current_page} está siendo bloqueada. Aumentando delays...")
                        random_delay(10, 20)  # Delay muy largo si hay bloqueo
                    
                    # Extraer propiedades de la página con reintentos
                    page_properties = []
                    for attempt in range(SCRAPER_CONFIG["max_retries"]):
                        page_properties = self._scrape_current_page()
                        if page_properties:
                            break
                        else:
                            logger.warning(f"Intento {attempt + 1} de extracción en página {current_page} falló")
                            if attempt < SCRAPER_CONFIG["max_retries"] - 1:
                                random_delay(3, 7)  # Delay más largo entre reintentos
                                # Recargar la página
                                self.driver.refresh()
                                random_delay(2, 4)
                    
                    if not page_properties:
                        logger.warning(f"No se encontraron propiedades en página {current_page}")
                        # Si no hay propiedades, podríamos haber llegado al final
                        if current_page > start_page:  # Solo si no es la primera página
                            logger.info("Posiblemente llegamos al final de las páginas disponibles")
                            break
                    else:
                        self.scraped_properties.extend(page_properties)
                        total_properties += len(page_properties)
                        
                        logger.info(f"Página {current_page}: {len(page_properties)} propiedades. Total: {total_properties}")
                    
                    # Delay entre páginas
                    if current_page < self.max_pages:
                        random_delay(*SCRAPER_CONFIG["delay_between_pages"])
                    
                    current_page += 1
                    
                except Exception as e:
                    logger.error(f"Error procesando página {current_page}: {e}")
                    current_page += 1
                    continue
            
            logger.info(f"Scraping completado. Total de propiedades extraídas: {len(self.scraped_properties)}")
            
            return self.scraped_properties
            
        except Exception as e:
            logger.error(f"Error durante el scraping: {e}")
            raise
        finally:
            self.close()
    
    def save_data(self, filename: str = None, format: str = "csv") -> str:
        """
        Guarda los datos extraídos en archivo
        
        Args:
            filename: Nombre del archivo (sin extensión)
            format: Formato del archivo ("csv" o "json")
            
        Returns:
            Ruta del archivo guardado
        """
        if not self.scraped_properties:
            logger.warning("No hay datos para guardar")
            return None
        
        # Crear directorio si no existe
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generar nombre de archivo si no se proporciona
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"zonaprop_properties_{timestamp}"
        
        # Guardar según formato (sin pandas temporalmente)
        import json
        import csv
        
        if format.lower() == "csv":
            filepath = RAW_DATA_DIR / f"{filename}.csv"
            if self.scraped_properties:
                with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = DATA_FIELDS
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for prop in self.scraped_properties:
                        # Escribir solo campos que existen en DATA_FIELDS
                        filtered_prop = {k: v for k, v in prop.items() if k in fieldnames}
                        writer.writerow(filtered_prop)
        elif format.lower() == "json":
            filepath = RAW_DATA_DIR / f"{filename}.json"
            with open(filepath, 'w', encoding='utf-8') as jsonfile:
                json.dump(self.scraped_properties, jsonfile, indent=2, ensure_ascii=False, default=str)
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        logger.info(f"Datos guardados en: {filepath}")
        logger.info(f"Total de registros: {len(self.scraped_properties)}")
        
        return str(filepath)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de los datos extraídos
        
        Returns:
            Diccionario con estadísticas
        """
        if not self.scraped_properties:
            return {"total_properties": 0}
        
        # Estadísticas básicas sin pandas
        total = len(self.scraped_properties)
        neighborhoods = set()
        prices = []
        currencies = {}
        property_types = {}
        rooms_dist = {}
        
        for prop in self.scraped_properties:
            if prop.get('neighborhood'):
                neighborhoods.add(prop['neighborhood'])
            if prop.get('price') and isinstance(prop['price'], (int, float)):
                prices.append(prop['price'])
            if prop.get('currency'):
                currencies[prop['currency']] = currencies.get(prop['currency'], 0) + 1
            if prop.get('property_type'):
                pt = prop['property_type']
                property_types[pt] = property_types.get(pt, 0) + 1
            if prop.get('rooms'):
                rooms = prop['rooms']
                rooms_dist[rooms] = rooms_dist.get(rooms, 0) + 1
        
        stats = {
            "total_properties": total,
            "unique_neighborhoods": len(neighborhoods),
            "price_range": {
                "min": min(prices) if prices else None,
                "max": max(prices) if prices else None,
                "mean": sum(prices) / len(prices) if prices else None
            },
            "currency_distribution": currencies,
            "property_types": property_types,
            "rooms_distribution": rooms_dist
        }
        
        return stats
    
    def close(self):
        """Cierra el WebDriver y limpia recursos"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("WebDriver cerrado exitosamente")
            except Exception as e:
                logger.error(f"Error cerrando WebDriver: {e}")
        
        self.driver = None
        self.extractor = None

def main():
    """Función principal para ejecutar el scraper desde línea de comandos"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scraper de propiedades de ZonaProp")
    parser.add_argument("--pages", type=int, default=10, help="Número máximo de páginas a scrapear")
    parser.add_argument("--start-page", type=int, default=1, help="Página desde la cual comenzar")
    parser.add_argument("--headless", action="store_true", help="Ejecutar en modo headless")
    parser.add_argument("--output", type=str, help="Nombre del archivo de salida")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Formato del archivo de salida")
    
    args = parser.parse_args()
    
    # Crear scraper
    scraper = ZonaPropScraper(headless=args.headless, max_pages=args.pages)
    
    try:
        # Ejecutar scraping
        properties = scraper.scrape(start_page=args.start_page)
        
        # Guardar datos
        if properties:
            filepath = scraper.save_data(filename=args.output, format=args.format)
            
            # Mostrar estadísticas
            stats = scraper.get_statistics()
            print(f"\n=== Estadísticas del Scraping ===")
            print(f"Total de propiedades: {stats['total_properties']}")
            print(f"Barrios únicos: {stats['unique_neighborhoods']}")
            if stats['price_range']['min']:
                print(f"Rango de precios: ${stats['price_range']['min']:,.0f} - ${stats['price_range']['max']:,.0f}")
                print(f"Precio promedio: ${stats['price_range']['mean']:,.0f}")
            print(f"Archivo guardado: {filepath}")
        else:
            print("No se extrajeron propiedades")
            
    except KeyboardInterrupt:
        logger.info("Scraping interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")
        sys.exit(1)
    finally:
        scraper.close()

if __name__ == "__main__":
    main()

