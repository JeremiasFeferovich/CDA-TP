"""
Extractor de datos para propiedades de Properati
"""
from typing import List, Dict, Any, Optional
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import logging

from .config import SELECTORS
from .utils import (
    clean_price, extract_bedrooms, extract_bathrooms, extract_area,
    extract_location_info, extract_amenities, extract_property_type,
    get_current_timestamp, safe_extract_text, safe_extract_attribute
)

class PropertyDataExtractor:
    """
    Clase para extraer datos de propiedades desde las páginas de Properati
    """
    
    def __init__(self, driver):
        """
        Inicializa el extractor con el driver de Selenium
        
        Args:
            driver: Instancia del WebDriver de Selenium
        """
        self.driver = driver
        self.wait = WebDriverWait(driver, 10)
    
    def extract_properties_from_page(self, page_number: int = 1) -> List[Dict[str, Any]]:
        """
        Extrae todas las propiedades de la página actual
        
        Args:
            page_number: Número de página actual
        
        Returns:
            Lista de diccionarios con datos de propiedades
        """
        properties = []
        
        try:
            # Esperar a que se carguen las propiedades
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, SELECTORS["property_container"]))
            )
            
            # Encontrar todos los contenedores de propiedades
            property_containers = self.driver.find_elements(By.CSS_SELECTOR, SELECTORS["property_container"])
            
            logging.info(f"Encontradas {len(property_containers)} propiedades en la página {page_number}")
            
            for i, container in enumerate(property_containers):
                try:
                    property_data = self._extract_single_property(container, page_number)
                    if property_data:
                        properties.append(property_data)
                        logging.debug(f"Propiedad {i+1} extraída exitosamente")
                    else:
                        logging.warning(f"No se pudo extraer la propiedad {i+1}")
                        
                except Exception as e:
                    logging.error(f"Error extrayendo propiedad {i+1}: {e}")
                    continue
            
            logging.info(f"Extraídas {len(properties)} propiedades válidas de la página {page_number}")
            
        except TimeoutException:
            logging.error(f"Timeout esperando que se carguen las propiedades en página {page_number}")
        except Exception as e:
            logging.error(f"Error general extrayendo propiedades de página {page_number}: {e}")
        
        return properties
    
    def _extract_single_property(self, container, page_number: int) -> Optional[Dict[str, Any]]:
        """
        Extrae datos de una sola propiedad
        
        Args:
            container: Elemento contenedor de la propiedad
            page_number: Número de página actual
            
        Returns:
            Diccionario con datos de la propiedad o None si hay error
        """
        try:
            property_data = {}
            
            # ID de la propiedad
            property_data["property_id"] = safe_extract_attribute(container, "data-idanuncio")
            
            # URL de detalle
            detail_url = safe_extract_attribute(container, "data-url")
            property_data["detail_url"] = detail_url if detail_url.startswith("http") else f"https://www.properati.com.ar{detail_url}"
            
            # Buscar el contenedor de información
            info_container = container.find_element(By.CSS_SELECTOR, SELECTORS["information2_container"])
            
            # Título
            title_element = info_container.find_element(By.CSS_SELECTOR, SELECTORS["title"])
            title = safe_extract_text(title_element)
            property_data["title"] = title
            
            # Precio
            price_element = info_container.find_element(By.CSS_SELECTOR, SELECTORS["price"])
            price_text = safe_extract_text(price_element)
            price_info = clean_price(price_text)
            property_data["price"] = price_info["amount"]
            property_data["currency"] = price_info["currency"]
            
            # Ubicación
            location_element = info_container.find_element(By.CSS_SELECTOR, SELECTORS["location"])
            location_text = safe_extract_text(location_element)
            location_info = extract_location_info(location_text)
            property_data["location"] = location_info["full_location"]
            property_data["neighborhood"] = location_info["neighborhood"]
            
            # Propiedades (dormitorios, baños, área, amenities)
            try:
                properties_container = container.find_element(By.CSS_SELECTOR, SELECTORS["properties_container"])
                properties_text = safe_extract_text(properties_container)
                
                # Dormitorios
                try:
                    bedrooms_element = properties_container.find_element(By.CSS_SELECTOR, SELECTORS["bedrooms"])
                    bedrooms_text = safe_extract_text(bedrooms_element)
                    property_data["bedrooms"] = extract_bedrooms(bedrooms_text)
                except NoSuchElementException:
                    property_data["bedrooms"] = None
                
                # Baños
                try:
                    bathrooms_element = properties_container.find_element(By.CSS_SELECTOR, SELECTORS["bathrooms"])
                    bathrooms_text = safe_extract_text(bathrooms_element)
                    property_data["bathrooms"] = extract_bathrooms(bathrooms_text)
                except NoSuchElementException:
                    property_data["bathrooms"] = None
                
                # Área
                try:
                    area_element = properties_container.find_element(By.CSS_SELECTOR, SELECTORS["area"])
                    area_text = safe_extract_text(area_element)
                    property_data["area"] = extract_area(area_text)
                except NoSuchElementException:
                    property_data["area"] = None
                
                # Amenities
                property_data["amenities"] = extract_amenities(properties_text)
                
            except NoSuchElementException:
                logging.warning(f"No se encontró contenedor de propiedades para {property_data.get('property_id', 'unknown')}")
                property_data["bedrooms"] = None
                property_data["bathrooms"] = None
                property_data["area"] = None
                property_data["amenities"] = []
            
            # Tipo de propiedad (extraído del título)
            property_data["property_type"] = extract_property_type(title)
            
            # Descripción (si existe)
            try:
                description_element = container.find_element(By.CSS_SELECTOR, SELECTORS["description"])
                property_data["description"] = safe_extract_text(description_element)
            except NoSuchElementException:
                property_data["description"] = None
            
            # Publisher (si existe)
            try:
                publisher_element = container.find_element(By.CSS_SELECTOR, SELECTORS["publisher"])
                property_data["publisher"] = safe_extract_text(publisher_element)
            except NoSuchElementException:
                property_data["publisher"] = None
            
            # Información adicional
            property_data["page_number"] = page_number
            property_data["scraping_date"] = get_current_timestamp()
            
            return property_data
            
        except Exception as e:
            logging.error(f"Error extrayendo datos de propiedad: {e}")
            return None
    
    def get_current_page_number(self) -> Optional[int]:
        """
        Obtiene el número de página actual (si está disponible)
        
        Returns:
            Número de página actual o None si no se puede determinar
        """
        try:
            current_page_element = self.driver.find_element(By.CSS_SELECTOR, SELECTORS["current_page"])
            page_text = safe_extract_text(current_page_element)
            return int(page_text) if page_text.isdigit() else None
        except (NoSuchElementException, ValueError):
            return None
    
    def has_next_page(self) -> bool:
        """
        Verifica si existe una página siguiente
        
        Returns:
            True si existe página siguiente, False en caso contrario
        """
        try:
            next_page_element = self.driver.find_element(By.CSS_SELECTOR, SELECTORS["next_page"])
            return next_page_element.is_enabled()
        except NoSuchElementException:
            return False
    
    def click_next_page(self) -> bool:
        """
        Hace clic en el botón de página siguiente
        
        Returns:
            True si se pudo hacer clic, False en caso contrario
        """
        try:
            next_page_element = self.driver.find_element(By.CSS_SELECTOR, SELECTORS["next_page"])
            if next_page_element.is_enabled():
                next_page_element.click()
                return True
            return False
        except NoSuchElementException:
            return False
