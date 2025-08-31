"""
Extractor de datos para propiedades de ZonaProp
"""
from typing import List, Dict, Any, Optional
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from loguru import logger

from .config import SELECTORS
from .utils import (
    clean_price, clean_expenses, extract_surface_info, 
    extract_location_info, extract_amenities, extract_property_status,
    get_current_timestamp, safe_extract_text, safe_extract_attribute
)

class PropertyDataExtractor:
    """
    Clase para extraer datos de propiedades desde las páginas de ZonaProp
    """
    
    def __init__(self, driver):
        """
        Inicializa el extractor con el driver de Selenium
        
        Args:
            driver: Instancia del WebDriver de Selenium
        """
        self.driver = driver
        self.wait = WebDriverWait(driver, 5)
    
    def extract_properties_from_page(self) -> List[Dict[str, Any]]:
        """
        Extrae todas las propiedades de la página actual
        
        Returns:
            Lista de diccionarios con datos de propiedades
        """
        properties = []
        
        try:
            # Esperar a que se carguen las propiedades con timeout más largo
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, SELECTORS["property_container"]))
            )
            
            # Encontrar todos los contenedores de propiedades
            property_containers = self.driver.find_elements(By.CSS_SELECTOR, SELECTORS["property_container"])
            
            logger.info(f"Encontradas {len(property_containers)} propiedades en la página")
            
            for i, container in enumerate(property_containers):
                try:
                    property_data = self._extract_single_property(container)
                    if property_data:
                        properties.append(property_data)
                        logger.debug(f"Propiedad {i+1} extraída exitosamente")
                    else:
                        logger.warning(f"No se pudo extraer la propiedad {i+1}")
                        
                except Exception as e:
                    logger.error(f"Error extrayendo propiedad {i+1}: {e}")
                    continue
            
            logger.info(f"Extraídas {len(properties)} propiedades válidas")
            
        except TimeoutException:
            logger.error("Timeout esperando que se carguen las propiedades")
        except Exception as e:
            logger.error(f"Error general extrayendo propiedades: {e}")
        
        return properties
    
    def _extract_single_property(self, container) -> Optional[Dict[str, Any]]:
        """
        Extrae datos de una sola propiedad
        
        Args:
            container: Elemento contenedor de la propiedad
            
        Returns:
            Diccionario con datos de la propiedad o None si hay error
        """
        try:
            property_data = {}
            
            # ID de la propiedad
            property_data["property_id"] = safe_extract_attribute(container, "data-id")
            
            # Precio
            price_element = container.find_element(By.CSS_SELECTOR, SELECTORS["price"])
            price_info = clean_price(safe_extract_text(price_element))
            property_data["price"] = price_info["amount"]
            property_data["currency"] = price_info["currency"]
            
            # Expensas
            expenses_elements = container.find_elements(By.CSS_SELECTOR, SELECTORS["expenses"])
            if expenses_elements:
                property_data["expenses"] = clean_expenses(safe_extract_text(expenses_elements[0]))
            else:
                property_data["expenses"] = None
            
            # Ubicación
            address_element = container.find_element(By.CSS_SELECTOR, SELECTORS["address"])
            location_element = container.find_element(By.CSS_SELECTOR, SELECTORS["location"])
            
            location_info = extract_location_info(
                safe_extract_text(address_element),
                safe_extract_text(location_element)
            )
            property_data.update(location_info)
            
            # Características principales
            feature_elements = container.find_elements(By.CSS_SELECTOR, SELECTORS["main_features"])
            features_text = [safe_extract_text(elem) for elem in feature_elements]
            
            surface_info = extract_surface_info(features_text)
            property_data.update(surface_info)
            
            # Pills (amenities y características)
            pill_elements = container.find_elements(By.CSS_SELECTOR, SELECTORS["pills"])
            pills_text = [safe_extract_text(elem) for elem in pill_elements]
            
            # Descripción
            description_elements = container.find_elements(By.CSS_SELECTOR, SELECTORS["description"])
            description = safe_extract_text(description_elements[0]) if description_elements else ""
            
            property_data["description"] = description
            
            # Amenities
            property_data["amenities"] = extract_amenities(pills_text, description)
            
            # Estado de la propiedad
            property_data["property_status"] = extract_property_status(pills_text, description)
            
            # Tipo de propiedad (por defecto departamento ya que estamos en esa sección)
            property_data["property_type"] = "Departamento"
            
            # Publisher/Inmobiliaria
            publisher_elements = container.find_elements(By.CSS_SELECTOR, SELECTORS["publisher"])
            property_data["publisher"] = safe_extract_text(publisher_elements[0]) if publisher_elements else None
            
            # URL de detalle
            property_data["detail_url"] = safe_extract_attribute(container, "data-to-posting")
            if property_data["detail_url"] and not property_data["detail_url"].startswith("http"):
                property_data["detail_url"] = f"https://www.zonaprop.com.ar{property_data['detail_url']}"
            
            # Número de fotos
            photo_elements = container.find_elements(By.CSS_SELECTOR, SELECTORS["photo_count"])
            if photo_elements:
                photo_text = safe_extract_text(photo_elements[0])
                property_data["photo_count"] = int(photo_text) if photo_text.isdigit() else None
            else:
                property_data["photo_count"] = None
            
            # Campos adicionales que se pueden extraer de la descripción
            property_data["orientation"] = self._extract_orientation(description)
            property_data["floor"] = self._extract_floor(description)
            
            # Información de contacto (generalmente no disponible en listado)
            property_data["phone"] = None
            property_data["email"] = None
            
            # Timestamp
            property_data["scraping_date"] = get_current_timestamp()
            
            return property_data
            
        except Exception as e:
            logger.error(f"Error extrayendo datos de propiedad: {e}")
            return None
    
    def _extract_orientation(self, description: str) -> Optional[str]:
        """
        Extrae orientación de la descripción
        
        Args:
            description: Descripción de la propiedad
            
        Returns:
            Orientación encontrada o None
        """
        if not description:
            return None
        
        orientations = ["norte", "sur", "este", "oeste", "noreste", "noroeste", "sureste", "suroeste"]
        desc_lower = description.lower()
        
        for orientation in orientations:
            if orientation in desc_lower:
                return orientation.title()
        
        return None
    
    def _extract_floor(self, description: str) -> Optional[str]:
        """
        Extrae información del piso de la descripción
        
        Args:
            description: Descripción de la propiedad
            
        Returns:
            Información del piso o None
        """
        if not description:
            return None
        
        import re
        
        # Buscar patrones como "5to piso", "piso 5", "planta baja", etc.
        floor_patterns = [
            r'(\d+)(?:to|do|er)?\s*piso',
            r'piso\s*(\d+)',
            r'planta\s*baja',
            r'pb\b',
            r'(\d+)(?:°|º)\s*piso'
        ]
        
        desc_lower = description.lower()
        
        for pattern in floor_patterns:
            match = re.search(pattern, desc_lower)
            if match:
                if "baja" in pattern or "pb" in pattern:
                    return "Planta Baja"
                elif match.group(1):
                    return f"Piso {match.group(1)}"
        
        return None
    
    def get_current_page_number(self) -> Optional[int]:
        """
        Obtiene el número de página actual
        
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
