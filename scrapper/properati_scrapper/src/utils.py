"""
Utilidades y funciones helper para el scraper de Properati
"""
import time
import random
import re
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

def random_delay(min_seconds: int, max_seconds: int) -> None:
    """
    Genera un delay aleatorio entre min_seconds y max_seconds
    
    Args:
        min_seconds: Mínimo número de segundos
        max_seconds: Máximo número de segundos
    """
    delay = random.uniform(min_seconds, max_seconds)
    logging.debug(f"Esperando {delay:.2f} segundos...")
    time.sleep(delay)

def clean_price(price_text: str) -> Dict[str, Any]:
    """
    Extrae y limpia el precio de una propiedad de Properati
    
    Args:
        price_text: Texto del precio (ej: "USD 195.000")
        
    Returns:
        Dict con currency y amount
    """
    if not price_text:
        return {"currency": None, "amount": None}
    
    # Remover espacios y caracteres especiales
    clean_text = price_text.strip().replace(".", "").replace(",", "")
    
    # Buscar patrón de moneda y monto
    patterns = [
        r'(USD)\s*(\d+)',
        r'(ARS|\$)\s*(\d+)',
        r'U\$S\s*(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, clean_text)
        if match:
            if len(match.groups()) == 2:
                currency = match.group(1)
                amount = int(match.group(2))
            else:
                currency = "USD"
                amount = int(match.group(1))
            
            # Normalizar moneda
            if currency == "$":
                currency = "ARS"
            elif currency == "U$S":
                currency = "USD"
                
            return {"currency": currency, "amount": amount}
    
    return {"currency": None, "amount": None}

def extract_bedrooms(bedrooms_text: str) -> Optional[int]:
    """
    Extrae número de dormitorios
    
    Args:
        bedrooms_text: Texto de dormitorios (ej: "2 dormitorios")
        
    Returns:
        Número de dormitorios como entero o None
    """
    if not bedrooms_text:
        return None
    
    # Buscar números en el texto
    numbers = re.findall(r'\d+', bedrooms_text)
    
    if numbers:
        return int(numbers[0])
    
    return None

def extract_bathrooms(bathrooms_text: str) -> Optional[int]:
    """
    Extrae número de baños
    
    Args:
        bathrooms_text: Texto de baños (ej: "1 baño")
        
    Returns:
        Número de baños como entero o None
    """
    if not bathrooms_text:
        return None
    
    # Buscar números en el texto
    numbers = re.findall(r'\d+', bathrooms_text)
    
    if numbers:
        return int(numbers[0])
    
    return None

def extract_area(area_text: str) -> Optional[int]:
    """
    Extrae superficie en metros cuadrados
    
    Args:
        area_text: Texto de área (ej: "62 m²")
        
    Returns:
        Área en m² como entero o None
    """
    if not area_text:
        return None
    
    # Buscar números seguidos de m² o m2
    match = re.search(r'(\d+)\s*m[²2]', area_text)
    
    if match:
        return int(match.group(1))
    
    return None

def extract_location_info(location_text: str) -> Dict[str, str]:
    """
    Extrae información de ubicación
    
    Args:
        location_text: Texto de ubicación (ej: "Belgrano, Capital Federal")
        
    Returns:
        Dict con información de ubicación estructurada
    """
    result = {
        "full_location": location_text.strip() if location_text else None,
        "neighborhood": None,
        "city": None
    }
    
    if location_text:
        # Separar barrio y ciudad (ej: "Belgrano, Capital Federal")
        parts = [part.strip() for part in location_text.split(",")]
        
        if len(parts) >= 1:
            result["neighborhood"] = parts[0]
        
        if len(parts) >= 2:
            result["city"] = parts[1]
    
    return result

def extract_amenities(properties_container_text: str) -> List[str]:
    """
    Extrae amenities de la sección de propiedades
    
    Args:
        properties_container_text: Texto completo del contenedor de propiedades
        
    Returns:
        Lista de amenities encontrados
    """
    amenities = []
    
    if not properties_container_text:
        return amenities
    
    # Amenities comunes a buscar en Properati
    common_amenities = {
        "balcón": "Balcón",
        "balcon": "Balcón",
        "terraza": "Terraza", 
        "jardín": "Jardín",
        "jardin": "Jardín",
        "patio": "Patio",
        "cochera": "Cochera",
        "garage": "Garage",
        "pileta": "Pileta",
        "piscina": "Pileta",
        "gimnasio": "Gimnasio",
        "gym": "Gimnasio",
        "sum": "SUM",
        "parrilla": "Parrilla",
        "seguridad": "Seguridad",
        "portero": "Portero",
        "ascensor": "Ascensor",
        "aire acondicionado": "Aire Acondicionado",
        "calefacción": "Calefacción",
        "calefaccion": "Calefacción",
    }
    
    text_lower = properties_container_text.lower()
    for keyword, amenity in common_amenities.items():
        if keyword in text_lower:
            amenities.append(amenity)
    
    return list(set(amenities))  # Remove duplicates

def extract_property_type(title: str) -> str:
    """
    Extrae el tipo de propiedad del título
    
    Args:
        title: Título de la propiedad
        
    Returns:
        Tipo de propiedad
    """
    if not title:
        return "Desconocido"
    
    title_lower = title.lower()
    
    property_types = {
        "departamento": "Departamento",
        "depto": "Departamento", 
        "casa": "Casa",
        "ph": "PH",
        "loft": "Loft",
        "oficina": "Oficina",
        "local": "Local",
        "cochera": "Cochera",
        "terreno": "Terreno",
        "lote": "Lote",
    }
    
    for keyword, prop_type in property_types.items():
        if keyword in title_lower:
            return prop_type
    
    return "Departamento"  # Default para Capital Federal

def get_current_timestamp() -> str:
    """
    Obtiene el timestamp actual en formato ISO
    
    Returns:
        Timestamp actual como string
    """
    return datetime.now().isoformat()

def validate_property_data(property_data: Dict[str, Any]) -> bool:
    """
    Valida que los datos de una propiedad tengan los campos mínimos requeridos
    
    Args:
        property_data: Diccionario con datos de la propiedad
        
    Returns:
        True si los datos son válidos, False en caso contrario
    """
    required_fields = ["property_id", "title", "price"]
    
    for field in required_fields:
        if not property_data.get(field):
            logging.warning(f"Campo requerido faltante: {field}")
            return False
    
    return True

def safe_extract_text(element, default: str = "") -> str:
    """
    Extrae texto de un elemento de forma segura
    
    Args:
        element: Elemento web de Selenium
        default: Valor por defecto si no se puede extraer
        
    Returns:
        Texto extraído o valor por defecto
    """
    try:
        return element.text.strip() if element else default
    except Exception as e:
        logging.debug(f"Error extrayendo texto: {e}")
        return default

def safe_extract_attribute(element, attribute: str, default: str = "") -> str:
    """
    Extrae atributo de un elemento de forma segura
    
    Args:
        element: Elemento web de Selenium
        attribute: Nombre del atributo
        default: Valor por defecto si no se puede extraer
        
    Returns:
        Valor del atributo o valor por defecto
    """
    try:
        return element.get_attribute(attribute) if element else default
    except Exception as e:
        logging.debug(f"Error extrayendo atributo {attribute}: {e}")
        return default
