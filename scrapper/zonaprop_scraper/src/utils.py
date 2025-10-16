"""
Utilidades y funciones helper para el scraper de ZonaProp
"""
import time
import random
import re
from datetime import datetime
from typing import Optional, List, Dict, Any
from loguru import logger

def random_delay(min_seconds: int, max_seconds: int) -> None:
    """
    Genera un delay aleatorio entre min_seconds y max_seconds
    
    Args:
        min_seconds: Mínimo número de segundos
        max_seconds: Máximo número de segundos
    """
    delay = random.uniform(min_seconds, max_seconds)
    logger.debug(f"Esperando {delay:.2f} segundos...")
    time.sleep(delay)

def clean_price(price_text: str) -> Dict[str, Any]:
    """
    Extrae y limpia el precio de una propiedad
    
    Args:
        price_text: Texto del precio (ej: "USD 265.000")
        
    Returns:
        Dict con currency y amount
    """
    if not price_text:
        return {"currency": None, "amount": None}
    
    # Remover espacios y caracteres especiales
    clean_text = price_text.strip().replace(".", "").replace(",", "")
    
    # Buscar patrón de moneda y monto
    pattern = r'(USD|ARS|\$)\s*(\d+)'
    match = re.search(pattern, clean_text)
    
    if match:
        currency = match.group(1)
        amount = int(match.group(2))
        
        # Normalizar moneda
        if currency == "$":
            currency = "ARS"
            
        return {"currency": currency, "amount": amount}
    
    return {"currency": None, "amount": None}

def clean_expenses(expenses_text: str) -> Optional[int]:
    """
    Extrae y limpia las expensas
    
    Args:
        expenses_text: Texto de expensas (ej: "$ 180.000 Expensas")
        
    Returns:
        Monto de expensas como entero o None
    """
    if not expenses_text:
        return None
    
    # Buscar números en el texto
    numbers = re.findall(r'\d+', expenses_text.replace(".", "").replace(",", ""))
    
    if numbers:
        return int("".join(numbers))
    
    return None

def extract_surface_info(features_list: List[str]) -> Dict[str, Any]:
    """
    Extrae información de superficie de la lista de características
    
    Args:
        features_list: Lista de características (ej: ["95 m² tot.", "4 amb."])
        
    Returns:
        Dict con información de superficie y ambientes
    """
    result = {
        "total_surface": None,
        "rooms": None,
        "bedrooms": None,
        "bathrooms": None,
        "parking_spaces": None
    }
    
    for feature in features_list:
        feature = feature.strip().lower()
        
        # Superficie total
        if "m²" in feature and "tot" in feature:
            numbers = re.findall(r'\d+', feature)
            if numbers:
                result["total_surface"] = int(numbers[0])
        
        # Ambientes
        elif "amb" in feature:
            numbers = re.findall(r'\d+', feature)
            if numbers:
                result["rooms"] = int(numbers[0])
        
        # Dormitorios
        elif "dorm" in feature:
            numbers = re.findall(r'\d+', feature)
            if numbers:
                result["bedrooms"] = int(numbers[0])
        
        # Baños
        elif "baño" in feature:
            numbers = re.findall(r'\d+', feature)
            if numbers:
                result["bathrooms"] = int(numbers[0])
        
        # Cocheras
        elif "coch" in feature:
            numbers = re.findall(r'\d+', feature)
            if numbers:
                result["parking_spaces"] = int(numbers[0])
    
    return result

def extract_location_info(address: str, location: str) -> Dict[str, str]:
    """
    Extrae información de ubicación
    
    Args:
        address: Dirección de la propiedad
        location: Ubicación (barrio, zona)
        
    Returns:
        Dict con información de ubicación estructurada
    """
    result = {
        "address": address.strip() if address else None,
        "neighborhood": None,
        "zone": None
    }
    
    if location:
        # Separar barrio y zona (ej: "Villa del Parque, Capital Federal")
        parts = [part.strip() for part in location.split(",")]
        
        if len(parts) >= 1:
            result["neighborhood"] = parts[0]
        
        if len(parts) >= 2:
            result["zone"] = parts[1]
    
    return result

def extract_amenities(pills_list: List[str], description: str = "") -> List[str]:
    """
    Extrae amenities de las pills y descripción
    
    Args:
        pills_list: Lista de pills/etiquetas
        description: Descripción de la propiedad
        
    Returns:
        Lista de amenities encontrados
    """
    amenities = set()
    
    # Amenities comunes a buscar
    common_amenities = [
        "pileta", "piscina", "gimnasio", "gym", "sum", "parrilla", 
        "solarium", "terraza", "balcón", "balcon", "jacuzzi", 
        "sauna", "laundry", "lavadero", "seguridad", "portero",
        "ascensor", "cochera", "baulera", "aire acondicionado"
    ]
    
    # Buscar en pills
    for pill in pills_list:
        pill_lower = pill.lower().strip()
        for amenity in common_amenities:
            if amenity in pill_lower:
                amenities.add(amenity.title())
    
    # Buscar en descripción
    if description:
        desc_lower = description.lower()
        for amenity in common_amenities:
            if amenity in desc_lower:
                amenities.add(amenity.title())
    
    return sorted(list(amenities))

def extract_property_status(pills_list: List[str], description: str = "") -> Optional[str]:
    """
    Extrae el estado de la propiedad
    
    Args:
        pills_list: Lista de pills/etiquetas
        description: Descripción de la propiedad
        
    Returns:
        Estado de la propiedad o None
    """
    status_keywords = {
        "a estrenar": "Nuevo",
        "estrenar": "Nuevo", 
        "construcción": "En Construcción",
        "en construcción": "En Construcción",
        "entrega": "En Construcción",
        "usado": "Usado",
        "excelente estado": "Excelente Estado"
    }
    
    # Buscar en pills
    for pill in pills_list:
        pill_lower = pill.lower().strip()
        for keyword, status in status_keywords.items():
            if keyword in pill_lower:
                return status
    
    # Buscar en descripción
    if description:
        desc_lower = description.lower()
        for keyword, status in status_keywords.items():
            if keyword in desc_lower:
                return status
    
    return "Usado"  # Default

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
    required_fields = ["property_id", "price", "address", "neighborhood"]
    
    for field in required_fields:
        if not property_data.get(field):
            logger.warning(f"Campo requerido faltante: {field}")
            return False
    
    # Validar que el precio sea numérico
    if not isinstance(property_data.get("price"), (int, float)):
        logger.warning("Precio no es numérico")
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
        logger.debug(f"Error extrayendo texto: {e}")
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
        logger.debug(f"Error extrayendo atributo {attribute}: {e}")
        return default
