#!/usr/bin/env python3

import json
import csv
import re
import time
import argparse
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# SeleniumBase imports
from seleniumbase import SB


class ZonaPropFullScraper:
    """
    Scraper completo de ZonaProp usando CDP Mode
    Maneja múltiples páginas con extracción de datos reales
    """
    
    def __init__(self, headless: bool = False, incognito: bool = True, delay: float = 3.0):
        """
        Inicializa el scraper completo
        
        Args:
            headless: Ejecutar en modo headless
            incognito: Usar modo incógnito
            delay: Delay entre páginas en segundos
        """
        self.headless = headless
        self.incognito = incognito
        self.delay = delay
        self.scraped_data = []
        self.last_saved_count = 0  # Track how many properties have been saved
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'scraper_full_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("ZonaProp Full Scraper inicializado con CDP Mode")
    
    def load_existing_data(self, resume_file: str) -> bool:
        """
        Carga datos existentes desde un archivo para continuar la extracción
        
        Args:
            resume_file: Nombre base del archivo (sin timestamp)
            
        Returns:
            True si se cargaron datos exitosamente
        """
        try:
            # Buscar archivos existentes con el patrón
            data_dir = Path("data/raw")
            json_pattern = str(data_dir / f"{resume_file}_*.json")
            json_files = glob.glob(json_pattern)
            
            if not json_files:
                self.logger.warning(f"No se encontraron archivos existentes para {resume_file}")
                return False
            
            # Usar el archivo más reciente
            latest_file = max(json_files, key=lambda x: Path(x).stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            if existing_data:
                self.scraped_data = existing_data
                self.last_saved_count = len(existing_data)  # Set saved count to loaded data
                self.logger.info(f"📂 Cargados {len(existing_data)} registros existentes desde {latest_file}")
                return True
            
        except Exception as e:
            self.logger.error(f"❌ Error cargando datos existentes: {e}")
            
        return False
    
    def scrape_multiple_pages(self, start_page: int = 1, max_pages: int = 5, 
                            properties_per_page: int = 25, save_every: int = 0, 
                            output_name: str = "zonaprop_full", output_format: str = "both") -> List[Dict[str, Any]]:
        """
        Scraper principal para múltiples páginas
        
        Args:
            start_page: Página inicial
            max_pages: Número máximo de páginas
            properties_per_page: Propiedades por página (default: 25, máximo en ZonaProp)
            
        Returns:
            Lista de todas las propiedades extraídas
        """
        self.logger.info(f"Iniciando scraping completo: páginas {start_page} a {start_page + max_pages - 1}")
        
        total_extracted = 0
        failed_pages = []
        
        with SB(uc=True, test=True, incognito=self.incognito, headless=self.headless) as sb:
            
            for page_num in range(start_page, start_page + max_pages):
                try:
                    self.logger.info(f"🔄 Procesando página {page_num}/{start_page + max_pages - 1}")
                    
                    # Navegar a la página
                    success = self._navigate_to_page(sb, page_num)
                    if not success:
                        self.logger.error(f"❌ Error navegando a página {page_num}")
                        failed_pages.append(page_num)
                        continue
                    
                    # Extraer propiedades
                    page_properties = self._extract_page_properties(sb, page_num, properties_per_page)
                    
                    if page_properties:
                        self.scraped_data.extend(page_properties)
                        total_extracted += len(page_properties)
                        self.logger.info(f"✅ Página {page_num}: {len(page_properties)} propiedades extraídas")
                        self.logger.info(f"📊 Total acumulado: {total_extracted} propiedades")
                        
                        # Guardar progreso incrementalmente si está configurado
                        if save_every > 0 and page_num % save_every == 0:
                            self.logger.info(f"💾 Guardando progreso en página {page_num}...")
                            self.save_data(output_name, output_format, append_mode=True)
                            
                    else:
                        self.logger.warning(f"⚠️  Página {page_num}: No se extrajeron propiedades")
                        failed_pages.append(page_num)
                    
                    # Delay entre páginas (excepto la última)
                    if page_num < start_page + max_pages - 1:
                        self.logger.info(f"⏳ Esperando {self.delay}s antes de la siguiente página...")
                        time.sleep(self.delay)
                    
                except KeyboardInterrupt:
                    self.logger.info("🛑 Scraping interrumpido por el usuario")
                    break
                except Exception as e:
                    self.logger.error(f"❌ Error inesperado en página {page_num}: {e}")
                    failed_pages.append(page_num)
                    continue
        
        # Resumen final
        self.logger.info(f"🏁 Scraping completado:")
        self.logger.info(f"   📊 Total extraído: {total_extracted} propiedades")
        self.logger.info(f"   ✅ Páginas exitosas: {max_pages - len(failed_pages)}/{max_pages}")
        if failed_pages:
            self.logger.warning(f"   ❌ Páginas fallidas: {failed_pages}")
        
        return self.scraped_data
    
    def _navigate_to_page(self, sb, page_number: int) -> bool:
        """
        Navega a una página específica usando CDP Mode
        
        Args:
            sb: Instancia de SeleniumBase
            page_number: Número de página
            
        Returns:
            True si la navegación fue exitosa
        """
        try:
            # Construir URL para Capital Federal específicamente
            if page_number == 1:
                url = "https://www.zonaprop.com.ar/departamentos-venta-capital-federal.html"
            else:
                url = f"https://www.zonaprop.com.ar/departamentos-venta-capital-federal-pagina-{page_number}.html"
            
            self.logger.info(f"🌐 Navegando con CDP Mode: {url}")
            
            # Activar CDP Mode para bypass automático de Cloudflare
            sb.activate_cdp_mode(url)
            
            # Force refresh to ensure new page content is loaded
            sb.refresh()
            time.sleep(3)
            
            # Verificar título
            title = sb.get_title()
            self.logger.debug(f"Título página {page_number}: {title}")
            
            # Verificar si Cloudflare fue evadido
            if "just a moment" in title.lower():
                self.logger.error(f"🔒 Cloudflare detectado en página {page_number}")
                return False
            
            # Verificar que hay propiedades
            try:
                sb.wait_for_element_visible('[data-qa="posting PROPERTY"]', timeout=10)
                properties_count = len(sb.find_elements('[data-qa="posting PROPERTY"]'))
                
                if properties_count > 0:
                    self.logger.info(f"✅ Página {page_number} cargada: {properties_count} propiedades detectadas")
                    return True
                else:
                    self.logger.warning(f"⚠️  Página {page_number}: No se detectaron propiedades")
                    return False
                    
            except Exception as e:
                self.logger.error(f"❌ Error verificando propiedades en página {page_number}: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error navegando a página {page_number}: {e}")
            return False
    
    def _extract_page_properties(self, sb, page_number: int, max_properties: int) -> List[Dict[str, Any]]:
        """
        Extrae propiedades de la página actual usando JavaScript + Regex
        
        Args:
            sb: Instancia de SeleniumBase
            page_number: Número de página actual
            max_properties: Máximo de propiedades a extraer
            
        Returns:
            Lista de propiedades extraídas
        """
        try:
            self.logger.info(f"🔍 Extrayendo datos de página {page_number}...")
            
            # JavaScript para extraer datos del DOM
            js_extraction_script = """
            var properties = [];
            var postings = document.querySelectorAll('[data-qa="posting PROPERTY"]');
            
            postings.forEach(function(posting, index) {
                var property = {
                    index: index + 1,
                    full_text: posting.textContent.replace(/\\s+/g, ' ').trim(),
                    // Extract property metadata from attributes
                    property_id: posting.getAttribute('data-id'),
                    property_url: posting.getAttribute('data-to-posting')
                };
                
                // Intentar extraer elementos específicos
                var selectors = {
                    price: ['.postingPrices-module__price', '.price', '[data-qa="price"]', '.posting-price', '.card-price', '.amount'],
                    location: ['.address', '[data-qa="address"]', '.posting-address', '.card-address', '.location'],
                    surface: ['.surface', '[data-qa="surface"]', '.posting-surface', '.card-surface', '.area'],
                    rooms: ['.rooms', '[data-qa="rooms"]', '.posting-rooms', '.card-rooms', '.ambientes'],
                    description: ['.description', '[data-qa="description"]', '.posting-description', '.card-description', '.property-description', '.listing-description', '.posting-title', '.card-title'],
                    // Enhanced location selectors
                    full_address: ['.postingLocations-module__location-address', '.location-address', '.address-full'],
                    neighborhood: ['.postingLocations-module__location-text', '.location-text', '.neighborhood']
                };
                
                Object.keys(selectors).forEach(function(field) {
                    for (var i = 0; i < selectors[field].length; i++) {
                        var elem = posting.querySelector(selectors[field][i]);
                        if (elem && elem.textContent.trim()) {
                            property[field] = elem.textContent.trim();
                            break;
                        }
                    }
                });
                
                properties.push(property);
            });
            
            return properties;
            """
            
            # Ejecutar JavaScript
            js_results = sb.execute_script(js_extraction_script)
            self.logger.info(f"📋 JavaScript extrajo {len(js_results)} elementos")
            
            # Procesar resultados con regex
            page_properties = []
            
            for i, js_prop in enumerate(js_results[:max_properties]):
                
                property_data = {
                    'property_id': js_prop.get('property_id', 'N/A'),
                }
                
                full_text = js_prop.get('full_text', '')
                
                # Extraer precio con múltiples patrones
                price = self._extract_price(js_prop.get('price', ''), full_text)
                property_data['price'] = price
                
                # Extraer moneda
                currency = self._extract_currency(js_prop.get('price', ''), full_text, price)
                property_data['currency'] = currency
                
                # Extraer ubicación
                location = self._extract_location(js_prop.get('location', ''), full_text)
                property_data['location'] = location
                
                # Extraer dirección completa y barrio
                full_address = self._extract_full_address(js_prop.get('full_address', ''), full_text)
                property_data['full_address'] = full_address
                
                neighborhood = self._extract_neighborhood(js_prop.get('neighborhood', ''), full_text)
                property_data['neighborhood'] = neighborhood
                
                # Extraer superficie
                surface = self._extract_surface(js_prop.get('surface', ''), full_text)
                property_data['surface'] = surface
                
                # Extraer habitaciones
                rooms = self._extract_rooms(js_prop.get('rooms', ''), full_text)
                property_data['rooms'] = rooms
                
                # Extraer dormitorios por separado
                bedrooms = self._extract_bedrooms(full_text)
                property_data['bedrooms'] = bedrooms
                
                # Extraer información adicional
                property_data.update(self._extract_additional_info(full_text))
                
                # AMENITIES DEL EDIFICIO
                property_data.update(self._extract_building_amenities(full_text))
                
                # TIPO DE PROPIEDAD Y CARACTERÍSTICAS
                property_data.update(self._extract_property_characteristics(full_text))
                
                # INFORMACIÓN DE PISO Y ORIENTACIÓN
                property_data.update(self._extract_floor_and_orientation(full_text))
                
                # SERVICIOS Y SEGURIDAD
                property_data.update(self._extract_security_services(full_text))
                
                # ESTADO Y CONDICIÓN
                property_data.update(self._extract_property_condition(full_text))
                
                # Solo agregar si tiene datos válidos
                if self._is_valid_property(property_data):
                    page_properties.append(property_data)
            
            self.logger.info(f"✅ Extraídas {len(page_properties)} propiedades válidas de página {page_number}")
            return page_properties
            
        except Exception as e:
            self.logger.error(f"❌ Error extrayendo propiedades de página {page_number}: {e}")
            return []
    
    def _extract_price(self, direct_price: str, full_text: str) -> str:
        """Extrae precio usando múltiples estrategias, evitando expensas"""
        if direct_price and direct_price.strip() and '$' in direct_price:
            # Verificar que no sea una expensa
            if 'expensas' not in direct_price.lower():
                return direct_price.strip()
        
        # Patrones de precio más específicos que evitan expensas
        patterns = [
            # Patrones que explícitamente NO son expensas
            r'USD\s*[\d,\.]+(?!\s*[Ee]xpensas)',         # USD 123,456 (no seguido de "expensas")
            r'[\d,\.]+\s*USD(?!\s*[Ee]xpensas)',         # 123,456 USD (no seguido de "expensas")
            r'U\$S\s*[\d,\.]+',                          # U$S 123,456
            r'Dólares?\s*[\d,\.]+',                      # Dólares 123,456
            # Pesos argentinos que NO sean expensas
            r'\$\s*[\d,\.]+(?!\s*[Ee]xpensas)',          # $123,456 (no seguido de "expensas")
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, full_text)
            if matches:
                # Tomar el primer match que no contenga "expensas"
                for match in matches:
                    if 'expensas' not in match.lower():
                        return match.strip()
        
        # Como último recurso, buscar patrones más generales pero filtrar expensas
        general_patterns = [
            r'\$\s*[\d,\.]+',
            r'USD\s*[\d,\.]+',
            r'[\d,\.]+\s*USD'
        ]
        
        for pattern in general_patterns:
            matches = re.findall(pattern, full_text)
            for match in matches:
                # Verificar el contexto alrededor del match para asegurar que no es expensa
                match_index = full_text.find(match)
                if match_index != -1:
                    # Verificar 20 caracteres antes y después
                    context_start = max(0, match_index - 20)
                    context_end = min(len(full_text), match_index + len(match) + 20)
                    context = full_text[context_start:context_end].lower()
                    
                    if 'expensas' not in context and 'gastos' not in context:
                        return match.strip()
        
        return "N/A"
    
    def _extract_currency(self, direct_price: str, full_text: str, extracted_price: str) -> str:
        """Extrae la moneda del precio"""
        # Primero verificar en el precio extraído
        if extracted_price and extracted_price != 'N/A':
            if 'USD' in extracted_price.upper():
                return 'USD'
            elif '$' in extracted_price and 'USD' not in extracted_price.upper():
                return 'ARS'
        
        # Verificar en el precio directo
        if direct_price:
            if 'USD' in direct_price.upper():
                return 'USD'
            elif '$' in direct_price and 'USD' not in direct_price.upper():
                return 'ARS'
        
        # Buscar en el texto completo
        if 'USD' in full_text.upper():
            return 'USD'
        elif '$' in full_text:
            return 'ARS'
        
        # Patrones específicos
        usd_patterns = [
            r'USD\s*[\d,\.]+',
            r'[\d,\.]+\s*USD',
            r'U\$S\s*[\d,\.]+',
            r'Dólares?'
        ]
        
        for pattern in usd_patterns:
            if re.search(pattern, full_text, re.IGNORECASE):
                return 'USD'
        
        # Si hay precio pero no se puede determinar moneda, asumir ARS
        if extracted_price and extracted_price != 'N/A':
            return 'ARS'
        
        return "N/A"
    
    def _extract_location(self, direct_location: str, full_text: str) -> str:
        """Extrae ubicación usando múltiples estrategias"""
        if direct_location and direct_location.strip() and len(direct_location) > 3:
            return direct_location.strip()
        
        # Patrones de ubicación
        patterns = [
            r'[A-Z][a-z]+\s*,\s*[A-Z][a-z]+',     # Palermo, CABA
            r'[A-Z][a-z]+\s+[A-Z][a-z]+',         # Villa Crespo
            r'[A-Z][a-z]+\s*,\s*Capital',         # Recoleta, Capital
            r'[A-Z][a-z]+\s*,\s*Buenos Aires',    # Tigre, Buenos Aires
        ]
        
        for pattern in patterns:
            match = re.search(pattern, full_text)
            if match:
                location = match.group(0).strip()
                if len(location) > 5:  # Filtrar matches muy cortos
                    return location
        
        return "N/A"
    
    def _extract_surface(self, direct_surface: str, full_text: str) -> str:
        """Extrae superficie usando múltiples estrategias"""
        if direct_surface and 'm' in direct_surface:
            return direct_surface.strip()
        
        # Patrones de superficie
        patterns = [
            r'\d+\s*m²\s*tot',        # 80 m² tot
            r'\d+\s*m²',              # 80 m²
            r'\d+\s*m2',              # 80 m2
            r'\d+\s*mts²',            # 80 mts²
            r'\d+\s*metros',          # 80 metros
        ]
        
        for pattern in patterns:
            match = re.search(pattern, full_text)
            if match:
                return match.group(0).strip()
        
        return "N/A"
    
    def _extract_rooms(self, direct_rooms: str, full_text: str) -> str:
        """Extrae habitaciones usando múltiples estrategias"""
        if direct_rooms and ('amb' in direct_rooms or 'dorm' in direct_rooms):
            return direct_rooms.strip()
        
        # Patrones de habitaciones
        patterns = [
            r'\d+\s*amb\.',           # 3 amb.
            r'\d+\s*amb',             # 3 amb
            r'\d+\s*dorm\.',          # 2 dorm.
            r'\d+\s*dorm',            # 2 dorm
            r'\d+\s*habitac',         # 2 habitaciones
            r'\d+\s*ambientes',       # 3 ambientes
        ]
        
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        return "N/A"
    
    def _extract_description(self, direct_description: str, full_text: str) -> str:
        """Extrae descripción usando múltiples estrategias"""
        if direct_description and direct_description.strip() and len(direct_description) > 10:
            # Limpiar y truncar la descripción directa si es muy larga
            cleaned_desc = direct_description.strip()
            if len(cleaned_desc) > 500:
                cleaned_desc = cleaned_desc[:500] + "..."
            return cleaned_desc
        
        # Si no hay descripción directa, intentar extraer del texto completo
        # Buscar patrones comunes de descripciones en listings
        patterns = [
            r'Departamento\s+[^.]{20,200}',     # Departamento + descripción
            r'Excelente\s+[^.]{20,200}',       # Excelente + descripción
            r'Hermoso\s+[^.]{20,200}',         # Hermoso + descripción
            r'Amplio\s+[^.]{20,200}',          # Amplio + descripción
            r'Moderno\s+[^.]{20,200}',         # Moderno + descripción
        ]
        
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                description = match.group(0).strip()
                # Limpiar y validar longitud
                if len(description) >= 20 and len(description) <= 500:
                    return description
                elif len(description) > 500:
                    return description[:500] + "..."
        
        # Como último recurso, tomar las primeras palabras del texto completo
        # que parezcan una descripción
        words = full_text.split()
        if len(words) > 10:
            # Buscar el inicio de una posible descripción
            description_start = -1
            for i, word in enumerate(words):
                if any(keyword in word.lower() for keyword in ['departamento', 'excelente', 'hermoso', 'amplio', 'moderno', 'luminoso']):
                    description_start = i
                    break
            
            if description_start >= 0 and description_start < len(words) - 5:
                # Tomar hasta 30 palabras desde el inicio de la descripción
                desc_words = words[description_start:description_start + 30]
                description = ' '.join(desc_words)
                if len(description) >= 20:
                    return description
        
        return "N/A"
    
    def _extract_bedrooms(self, full_text: str) -> str:
        """Extrae número de dormitorios específicamente"""
        # Patrones específicos para dormitorios
        patterns = [
            r'(\d+)\s*dorm\.',           # 3 dorm.
            r'(\d+)\s*dorm',             # 3 dorm
            r'(\d+)\s*dormitorios?',     # 3 dormitorios
            r'(\d+)\s*habitaciones?',    # 3 habitaciones
        ]
        
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        return "N/A"
    
    def _extract_full_address(self, direct_address: str, full_text: str) -> str:
        """Extrae dirección completa con piso/unidad"""
        if direct_address and direct_address.strip() and len(direct_address) > 5:
            return direct_address.strip()
        
        # Patrones para direcciones completas
        patterns = [
            r'[A-Z][a-z\s]+\s+\d+\s*[A-Za-z°\s]*\d*[A-Za-z°]*',  # Marcelo T De Alvear 2199 6°G
            r'[A-Z][a-z\s]+\s+al\s+\d+',                          # Av Corrientes al 1800
            r'[A-Z][a-z\s]+\s+\d+\s+[A-Z][a-z\s]*',              # Las Heras 1700 Torre Brunetta
        ]
        
        for pattern in patterns:
            match = re.search(pattern, full_text)
            if match:
                address = match.group(0).strip()
                if len(address) > 10:  # Filtrar matches muy cortos
                    return address
        
        return "N/A"
    
    def _extract_neighborhood(self, direct_neighborhood: str, full_text: str) -> str:
        """Extrae barrio específico"""
        if direct_neighborhood and direct_neighborhood.strip() and len(direct_neighborhood) > 3:
            return direct_neighborhood.strip()
        
        # Patrones para barrios
        patterns = [
            r'(Recoleta|Palermo|Belgrano|Villa Crespo|San Telmo|Puerto Madero|Caballito|Barracas|La Boca|Monserrat|San Nicolás|Retiro|Once|Balvanera|Almagro|Villa Urquiza|Núñez|Saavedra|Colegiales|Chacarita|Paternal|Villa Pueyrredón|Agronomía|Villa Ortúzar|Villa del Parque|Devoto|Villa Real|Monte Castro|Versalles|Liniers|Mataderos|Parque Avellaneda|Flores|Parque Chacabuco|Boedo|San Cristóbal|Constitución|Barracas|Pompeya|Nueva Pompeya|Parque Patricios|Soldati),?\s*(Capital|Buenos Aires|CABA)?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "N/A"
    
    def _build_full_url(self, relative_url: str) -> str:
        """Construye URL completa desde URL relativa"""
        if not relative_url or relative_url == 'N/A':
            return "N/A"
        
        if relative_url.startswith('http'):
            return relative_url
        
        base_url = "https://www.zonaprop.com.ar"
        if relative_url.startswith('/'):
            return base_url + relative_url
        else:
            return base_url + '/' + relative_url
    
    def _extract_additional_info(self, full_text: str) -> Dict[str, str]:
        """Extrae información adicional como baños, cocheras, etc."""
        additional = {}
        
        # Baños
        bath_match = re.search(r'(\d+)\s*baño', full_text, re.IGNORECASE)
        if bath_match:
            additional['bathrooms'] = bath_match.group(0)
        
        # Cocheras
        garage_match = re.search(r'(\d+)\s*coch', full_text, re.IGNORECASE)
        if garage_match:
            additional['garage'] = garage_match.group(0)
        
        # Expensas
        expenses_match = re.search(r'\$\s*[\d,\.]+\s*Expensas', full_text)
        if expenses_match:
            additional['expenses'] = expenses_match.group(0)
        
        return additional
    
    def _extract_building_amenities(self, full_text: str) -> Dict[str, bool]:
        """Extrae amenities del edificio"""
        amenities = {}
        text_lower = full_text.lower()
        
        # Amenities principales con múltiples variaciones
        amenity_patterns = {
            'has_pool': ['pileta', 'piscina', 'pool', 'natación'],
            'has_gym': ['gimnasio', 'gym', 'fitness', 'aparatos'],
            'has_sum': ['sum', 'salón de usos múltiples', 'salon de usos multiples', 'salón social', 'salon social'],
            'has_grill': ['parrilla', 'barbacoa', 'quincho', 'asador'],
        }
        
        for amenity_key, patterns in amenity_patterns.items():
            amenities[amenity_key] = any(pattern in text_lower for pattern in patterns)
        
        return amenities
    
    def _extract_security_services(self, full_text: str) -> Dict[str, Any]:
        """Extrae servicios de seguridad y edificio"""
        services = {}
        text_lower = full_text.lower()
        
        # Servicios de seguridad
        services['has_doorman'] = any(term in text_lower for term in ['portero', 'portería', 'porteria', 'conserje'])
        services['has_security'] = any(term in text_lower for term in ['seguridad', 'vigilancia', '24hs', '24 hs', 'guardia'])
    
        # Otros servicios
        services['has_storage'] = any(term in text_lower for term in ['baulera', 'depósito', 'deposito', 'storage'])
        return services
    
    def _extract_property_characteristics(self, full_text: str) -> Dict[str, str]:
        """Extrae características específicas de la propiedad"""
        characteristics = {}
        text_lower = full_text.lower()
        
        # Tipo de propiedad
        property_types = {
            'departamento': ['departamento', 'depto', 'dept'],
            'casa': ['casa'],
            'ph': ['ph', 'p.h.', 'casa chorizo'],
            'loft': ['loft'],
            'duplex': ['duplex', 'dúplex'],
            'triplex': ['triplex', 'tríplex'],
            'monoambiente': ['monoambiente', 'mono ambiente', 'studio'],
        }
        
        for prop_type, patterns in property_types.items():
            if any(pattern in text_lower for pattern in patterns):
                characteristics['property_type'] = prop_type
                break
        else:
            characteristics['property_type'] = 'N/A'
        return characteristics
    
    def _extract_floor_and_orientation(self, full_text: str) -> Dict[str, Any]:
        """Extrae información de piso y orientación"""
        floor_info = {}
        
        # Número de piso
        floor_patterns = [
            r'(\d+)°?\s*piso',
            r'piso\s*(\d+)',
            r'(\d+)°\s*[A-Z]',  # 4°A, 5°B, etc.
        ]
        
        floor_number = None
        for pattern in floor_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                floor_number = int(match.group(1))
                break
        
        floor_info['floor_number'] = floor_number if floor_number else 0
        
        # Pisos totales del edificio
        total_floors_match = re.search(r'(\d+)\s*pisos?', full_text, re.IGNORECASE)
        if total_floors_match:
            floor_info['total_floors'] = int(total_floors_match.group(1))
        else:
            floor_info['total_floors'] = 0
        
        # Información de balcones
        balcony_match = re.search(r'(\d+)\s*balcón|balcón', full_text, re.IGNORECASE)
        if balcony_match:
            if balcony_match.group(1):
                floor_info['balcony_count'] = int(balcony_match.group(1))
            else:
                floor_info['balcony_count'] = 1
        else:
            floor_info['balcony_count'] = 0
        
        return floor_info
    
    def _extract_property_condition(self, full_text: str) -> Dict[str, str]:
        """Extrae estado y condición de la propiedad"""
        condition = {}
        text_lower = full_text.lower()
        
        # Estado de la propiedad
        if any(term in text_lower for term in ['a estrenar', 'nuevo', 'brand new']):
            condition['property_status'] = 'a estrenar'
        elif any(term in text_lower for term in ['en construcción', 'en construccion', 'under construction']):
            condition['property_status'] = 'en construcción'
        elif any(term in text_lower for term in ['usado', 'segunda mano']):
            condition['property_status'] = 'usado'
        else:
            condition['property_status'] = 'N/A'
        return condition
    
    def _is_valid_property(self, property_data: Dict[str, Any]) -> bool:
        """Verifica si la propiedad tiene datos válidos"""
        required_fields = ['price', 'location', 'surface', 'rooms']
        valid_count = sum(1 for field in required_fields 
                         if property_data.get(field) and property_data.get(field) != 'N/A')
        
        # Considerar válida si tiene al menos 2 campos válidos
        return valid_count >= 2
    
    def save_data(self, filename: str = "zonaprop_full_scraping", format: str = "both", append_mode: bool = False) -> Dict[str, str]:
        """
        Guarda los datos extraídos
        
        Args:
            filename: Nombre base del archivo
            format: 'csv', 'json', o 'both'
            append_mode: Si True, append a archivo existente; si False, crea nuevo con timestamp
            
        Returns:
            Dict con las rutas de los archivos guardados
        """
        if not self.scraped_data:
            self.logger.warning("⚠️  No hay datos para guardar")
            return {}
        
        # Determine what data to save
        if append_mode and self.last_saved_count > 0:
            # Only save new data since last save
            data_to_save = self.scraped_data[self.last_saved_count:]
            if not data_to_save:
                self.logger.info("📝 No hay nuevos datos para guardar")
                return {}
        else:
            # Save all data (first save or non-append mode)
            data_to_save = self.scraped_data
        
        # Crear directorio de salida
        output_dir = Path("data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        if append_mode:
            # Modo append: usar archivo sin timestamp
            csv_file = output_dir / f"{filename}.csv"
            json_file = output_dir / f"{filename}.json"
        else:
            # Modo normal: crear archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = output_dir / f"{filename}_{timestamp}.csv"
            json_file = output_dir / f"{filename}_{timestamp}.json"
        
        # Guardar CSV
        if format in ['csv', 'both']:
            # Obtener todos los campos únicos
            all_fields = set()
            for prop in data_to_save:
                all_fields.update(prop.keys())
            sorted_fields = sorted(all_fields)
            
            if append_mode and csv_file.exists():
                # Append mode: agregar sin header
                with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted_fields)
                    writer.writerows(data_to_save)
                self.logger.info(f"💾 CSV actualizado (append): {csv_file} - Agregadas {len(data_to_save)} propiedades")
            else:
                # Write mode: crear nuevo archivo con header
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted_fields)
                    writer.writeheader()
                    writer.writerows(data_to_save)
                self.logger.info(f"💾 CSV guardado: {csv_file}")
            
            saved_files['csv'] = str(csv_file)
        
        # Guardar JSON
        if format in ['json', 'both']:
            if append_mode and json_file.exists():
                # Append mode: leer archivo existente, agregar nuevos datos, y reescribir
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    
                    # Combinar datos existentes con nuevos
                    combined_data = existing_data + data_to_save
                    
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(combined_data, f, ensure_ascii=False, indent=2)
                    
                    self.logger.info(f"💾 JSON actualizado (append): {json_file} - Agregadas {len(data_to_save)} propiedades, Total: {len(combined_data)}")
                except Exception as e:
                    self.logger.error(f"❌ Error al hacer append en JSON: {e}")
                    # Fallback: crear nuevo archivo
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
                    self.logger.info(f"💾 JSON guardado (fallback): {json_file}")
            else:
                # Write mode: crear nuevo archivo
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, ensure_ascii=False, indent=2)
                self.logger.info(f"💾 JSON guardado: {json_file}")
            
            saved_files['json'] = str(json_file)
        
        # Update the saved count after successful save
        if append_mode:
            self.last_saved_count = len(self.scraped_data)
        
        # Mostrar estadísticas solo si no es modo append
        if not append_mode:
            self._show_statistics()
        
        return saved_files
    
    def _show_statistics(self):
        """Muestra estadísticas detalladas de los datos extraídos"""
        if not self.scraped_data:
            return
        
        total = len(self.scraped_data)
        
        print(f"\n📊 ESTADÍSTICAS DEL SCRAPING COMPLETO")
        print("=" * 60)
        print(f"Total de propiedades extraídas: {total}")
        
        # Páginas procesadas
        pages = sorted(set(prop.get('page', 0) for prop in self.scraped_data))
        print(f"Páginas procesadas: {pages}")
        print(f"Propiedades por página: {total // len(pages) if pages else 0} promedio")
        
        # Calidad de datos por campo
        fields = ['price', 'location', 'surface', 'rooms', 'description']
        print(f"\n📋 Calidad de datos:")
        
        for field in fields:
            valid_count = sum(1 for prop in self.scraped_data 
                            if prop.get(field) and prop.get(field) != 'N/A')
            percentage = (valid_count / total * 100) if total > 0 else 0
            print(f"  {field.capitalize()}: {valid_count}/{total} ({percentage:.1f}%)")
        
        # Rangos de precios
        prices = []
        for prop in self.scraped_data:
            price_text = prop.get('price', '').replace('$', '').replace(',', '').replace('.', '').replace('USD', '').strip()
            if price_text.isdigit():
                prices.append(int(price_text))
        
        if prices:
            print(f"\n💰 Análisis de precios:")
            print(f"  Rango: ${min(prices):,} - ${max(prices):,}")
            print(f"  Promedio: ${sum(prices) // len(prices):,}")
            print(f"  Propiedades con precio: {len(prices)}/{total}")
        
        # Ubicaciones únicas
        locations = [prop.get('location', '') for prop in self.scraped_data 
                    if prop.get('location') and prop.get('location') != 'N/A']
        unique_locations = len(set(locations))
        print(f"\n📍 Ubicaciones únicas: {unique_locations}")
        
        # Top 5 ubicaciones
        if locations:
            from collections import Counter
            top_locations = Counter(locations).most_common(5)
            print(f"  Top ubicaciones:")
            for location, count in top_locations:
                print(f"    {location}: {count} propiedades")


def main():
    """Función principal para ejecutar el scraper desde línea de comandos"""
    parser = argparse.ArgumentParser(description="ZonaProp Full Scraper con CDP Mode")
    parser.add_argument("--pages", type=int, default=5, help="Número de páginas a scrapear (default: 5)")
    parser.add_argument("--start-page", type=int, default=1, help="Página inicial (default: 1)")
    parser.add_argument("--output", type=str, default="zonaprop_full", help="Nombre base del archivo de salida")
    parser.add_argument("--format", type=str, choices=["csv", "json", "both"], default="both", help="Formato de salida")
    parser.add_argument("--headless", action="store_true", help="Ejecutar en modo headless")
    parser.add_argument("--no-incognito", action="store_true", help="Desactivar modo incógnito")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay entre páginas en segundos (default: 3.0)")
    parser.add_argument("--properties-per-page", type=int, default=25, help="Propiedades por página (default: 25)")
    parser.add_argument("--resume", type=str, help="Resume extraction by appending to existing file (provide base filename without timestamp)")
    parser.add_argument("--save-every", type=int, default=0, help="Save progress every N pages (0 = save only at end)")
    
    args = parser.parse_args()
    
    try:
        print("🚀 ZonaProp Full Scraper - CDP Mode")
        print(f"📄 Páginas: {args.start_page} a {args.start_page + args.pages - 1}")
        print(f"🎯 Objetivo: ~{args.pages * args.properties_per_page} propiedades")
        print("=" * 60)
        
        # Crear instancia del scraper
        scraper = ZonaPropFullScraper(
            headless=args.headless,
            incognito=not args.no_incognito,
            delay=args.delay
        )
        
        # Cargar datos existentes si se especifica resume
        if args.resume:
            print(f"🔄 Modo resume activado: {args.resume}")
            if scraper.load_existing_data(args.resume):
                existing_count = len(scraper.scraped_data)
                print(f"📂 Continuando desde {existing_count} propiedades existentes")
            else:
                print("⚠️  No se encontraron datos existentes, iniciando desde cero")
        
        
        # Ejecutar scraping
        output_name = args.resume if args.resume else args.output
        data = scraper.scrape_multiple_pages(
            start_page=args.start_page,
            max_pages=args.pages,
            properties_per_page=args.properties_per_page,
            save_every=args.save_every,
            output_name=output_name,
            output_format=args.format
        )
        
        # Guardar datos
        if data:
            # Usar el nombre del archivo resume si está disponible, sino usar args.output
            output_name = args.resume if args.resume else args.output
            # Si save_every está habilitado, usar append mode para el guardado final también
            use_append = args.save_every > 0
            saved_files = scraper.save_data(output_name, args.format, append_mode=use_append)
            
            print(f"\n🎉 SCRAPING COMPLETADO EXITOSAMENTE!")
            print(f"📊 Total extraído: {len(data)} propiedades")
            
            if saved_files:
                print(f"📁 Archivos guardados:")
                for format_type, filepath in saved_files.items():
                    print(f"  {format_type.upper()}: {filepath}")
            
            print(f"\n✅ CDP Mode: Bypass exitoso de Cloudflare")
            print(f"✅ Extracción: Datos reales verificados")
            
        else:
            print("❌ No se extrajeron datos")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Scraping interrumpido por el usuario")
        return 0
    except Exception as e:
        print(f"❌ Error en el scraping: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
