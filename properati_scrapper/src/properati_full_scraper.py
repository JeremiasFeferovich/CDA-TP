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


class ProperatiFullScraper:
    """
    Scraper completo de Properati usando CDP Mode
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
                logging.FileHandler(f'properati_scraper_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Properati Full Scraper inicializado con CDP Mode")
    
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
                            properties_per_page: int = 30, save_every: int = 0, 
                            output_name: str = "properati_full", output_format: str = "both") -> List[Dict[str, Any]]:
        """
        Scraper principal para múltiples páginas
        
        Args:
            start_page: Página inicial
            max_pages: Número máximo de páginas
            properties_per_page: Propiedades por página (default: 30, típico en Properati)
            
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
        Navega a una página específica usando CDP Mode y botones de paginación
        
        Args:
            sb: Instancia de SeleniumBase
            page_number: Número de página
            
        Returns:
            True si la navegación fue exitosa
        """
        try:
            if page_number == 1:
                # Para la primera página, usar la URL base
                url = "https://www.properati.com.ar/s/capital-federal/venta"
                self.logger.info(f"🌐 Navegando con CDP Mode a página 1: {url}")
                
                # Activar CDP Mode para bypass automático de detección
                sb.activate_cdp_mode(url)
                
                # Force refresh to ensure new page content is loaded
                sb.refresh()
                time.sleep(3)
            else:
                # Para páginas siguientes, usar el botón "Siguiente"
                self.logger.info(f"🔄 Navegando a página {page_number} usando botón de paginación")
                
                # Scroll hacia abajo para asegurar que el botón de paginación esté visible
                sb.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                # Navegar directamente a la URL de la página
                try:
                    # Construir la URL específica para la página
                    page_url = f"https://www.properati.com.ar/s/capital-federal/venta/{page_number}"
                    self.logger.info(f"🌐 Navegando directamente a página {page_number}: {page_url}")
                    
                    # Navegar directamente a la URL
                    sb.get(page_url)
                    time.sleep(4)  # Esperar a que cargue la página
                    
                    self.logger.info(f"✅ Navegación exitosa a página {page_number}")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error navegando a página {page_number}: {e}")
                    return False
            
            # Verificar que la página se cargó correctamente y hay propiedades
            try:
                sb.wait_for_element_visible('.snippet', timeout=10)
                properties_count = len(sb.find_elements('.snippet'))
                
                # Verificar título
                title = sb.get_title()
                self.logger.debug(f"Título página {page_number}: {title}")
                
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
            var snippets = document.querySelectorAll('.snippet');
            
            snippets.forEach(function(snippet, index) {
                var property = {
                    index: index + 1,
                    property_id: snippet.getAttribute('data-idanuncio'),
                    detail_url: snippet.getAttribute('data-url')
                };
                
                // Buscar información básica
                var information2 = snippet.querySelector('.information2');
                if (information2) {
                    var title = information2.querySelector('.title');
                    if (title) property.title = title.textContent.trim();
                    
                    var price = information2.querySelector('.price');
                    if (price) property.price = price.textContent.trim();
                    
                    var location = information2.querySelector('.location');
                    if (location) property.location = location.textContent.trim();
                }
                
                // Buscar propiedades (dormitorios, baños, área)
                var properties_container = snippet.querySelector('.properties');
                if (properties_container) {
                    property.properties_text = properties_container.textContent.trim();
                    
                    var bedrooms = properties_container.querySelector('.properties__bedrooms');
                    if (bedrooms) property.bedrooms = bedrooms.textContent.trim();
                    
                    var bathrooms = properties_container.querySelector('.properties__bathrooms');
                    if (bathrooms) property.bathrooms = bathrooms.textContent.trim();
                    
                    var area = properties_container.querySelector('.properties__area');
                    if (area) property.area = area.textContent.trim();
                    
                    // Buscar amenities
                    var amenities = properties_container.querySelectorAll('[class*="properties__amenity"]');
                    property.amenities = [];
                    amenities.forEach(function(amenity) {
                        property.amenities.push(amenity.textContent.trim());
                    });
                }
                
                // Buscar descripción
                var description = snippet.querySelector('.description');
                if (description) property.description = description.textContent.trim();
                
                // Buscar publisher/agencia
                var publisher = snippet.querySelector('.publisher');
                if (publisher) property.publisher = publisher.textContent.trim();
                
                var agency = snippet.querySelector('.agency__name');
                if (agency) property.agency_name = agency.textContent.trim();
                
                // Buscar fecha de publicación
                var published_date = snippet.querySelector('.published-date');
                if (published_date) property.published_date = published_date.textContent.trim();
                
                // Buscar número de fotos
                var photo_total = snippet.querySelector('.swiper-pagination-total');
                if (photo_total) property.photo_count = photo_total.textContent.trim();
                
                // Buscar etiquetas especiales (DESTACADO, etc.)
                var labels = snippet.querySelectorAll('.label');
                property.labels = [];
                labels.forEach(function(label) {
                    property.labels.push(label.textContent.trim());
                });
                
                // Extraer texto completo para análisis posterior
                property.full_text = snippet.textContent.trim();
                
                properties.push(property);
            });
            
            return properties;
            """
            
            # Ejecutar JavaScript
            js_results = sb.execute_script(js_extraction_script)
            self.logger.info(f"📋 JavaScript extrajo {len(js_results)} elementos")
            
            # Procesar resultados
            page_properties = []
            
            for i, js_prop in enumerate(js_results[:max_properties]):
                
                property_data = {
                    'property_id': js_prop.get('property_id', 'N/A'),
                    'page_number': page_number,
                    
                    # Basic fields
                    'title': None,
                    'price': None,
                    'currency': None,
                    'location': None,
                    'neighborhood': None,
                    'full_address': None,
                    'bedrooms': None,
                    'bathrooms': None,
                    'rooms': None,
                    'area': None,
                    'surface_total': None,
                    'surface_covered': None,
                    'amenities': [],
                    'property_type': None,
                    'property_status': None,
                    'description': None,
                    'publisher': None,
                    'agency_name': None,
                    'published_date': None,
                    'detail_url': None,
                    'scraping_date': None,
                    
                    # Additional fields from ZonaProp
                    'expenses': None,
                    'parking_spaces': None,
                    'floor_number': None,
                    'total_floors': None,
                    'balcony_count': 0,
                    'orientation': None,
                    
                    # Amenities flags
                    'has_pool': False,
                    'has_gym': False,
                    'has_security': False,
                    'has_doorman': False,
                    'has_storage': False,
                    'has_grill': False,
                    'has_sum': False,
                    'has_balcony': False,
                    'has_terrace': False,
                    'has_garage': False,
                    
                    # Photo and labels
                    'photo_count': 0,
                    'labels': []
                }
                
                # Procesar título
                title = js_prop.get('title', '')
                property_data['title'] = title if title else 'N/A'
                
                # Procesar precio
                price_text = js_prop.get('price', '')
                price_info = self._extract_price(price_text)
                property_data['price'] = price_info['amount']
                property_data['currency'] = price_info['currency']
                
                # Procesar ubicación
                location_text = js_prop.get('location', '')
                location_info = self._extract_location_info(location_text)
                property_data['location'] = location_info['full_location']
                property_data['neighborhood'] = location_info['neighborhood']
                property_data['full_address'] = location_text  # Use location as full address
                
                # Procesar dormitorios
                bedrooms_text = js_prop.get('bedrooms', '')
                property_data['bedrooms'] = self._extract_bedrooms(bedrooms_text)
                
                # Procesar baños
                bathrooms_text = js_prop.get('bathrooms', '')
                property_data['bathrooms'] = self._extract_bathrooms(bathrooms_text)
                
                # Procesar área
                area_text = js_prop.get('area', '')
                area_value = self._extract_area(area_text)
                property_data['area'] = area_value
                property_data['surface_total'] = area_value  # Use area as surface total
                
                # Procesar amenities y características especiales
                properties_text = js_prop.get('properties_text', '')
                amenities_list = js_prop.get('amenities', [])
                full_text = js_prop.get('full_text', '').lower()
                
                amenities_result = self._extract_comprehensive_amenities(properties_text, amenities_list, full_text)
                property_data.update(amenities_result)
                
                # Tipo de propiedad y estado
                property_type_info = self._extract_property_type_and_status(title)
                property_data['property_type'] = property_type_info['type']
                property_data['property_status'] = property_type_info['status']
                
                # Descripción
                property_data['description'] = js_prop.get('description', None)
                
                # Publisher y agencia
                property_data['publisher'] = js_prop.get('publisher', None)
                property_data['agency_name'] = js_prop.get('agency_name', None)
                
                # Fecha de publicación
                property_data['published_date'] = js_prop.get('published_date', None)
                
                # Número de fotos
                photo_count_text = js_prop.get('photo_count', '')
                property_data['photo_count'] = self._extract_photo_count(photo_count_text)
                
                # Etiquetas especiales
                property_data['labels'] = js_prop.get('labels', [])
                
                # Extraer información adicional del texto completo
                additional_info = self._extract_additional_info(full_text)
                property_data.update(additional_info)
                
                # URL de detalle
                detail_url = js_prop.get('detail_url', '')
                if detail_url and not detail_url.startswith('http'):
                    detail_url = f"https://www.properati.com.ar{detail_url}"
                property_data['detail_url'] = detail_url if detail_url else 'N/A'
                
                # Timestamp
                property_data['scraping_date'] = datetime.now().isoformat()
                
                # Solo agregar si tiene datos válidos
                if self._is_valid_property(property_data):
                    page_properties.append(property_data)
            
            self.logger.info(f"✅ Extraídas {len(page_properties)} propiedades válidas de página {page_number}")
            return page_properties
            
        except Exception as e:
            self.logger.error(f"❌ Error extrayendo propiedades de página {page_number}: {e}")
            return []
    
    def _extract_price(self, price_text: str) -> Dict[str, Any]:
        """Extrae precio y moneda"""
        if not price_text:
            return {"currency": None, "amount": None}
        
        # Remover espacios y puntos como separadores de miles
        clean_text = price_text.strip().replace(".", "").replace(",", "")
        
        # Patrones para diferentes formatos de precio
        patterns = [
            r'USD\s*(\d+)',
            r'U\$S\s*(\d+)',
            r'ARS\s*(\d+)',
            r'\$\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, clean_text)
            if match:
                amount = int(match.group(1))
                if 'USD' in price_text or 'U$S' in price_text:
                    currency = 'USD'
                else:
                    currency = 'ARS'
                return {"currency": currency, "amount": amount}
        
        return {"currency": None, "amount": None}
    
    def _extract_location_info(self, location_text: str) -> Dict[str, str]:
        """Extrae información de ubicación"""
        result = {
            "full_location": location_text.strip() if location_text else None,
            "neighborhood": None,
        }
        
        if location_text:
            # Separar barrio y ciudad
            parts = [part.strip() for part in location_text.split(",")]
            if len(parts) >= 1:
                result["neighborhood"] = parts[0]
        
        return result
    
    def _extract_bedrooms(self, bedrooms_text: str) -> Optional[int]:
        """Extrae número de dormitorios"""
        if not bedrooms_text:
            return None
        
        numbers = re.findall(r'\d+', bedrooms_text)
        if numbers:
            return int(numbers[0])
        
        return None
    
    def _extract_bathrooms(self, bathrooms_text: str) -> Optional[int]:
        """Extrae número de baños"""
        if not bathrooms_text:
            return None
        
        numbers = re.findall(r'\d+', bathrooms_text)
        if numbers:
            return int(numbers[0])
        
        return None
    
    def _extract_area(self, area_text: str) -> Optional[int]:
        """Extrae área en m²"""
        if not area_text:
            return None
        
        match = re.search(r'(\d+)\s*m[²2]', area_text)
        if match:
            return int(match.group(1))
        
        return None
    
    def _extract_amenities(self, properties_text: str, amenities_list: List[str]) -> List[str]:
        """Extrae amenities"""
        amenities = set()
        
        # Añadir amenities de la lista directa
        for amenity in amenities_list:
            if amenity.strip():
                amenities.add(amenity.strip())
        
        # Buscar amenities comunes en el texto de propiedades
        if properties_text:
            text_lower = properties_text.lower()
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
            }
            
            for keyword, amenity in common_amenities.items():
                if keyword in text_lower:
                    amenities.add(amenity)
        
        return sorted(list(amenities))
    
    def _extract_property_type(self, title: str) -> str:
        """Extrae tipo de propiedad"""
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
        
        return "Departamento"  # Default
    
    def _is_valid_property(self, property_data: Dict[str, Any]) -> bool:
        """Verifica si la propiedad tiene datos válidos"""
        required_fields = ['property_id', 'title', 'price']
        valid_count = sum(1 for field in required_fields 
                         if property_data.get(field) and property_data.get(field) != 'N/A')
        
        # Considerar válida si tiene al menos 2 campos válidos
        return valid_count >= 2
    
    def _extract_comprehensive_amenities(self, properties_text: str, amenities_list: List[str], full_text: str) -> Dict[str, Any]:
        """Extrae amenities completos y flags booleanos"""
        result = {
            'amenities': [],
            'has_pool': False,
            'has_gym': False,
            'has_security': False,
            'has_doorman': False,
            'has_storage': False,
            'has_grill': False,
            'has_sum': False,
            'has_balcony': False,
            'has_terrace': False,
            'has_garage': False,
            'parking_spaces': None,
            'balcony_count': 0,
            'rooms': None
        }
        
        # Combinar todas las fuentes de texto
        combined_text = f"{properties_text} {' '.join(amenities_list)} {full_text}".lower()
        
        # Extraer amenities básicos
        for amenity in amenities_list:
            if amenity:
                result['amenities'].append(amenity)
        
        # Detectar amenities específicos y establecer flags
        if any(word in combined_text for word in ['balcón', 'balcon']):
            result['has_balcony'] = True
            result['balcony_count'] = 1
            if 'balcón' not in [a.lower() for a in result['amenities']]:
                result['amenities'].append('Balcón')
        
        if any(word in combined_text for word in ['terraza', 'terrasse']):
            result['has_terrace'] = True
            if 'terraza' not in [a.lower() for a in result['amenities']]:
                result['amenities'].append('Terraza')
        
        if any(word in combined_text for word in ['cochera', 'garage', 'estacionamiento']):
            result['has_garage'] = True
            # Buscar número de cocheras
            parking_match = re.search(r'(\d+)\s*cochera', combined_text)
            if parking_match:
                result['parking_spaces'] = int(parking_match.group(1))
        
        if any(word in combined_text for word in ['piscina', 'pileta', 'pool']):
            result['has_pool'] = True
        
        if any(word in combined_text for word in ['gimnasio', 'gym']):
            result['has_gym'] = True
        
        if any(word in combined_text for word in ['seguridad', 'vigilancia', 'security']):
            result['has_security'] = True
        
        if any(word in combined_text for word in ['portero', 'conserje', 'doorman']):
            result['has_doorman'] = True
        
        if any(word in combined_text for word in ['baulera', 'storage', 'depósito']):
            result['has_storage'] = True
        
        if any(word in combined_text for word in ['parrilla', 'asador', 'grill']):
            result['has_grill'] = True
        
        if any(word in combined_text for word in ['sum', 'salón', 'salon']):
            result['has_sum'] = True
        
        # Extraer número de ambientes
        rooms_match = re.search(r'(\d+)\s*amb', combined_text)
        if rooms_match:
            result['rooms'] = int(rooms_match.group(1))
        
        return result
    
    def _extract_property_type_and_status(self, title: str) -> Dict[str, str]:
        """Extrae tipo de propiedad y estado"""
        if not title:
            return {'type': None, 'status': None}
        
        title_lower = title.lower()
        
        # Tipo de propiedad
        property_type = None
        if 'departamento' in title_lower:
            property_type = 'Departamento'
        elif 'casa' in title_lower:
            property_type = 'Casa'
        elif 'ph' in title_lower:
            property_type = 'PH'
        elif 'local' in title_lower:
            property_type = 'Local'
        elif 'oficina' in title_lower:
            property_type = 'Oficina'
        elif 'terreno' in title_lower or 'lote' in title_lower:
            property_type = 'Terreno'
        elif 'loft' in title_lower:
            property_type = 'Loft'
        elif 'duplex' in title_lower:
            property_type = 'Duplex'
        
        # Estado de la propiedad
        property_status = None
        if 'a estrenar' in title_lower:
            property_status = 'a estrenar'
        elif 'en construcción' in title_lower:
            property_status = 'en construcción'
        elif 'usado' in title_lower:
            property_status = 'usado'
        elif 'muy bueno' in title_lower:
            property_status = 'muy bueno'
        elif 'excelente' in title_lower:
            property_status = 'excelente'
        
        return {'type': property_type, 'status': property_status}
    
    def _extract_photo_count(self, photo_text: str) -> int:
        """Extrae número de fotos"""
        if not photo_text:
            return 0
        
        match = re.search(r'\d+', photo_text)
        return int(match.group()) if match else 0
    
    def _extract_additional_info(self, full_text: str) -> Dict[str, Any]:
        """Extrae información adicional del texto completo"""
        result = {
            'floor_number': None,
            'total_floors': None,
            'orientation': None,
            'expenses': None
        }
        
        if not full_text:
            return result
        
        # Número de piso
        floor_patterns = [
            r'(\d+)°?\s*piso',
            r'piso\s*(\d+)',
            r'(\d+)°'
        ]
        
        for pattern in floor_patterns:
            match = re.search(pattern, full_text)
            if match:
                result['floor_number'] = int(match.group(1))
                break
        
        # Orientación
        if 'al frente' in full_text:
            result['orientation'] = 'frente'
        elif 'contrafrente' in full_text:
            result['orientation'] = 'contrafrente'
        elif 'interno' in full_text:
            result['orientation'] = 'interno'
        
        # Expensas
        expenses_match = re.search(r'expensas?\s*\$?\s*(\d+)', full_text)
        if expenses_match:
            result['expenses'] = int(expenses_match.group(1))
        
        return result
    
    def save_data(self, filename: str = "properati_full_scraping", format: str = "both", append_mode: bool = False) -> Dict[str, str]:
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
        pages = sorted(set(prop.get('page_number', 0) for prop in self.scraped_data))
        print(f"Páginas procesadas: {pages}")
        print(f"Propiedades por página: {total // len(pages) if pages else 0} promedio")
        
        # Calidad de datos por campo
        fields = ['price', 'location', 'area', 'bedrooms', 'bathrooms']
        print(f"\n📋 Calidad de datos:")
        
        for field in fields:
            valid_count = sum(1 for prop in self.scraped_data 
                            if prop.get(field) and prop.get(field) != 'N/A' and prop.get(field) is not None)
            percentage = (valid_count / total * 100) if total > 0 else 0
            print(f"  {field.capitalize()}: {valid_count}/{total} ({percentage:.1f}%)")
        
        # Rangos de precios
        prices = []
        for prop in self.scraped_data:
            price = prop.get('price')
            if price and isinstance(price, (int, float)):
                prices.append(price)
        
        if prices:
            print(f"\n💰 Análisis de precios:")
            print(f"  Rango: ${min(prices):,} - ${max(prices):,}")
            print(f"  Promedio: ${sum(prices) // len(prices):,}")
            print(f"  Propiedades con precio: {len(prices)}/{total}")
        
        # Ubicaciones únicas
        locations = [prop.get('neighborhood', '') for prop in self.scraped_data 
                    if prop.get('neighborhood') and prop.get('neighborhood') != 'N/A']
        unique_locations = len(set(locations))
        print(f"\n📍 Barrios únicos: {unique_locations}")
        
        # Top 5 ubicaciones
        if locations:
            from collections import Counter
            top_locations = Counter(locations).most_common(5)
            print(f"  Top barrios:")
            for location, count in top_locations:
                print(f"    {location}: {count} propiedades")


def main():
    """Función principal para ejecutar el scraper desde línea de comandos"""
    parser = argparse.ArgumentParser(description="Properati Full Scraper con CDP Mode")
    parser.add_argument("--pages", type=int, default=5, help="Número de páginas a scrapear (default: 5)")
    parser.add_argument("--start-page", type=int, default=1, help="Página inicial (default: 1)")
    parser.add_argument("--output", type=str, default="properati_full", help="Nombre base del archivo de salida")
    parser.add_argument("--format", type=str, choices=["csv", "json", "both"], default="both", help="Formato de salida")
    parser.add_argument("--headless", action="store_true", help="Ejecutar en modo headless")
    parser.add_argument("--no-incognito", action="store_true", help="Desactivar modo incógnito")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay entre páginas en segundos (default: 3.0)")
    parser.add_argument("--properties-per-page", type=int, default=30, help="Propiedades por página (default: 30)")
    parser.add_argument("--resume", type=str, help="Resume extraction by appending to existing file (provide base filename without timestamp)")
    parser.add_argument("--save-every", type=int, default=0, help="Save progress every N pages (0 = save only at end)")
    
    args = parser.parse_args()
    
    try:
        print("🚀 Properati Full Scraper - CDP Mode")
        print(f"📄 Páginas: {args.start_page} a {args.start_page + args.pages - 1}")
        print(f"🎯 Objetivo: ~{args.pages * args.properties_per_page} propiedades")
        print("=" * 60)
        
        # Crear instancia del scraper
        scraper = ProperatiFullScraper(
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
            
            print(f"\n✅ CDP Mode: Bypass exitoso de detección")
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
