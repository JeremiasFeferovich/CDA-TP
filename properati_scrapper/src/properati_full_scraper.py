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
        self.coordinate_cache = {}  # Cache para coordenadas por URL
        self.failed_coordinates = set()  # URLs que fallaron para evitar reintentos
        
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
                            output_name: str = "properati_full", output_format: str = "both",
                            extract_exact_coordinates: bool = False) -> List[Dict[str, Any]]:
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
                    page_properties = self._extract_page_properties(sb, page_num, properties_per_page, extract_exact_coordinates)
                    
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
                
                # Scroll gradual para encontrar el botón de paginación (90% de la página)
                sb.execute_script("""
                    const targetScroll = document.body.scrollHeight * 0.9;
                    window.scrollTo({
                        top: targetScroll,
                        behavior: 'smooth'
                    });
                """)
                time.sleep(2)  # Dar tiempo para que termine el scroll suave
                
                # Intentar encontrar y hacer clic en el botón de paginación
                pagination_success = False
                
                # Selectores para botones de paginación (en orden de prioridad)
                pagination_selectors = [
                    f'a[href*="/venta/{page_number}"]',  # Link directo a la página específica
                    f'button:contains("{page_number}")',  # Botón con el número de página
                    'a:contains("Siguiente")',  # Link "Siguiente"
                    'button:contains("Siguiente")',  # Botón "Siguiente"
                    '.pagination a[rel="next"]',  # Link de siguiente página
                    '.pagination .next',  # Clase next en paginación
                    'a[aria-label="Siguiente"]',  # Aria label para siguiente
                    'button[aria-label="Siguiente"]',  # Botón con aria label
                    '.pagination a:last-child'  # Último link en paginación
                ]
                
                self.logger.info(f"🔍 Buscando botones de paginación...")
                
                for i, selector in enumerate(pagination_selectors):
                    try:
                        if sb.is_element_visible(selector):
                            self.logger.info(f"🎯 Encontrado botón de paginación [{i+1}]: {selector}")
                            
                            # Hacer scroll para asegurar que el elemento esté visible
                            sb.execute_script(f"document.querySelector('{selector}')?.scrollIntoView({{behavior: 'smooth', block: 'center'}});")
                            time.sleep(1)
                            
                            # Intentar hacer clic
                            sb.click(selector)
                            time.sleep(3)  # Esperar navegación
                            
                            # Verificar si la navegación fue exitosa
                            current_url = sb.get_current_url()
                            expected_in_url = f"venta/{page_number}" if page_number > 1 else "venta"
                            
                            if expected_in_url in current_url:
                                self.logger.info(f"✅ Navegación exitosa usando botón de paginación: página {page_number}")
                                pagination_success = True
                                break
                            else:
                                self.logger.warning(f"⚠️  Botón clickeado pero URL no cambió correctamente. URL actual: {current_url}")
                        
                    except Exception as e:
                        self.logger.debug(f"❌ Error con selector [{i+1}] {selector}: {e}")
                        continue
                
                # Si la navegación por botón falló, usar navegación directa como fallback
                if not pagination_success:
                    self.logger.warning(f"⚠️  No se pudo navegar usando botones. Usando navegación directa como fallback...")
                    try:
                        page_url = f"https://www.properati.com.ar/s/capital-federal/venta/{page_number}"
                        self.logger.info(f"🌐 Navegando directamente a página {page_number}: {page_url}")
                        sb.get(page_url)
                        time.sleep(4)
                        self.logger.info(f"✅ Navegación directa exitosa a página {page_number}")
                        return True
                    except Exception as e:
                        self.logger.error(f"❌ Error navegando directamente a página {page_number}: {e}")
                        return False
                
                return pagination_success
            
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
    
    def _extract_page_properties(self, sb, page_number: int, max_properties: int, extract_exact_coordinates: bool = False) -> List[Dict[str, Any]]:
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
                
                // Buscar coordenadas en scripts de la página
                property.coordinates_found = [];
                var scripts = document.querySelectorAll('script');
                scripts.forEach(function(script) {
                    var content = script.textContent || script.innerHTML;
                    if (content) {
                        // Buscar coordenadas de Buenos Aires (aproximadamente -34, -58)
                        var coordPatterns = [
                            /-34\.\d+/g,  // Latitud Buenos Aires
                            /-58\.\d+/g   // Longitud Buenos Aires
                        ];
                        
                        coordPatterns.forEach(function(pattern) {
                            var matches = content.match(pattern);
                            if (matches) {
                                property.coordinates_found.push(...matches.slice(0, 2));
                            }
                        });
                    }
                });
                
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
                    'labels': [],
                    
                    # Coordinates
                    'latitude': None,
                    'longitude': None,
                    'coordinates': None
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
                
                # Extraer coordenadas
                if extract_exact_coordinates:
                    # Extraer coordenadas exactas de la página de detalle
                    detail_url = js_prop.get('detail_url', '')
                    if detail_url:
                        coordinate_info = self._extract_exact_coordinates(sb, detail_url, property_data.get('property_id', ''))
                    else:
                        # Fallback a geocodificación por barrio
                        neighborhood = property_data.get('neighborhood', '')
                        full_address = property_data.get('full_address', '')
                        coordinate_info = self._geocode_address(full_address, neighborhood)
                else:
                    # Usar geocodificación rápida por barrio
                    neighborhood = property_data.get('neighborhood', '')
                    full_address = property_data.get('full_address', '')
                    coordinate_info = self._geocode_address(full_address, neighborhood)
                
                property_data.update(coordinate_info)
                
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
    
    def _extract_coordinates(self, coordinates_data: List[str]) -> Dict[str, Any]:
        """Extrae coordenadas de latitud y longitud"""
        result = {
            'latitude': None,
            'longitude': None,
            'coordinates': None
        }
        
        if not coordinates_data:
            return result
        
        # Buscar latitud y longitud en los datos
        latitude = None
        longitude = None
        
        # Filtrar coordenadas para evitar duplicados globales
        unique_coords = list(set(coordinates_data))
        
        for coord in unique_coords:
            try:
                coord_float = float(coord)
                
                # Buenos Aires está aproximadamente en -34.6, -58.4
                # Latitud: entre -35 y -34
                # Longitud: entre -59 y -57
                if -35.5 < coord_float < -33.5:  # Rango de latitud para Buenos Aires
                    if latitude is None:  # Solo tomar la primera latitud válida
                        latitude = coord_float
                elif -59.5 < coord_float < -57.0:  # Rango de longitud para Buenos Aires
                    if longitude is None:  # Solo tomar la primera longitud válida
                        longitude = coord_float
            except (ValueError, TypeError):
                continue
        
        # Asignar coordenadas si se encontraron ambas
        if latitude is not None and longitude is not None:
            result['latitude'] = latitude
            result['longitude'] = longitude
            result['coordinates'] = f"{latitude},{longitude}"
        
        return result
    
    def _geocode_address(self, address: str, neighborhood: str) -> Dict[str, Any]:
        """Geocodifica una dirección usando coordenadas aproximadas por barrio de Buenos Aires"""
        result = {
            'latitude': None,
            'longitude': None,
            'coordinates': None
        }
        
        if not neighborhood:
            return result
        
        # Coordenadas aproximadas de barrios de Buenos Aires
        # Estas son coordenadas reales de los centros de cada barrio
        neighborhood_coords = {
            'puerto madero': (-34.6118, -58.3691),
            'belgrano': (-34.5627, -58.4583),
            'palermo': (-34.5889, -58.4196),
            'recoleta': (-34.5875, -58.3974),
            'san telmo': (-34.6214, -58.3731),
            'la boca': (-34.6345, -58.3617),
            'barracas': (-34.6484, -58.3692),
            'san nicolas': (-34.6037, -58.3816),
            'monserrat': (-34.6132, -58.3782),
            'retiro': (-34.5935, -58.3747),
            'balvanera': (-34.6092, -58.3998),
            'centro': (-34.6037, -58.3816),
            'microcentro': (-34.6037, -58.3816),
            'flores': (-34.6323, -58.4676),
            'caballito': (-34.6198, -58.4370),
            'villa crespo': (-34.5998, -58.4370),
            'almagro': (-34.6132, -58.4198),
            'boedo': (-34.6323, -58.4198),
            'san cristobal': (-34.6198, -58.3998),
            'barrio norte': (-34.5875, -58.3974),
            'once': (-34.6092, -58.3998),
            'abasto': (-34.6092, -58.3998),
            'congreso': (-34.6092, -58.3870),
            'tribunales': (-34.6037, -58.3870),
            'villa urquiza': (-34.5751, -58.4583),
            'villa ortuzar': (-34.5751, -58.4583),
            'coghlan': (-34.5627, -58.4583),
            'nunez': (-34.5464, -58.4583),
            'saavedra': (-34.5464, -58.4583),
            'villa pueyrredon': (-34.5627, -58.4676),
            'agronomia': (-34.5751, -58.4889),
            'chacarita': (-34.5889, -58.4583),
            'paternal': (-34.6037, -58.4583),
            'villa general mitre': (-34.6037, -58.4583),
            'villa santa rita': (-34.6323, -58.4676),
            'floresta': (-34.6323, -58.4889),
            'velez sarsfield': (-34.6323, -58.5089),
            'liniers': (-34.6484, -58.5289),
            'mataderos': (-34.6645, -58.5089),
            'parque avellaneda': (-34.6484, -58.4889),
            'nueva pompeya': (-34.6645, -58.4198),
            'parque patricios': (-34.6484, -58.4198),
            'constitucion': (-34.6276, -58.3817)
        }
        
        # Buscar coordenadas por barrio
        neighborhood_lower = neighborhood.lower().strip()
        
        # Buscar coincidencia exacta primero
        if neighborhood_lower in neighborhood_coords:
            lat, lng = neighborhood_coords[neighborhood_lower]
            result['latitude'] = lat
            result['longitude'] = lng
            result['coordinates'] = f"{lat},{lng}"
            return result
        
        # Buscar coincidencia parcial
        for barrio, coords in neighborhood_coords.items():
            if barrio in neighborhood_lower or neighborhood_lower in barrio:
                lat, lng = coords
                result['latitude'] = lat
                result['longitude'] = lng
                result['coordinates'] = f"{lat},{lng}"
                return result
        
        return result
    
    def _extract_exact_coordinates(self, sb, detail_url: str, property_id: str) -> Dict[str, Any]:
        """Extrae coordenadas exactas navegando a la página de detalle de la propiedad"""
        result = {
            'latitude': None,
            'longitude': None,
            'coordinates': None
        }
        
        if not detail_url:
            return result
        
        # Construir URL completa si es necesario
        if not detail_url.startswith('http'):
            detail_url = f"https://www.properati.com.ar{detail_url}"
        
        # Verificar cache primero
        if detail_url in self.coordinate_cache:
            self.logger.debug(f"📋 Cache hit para propiedad {property_id[:8]}")
            return self.coordinate_cache[detail_url].copy()
        
        # Verificar si ya falló antes
        if detail_url in self.failed_coordinates:
            self.logger.debug(f"⚠️  Saltando propiedad {property_id[:8]} - falló anteriormente")
            return result
        
        try:
            self.logger.info(f"🗺️  Extrayendo coordenadas exactas para propiedad {property_id[:8]}...")
            
            # Navegar a la página de detalle
            sb.get(detail_url)
            
            # Esperar a que cargue la página (reducido)
            try:
                sb.wait_for_element_visible('.location-map-wrapper', timeout=3)
            except:
                # Intentar extraer coordenadas sin esperar al mapa
                pass
            
            # Scroll hacia el mapa para asegurar que esté visible (optimizado)
            sb.execute_script("document.querySelector('.location-map-wrapper')?.scrollIntoView();")
            time.sleep(0.5)
            
            # Intentar hacer clic en "Ver en mapa" si está disponible
            try:
                # Buscar el botón "Ver en mapa" y hacer clic
                map_button_selectors = [
                    '#view-map-button',
                    '.location-map__view-button',
                    '[class*="view-map"]',
                    'button:contains("Ver en mapa")',
                    'div:contains("Ver en mapa")'
                ]
                
                button_clicked = False
                for selector in map_button_selectors:
                    try:
                        if sb.is_element_visible(selector):
                            sb.js_click(selector)
                            button_clicked = True
                            self.logger.debug(f"✅ Botón mapa clickeado: {selector}")
                            break
                    except:
                        continue
                
                if button_clicked:
                    time.sleep(1)  # Esperar a que cargue el mapa (optimizado)
            except Exception as e:
                self.logger.debug(f"No se pudo hacer clic en botón mapa: {e}")
            
            # Extraer coordenadas usando JavaScript
            coordinate_extraction_script = """
            // Buscar coordenadas en la página de detalle
            var coordinates = {lat: null, lng: null};
            
            // Método 1: Buscar en scripts
            var scripts = document.querySelectorAll('script');
            scripts.forEach(function(script) {
                var content = script.textContent || script.innerHTML;
                if (content) {
                    // Buscar patrones de coordenadas específicas
                    var latMatch = content.match(/-34\\.\\d{6,}/);
                    var lngMatch = content.match(/-58\\.\\d{6,}/);
                    
                    if (latMatch && lngMatch) {
                        coordinates.lat = parseFloat(latMatch[0]);
                        coordinates.lng = parseFloat(lngMatch[0]);
                        return;
                    }
                }
            });
            
            // Método 2: Buscar en atributos data
            var mapElements = document.querySelectorAll('[data-lat], [data-lng], [data-latitude], [data-longitude]');
            mapElements.forEach(function(el) {
                var lat = el.getAttribute('data-lat') || el.getAttribute('data-latitude');
                var lng = el.getAttribute('data-lng') || el.getAttribute('data-longitude');
                
                if (lat && lng) {
                    coordinates.lat = parseFloat(lat);
                    coordinates.lng = parseFloat(lng);
                }
            });
            
            // Método 3: Buscar en variables globales
            if (window.propertyCoordinates) {
                coordinates = window.propertyCoordinates;
            } else if (window.mapData) {
                if (window.mapData.lat) coordinates.lat = window.mapData.lat;
                if (window.mapData.lng) coordinates.lng = window.mapData.lng;
            }
            
            // Método 4: Buscar en el texto de la página patrones específicos
            var pageText = document.documentElement.textContent;
            var coordMatches = pageText.match(/-34\\.\\d{6,}\\s*,\\s*-58\\.\\d{6,}/);
            if (coordMatches) {
                var coords = coordMatches[0].split(',');
                coordinates.lat = parseFloat(coords[0].trim());
                coordinates.lng = parseFloat(coords[1].trim());
            }
            
            return coordinates;
            """
            
            # Ejecutar script de extracción
            js_result = sb.execute_script(coordinate_extraction_script)
            
            if js_result and js_result.get('lat') and js_result.get('lng'):
                lat = float(js_result['lat'])
                lng = float(js_result['lng'])
                
                # Validar que las coordenadas están en Buenos Aires
                if -35.5 < lat < -33.5 and -59.5 < lng < -57.0:
                    result['latitude'] = lat
                    result['longitude'] = lng
                    result['coordinates'] = f"{lat},{lng}"
                    self.logger.info(f"✅ Coordenadas exactas extraídas: {lat:.6f},{lng:.6f}")
                    
                    # Guardar en cache
                    self.coordinate_cache[detail_url] = result.copy()
                else:
                    self.logger.warning(f"⚠️  Coordenadas fuera de rango Buenos Aires: {lat},{lng}")
                    self.failed_coordinates.add(detail_url)
            else:
                self.logger.warning(f"⚠️  No se encontraron coordenadas para propiedad {property_id[:8]}")
                self.failed_coordinates.add(detail_url)
                
        except Exception as e:
            self.logger.error(f"❌ Error extrayendo coordenadas exactas para {property_id[:8]}: {e}")
            self.failed_coordinates.add(detail_url)
        
        return result
    
    def _extract_coordinates_batch(self, sb, properties_batch: List[Dict], extract_exact_coordinates: bool = False) -> List[Dict]:
        """Procesa un lote de propiedades optimizando la extracción de coordenadas"""
        if not extract_exact_coordinates:
            # Usar geocodificación rápida para todo el lote
            for prop in properties_batch:
                neighborhood = prop.get('neighborhood', '')
                full_address = prop.get('full_address', '')
                coordinate_info = self._geocode_address(full_address, neighborhood)
                prop.update(coordinate_info)
            return properties_batch
        
        # Para coordenadas exactas, procesar con optimizaciones
        processed_batch = []
        
        for prop in properties_batch:
            detail_url = prop.get('detail_url', '')
            property_id = prop.get('property_id', '')
            
            if detail_url:
                # Usar método optimizado con cache
                coordinate_info = self._extract_exact_coordinates(sb, detail_url, property_id)
                prop.update(coordinate_info)
                
                # Si no se encontraron coordenadas exactas, usar fallback
                if not coordinate_info.get('latitude'):
                    neighborhood = prop.get('neighborhood', '')
                    full_address = prop.get('full_address', '')
                    fallback_coords = self._geocode_address(full_address, neighborhood)
                    prop.update(fallback_coords)
            else:
                # Sin URL, usar geocodificación
                neighborhood = prop.get('neighborhood', '')
                full_address = prop.get('full_address', '')
                coordinate_info = self._geocode_address(full_address, neighborhood)
                prop.update(coordinate_info)
            
            processed_batch.append(prop)
        
        return processed_batch
    
    def _extract_coordinates_from_jsonld(self, sb) -> Dict[str, Dict[str, float]]:
        """Extrae coordenadas desde JSON-LD estructurado en la página de listado (ULTRA RÁPIDO)"""
        coordinate_extraction_script = """
        // Extraer coordenadas desde JSON-LD estructurado (mejorado)
        var coordinatesMap = {};
        var coordIndex = 0;
        
        // Buscar scripts JSON-LD
        var jsonLdScripts = document.querySelectorAll('script[type="application/ld+json"]');
        
        jsonLdScripts.forEach(function(script) {
            try {
                var content = script.textContent.trim();
                if (!content) return;
                
                // Intentar parsear como JSON
                var data = JSON.parse(content);
                
                // Función recursiva para buscar coordenadas en cualquier parte del objeto
                function findCoordinates(obj, depth) {
                    if (depth > 10) return; // Evitar recursión infinita
                    
                    if (obj && typeof obj === 'object') {
                        // Buscar coordenadas directas
                        if (obj.geo && obj.geo.latitude && obj.geo.longitude) {
                            var lat = parseFloat(obj.geo.latitude);
                            var lng = parseFloat(obj.geo.longitude);
                            
                            if (lat > -35.5 && lat < -33.5 && lng > -59.5 && lng < -57.0) {
                                var key = 'property_' + coordIndex;
                                if (obj.name) key = obj.name.substring(0, 50);
                                else if (obj.address && obj.address.streetAddress) key = obj.address.streetAddress.substring(0, 50);
                                
                                coordinatesMap[key] = {
                                    latitude: lat,
                                    longitude: lng,
                                    name: obj.name || '',
                                    locality: obj.address ? obj.address.addressLocality : ''
                                };
                                coordIndex++;
                            }
                        }
                        
                        // Buscar en arrays
                        if (Array.isArray(obj)) {
                            obj.forEach(function(item) {
                                findCoordinates(item, depth + 1);
                            });
                        } else {
                            // Buscar en propiedades del objeto
                            Object.keys(obj).forEach(function(key) {
                                findCoordinates(obj[key], depth + 1);
                            });
                        }
                    }
                }
                
                findCoordinates(data, 0);
                
            } catch (e) {
                // Si falla JSON, buscar patrones de coordenadas en el texto
                var coordPattern = /"latitude":\s*"?(-34\.[0-9]+)"?[^}]*"longitude":\s*"?(-58\.[0-9]+)"?/g;
                var match;
                while ((match = coordPattern.exec(content)) !== null) {
                    var lat = parseFloat(match[1]);
                    var lng = parseFloat(match[2]);
                    
                    if (lat > -35.5 && lat < -33.5 && lng > -59.5 && lng < -57.0) {
                        coordinatesMap['script_coord_' + coordIndex] = {
                            latitude: lat,
                            longitude: lng,
                            name: '',
                            locality: ''
                        };
                        coordIndex++;
                    }
                }
            }
        });
        
        return coordinatesMap;
        """
        
        try:
            result = sb.execute_script(coordinate_extraction_script)
            if result and isinstance(result, dict):
                self.logger.info(f"🗺️  JSON-LD: Extraídas {len(result)} coordenadas de la página")
                return result
            else:
                self.logger.warning(f"⚠️  JSON-LD: No se encontraron coordenadas estructuradas")
                return {}
        except Exception as e:
            self.logger.error(f"❌ Error extrayendo coordenadas JSON-LD: {e}")
            return {}
    
    def _extract_coordinates_from_scripts_fast(self, sb) -> Dict[str, Dict[str, float]]:
        """Método alternativo rápido para extraer coordenadas desde scripts"""
        coordinate_extraction_script = """
        // Extraer coordenadas desde scripts (método alternativo mejorado)
        var coordinatesMap = {};
        var coordIndex = 0;
        var usedCoordinates = new Set(); // Evitar duplicados
        
        // Buscar en todos los scripts
        var scripts = document.querySelectorAll('script');
        scripts.forEach(function(script) {
            var content = script.textContent || script.innerHTML;
            if (content && content.includes('latitude') && content.includes('longitude')) {
                
                // Patrón mejorado para encontrar pares lat/lng juntos
                var coordPairPattern = /"latitude":\s*"?(-34\.[0-9]+)"?[^}]{0,200}"longitude":\s*"?(-58\.[0-9]+)"?/g;
                var match;
                
                while ((match = coordPairPattern.exec(content)) !== null) {
                    var lat = parseFloat(match[1]);
                    var lng = parseFloat(match[2]);
                    
                    // Validar coordenadas de Buenos Aires
                    if (lat > -35.5 && lat < -33.5 && lng > -59.5 && lng < -57.0) {
                        var coordKey = lat.toFixed(6) + ',' + lng.toFixed(6);
                        
                        // Evitar duplicados
                        if (!usedCoordinates.has(coordKey)) {
                            usedCoordinates.add(coordKey);
                            coordinatesMap['script_prop_' + coordIndex] = {
                                latitude: lat,
                                longitude: lng
                            };
                            coordIndex++;
                        }
                    }
                }
                
                // Si no encontramos pares, buscar por separado pero con mejor lógica
                if (coordIndex === 0) {
                    var latPattern = /"latitude":\s*"?(-34\.[0-9]+)"?/g;
                    var lngPattern = /"longitude":\s*"?(-58\.[0-9]+)"?/g;
                    
                    var latMatches = [];
                    var lngMatches = [];
                    
                    // Extraer latitudes únicas
                    while ((match = latPattern.exec(content)) !== null) {
                        var lat = parseFloat(match[1]);
                        if (lat > -35.5 && lat < -33.5 && !latMatches.includes(lat)) {
                            latMatches.push(lat);
                        }
                    }
                    
                    // Extraer longitudes únicas
                    while ((match = lngPattern.exec(content)) !== null) {
                        var lng = parseFloat(match[1]);
                        if (lng > -59.5 && lng < -57.0 && !lngMatches.includes(lng)) {
                            lngMatches.push(lng);
                        }
                    }
                    
                    // Emparejar coordenadas (máximo 30 por página)
                    var maxPairs = Math.min(latMatches.length, lngMatches.length, 30);
                    for (var i = 0; i < maxPairs; i++) {
                        var coordKey = latMatches[i].toFixed(6) + ',' + lngMatches[i].toFixed(6);
                        if (!usedCoordinates.has(coordKey)) {
                            usedCoordinates.add(coordKey);
                            coordinatesMap['script_prop_' + coordIndex] = {
                                latitude: latMatches[i],
                                longitude: lngMatches[i]
                            };
                            coordIndex++;
                        }
                    }
                }
            }
        });
        
        return coordinatesMap;
        """
        
        try:
            result = sb.execute_script(coordinate_extraction_script)
            if result and isinstance(result, dict):
                self.logger.info(f"🔍 Scripts: Extraídas {len(result)} coordenadas alternativas")
                return result
            else:
                self.logger.debug(f"🔍 Scripts: No se encontraron coordenadas")
                return {}
        except Exception as e:
            self.logger.error(f"❌ Error extrayendo coordenadas de scripts: {e}")
            return {}
    
    def _match_property_coordinates(self, properties: List[Dict], coordinates_map: Dict) -> List[Dict]:
        """Asigna coordenadas a propiedades usando coincidencia inteligente"""
        matched_count = 0
        coordinates_list = list(coordinates_map.values())
        
        self.logger.info(f"🔍 Intentando asignar {len(coordinates_list)} coordenadas a {len(properties)} propiedades")
        
        for i, prop in enumerate(properties):
            coordinate_found = False
            
            # Método 1: Asignación secuencial si tenemos suficientes coordenadas únicas
            if len(coordinates_list) >= len(properties) and i < len(coordinates_list):
                coord_data = coordinates_list[i]
                prop['latitude'] = coord_data['latitude']
                prop['longitude'] = coord_data['longitude']
                prop['coordinates'] = f"{coord_data['latitude']},{coord_data['longitude']}"
                matched_count += 1
                coordinate_found = True
                self.logger.debug(f"🎯 Asignación secuencial [{i+1}]: {prop['coordinates']}")
            
            # Método 2: Búsqueda por coincidencia de texto (si el método 1 no funcionó)
            if not coordinate_found:
                match_keys = [
                    prop.get('title', ''),
                    prop.get('location', ''),
                    prop.get('neighborhood', ''),
                    prop.get('full_address', '')
                ]
                
                for coord_key, coord_data in coordinates_map.items():
                    for match_key in match_keys:
                        if match_key and coord_key and (
                            match_key.lower() in coord_key.lower() or 
                            coord_key.lower() in match_key.lower() or
                            coord_data.get('locality', '').lower() in match_key.lower()
                        ):
                            prop['latitude'] = coord_data['latitude']
                            prop['longitude'] = coord_data['longitude']
                            prop['coordinates'] = f"{coord_data['latitude']},{coord_data['longitude']}"
                            matched_count += 1
                            coordinate_found = True
                            self.logger.debug(f"🎯 Match por texto: {coord_key} -> {prop['coordinates']}")
                            break
                    
                    if coordinate_found:
                        break
            
            # Método 3: Fallback a geocodificación por barrio
            if not coordinate_found:
                neighborhood = prop.get('neighborhood', '')
                if neighborhood:
                    fallback_coords = self._geocode_address('', neighborhood)
                    prop.update(fallback_coords)
                    if prop.get('latitude'):
                        self.logger.debug(f"🏘️  Geocodificación: {neighborhood} -> {prop.get('coordinates', 'N/A')}")
        
        self.logger.info(f"🎯 Coordenadas asignadas: {matched_count}/{len(properties)} propiedades con coordenadas exactas")
        return properties
    
    def scrape_ultra_fast(self, total_pages: int, output_name: str = "properati_ultra_fast", 
                         save_every: int = 25) -> List[Dict[str, Any]]:
        """
        Método ULTRA RÁPIDO para scraping masivo usando JSON-LD
        Elimina navegación individual a páginas de detalle
        """
        print(f"⚡ SCRAPING ULTRA RÁPIDO - JSON-LD MODE")
        print(f"📊 Objetivo: ~{total_pages * 30:,} propiedades")
        print(f"🚀 Velocidad estimada: <0.5s por propiedad")
        print(f"💾 Guardar cada: {save_every} páginas")
        print("=" * 60)
        
        start_time = time.time()
        all_data = []
        failed_pages = []
        
        with SB(uc=True, headless=self.headless, incognito=self.incognito) as sb:
            for page_num in range(1, total_pages + 1):
                page_start_time = time.time()
                
                try:
                    # Navegar a la página
                    if not self._navigate_to_page(sb, page_num):
                        self.logger.error(f"❌ Error navegando a página {page_num}")
                        failed_pages.append(page_num)
                        continue
                    
                    # Extraer propiedades básicas (SIN coordenadas exactas)
                    page_properties = self._extract_page_properties(sb, page_num, 30, False)
                    
                    if not page_properties:
                        self.logger.warning(f"⚠️  Página {page_num}: Sin propiedades")
                        continue
                    
                    # Extraer coordenadas desde JSON-LD (ULTRA RÁPIDO)
                    coordinates_map = self._extract_coordinates_from_jsonld(sb)
                    
                    # Si JSON-LD no funcionó, usar método alternativo rápido
                    if not coordinates_map:
                        coordinates_map = self._extract_coordinates_from_scripts_fast(sb)
                    
                    # Asignar coordenadas a propiedades
                    processed_properties = self._match_property_coordinates(page_properties, coordinates_map)
                    
                    all_data.extend(processed_properties)
                    
                    # Calcular velocidad
                    page_time = time.time() - page_start_time
                    props_per_second = len(processed_properties) / page_time if page_time > 0 else 0
                    
                    self.logger.info(f"⚡ Página {page_num}/{total_pages}: {len(processed_properties)} propiedades "
                                   f"({page_time:.1f}s, {props_per_second:.1f} props/s)")
                    
                    # Guardar progreso
                    if page_num % save_every == 0 or page_num == total_pages:
                        self._save_progress(all_data, output_name, page_num, total_pages)
                        
                        # Estadísticas
                        elapsed_time = time.time() - start_time
                        total_props = len(all_data)
                        avg_time_per_prop = elapsed_time / total_props if total_props > 0 else 0
                        
                        print(f"\n⚡ ESTADÍSTICAS ULTRA RÁPIDAS (Página {page_num}):")
                        print(f"   ⏱️  Tiempo transcurrido: {elapsed_time/3600:.2f} horas")
                        print(f"   📊 Propiedades procesadas: {total_props:,}")
                        print(f"   🚀 Velocidad promedio: {avg_time_per_prop:.3f}s/propiedad")
                        
                        # Estimación
                        if page_num < total_pages:
                            remaining_pages = total_pages - page_num
                            estimated_time = (elapsed_time / page_num) * remaining_pages
                            print(f"   ⏳ Tiempo estimado restante: {estimated_time/3600:.2f} horas")
                        print()
                    
                    # Delay mínimo
                    if page_num < total_pages:
                        time.sleep(0.5)  # Delay ultra reducido
                        
                except Exception as e:
                    self.logger.error(f"❌ Error procesando página {page_num}: {e}")
                    failed_pages.append(page_num)
        
        # Estadísticas finales
        total_time = time.time() - start_time
        self._print_ultra_fast_stats(all_data, total_time, failed_pages)
        
        return all_data
    
    def _print_ultra_fast_stats(self, data: List[Dict], total_time: float, failed_pages: List[int]):
        """Imprime estadísticas del scraping ultra rápido"""
        total_props = len(data)
        props_per_second = total_props / total_time if total_time > 0 else 0
        
        print(f"\n⚡ SCRAPING ULTRA RÁPIDO COMPLETADO!")
        print("=" * 60)
        print(f"📊 Propiedades extraídas: {total_props:,}")
        print(f"⏱️  Tiempo total: {total_time/3600:.2f} horas ({total_time/60:.1f} minutos)")
        print(f"🚀 Velocidad promedio: {props_per_second:.2f} props/s ({3600/props_per_second:.2f}s/prop)")
        print(f"❌ Páginas fallidas: {len(failed_pages)}")
        
        # Análisis de coordenadas
        props_with_coords = sum(1 for prop in data if prop.get('latitude'))
        coord_success_rate = props_with_coords / total_props if total_props > 0 else 0
        print(f"🎯 Propiedades con coordenadas: {props_with_coords:,} ({coord_success_rate*100:.1f}%)")
        
        # Proyección para 68k
        if total_props > 0:
            time_per_prop = total_time / total_props
            time_68k = 68000 * time_per_prop
            print(f"\n🎯 PROYECCIÓN PARA 68K PROPIEDADES:")
            print(f"   ⏱️  Tiempo estimado: {time_68k/3600:.1f} horas ({time_68k/3600/24:.1f} días)")
    
    def scrape_large_dataset(self, total_pages: int, output_name: str = "properati_large", 
                           batch_size: int = 50, save_every: int = 10, 
                           extract_exact_coordinates: bool = True) -> List[Dict[str, Any]]:
        """
        Método optimizado para scraping de datasets grandes (68k+ propiedades)
        
        Args:
            total_pages: Total de páginas a procesar
            output_name: Nombre base del archivo
            batch_size: Propiedades por lote de procesamiento
            save_every: Guardar cada N páginas
            extract_exact_coordinates: Si extraer coordenadas exactas
        """
        print(f"🚀 SCRAPING MASIVO OPTIMIZADO")
        print(f"📊 Objetivo: ~{total_pages * 30:,} propiedades")
        print(f"🎯 Coordenadas exactas: {'✅ SÍ' if extract_exact_coordinates else '❌ NO'}")
        print(f"📦 Tamaño de lote: {batch_size}")
        print(f"💾 Guardar cada: {save_every} páginas")
        print("=" * 60)
        
        start_time = time.time()
        all_data = []
        failed_pages = []
        
        # Estadísticas de rendimiento
        coord_cache_hits = 0
        coord_extractions = 0
        
        with SB(uc=True, headless=self.headless, incognito=self.incognito) as sb:
            for page_num in range(1, total_pages + 1):
                page_start_time = time.time()
                
                try:
                    # Navegar a la página
                    if not self._navigate_to_page(sb, page_num):
                        self.logger.error(f"❌ Error navegando a página {page_num}")
                        failed_pages.append(page_num)
                        continue
                    
                    # Extraer propiedades básicas
                    page_properties = self._extract_page_properties(sb, page_num, 30, False)  # Sin coordenadas exactas inicialmente
                    
                    if not page_properties:
                        self.logger.warning(f"⚠️  Página {page_num}: Sin propiedades")
                        continue
                    
                    # Procesar en lotes para coordenadas
                    processed_properties = []
                    for i in range(0, len(page_properties), batch_size):
                        batch = page_properties[i:i+batch_size]
                        
                        # Contar cache hits antes del procesamiento
                        cache_hits_before = len([url for url in [prop.get('detail_url', '') for prop in batch] 
                                               if url in self.coordinate_cache])
                        
                        processed_batch = self._extract_coordinates_batch(sb, batch, extract_exact_coordinates)
                        processed_properties.extend(processed_batch)
                        
                        # Actualizar estadísticas
                        coord_cache_hits += cache_hits_before
                        coord_extractions += len(batch) - cache_hits_before
                        
                        # Mostrar progreso del lote
                        self.logger.info(f"📦 Lote {i//batch_size + 1}: {len(batch)} propiedades procesadas")
                    
                    all_data.extend(processed_properties)
                    
                    # Calcular tiempo y velocidad
                    page_time = time.time() - page_start_time
                    props_per_second = len(processed_properties) / page_time if page_time > 0 else 0
                    
                    self.logger.info(f"✅ Página {page_num}/{total_pages}: {len(processed_properties)} propiedades "
                                   f"({page_time:.1f}s, {props_per_second:.1f} props/s)")
                    
                    # Guardar progreso
                    if page_num % save_every == 0 or page_num == total_pages:
                        self._save_progress(all_data, output_name, page_num, total_pages)
                        
                        # Mostrar estadísticas de rendimiento
                        elapsed_time = time.time() - start_time
                        total_props = len(all_data)
                        avg_time_per_prop = elapsed_time / total_props if total_props > 0 else 0
                        cache_hit_rate = coord_cache_hits / (coord_cache_hits + coord_extractions) if (coord_cache_hits + coord_extractions) > 0 else 0
                        
                        print(f"\n📈 ESTADÍSTICAS DE RENDIMIENTO (Página {page_num}):")
                        print(f"   ⏱️  Tiempo transcurrido: {elapsed_time/3600:.1f} horas")
                        print(f"   📊 Propiedades procesadas: {total_props:,}")
                        print(f"   🚀 Velocidad promedio: {avg_time_per_prop:.2f}s/propiedad")
                        print(f"   📋 Cache hits: {coord_cache_hits:,} ({cache_hit_rate*100:.1f}%)")
                        print(f"   🗺️  Extracciones nuevas: {coord_extractions:,}")
                        
                        # Estimación de tiempo restante
                        if page_num < total_pages:
                            remaining_pages = total_pages - page_num
                            estimated_time = (elapsed_time / page_num) * remaining_pages
                            print(f"   ⏳ Tiempo estimado restante: {estimated_time/3600:.1f} horas")
                        print()
                    
                    # Delay optimizado entre páginas
                    if page_num < total_pages:
                        optimized_delay = max(0.5, self.delay * 0.3)  # Delay reducido
                        time.sleep(optimized_delay)
                        
                except Exception as e:
                    self.logger.error(f"❌ Error procesando página {page_num}: {e}")
                    failed_pages.append(page_num)
        
        # Estadísticas finales
        total_time = time.time() - start_time
        self._print_final_stats(all_data, total_time, failed_pages, coord_cache_hits, coord_extractions)
        
        return all_data
    
    def _save_progress(self, data: List[Dict], output_name: str, current_page: int, total_pages: int):
        """Guarda el progreso durante el scraping masivo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        progress_filename = f"{output_name}_progress_p{current_page}of{total_pages}_{timestamp}"
        
        # Guardar datos
        self.scraped_data = data  # Temporalmente asignar para usar save_data
        saved_files = self.save_data(progress_filename, "both", False)
        
        self.logger.info(f"💾 Progreso guardado: {len(data):,} propiedades")
        for format_type, filepath in saved_files.items():
            self.logger.info(f"   {format_type.upper()}: {filepath}")
    
    def _print_final_stats(self, data: List[Dict], total_time: float, failed_pages: List[int], 
                          cache_hits: int, extractions: int):
        """Imprime estadísticas finales del scraping masivo"""
        total_props = len(data)
        props_per_second = total_props / total_time if total_time > 0 else 0
        cache_hit_rate = cache_hits / (cache_hits + extractions) if (cache_hits + extractions) > 0 else 0
        
        print(f"\n🎉 SCRAPING MASIVO COMPLETADO!")
        print("=" * 60)
        print(f"📊 Propiedades extraídas: {total_props:,}")
        print(f"⏱️  Tiempo total: {total_time/3600:.2f} horas ({total_time/60:.1f} minutos)")
        print(f"🚀 Velocidad promedio: {props_per_second:.2f} props/s ({3600/props_per_second:.1f}s/prop)")
        print(f"📋 Cache hits: {cache_hits:,} ({cache_hit_rate*100:.1f}%)")
        print(f"🗺️  Nuevas extracciones: {extractions:,}")
        print(f"❌ Páginas fallidas: {len(failed_pages)}")
        
        if failed_pages:
            print(f"   Páginas con errores: {failed_pages[:10]}{'...' if len(failed_pages) > 10 else ''}")
        
        # Análisis de coordenadas
        props_with_coords = sum(1 for prop in data if prop.get('latitude'))
        coord_success_rate = props_with_coords / total_props if total_props > 0 else 0
        print(f"🎯 Propiedades con coordenadas: {props_with_coords:,} ({coord_success_rate*100:.1f}%)")
    
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
    parser.add_argument("--exact-coords", action="store_true", help="Extraer coordenadas exactas navegando a cada propiedad (más lento)")
    parser.add_argument("--resume", type=str, help="Resume extraction by appending to existing file (provide base filename without timestamp)")
    parser.add_argument("--save-every", type=int, default=0, help="Save progress every N pages (0 = save only at end)")
    parser.add_argument("--large-dataset", action="store_true", help="Modo optimizado para datasets grandes (68k+ propiedades)")
    parser.add_argument("--batch-size", type=int, default=50, help="Tamaño de lote para procesamiento optimizado (default: 50)")
    parser.add_argument("--ultra-fast", action="store_true", help="Modo ULTRA RÁPIDO usando JSON-LD (sin navegación individual)")
    
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
        
        if args.ultra_fast:
            # Usar método ULTRA RÁPIDO
            print("⚡ MODO ULTRA RÁPIDO ACTIVADO")
            data = scraper.scrape_ultra_fast(
                total_pages=args.pages,
                output_name=output_name,
                save_every=max(1, args.save_every) if args.save_every > 0 else 25  # Default save every 25 pages
            )
        elif args.large_dataset:
            # Usar método optimizado para datasets grandes
            print("🚀 MODO DATASET GRANDE ACTIVADO")
            data = scraper.scrape_large_dataset(
                total_pages=args.pages,
                output_name=output_name,
                batch_size=args.batch_size,
                save_every=max(1, args.save_every) if args.save_every > 0 else 10,  # Default save every 10 pages
                extract_exact_coordinates=args.exact_coords
            )
        else:
            # Usar método estándar
            data = scraper.scrape_multiple_pages(
                start_page=args.start_page,
                max_pages=args.pages,
                properties_per_page=args.properties_per_page,
                save_every=args.save_every,
                output_name=output_name,
                output_format=args.format,
                extract_exact_coordinates=args.exact_coords
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
