#!/usr/bin/env python3

import json
import csv
import re
import time
import argparse
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
    
    def scrape_multiple_pages(self, start_page: int = 1, max_pages: int = 5, 
                            properties_per_page: int = 25) -> List[Dict[str, Any]]:
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
            # Construir URL
            if page_number == 1:
                url = "https://www.zonaprop.com.ar/departamentos-venta.html"
            else:
                url = f"https://www.zonaprop.com.ar/departamentos-venta-pagina-{page_number}.html"
            
            self.logger.info(f"🌐 Navegando con CDP Mode: {url}")
            
            # Activar CDP Mode para bypass automático de Cloudflare
            sb.activate_cdp_mode(url)
            time.sleep(2)
            
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
                    full_text: posting.textContent.replace(/\\s+/g, ' ').trim()
                };
                
                // Intentar extraer elementos específicos
                var selectors = {
                    price: ['.price', '[data-qa="price"]', '.posting-price', '.card-price', '.amount'],
                    location: ['.address', '[data-qa="address"]', '.posting-address', '.card-address', '.location'],
                    surface: ['.surface', '[data-qa="surface"]', '.posting-surface', '.card-surface', '.area'],
                    rooms: ['.rooms', '[data-qa="rooms"]', '.posting-rooms', '.card-rooms', '.ambientes']
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
                    'id': f"prop_{page_number}_{i+1}",
                    'page': page_number,
                    'position': i + 1,
                    'scraping_date': datetime.now().isoformat(),
                    'extraction_method': 'cdp_javascript_regex'
                }
                
                full_text = js_prop.get('full_text', '')
                
                # Extraer precio con múltiples patrones
                price = self._extract_price(js_prop.get('price', ''), full_text)
                property_data['price'] = price
                
                # Extraer ubicación
                location = self._extract_location(js_prop.get('location', ''), full_text)
                property_data['location'] = location
                
                # Extraer superficie
                surface = self._extract_surface(js_prop.get('surface', ''), full_text)
                property_data['surface'] = surface
                
                # Extraer habitaciones
                rooms = self._extract_rooms(js_prop.get('rooms', ''), full_text)
                property_data['rooms'] = rooms
                
                # Extraer información adicional
                property_data.update(self._extract_additional_info(full_text))
                
                # Solo agregar si tiene datos válidos
                if self._is_valid_property(property_data):
                    page_properties.append(property_data)
            
            self.logger.info(f"✅ Extraídas {len(page_properties)} propiedades válidas de página {page_number}")
            return page_properties
            
        except Exception as e:
            self.logger.error(f"❌ Error extrayendo propiedades de página {page_number}: {e}")
            return []
    
    def _extract_price(self, direct_price: str, full_text: str) -> str:
        """Extrae precio usando múltiples estrategias"""
        if direct_price and direct_price.strip() and '$' in direct_price:
            return direct_price.strip()
        
        # Patrones de precio
        patterns = [
            r'\$\s*[\d,\.]+',          # $123,456
            r'USD\s*[\d,\.]+',         # USD 123,456
            r'[\d,\.]+\s*USD',         # 123,456 USD
            r'\$\s*[\d,\.]+\s*USD',    # $ 123,456 USD
        ]
        
        for pattern in patterns:
            match = re.search(pattern, full_text)
            if match:
                return match.group(0).strip()
        
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
    
    def _is_valid_property(self, property_data: Dict[str, Any]) -> bool:
        """Verifica si la propiedad tiene datos válidos"""
        required_fields = ['price', 'location', 'surface', 'rooms']
        valid_count = sum(1 for field in required_fields 
                         if property_data.get(field) and property_data.get(field) != 'N/A')
        
        # Considerar válida si tiene al menos 2 campos válidos
        return valid_count >= 2
    
    def save_data(self, filename: str = "zonaprop_full_scraping", format: str = "both") -> Dict[str, str]:
        """
        Guarda los datos extraídos
        
        Args:
            filename: Nombre base del archivo
            format: 'csv', 'json', o 'both'
            
        Returns:
            Dict con las rutas de los archivos guardados
        """
        if not self.scraped_data:
            self.logger.warning("⚠️  No hay datos para guardar")
            return {}
        
        # Crear directorio de salida
        output_dir = Path("data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        # Guardar CSV
        if format in ['csv', 'both']:
            csv_file = output_dir / f"{filename}_{timestamp}.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                if self.scraped_data:
                    # Obtener todos los campos únicos de todas las propiedades
                    all_fields = set()
                    for prop in self.scraped_data:
                        all_fields.update(prop.keys())
                    
                    writer = csv.DictWriter(f, fieldnames=sorted(all_fields))
                    writer.writeheader()
                    writer.writerows(self.scraped_data)
            saved_files['csv'] = str(csv_file)
            self.logger.info(f"💾 CSV guardado: {csv_file}")
        
        # Guardar JSON
        if format in ['json', 'both']:
            json_file = output_dir / f"{filename}_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(self.scraped_data, f, ensure_ascii=False, indent=2)
            saved_files['json'] = str(json_file)
            self.logger.info(f"💾 JSON guardado: {json_file}")
        
        # Mostrar estadísticas
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
        fields = ['price', 'location', 'surface', 'rooms']
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
        
        # Ejecutar scraping
        data = scraper.scrape_multiple_pages(
            start_page=args.start_page,
            max_pages=args.pages,
            properties_per_page=args.properties_per_page
        )
        
        # Guardar datos
        if data:
            saved_files = scraper.save_data(args.output, args.format)
            
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
