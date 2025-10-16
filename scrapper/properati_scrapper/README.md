# Properati Scraper

Scraper completo para extraer propiedades inmobiliarias de Properati.com.ar usando SeleniumBase con CDP Mode para evadir detección.

## Características

- ✅ **CDP Mode**: Bypass automático de sistemas de detección
- ✅ **Extracción completa**: Precio, ubicación, características, amenities
- ✅ **Multi-página**: Scraping de múltiples páginas con manejo de errores
- ✅ **Datos estructurados**: Exportación a CSV y JSON
- ✅ **Logging detallado**: Seguimiento completo del proceso
- ✅ **Modo resume**: Continuar scraping desde donde se dejó
- ✅ **Guardado incremental**: Guardar progreso cada N páginas

## Instalación

### Prerrequisitos

```bash
pip install seleniumbase selenium requests beautifulsoup4 pandas
```

### Instalación del proyecto

```bash
cd properati_scrapper
pip install -e .
```

## Uso

### Uso básico

```bash
cd src
python properati_full_scraper.py --pages 5
```

### Opciones avanzadas

```bash
# Scraper 10 páginas empezando desde la página 2
python properati_full_scraper.py --pages 10 --start-page 2

# Modo headless (sin interfaz gráfica)
python properati_full_scraper.py --pages 5 --headless

# Guardar solo en formato JSON
python properati_full_scraper.py --pages 5 --format json

# Guardar progreso cada 2 páginas
python properati_full_scraper.py --pages 10 --save-every 2

# Continuar scraping desde archivo existente
python properati_full_scraper.py --pages 5 --resume properati_full_20231224_120000

# Cambiar delay entre páginas (segundos)
python properati_full_scraper.py --pages 5 --delay 5.0
```

### Parámetros completos

- `--pages`: Número de páginas a scrapear (default: 5)
- `--start-page`: Página inicial (default: 1)
- `--output`: Nombre base del archivo de salida (default: "properati_full")
- `--format`: Formato de salida: "csv", "json", o "both" (default: "both")
- `--headless`: Ejecutar en modo headless
- `--no-incognito`: Desactivar modo incógnito
- `--delay`: Delay entre páginas en segundos (default: 3.0)
- `--properties-per-page`: Propiedades esperadas por página (default: 30)
- `--resume`: Continuar desde archivo existente
- `--save-every`: Guardar progreso cada N páginas (default: 0)

## Estructura de datos

### Campos extraídos

```json
{
  "property_id": "01985f19-d980-7657-a8f9-8d3aca4ccecb",
  "title": "Departamento en Venta en Belgrano",
  "price": 195000,
  "currency": "USD",
  "location": "Belgrano, Capital Federal",
  "neighborhood": "Belgrano",
  "area": 62,
  "bedrooms": 2,
  "bathrooms": 1,
  "amenities": ["Balcón", "Ascensor"],
  "property_type": "Departamento",
  "description": "Hermoso departamento...",
  "publisher": "Inmobiliaria XYZ",
  "detail_url": "https://www.properati.com.ar/detalle/...",
  "page_number": 1,
  "scraping_date": "2023-12-24T12:00:00"
}
```

## Archivos de salida

Los archivos se guardan en `data/raw/` con timestamp:

- `properati_full_20231224_120000.csv`
- `properati_full_20231224_120000.json`

## Logging

Los logs se guardan automáticamente en:
- `properati_scraper_20231224_120000.log`

## Características técnicas

### CDP Mode (Chrome DevTools Protocol)
- Bypass automático de sistemas anti-bot
- Navegación nativa sin detección
- Manejo automático de Cloudflare y similares

### Extracción de datos
- **JavaScript + DOM**: Extracción directa desde el DOM
- **Selectores específicos**: Adaptados a la estructura de Properati
- **Validación**: Verificación de datos antes de guardar
- **Limpieza**: Procesamiento automático de precios, números, etc.

### Manejo de errores
- Reintentos automáticos
- Logging detallado de errores
- Continuación en caso de fallos parciales
- Guardado de progreso

## Ejemplo de uso completo

```bash
# Scraping completo con todas las características
python properati_full_scraper.py \
  --pages 20 \
  --start-page 1 \
  --output "properati_capital_federal" \
  --format both \
  --delay 4.0 \
  --save-every 5 \
  --properties-per-page 30
```

## Estructura del proyecto

```
properati_scrapper/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuración y selectores
│   ├── utils.py               # Utilidades y funciones helper
│   ├── data_extractor.py      # Extractor de datos (no usado en versión actual)
│   └── properati_full_scraper.py  # Scraper principal
├── data/
│   └── raw/                   # Datos extraídos
├── setup.py
└── README.md
```

## Notas importantes

1. **Respeto por el sitio**: El scraper incluye delays para no sobrecargar el servidor
2. **Detección**: Usa CDP Mode para evitar sistemas anti-bot
3. **Datos reales**: Extrae datos directamente del DOM, no de APIs
4. **Capital Federal**: Configurado específicamente para propiedades en CABA
5. **Formato de URL**: Usa el formato `/s/capital-federal/venta/{page}`

## Solución de problemas

### Error de navegación
- Verificar conexión a internet
- Aumentar delay entre páginas
- Verificar que la URL esté accesible

### Propiedades no encontradas
- Verificar selectores CSS en `config.py`
- Comprobar cambios en la estructura del sitio
- Revisar logs para errores específicos

### Archivos no guardados
- Verificar permisos de escritura en directorio `data/raw/`
- Comprobar espacio en disco
- Revisar logs de errores

## Desarrollo

Para contribuir al proyecto:

1. Fork del repositorio
2. Crear branch para feature
3. Realizar cambios
4. Agregar tests si es necesario
5. Crear Pull Request

## Licencia

MIT License - Ver archivo LICENSE para detalles.
