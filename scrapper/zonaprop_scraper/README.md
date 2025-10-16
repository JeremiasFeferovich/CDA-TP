# ZonaProp Scraper

Scraper automatizado para extraer datos de propiedades inmobiliarias de ZonaProp.com.ar, desarrollado para el proyecto de **Predicción de Precios de Viviendas en Buenos Aires**.

## 📋 Descripción

Este proyecto forma parte del trabajo de **Ciencia de Datos Aplicada** para desarrollar un modelo de predicción de precios de viviendas. El scraper extrae información detallada de departamentos en venta en Buenos Aires desde ZonaProp.

## 🎯 Objetivos

- Recopilar datos de al menos 5,000 propiedades en Buenos Aires
- Extraer características relevantes para predicción de precios
- Generar dataset limpio y estructurado para análisis posterior
- Cumplir con buenas prácticas de web scraping ético

## 🏗️ Arquitectura del Proyecto

```
zonaprop_scraper/
├── src/
│   ├── __init__.py          # Inicialización del paquete
│   ├── scraper.py           # Clase principal del scraper
│   ├── data_extractor.py    # Extractor de datos de propiedades
│   ├── utils.py            # Utilidades y funciones helper
│   └── config.py           # Configuraciones del sistema
├── data/
│   ├── raw/                # Datos crudos extraídos
│   ├── processed/          # Datos procesados y limpios
│   └── logs/               # Logs del proceso de scraping
├── example_usage.py        # Ejemplo de uso del scraper
├── requirements.txt        # Dependencias del proyecto
└── README.md              # Esta documentación
```

## 📊 Datos Extraídos

### Información Básica
- **ID de propiedad**: Identificador único
- **Precio**: Valor en USD o ARS
- **Expensas**: Gastos mensuales adicionales

### Ubicación
- **Dirección**: Dirección específica
- **Barrio**: Barrio de Buenos Aires
- **Zona**: Capital Federal, GBA Norte, etc.

### Características Físicas
- **Superficie total**: Metros cuadrados
- **Ambientes**: Número de ambientes
- **Dormitorios**: Número de dormitorios
- **Baños**: Número de baños
- **Cocheras**: Espacios de estacionamiento

### Características Adicionales
- **Amenities**: Pileta, gimnasio, SUM, etc.
- **Estado**: Nuevo, usado, en construcción
- **Orientación**: Norte, sur, este, oeste
- **Piso**: Número de piso
- **Inmobiliaria**: Empresa responsable

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd zonaprop_scraper
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# o
venv\\Scripts\\activate   # En Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Verificar instalación de Chrome
El scraper requiere Google Chrome instalado en el sistema. ChromeDriver se descarga automáticamente.

## 📖 Uso

### Uso Básico

```python
from src.scraper import ZonaPropScraper

# Crear instancia del scraper
scraper = ZonaPropScraper(
    headless=True,    # Ejecutar sin mostrar navegador
    max_pages=10      # Máximo 10 páginas
)

# Ejecutar scraping
properties = scraper.scrape(start_page=1)

# Guardar datos
scraper.save_data(filename="propiedades_ba", format="csv")

# Obtener estadísticas
stats = scraper.get_statistics()
print(f"Propiedades extraídas: {stats['total_properties']}")

# Cerrar scraper
scraper.close()
```

### Ejemplo Completo

```bash
# Ejecutar ejemplo incluido
python example_usage.py
```

### Línea de Comandos

```bash
# Scraping básico
python -m src.scraper --pages 5 --headless

# Scraping avanzado
python -m src.scraper \\
    --pages 20 \\
    --start-page 1 \\
    --output "propiedades_caba" \\
    --format csv \\
    --headless
```

#### Parámetros disponibles:
- `--pages`: Número máximo de páginas (default: 10)
- `--start-page`: Página inicial (default: 1)
- `--headless`: Ejecutar sin mostrar navegador
- `--output`: Nombre del archivo de salida
- `--format`: Formato de salida (csv/json)

## ⚙️ Configuración

### Modificar configuraciones en `src/config.py`:

```python
# Delays entre requests
SCRAPER_CONFIG = {
    "delay_between_requests": (2, 5),  # 2-5 segundos
    "delay_between_pages": (5, 10),   # 5-10 segundos
    "max_retries": 3,
    "max_pages": 100,
    "max_properties_per_session": 1000
}

# Configuración de Selenium
SELENIUM_CONFIG = {
    "headless": True,
    "window_size": (1920, 1080),
    "implicit_wait": 10,
    "page_load_timeout": 30
}
```

## 🔍 Manejo de Errores

El scraper incluye manejo robusto de errores:

- **Timeouts**: Reintentos automáticos
- **Elementos faltantes**: Valores por defecto
- **Cloudflare**: Configuración anti-detección
- **Logging**: Registro detallado de errores

### Logs

Los logs se guardan automáticamente en `data/logs/` con:
- Timestamp de cada operación
- Errores y excepciones
- Estadísticas de progreso
- Información de debug

## 📈 Salida de Datos

### Formato CSV
```csv
property_id,price,currency,expenses,address,neighborhood,zone,total_surface,rooms,bathrooms,parking_spaces,amenities,property_status,scraping_date
56054832,265000,USD,180000,"Condarco al 3000","Villa del Parque","Capital Federal",95,4,2,1,"['Balcón']","Usado","2025-08-30T14:30:00"
```

### Formato JSON
```json
{
  "property_id": "56054832",
  "price": 265000,
  "currency": "USD",
  "expenses": 180000,
  "address": "Condarco al 3000",
  "neighborhood": "Villa del Parque",
  "zone": "Capital Federal",
  "total_surface": 95,
  "rooms": 4,
  "bathrooms": 2,
  "parking_spaces": 1,
  "amenities": ["Balcón"],
  "property_status": "Usado",
  "scraping_date": "2025-08-30T14:30:00"
}
```

## 🛡️ Consideraciones Éticas

### Buenas Prácticas Implementadas:
- **Rate Limiting**: Delays entre requests para no sobrecargar el servidor
- **User-Agent Rotation**: Rotación de navegadores para evitar detección
- **Respeto por robots.txt**: Verificación de políticas del sitio
- **Uso Académico**: Datos utilizados solo para investigación

### Límites Implementados:
- Máximo 1000 propiedades por sesión
- Delays de 2-5 segundos entre requests
- Delays de 5-10 segundos entre páginas
- Máximo 3 reintentos por página

## 🔧 Troubleshooting

### Problemas Comunes:

#### 1. Error de ChromeDriver
```bash
# Solución: Actualizar webdriver-manager
pip install --upgrade webdriver-manager
```

#### 2. Timeout en páginas
```python
# Aumentar timeout en config.py
SELENIUM_CONFIG["page_load_timeout"] = 60
```

#### 3. Cloudflare bloqueando requests
```python
# Usar modo no-headless y aumentar delays
scraper = ZonaPropScraper(headless=False)
SCRAPER_CONFIG["delay_between_requests"] = (5, 10)
```

#### 4. Elementos no encontrados
- Verificar que los selectores CSS en `config.py` estén actualizados
- ZonaProp puede cambiar su estructura HTML

## 📊 Estadísticas Esperadas

Para un scraping exitoso de 10 páginas (~200 propiedades):

- **Tiempo estimado**: 15-30 minutos
- **Datos válidos**: >95% de completitud
- **Errores esperados**: <5%
- **Barrios cubiertos**: 15-25 barrios de CABA

## 🤝 Contribución

### Para contribuir al proyecto:

1. Fork del repositorio
2. Crear branch para feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit de cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push al branch (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📝 Licencia

Este proyecto es para uso académico en el marco del curso de Ciencia de Datos Aplicada - ITBA.

## 👥 Autores

- **Ian Bernasconi** - Desarrollo y análisis
- **Jeremías Feferovich** - Desarrollo y análisis

## 📞 Contacto

Para consultas sobre el proyecto:
- Email: [contacto@proyecto.com]
- GitHub Issues: [link-to-issues]

## 🔄 Próximas Mejoras

- [ ] Scraping de casas y PHs
- [ ] Integración con base de datos
- [ ] API REST para acceso a datos
- [ ] Dashboard de monitoreo
- [ ] Scraping incremental
- [ ] Detección automática de cambios en estructura HTML

---

**Nota**: Este scraper está diseñado específicamente para ZonaProp.com.ar y puede requerir actualizaciones si el sitio cambia su estructura HTML.
