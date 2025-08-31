#!/usr/bin/env python3
"""
Script de configuración e instalación para ZonaProp Scraper
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Verifica que la versión de Python sea compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detectado")
    return True

def check_chrome_installation():
    """Verifica que Google Chrome esté instalado"""
    chrome_paths = [
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser", 
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
    ]
    
    for path in chrome_paths:
        if os.path.exists(path):
            print("✅ Google Chrome encontrado")
            return True
    
    print("⚠️  Google Chrome no encontrado en ubicaciones estándar")
    print("   Por favor, instala Google Chrome antes de continuar")
    print("   Descarga desde: https://www.google.com/chrome/")
    return False

def install_dependencies():
    """Instala las dependencias del proyecto"""
    print("\n📦 Instalando dependencias...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ Error: archivo requirements.txt no encontrado")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def create_directories():
    """Crea los directorios necesarios para el proyecto"""
    print("\n📁 Creando estructura de directorios...")
    
    base_path = Path(__file__).parent
    directories = [
        base_path / "data" / "raw",
        base_path / "data" / "processed", 
        base_path / "data" / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    return True

def test_scraper():
    """Ejecuta una prueba básica del scraper"""
    print("\n🧪 Ejecutando prueba básica...")
    
    try:
        # Agregar src al path
        src_path = Path(__file__).parent / "src"
        sys.path.insert(0, str(src_path))
        
        from scraper import ZonaPropScraper
        
        # Crear scraper de prueba
        scraper = ZonaPropScraper(headless=True, max_pages=1)
        
        # Intentar configurar el driver
        scraper._setup_driver()
        
        print("✅ Configuración del WebDriver exitosa")
        
        # Cerrar scraper
        scraper.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

def main():
    """Función principal de configuración"""
    print("🚀 ZonaProp Scraper - Script de Configuración")
    print("=" * 50)
    
    # Verificar Python
    if not check_python_version():
        return False
    
    # Verificar Chrome
    if not check_chrome_installation():
        return False
    
    # Instalar dependencias
    if not install_dependencies():
        return False
    
    # Crear directorios
    if not create_directories():
        return False
    
    # Ejecutar prueba
    if not test_scraper():
        print("\n⚠️  La prueba falló, pero la instalación básica está completa")
        print("   Puedes intentar ejecutar el scraper manualmente")
    
    print("\n🎉 ¡Configuración completada exitosamente!")
    print("\n📖 Próximos pasos:")
    print("   1. Ejecuta el ejemplo: python example_usage.py")
    print("   2. O usa el scraper desde línea de comandos:")
    print("      python -m src.scraper --pages 5 --headless")
    print("   3. Lee la documentación completa en README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
