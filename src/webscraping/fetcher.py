# fetcher.py
import undetected_chromedriver as uc
import random
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Lista de user-agents para rotar
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0"
]

def handle_cookies(driver, accept=True):
    try:
        # espera a que aparezca el contenedor de cookies
        wait = WebDriverWait(driver, 5)
        if accept:
            button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button#didomi-notice-agree-button"))
            )
        else:
            button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button#didomi-notice-disagree-button"))
            )
        button.click()
        print("Cookies gestionadas correctamente")
        time.sleep(1)  # pequeña pausa para que desaparezca el modal
    except Exception as e:
        print(f"No apareció banner de cookies o fallo al gestionarlo: {e}")


def get_random_user_agent():
    """Selecciona un user-agent aleatorio de la lista"""
    return random.choice(USER_AGENTS)

def fetch_page_simple(url):
    driver = None
    try:
        print(f"Obteniendo página: {url}")
        driver = uc.Chrome()

        # User-Agent aleatorio
        user_agent = get_random_user_agent()
        print(f"Usando User-Agent: {user_agent[:50]}...")
        driver.execute_script(f"""
            Object.defineProperty(navigator, 'userAgent', {{
                get: function() {{ return '{user_agent}'; }}
            }});
        """)

        driver.get(url)

        # --- aceptar/rechazar cookies ---
        try:
            WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button#didomi-notice-agree-button"))
            ).click()
            print("Cookies gestionadas correctamente")
            time.sleep(1)
        except:
            print("No apareció banner de cookies")

        # --- esperar a que cargue contenido principal ---
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.item-info-container"))
            )
        except:
            print("⚠️ Timeout esperando resultados, puede ser bloqueo")

        # --- scroll ligero ---
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/4);")
        time.sleep(1)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)

        page_source = driver.page_source
        print(f"Página obtenida, tamaño: {len(page_source)} caracteres")

        # --- comprobar bloqueo ---
        block_indicators = ['too many requests', 'uso indebido']
        if any(ind in page_source.lower() for ind in block_indicators):
            print("⚠️ Posible bloqueo detectado. Guardando debug.html")
            with open("debug.html", "w", encoding="utf-8") as f:
                f.write(page_source)
            return None

        return page_source

    except Exception as e:
        print(f"Error obteniendo página: {str(e)}")
        return None
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

def fetch_page(url, retry_count=2):
    """
    Método principal con reintentos
    """
    for attempt in range(retry_count):
        response = fetch_page_simple(url)

        if response:
            response_lower = response.lower()
            block_indicators = ['too many requests', 'uso indebido']
            
            # solo bloqueo si no hay listados de pisos
            if any(ind in response_lower for ind in block_indicators) and "item-info-container" not in response_lower:
                print("⚠️ Bloqueo real detectado")
                if attempt < retry_count - 1:
                    wait_time = random.uniform(30, 60)
                    print(f"Esperando {wait_time:.1f} segundos antes del siguiente intento...")
                    time.sleep(wait_time)
                continue  # siguiente intento

            print("✅ Página obtenida exitosamente")
            return response

        else:
            print("❌ No se pudo obtener la página")
            if attempt < retry_count - 1:
                wait_time = random.uniform(15, 30)
                print(f"Esperando {wait_time:.1f} segundos antes del siguiente intento...")
                time.sleep(wait_time)
    
    print(f"❌ Falló después de {retry_count} intentos")
    return None

def fetch_page_with_stealth(url):
    """
    Método stealth simplificado
    """
    driver = None
    try:
        print(f"🥷 Modo stealth para: {url}")
        
        # Crear driver básico
        driver = uc.Chrome()
        
        user_agent = get_random_user_agent()
        
        # Scripts anti-detección más agresivos
        stealth_script = f"""
            // Eliminar propiedades de webdriver
            Object.defineProperty(navigator, 'webdriver', {{
                get: () => undefined
            }});
            
            // Cambiar user agent
            Object.defineProperty(navigator, 'userAgent', {{
                get: () => '{user_agent}'
            }});
            
            // Simular plugins
            Object.defineProperty(navigator, 'plugins', {{
                get: () => [
                    {{ name: 'Chrome PDF Plugin', length: 1 }},
                    {{ name: 'Chrome PDF Viewer', length: 1 }},
                    {{ name: 'Native Client', length: 1 }}
                ]
            }});
            
            // Configurar idiomas
            Object.defineProperty(navigator, 'languages', {{
                get: () => ['es-ES', 'es', 'en-US', 'en']
            }});
            
            // Simular chrome object
            window.chrome = {{
                runtime: {{ onConnect: null, onMessage: null }}
            }};
            
            // Eliminar automation flags
            delete navigator.__proto__.webdriver;
        """
        
        driver.execute_script(stealth_script)
        
        # Configurar ventana de forma aleatoria
        resolutions = [(1920, 1080), (1366, 768), (1440, 900)]
        width, height = random.choice(resolutions)
        driver.set_window_size(width, height)
        
        # Navegar
        driver.get(url)
        handle_cookies(driver, accept=False)
        
        # Comportamiento humano más elaborado
        time.sleep(random.uniform(3, 5))
        
        # Múltiples scrolls pequeños
        for _ in range(3):
            scroll_y = random.randint(200, 800)
            driver.execute_script(f"window.scrollTo(0, {scroll_y});")
            time.sleep(random.uniform(0.5, 1.5))
        
        # Volver arriba
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(random.uniform(1, 2))
        
        return driver.page_source
        
    except Exception as e:
        print(f"Error en modo stealth: {str(e)}")
        return None
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass