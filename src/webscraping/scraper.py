from bs4 import BeautifulSoup
import time
from random import randint
import random
import os
import csv
try:
    from src.webscraping.fetcher import fetch_page, fetch_page_with_stealth
except ModuleNotFoundError:
    print(f'Ejecuta con "python -m src.webscraping.scraper" para importar correctamente las librerías.')
    quit()
except Exception as e:
    print(f'Error encontrado al importar librerías de fetch.py: {e}')
from dotenv import load_dotenv

load_dotenv()
raw = os.getenv("DISTRICTS", "")
DISTRICTS = raw.split(",") if raw else []

def list_to_csv(data, csv_filename):
    """
    Converts a list of dictionaries into a CSV file.

    Parameters:
        data (list): A list of dictionaries to be converted into CSV.
        csv_filename (str): The name of the output CSV file.
    """
    if not data:
        print("The input data is empty.")
        return

    # Extract the fieldnames from the keys of the first dictionary
    fieldnames = data[0].keys()

    try:
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=';')

            # Write header (fieldnames)
            writer.writeheader()

            # Write the rows
            for row in data:
                # Clean up the 'localidad' field to remove unwanted newline characters
                # Remove leading/trailing whitespaces and newlines
                row['localidad'] = row['localidad'].strip()
                writer.writerow(row)

        print(f"CSV file '{csv_filename}' has been created successfully!")
    except Exception as e:
        print(f"An error occurred while writing the CSV file: {e}")


def local_content_loader(page):
    with open(f'data/webscraping/index-{page}.html', 'r', encoding='utf-8') as file:
        content = file.read()
        print("Loaded content from local file 'data/webscraping/index.html'.")
        # Create a mock response object
        response = type('obj', (object,), {'text': content})
        return response
    return None


# Function to parse the page with BeautifulSoup and find the div by class
def parse_html(response, div_class):
    soup = BeautifulSoup(response, 'html.parser')
    divs = soup.select(f'div.{div_class}')
    return divs

# Function to extract all the inner HTML from each div
def extract_content(divs):
    content_list = []
    for div in divs:
        try:
            # Extraer caracteristicas
            char_div = div.find("div", class_="item-detail-char")
            caracteristicas = char_div.find_all("span", class_="item-detail") if char_div else []

            habitaciones = caracteristicas[0].get_text().split(" ")[0] if len(caracteristicas) > 0 else None
            tamanio = caracteristicas[1].get_text().split(" ")[0] if len(caracteristicas) > 1 else None
            descripcion = caracteristicas[2].get_text() if len(caracteristicas) > 2 else None

            # Precio
            precio_tag = div.find("span", class_="item-price h2-simulated")
            precio = precio_tag.get_text().split("€")[0].strip() if precio_tag else None

            # Dirección
            link_tag = div.find("a", class_="item-link")
            direccion = None
            link = None
            if link_tag:
                link = link_tag.get("href")
                raw_text = link_tag.get_text().replace("\n", "").strip()
                direccion = "".join(raw_text.split(",")[:2])

            # Descripción larga
            desc_div = div.find("div", class_="item-description description")
            caracteristicas_extendido = desc_div.find_all("p", class_="ellipsis") if desc_div else []
            descripcion_larga = caracteristicas_extendido[0].get_text() if caracteristicas_extendido else None

            content_list.append({
                "precio": precio,
                "localidad": direccion,
                "tamanio": tamanio,
                "habitaciones": habitaciones,
                "descripcion": descripcion,
                "link": link,
                "descripcion_larga": descripcion_larga
            })

        except Exception as e:
            print(f"⚠️ Error parsing div: {e}")
            continue

    return content_list


# Function to simulate human-like behavior by adding a delay
def random_sleep(min_seconds=10, max_seconds=30):
    """
    Duerme un tiempo aleatorio más largo para evitar detección
    
    Args:
        min_seconds: Mínimo tiempo de espera
        max_seconds: Máximo tiempo de espera
    """
    sleep_time = randint(min_seconds, max_seconds) + random.random()
    print(f"Esperando {sleep_time:.1f} segundos...")
    time.sleep(sleep_time)


def progressive_sleep(attempt_count):
    """
    Implementa una espera progresiva basada en el número de intentos
    """
    base_time = 15  # tiempo base en segundos
    progressive_time = base_time * (1.5 ** attempt_count)
    max_time = 300  # máximo 5 minutos
    
    sleep_time = min(progressive_time, max_time) + random.uniform(0, 10)
    print(f"Espera progresiva: {sleep_time:.1f} segundos (intento {attempt_count})")
    time.sleep(sleep_time)


# Function to save the page content to a local file
def save_page_locallyv1(url, content):
    filename = "data/webscraping/cached_pages/" + \
        url.replace("https://", "").replace("http://",
                                            "").replace("/", "_") + ".html"
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)
    print(f"Page saved locally as {filename}")


def save_page_locally(identifier, content):
    """
    Guarda el contenido HTML en la carpeta data/webscraping/cached_pages
    usando un nombre de fichero derivado del identificador.

    identifier: puede ser una URL o un número de página.
    content:    string con el HTML a guardar.
    """
    # Asegurar que existe la carpeta
    os.makedirs("data/webscraping/cached_pages", exist_ok=True)

    # Si el identificador parece una URL, limpiar
    if isinstance(identifier, str) and identifier.startswith(("http://", "https://")):
        filename = identifier.replace("https://", "").replace("http://", "").replace("/", "_")
    else:
        # En caso contrario usarlo tal cual (ej. número de página)
        filename = str(identifier)

    filepath = os.path.join("data/webscraping/cached_pages", f"{filename}.html")

    with open(filepath, "w", encoding="utf-8") as file:
        file.write(content)

    print(f"Page saved locally as {filepath}")

# Function to load the page content from a local file
def load_page_from_file(url):
    filename = url.replace("https://", "").replace("http://",
                                                   "").replace("/", "_") + ".html"
    filepath = os.path.join("data/webscraping/cached_pages", filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        return None


def check_for_captcha_or_block(content):
    """
    Verifica si el contenido indica un captcha o bloqueo
    """
    if not content:
        return True
    
    content_lower = content.lower()
    block_indicators = [
        'too many requests',
        'uso indebido'
    ]
    
    return any(indicator in content_lower for indicator in block_indicators)

se_usa_cache = False
def get_div_content(url, div_class, use_stealth=False, max_retries=2):
    """
    Obtiene el contenido de una página con manejo mejorado de errores
    
    Args:
        url: URL a obtener
        div_class: Clase CSS del div a buscar
        use_stealth: Si usar el modo stealth
        max_retries: Máximo número de reintentos
    """
    global se_usa_cache
    for attempt in range(max_retries):
        try:
            print(f"\n🔄 Intento {attempt + 1} de {max_retries} para: {url}")
            
            # Intentar cargar desde cache primero
            cached_content = load_page_from_file(url)
            if cached_content and not check_for_captcha_or_block(cached_content):
                se_usa_cache = True
                print("📁 Usando contenido desde cache")
                divs = parse_html(cached_content, div_class)
                content = extract_content(divs)
                if content and len(content) > 0:
                    print(f"✅ Cache válido con {len(content)} elementos")
                    return content
            else:
                se_usa_cache = False

            # Si no hay cache válido, obtener de internet
            response = None
            
            if use_stealth:
                if use_fast:
                    print("🥷⏩ Usando modo stealth y rápido...")
                else:
                    print("🥷 Usando modo stealth...")
                response = fetch_page_with_stealth(url)
            elif use_fast:
                print("⏩ Usando modo rápido...")
                response = fetch_page(url, retry_count=1)  # Un solo intento interno
            else:
                print("🌐 Usando modo estándar...")
                response = fetch_page(url, retry_count=1)  # Un solo intento interno
            
            if not response:
                print(f"❌ No se pudo obtener respuesta (intento {attempt + 1})")
                if attempt < max_retries - 1:
                    progressive_sleep(attempt)
                continue
            
            # Verificar si hay captcha/bloqueo adicional
            if check_for_captcha_or_block(response):
                print(f"🚫 Captcha o bloqueo detectado (intento {attempt + 1})")
                if attempt < max_retries - 1:
                    progressive_sleep(attempt)
                    use_stealth = True  # Cambiar a stealth en siguiente intento
                continue
            
            # Guardar página para cache
            try:
                save_page_locally(url, response)
            except Exception as e:
                print(f"⚠️  No se pudo guardar cache: {e}")
            
            # Parsear contenido
            divs = parse_html(response, div_class)
            content = extract_content(divs)
            
            if content and len(content) > 0:
                print(f"✅ Extraídos {len(content)} elementos exitosamente")
                return content
            else:
                print("⚠️  No se encontraron elementos válidos en la página")
                # No hacer retry si la página carga pero no tiene contenido
                # (probablemente sea la última página)
                return []
                    
        except Exception as e:
            print(f"💥 Error en intento {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                progressive_sleep(attempt)
    
    print(f"❌ No se pudo obtener contenido válido después de {max_retries} intentos")
    return None


def get_div_content_local(id, div_class):
    # Try fetching the page, or load from local file if available
    response = local_content_loader(id)
    if not response:  # If the request fails, try loading from local file
        print("Fetching failed. Trying to load from local file...")
        page_content = load_page_from_file(id)
        if page_content:
            print("Loaded page from local file.")
            # Create a mock response object
            response = type('obj', (object,), {'text': page_content})
        else:
            print("Page not found locally either.")
            return None

    # Save the fetched page locally for future use
    save_page_locally(url, response.text)

    # Parse the HTML and extract the div content
    divs = parse_html(response, div_class)
    content = extract_content(divs)  # Extract the content from the div
    return content


def get_url_for_district_page(district, page, venta_o_alquiler):
   return f"https://www.idealista.com/{venta_o_alquiler}/madrid/{district}/pagina-{page}.htm"


if __name__ == "__main__":

    lista_final = []
    while True:
        try:
            inicio = int(input('Introduce página de inicio: '))
            final = int(input('Introduce página final: '))
            venta_o_alquiler = int(input('¿Venta (1) o alquiler (2)?: '))
        except ValueError:
            print('Por favor, introduce un número entero.')
            continue
        break
    
    use_stealth = input('¿Usar modo stealth? (s/n): ').lower().startswith('s')
    use_fast = input('¿Usar modo rápido? (s/n): ').lower().startswith('s')
    if use_fast:
        use_fast = input('¡Atención! El modo rápido aumenta el riesgo de bloqueo de IP. ¿Continuar? (s/n): ').lower().startswith('s')
    
    if venta_o_alquiler == 1:
        venta_o_alquiler = 'venta-viviendas'
        print('--> Obteniendo datos de VENTA.')
    else:
        venta_o_alquiler = 'alquiler-viviendas'
        print('--> Obteniendo datos de ALQUILER.')

    total_errors = 0
    max_consecutive_errors = 5  # Parar si hay muchos errores consecutivos
    consecutive_errors = 0

    for district in DISTRICTS:
        print(f"\n=== Procesando distrito: {district} ===")
        district_data = []
        
        for page in range(inicio, final + 1):
            url = get_url_for_district_page(district, page, venta_o_alquiler)
            print(f"\nScraping página {page}: {url}")
            
            content = get_div_content(url, "item-info-container ", use_stealth=use_stealth)
            
            if not content:
                consecutive_errors += 1
                total_errors += 1
                print(f"Error obteniendo contenido. Errores consecutivos: {consecutive_errors}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print(f"Demasiados errores consecutivos ({consecutive_errors}). Cambiando al siguiente distrito.")
                    break
                    
                # Espera extra larga después de un error
                progressive_sleep(consecutive_errors)
                continue
            
            # Reset consecutive errors counter on success
            consecutive_errors = 0
            
            if len(content) == 0:
                print("No hay más páginas en este distrito")
                break  # ya no hay más páginas en este distrito

            for row in content:
                row["distrito"] = district
                district_data.append(row)
                lista_final.append(row)
            
            print(f"Extraídos {len(content)} elementos de página {page}")
            
            # Espera aleatoria entre páginas (si no es la última)
            if page != final and not se_usa_cache:
                if use_fast:
                    random_sleep(1, 5)
                else:
                    random_sleep(15, 45)  # Espera más larga entre páginas
        
        # Guardar datos del distrito
        if district_data:
            list_to_csv(district_data, f'data/webscraping/csv/distrito_{district}_{venta_o_alquiler}_pags_{inicio}-{final}.csv')
            print(f"Guardados {len(district_data)} elementos para distrito {district}")
        
        # Espera extra entre distritos
        if district != DISTRICTS[-1] and not se_usa_cache:  # Si no es el último distrito
            print("Esperando antes del siguiente distrito...")
            if use_fast:
                time.sleep(random.uniform(5, 10))
            else:
                time.sleep(random.uniform(60, 120)) # 1-2 minutos entre distritos

    # Guardar archivo final
    if lista_final:
        list_to_csv(lista_final, f'data/raw/properties_all_{venta_o_alquiler}_pags_{inicio}-{final}.csv')
        print(f"\n=== COMPLETADO ===")
        print(f"Total de propiedades extraídas: {len(lista_final)}")
        print(f"Total de errores: {total_errors}")
        print(f"Guardando csv final en data/raw/properties_all_{venta_o_alquiler}_pags_{inicio}-{final}.csv")
    else:
        print("No se pudo extraer ningún dato")