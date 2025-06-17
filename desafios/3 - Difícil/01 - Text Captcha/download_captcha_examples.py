import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin

# URL da página com os exemplos de CAPTCHA
EXAMPLES_URL = 'https://captcha.com/captcha-examples.html'
# Pasta para salvar as imagens baixadas
SAVE_DIR = 'captcha_example_images'
# User-Agent para simular um navegador
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def download_captcha_images():
    """Baixa imagens de CAPTCHA da página de exemplos."""
    print(f"Acessando {EXAMPLES_URL}...")
    try:
        response = requests.get(EXAMPLES_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()  # Levanta um erro para códigos HTTP 4xx/5xx
    except requests.exceptions.RequestException as e:
        print(f"Erro ao acessar a URL: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')

    # Cria o diretório para salvar as imagens, se não existir
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Diretório '{SAVE_DIR}' criado.")

    # Encontrar as imagens de CAPTCHA
    # O seletor foi ajustado com base no HTML fornecido pelo usuário.
    # Buscamos por imagens com a classe 'captcha_sample' dentro das divs corretas.
    captcha_images = soup.select('div.captcha_images_left img.captcha_sample, div.captcha_images_right img.captcha_sample')

    if not captcha_images:
        print("Ainda não foram encontradas imagens de CAPTCHA. O script pode precisar de atualização para o HTML do site.")
        return

    print(f"Encontradas {len(captcha_images)} imagens de CAPTCHA para baixar.")
    downloaded_count = 0

    for i, img_tag in enumerate(captcha_images):
        img_src = img_tag.get('src')
        if not img_src:
            continue

        # Constrói a URL completa da imagem
        img_url = urljoin(EXAMPLES_URL, img_src)
        
        print(f"Baixando imagem {i+1}/{len(captcha_images)}: {img_url}")
        try:
            img_response = requests.get(img_url, headers=HEADERS, timeout=10)
            img_response.raise_for_status()

            # Gera um nome de arquivo único
            # Pega a última parte da URL da imagem como nome base
            base_filename = os.path.basename(img_url)
            filename = os.path.join(SAVE_DIR, f"example_{i+1}_{base_filename}")
            
            # Garante que a extensão seja .png se não houver uma clara
            if '.' not in os.path.basename(filename):
                filename += '.png'

            with open(filename, 'wb') as f:
                f.write(img_response.content)
            print(f"Imagem salva como: {filename}")
            downloaded_count += 1

            # Pequena pausa para ser gentil com o servidor
            time.sleep(0.5) 

        except requests.exceptions.RequestException as e:
            print(f"Erro ao baixar {img_url}: {e}")
        except IOError as e:
            print(f"Erro ao salvar a imagem {filename}: {e}")

    print(f"\nDownload concluído. {downloaded_count} de {len(captcha_images)} imagens baixadas para '{SAVE_DIR}'.")

if __name__ == '__main__':
    download_captcha_images()
