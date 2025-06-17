import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.config import Driver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import requests
from PIL import Image, ImageEnhance
import pytesseract
import easyocr
import tensorflow as tf, string
from io import BytesIO
import re
import cv2
import numpy as np
import base64
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = '/home/guilherme/Documentos/GitHub/automate/desafios/3 - Difícil/01 - Text Captcha/final_model.h5'

class CaptchaSolver:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.CHARS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'
        self.char_to_index = {c: i for i, c in enumerate(self.CHARS)}
        self.index_to_char = {i: c for i, c in enumerate(self.CHARS)}
    
    def preprocess_image(self, img_array):
        """Pré-processa a imagem igual ao treinamento"""
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (120, 40))
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        return np.array([img])
    
    def solve(self, img_array):
        """Resolve CAPTCHA a partir de array de imagem"""
        processed_img = self.preprocess_image(img_array)
        preds = self.model.predict(processed_img, verbose=0)
        
        # Decodificar cada caractere
        captcha_text = ''
        for i in range(10):  # Supondo CAPTCHAs de 10 caracteres
            char_probs = preds[i][0]
            captcha_text += self.index_to_char[np.argmax(char_probs)]
        return captcha_text

class Desafio01:
    def __init__(self):
        self.session = requests.Session()
        self.solver = CaptchaSolver(MODEL_PATH)
        self.driver = Driver().driver
    
    def processar_imagem(self, img_base64):
        """Processa imagem base64 e retorna texto"""
        try:
            img_data = base64.b64decode(img_base64)
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return self.solver.solve(img)
        except Exception as e:
            logger.error(f"Erro ao processar imagem: {e}")
            return None
    
    def iniciar(self):
        try:
            print('Iniciando navegador...')
            self.driver.get('https://captcha.com/demos/features/captcha-demo.aspx')
            time.sleep(2)
            self.driver.maximize_window()
            acerto = 0
            tentativas = 0
            tentativas_ocr = 0
            time.sleep(2)
            while True:
                print('Procurando imagem do captcha...')
                try:
                    img = WebDriverWait(self.driver, 20).until(EC.presence_of_element_located((By.ID, 'demoCaptcha_CaptchaImage')))
                except Exception as e:
                    print('Não achou imagem do captcha:', e)
                    if "invalid session id" in str(e):
                        print('Sessão do navegador fechada, reiniciando...')
                        self.driver.quit()
                        self.driver = Driver().driver
                        self.driver.get('https://captcha.com/demos/features/captcha-demo.aspx')
                        self.driver.maximize_window()
                        continue
                    print('HTML da página para debug:')
                    print(self.driver.page_source)
                    raise
                src = img.get_attribute('src')
                if src.startswith('/'):
                    src = 'https://captcha.com' + src
                print('Captcha encontrado:', src)
                print('Baixando captcha:', src)

                # garante que content_type exista para base64 também
                content_type = ''
                try:
                    if src.startswith('data:image'):
                        print("Imagem base64 detectada")
                        _, b64 = src.split(',', 1)
                        b64_clean = re.sub(r'[^A-Za-z0-9+/=]', '', b64)
                        while len(b64_clean) % 4: b64_clean += '='
                        with open('captcha_b64.txt', 'w') as f: f.write(b64_clean)
                        try:
                            image_data = base64.b64decode(b64_clean)
                            # Salva a imagem para debug
                            with open('debug.png', 'wb') as f_debug:
                                f_debug.write(image_data)
                            # Tenta abrir com Pillow
                            try:
                                image = Image.open(BytesIO(image_data))
                            except:
                                # Se falhar, tenta com OpenCV
                                nparr = np.frombuffer(image_data, np.uint8)
                                img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                if img_cv is not None:
                                    # Converte de BGR (OpenCV) para RGB (Pillow)
                                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                                    image = Image.fromarray(img_cv)
                                else:
                                    raise Exception("OpenCV não conseguiu decodificar")
                        except Exception as e:
                            print(f"Erro ao abrir imagem base64: {e}")
                            print(f"Comprimento do base64: {len(b64_clean)}")
                            continue
                    else:
                        r = requests.get(src)
                        content_type = r.headers.get('Content-Type', '')
                        if not content_type.startswith('image/'):
                            print(f"Unexpected content type: {content_type}, recarregando...")
                            continue
                        try:
                            image = Image.open(BytesIO(r.content))
                        except Exception as e:
                            print(f"Erro ao abrir imagem: {e}, recarregando...")
                            continue

                                        # salva debug opcional
                    # image.save(f'debug_{int(time.time()*1000)}.png')

                    # resize 3x para facilitar OCR
                    img_rgb = image.resize((image.width * 3, image.height * 3), Image.BILINEAR)

                    # Detecta formato base64 para decidir whitelist
                    is_jpeg = src.startswith('data:image/jpeg') or content_type.endswith('jpeg')
                    if is_jpeg:
                        allow = '0123456789'
                    else:
                        allow = self.solver.CHARS

                    texto = self.processar_imagem(b64_clean)

                    # fallback binarizado + tesseract para JPEG numérico
                    if (len(texto) < 5 and is_jpeg):
                        gray = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2GRAY)
                        _, thr = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
                        texto = pytesseract.image_to_string(thr, config='--psm 6 -c tessedit_char_whitelist=0123456789').strip()
                        texto = ''.join(filter(str.isalnum, texto.upper()))

                    # fallback geral
                    if not 4 <= len(texto) <= 7:
                        gray = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2GRAY)
                        texto = pytesseract.image_to_string(gray, config='--psm 8 --oem 3 -c tessedit_char_whitelist=' + self.solver.CHARS).strip()
                        texto = ''.join(filter(str.isalnum, texto.upper()))

                    # Controle de falhas de OCR
                    if len(texto) < 4:
                        tentativas_ocr += 1
                        print(f'Texto OCR curto ({texto}), tentativas falhas: {tentativas_ocr}')
                        if tentativas_ocr > 5:
                            print("Muitas falhas de OCR, reiniciando a página...")
                            self.driver.refresh()
                            tentativas_ocr = 0
                        continue
                    else:
                        tentativas_ocr = 0  # reset sucesso

                    print('Texto lido:', texto)
                    print('Texto lido:', texto)
                    texto = ''.join(filter(str.isalnum, texto))
                    
                    # Validar texto OCR
                    if len(texto) < 4:
                        print('Texto inválido, recarregando...')
                        continue
                except Exception as e:
                    print('Erro OCR:', e)
                    continue
                try:
                    input_box = self.driver.find_element(By.ID, 'captchaCode')
                    input_box.clear()
                    input_box.send_keys(texto)
                    botao = self.driver.find_element(By.ID, 'validateCaptchaButton')
                    botao.click()
                    print('Enviado:', texto)
                except Exception as e:
                    print('Erro ao enviar:', e)
                    continue
                time.sleep(1)
                tentativas += 1
                try:
                    result = self.driver.find_element(By.ID, 'validationResult').text
                    print('Resultado:', result)
                except:
                    result = ''
                if result and 'correct' in result.lower() and 'incorrect' not in result.lower():
                    acerto += 1
                score = (acerto / tentativas) * 100
                print(f'Tentativas: {tentativas}  Acertos: {acerto}  Score: {score:.2f}%')
                if score >= 75 and tentativas >= 10:
                    print('Sucesso!')
                    break
                try:
                    reload = self.driver.find_element(By.ID, 'demoCaptcha_ReloadLink')
                    reload.click()
                    print('Reload captcha')
                    time.sleep(5)
                except Exception as e:
                    print('Erro reload:', e)
                    continue
        except Exception as e:
            print(f"Erro crítico: {str(e)}")
        finally:
            self.driver.quit()
            print('Fim do script.')

if __name__ == '__main__':
    Desafio01().iniciar()
