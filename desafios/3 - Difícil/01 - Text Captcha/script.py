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

CHARS = string.ascii_uppercase + string.digits
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'captcha_cnn', 'captcha_char_cnn.h5'))
cnn_model = None
reader = easyocr.Reader(['en'], gpu=False)

def cnn_ocr(thresh_img):
    # Remover ruídos finos
        # encontrar contornos dos caracteres
    blur = cv2.GaussianBlur(thresh_img, (3,3), 0)
    cnts, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) >= 80]
    boxes.sort(key=lambda b: b[0])

    # Define número esperado de caracteres (site usa 4 a 7)
    if 4 <= len(boxes) <= 7:
        expected_chars = len(boxes)
    else:
        expected_chars = 6  # fallback

    if len(boxes) != expected_chars:
        h, w = thresh_img.shape[:2]
        step = w // expected_chars
        boxes = [(i*step, 0, step, h) for i in range(expected_chars)]
    elif len(boxes) > expected_chars:
        boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)[:expected_chars]
        boxes.sort(key=lambda b: b[0])
    chars = []
    for (x, y, w, h) in boxes:
        roi = thresh_img[y:y+h, x:x+w]
        hp = max(w - h, 0); vp = max(h - w, 0)
        roi = cv2.copyMakeBorder(roi, hp//2, hp-hp//2, vp//2, vp-vp//2, cv2.BORDER_CONSTANT, value=0)
        roi = cv2.resize(roi, (28, 28))
        chars.append(roi.astype('float32')/255.0)
    X = np.expand_dims(np.array(chars), -1)
    preds = cnn_model.predict(X, verbose=0)
    return ''.join(CHARS[p.argmax()] for p in preds)

class Desafio01:
    def iniciar(self):
        try:
            print('Iniciando navegador...')
            driver = Driver().driver
            driver.get('https://captcha.com/demos/features/captcha-demo.aspx')
            time.sleep(2)
            driver.maximize_window()
            acerto = 0
            tentativas = 0
            tentativas_ocr = 0
            time.sleep(2)
            while True:
                print('Procurando imagem do captcha...')
                try:
                    img = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'demoCaptcha_CaptchaImage')))
                except Exception as e:
                    print('Não achou imagem do captcha:', e)
                    if "invalid session id" in str(e):
                        print('Sessão do navegador fechada, reiniciando...')
                        driver.quit()
                        driver = Driver().driver
                        driver.get('https://captcha.com/demos/features/captcha-demo.aspx')
                        driver.maximize_window()
                        continue
                    print('HTML da página para debug:')
                    print(driver.page_source)
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
                        allow = CHARS

                    texto = ''.join(reader.readtext(np.array(img_rgb), detail=0, paragraph=False, allowlist=allow))
                    texto = ''.join(filter(str.isalnum, texto.upper()))

                    # fallback binarizado + tesseract para JPEG numérico
                    if (len(texto) < 5 and is_jpeg):
                        gray = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2GRAY)
                        _, thr = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
                        texto = pytesseract.image_to_string(thr, config='--psm 6 -c tessedit_char_whitelist=0123456789').strip()
                        texto = ''.join(filter(str.isalnum, texto.upper()))

                    # fallback geral
                    if not 4 <= len(texto) <= 7:
                        gray = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2GRAY)
                        texto = pytesseract.image_to_string(gray, config='--psm 8 --oem 3 -c tessedit_char_whitelist=' + CHARS).strip()
                        texto = ''.join(filter(str.isalnum, texto.upper()))

                    # Controle de falhas de OCR
                    if len(texto) < 4:
                        tentativas_ocr += 1
                        print(f'Texto OCR curto ({texto}), tentativas falhas: {tentativas_ocr}')
                        if tentativas_ocr > 5:
                            print("Muitas falhas de OCR, reiniciando a página...")
                            driver.refresh()
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
                    input_box = driver.find_element(By.ID, 'captchaCode')
                    input_box.clear()
                    input_box.send_keys(texto)
                    botao = driver.find_element(By.ID, 'validateCaptchaButton')
                    botao.click()
                    print('Enviado:', texto)
                except Exception as e:
                    print('Erro ao enviar:', e)
                    continue
                time.sleep(1)
                tentativas += 1
                try:
                    result = driver.find_element(By.ID, 'validationResult').text
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
                    reload = driver.find_element(By.ID, 'demoCaptcha_ReloadLink')
                    reload.click()
                    print('Reload captcha')
                    time.sleep(5)
                except Exception as e:
                    print('Erro reload:', e)
                    continue
        except Exception as e:
            print(f"Erro crítico: {str(e)}")
        finally:
            driver.quit()
            print('Fim do script.')

if __name__ == '__main__':
    Desafio01().iniciar()
