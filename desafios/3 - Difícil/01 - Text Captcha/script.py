import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.config import Driver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import requests
from PIL import Image
import pytesseract
from io import BytesIO

class Desafio01:
    def iniciar(self):
        print('Iniciando navegador...')
        driver = Driver().driver
        driver.get('https://captcha.com/demos/features/captcha-demo.aspx')
        time.sleep(2)
        driver.maximize_window()
        acerto = 0
        tentativas = 0
        time.sleep(2)
        while True:
            print('Procurando imagem do captcha...')
            try:
                img = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'demoCaptcha_CaptchaImage')))
            except Exception as e:
                print('Não achou imagem do captcha:', e)
                print('HTML da página para debug:')
                print(driver.page_source)
                break
            src = img.get_attribute('src')
            if src.startswith('/'):
                src = 'https://captcha.com' + src
            print('Captcha encontrado:', src)
            print('Baixando captcha:', src)
            try:
                if src.startswith('data:image'):
                    import base64
                    header, b64data = src.split(',', 1)
                    print(header)
                    b64data = b64data.replace('\n', '').replace('\r', '')
                    print(b64data[:100])
                    print(b64data[-100:])
                    while len(b64data) % 4 != 0:
                        b64data += '='
                    image_data = base64.b64decode(b64data)
                    print(len(image_data))
                    with open('captcha_debug.png', 'wb') as f:
                        f.write(image_data)
                    image = Image.open('captcha_debug.png')
                else:
                    r = requests.get(src)
                    image = Image.open(BytesIO(r.content))
                # Pré-processamento para melhorar OCR
                gray = image.convert('L')
                # Binarização simples
                bw = gray.point(lambda x: 0 if x < 128 else 255, '1')
                texto = pytesseract.image_to_string(bw)
                print('Texto lido:', texto)
                texto = ''.join(filter(str.isalnum, texto))
            except Exception as e:
                print('Erro OCR:', e)
                break
            try:
                input_box = driver.find_element(By.ID, 'captchaCode')
                input_box.clear()
                input_box.send_keys(texto)
                botao = driver.find_element(By.ID, 'validateCaptchaButton')
                botao.click()
                print('Enviado:', texto)
            except Exception as e:
                print('Erro ao enviar:', e)
                break
            time.sleep(1)
            tentativas += 1
            try:
                result = driver.find_element(By.ID, 'validationResult').text
                print('Resultado:', result)
            except:
                result = ''
            if 'correct' in result.lower():
                acerto += 1
            score = (acerto / tentativas) * 100
            print(f'Tentativas: {tentativas}  Acertos: {acerto}  Score: {score:.2f}%')
            if score >= 75 and tentativas >= 4:
                print('Sucesso!')
                break
            try:
                reload = driver.find_element(By.ID, 'demoCaptcha_ReloadLink')
                reload.click()
                print('Reload captcha')
                time.sleep(1)
            except Exception as e:
                print('Erro reload:', e)
                break
        driver.quit()
        print('Fim do script.')

if __name__ == '__main__':
    Desafio01().iniciar()
