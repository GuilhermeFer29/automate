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
import tensorflow as tf, string
from io import BytesIO
import re
import cv2
import numpy as np
import base64
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = '/home/guilherme/Documentos/GitHub/automate/captcha_cnn/Desafio3Captcha_v2/final_model.keras'

class CaptchaSolver:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        # CARS deve ser exatamente o mesmo usado no treinamento, incluindo o caractere de preenchimento se houver.
        # Revertendo para o conjunto de caracteres mais provável com o qual o modelo foi treinado,
        # dado o erro np.int64(39). Este conjunto inclui maiúsculas.
        self.CHARS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'
        self.char_to_index = {c: i for i, c in enumerate(self.CHARS)}
        self.index_to_char = {i: c for i, c in enumerate(self.CHARS)}
        self.max_length = 10 # Comprimento do CAPTCHA que o modelo espera/prevê
    
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
        preds = self.model.predict(processed_img, verbose=0) # Retorna uma lista de 10 arrays (um por caractere)
        
        captcha_text_parts = []
        for i in range(self.max_length):
            char_probs = preds[i][0] # preds[i] é (1, num_chars), então preds[i][0] é (num_chars,)
            captcha_text_parts.append(self.index_to_char[np.argmax(char_probs)])
        return "".join(captcha_text_parts)

class Desafio01:
    def __init__(self):
        self.session = requests.Session()
        self.solver = CaptchaSolver(MODEL_PATH)
        self.driver = Driver().driver
    
    # Removida a função processar_imagem não utilizada
    
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

                content_type = ''
                img_array_para_modelo = None # Para o modelo CNN
                imagem_para_ocr = None # Para o Tesseract
                resolved_text = "" # Padronizado para resolved_text

                try:
                    timestamp = int(time.time() * 1000)
                    if src.startswith('data:image'):
                        logger.info("Imagem base64 detectada")
                        _, b64_data_str = src.split(',', 1)
                        
                        b64_data_str_clean = re.sub(r'[^A-Za-z0-9+/=]', '', b64_data_str)
                        while len(b64_data_str_clean) % 4:
                            b64_data_str_clean += '='
                        
                        image_bytes = base64.b64decode(b64_data_str_clean)
                        
                        debug_filename = f"debug_captcha_original_{timestamp}.png"
                        with open(debug_filename, 'wb') as f_debug:
                            f_debug.write(image_bytes)
                        logger.info(f"Imagem original salva para depuração em: {debug_filename}")

                        try:
                            pil_image = Image.open(BytesIO(image_bytes))
                            buffer = BytesIO()
                            pil_image.save(buffer, format="PNG")
                            buffer.seek(0)
                            pil_image_sanitized = Image.open(buffer)
                            
                            img_array_para_modelo = cv2.cvtColor(np.array(pil_image_sanitized.convert('RGB')), cv2.COLOR_RGB2BGR)
                            imagem_para_ocr = pil_image_sanitized.copy()
                        except Exception as e_pil:
                            logger.warning(f"Pillow falhou ao processar base64: {e_pil}. Tentando OpenCV diretamente.")
                            np_arr = np.frombuffer(image_bytes, np.uint8)
                            img_cv_direct = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            if img_cv_direct is not None:
                                img_array_para_modelo = img_cv_direct
                                imagem_para_ocr = Image.fromarray(cv2.cvtColor(img_cv_direct, cv2.COLOR_BGR2RGB))
                            else:
                                logger.error("OpenCV também não conseguiu decodificar base64.")
                                continue
                        
                    else: # Imagem via URL
                        r = requests.get(src, timeout=10)
                        r.raise_for_status()
                        content_type = r.headers.get('Content-Type', '')
                        if not content_type.startswith('image/'):
                            logger.warning(f"Conteúdo inesperado: {content_type}, recarregando...")
                            continue

                        image_bytes = r.content
                        extension = content_type.split('/')[-1] if '/' in content_type else 'png'
                        debug_filename = f"debug_captcha_original_{timestamp}.{extension}"
                        with open(debug_filename, 'wb') as f_debug:
                            f_debug.write(image_bytes)
                        logger.info(f"Imagem original salva para depuração em: {debug_filename}")

                        try:
                            pil_image = Image.open(BytesIO(image_bytes))
                            buffer = BytesIO()
                            pil_image.save(buffer, format="PNG")
                            buffer.seek(0)
                            pil_image_sanitized = Image.open(buffer)

                            img_array_para_modelo = cv2.cvtColor(np.array(pil_image_sanitized.convert('RGB')), cv2.COLOR_RGB2BGR)
                            imagem_para_ocr = pil_image_sanitized.copy()
                        except Exception as e_pil_url:
                            logger.error(f"Erro ao processar imagem de URL com Pillow: {e_pil_url}")
                            continue

                    if img_array_para_modelo is None: # Esta linha já está correta da edição anterior, mas a incluímos para garantir a substituição do bloco.
                        logger.warning("Não foi possível preparar a imagem para o modelo.")
                        continue # Pula para a próxima tentativa de CAPTCHA se a imagem não pôde ser preparada

                    # 1. Tentar resolver com o modelo CNN
                    captcha_solution_cnn = None
                    try:
                        # self.solver.solve espera um array CV BGR que é o que img_array_para_modelo é.
                        captcha_solution_cnn = self.solver.solve(img_array_para_modelo) 
                        logger.info(f"Predição do Modelo CNN: '{captcha_solution_cnn}'")
                        
                        # Validação da predição do CNN
                        # Remove placeholders '_' e verifica se o restante é válido e tem comprimento mínimo
                        resolved_text_cnn_cleaned = captcha_solution_cnn.replace('_', '')
                        
                        # O CAPTCHA real do site parece ter 6 caracteres. O modelo prevê 10.
                        # Se o CNN preencher com '_' além dos 6, resolved_text_cnn_cleaned pode ter 6.
                        # A validação deve ser se os caracteres EM resolved_text_cnn_cleaned são válidos.
                        if resolved_text_cnn_cleaned and len(resolved_text_cnn_cleaned) >= 3 and \
                           all(c in self.solver.CHARS for c in resolved_text_cnn_cleaned if c != '_'): # if c != '_' é redundante aqui
                            logger.info(f"Solução via CNN (limpa): {resolved_text_cnn_cleaned}")
                            resolved_text = resolved_text_cnn_cleaned # Usar a solução do CNN
                        else:
                            logger.warning(f"Solução do CNN '{captcha_solution_cnn}' (limpa: '{resolved_text_cnn_cleaned}') parece inválida. Tentando OCR.")
                            resolved_text = "" # Garante que resolved_text esteja vazio para forçar OCR
                    except Exception as e_cnn:
                        logger.error(f"Erro ao tentar resolver com CNN: {e_cnn}")
                        resolved_text = "" # Garante que resolved_text esteja vazio para forçar OCR
                    
                    # 2. Fallback para OCR se o CNN falhar ou resultado inválido (resolved_text ainda vazio)
                    if not resolved_text: 
                        logger.info("Tentando OCR como fallback...")
                        if imagem_para_ocr: # Verifica se temos uma imagem Pillow para OCR
                            ocr_image_L = imagem_para_ocr.convert('L') # Escala de cinza
                            # Aumenta o tamanho para melhor OCR
                            ocr_image_resized = ocr_image_L.resize((ocr_image_L.width * 3, ocr_image_L.height * 3), Image.Resampling.LANCZOS)
                            
                            ocr_np_array = np.array(ocr_image_resized)
                            # Aplicar limiar adaptativo Gaussiano. Ajuste os parâmetros 11 e 2 se necessário.
                            ocr_np_array_binarized = cv2.adaptiveThreshold(ocr_np_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                            
                            timestamp_ocr = int(time.time() * 1000) # Novo timestamp para esta imagem específica
                            ocr_debug_filename = f"debug_captcha_para_ocr_{timestamp_ocr}.png"
                            cv2.imwrite(ocr_debug_filename, ocr_np_array_binarized)
                            logger.info(f"Imagem para OCR salva em: {ocr_debug_filename}")

                            # Whitelist para Tesseract
                            whitelist_chars = string.ascii_letters + string.digits # A-Z, a-z, 0-9
                            # Alterando para --psm 8 (single word) ou --psm 7 (single line) pode ser melhor para CAPTCHAs
                            # Testando --psm 7 (tratar como uma única linha de texto)
                            ocr_text_raw = pytesseract.image_to_string(Image.fromarray(ocr_np_array_binarized), lang='eng', config=f'--psm 7 -c tessedit_char_whitelist={whitelist_chars}').strip()
                            # Limpeza adicional para remover quaisquer caracteres não alfanuméricos que possam ter passado
                            current_ocr_solution = re.sub(r'[^a-zA-Z0-9]', '', ocr_text_raw)
                            logger.info(f"Solução via OCR (raw: '{ocr_text_raw}', limpa: '{current_ocr_solution}')")
                            
                            # Verifica se a solução OCR é válida antes de atribuir a 'resolved_text'
                            if len(current_ocr_solution) >= 3: # Ou o comprimento mínimo que você espera
                                resolved_text = current_ocr_solution
                                # Não incrementa tentativas_ocr aqui, fazemos isso de forma geral para cada tentativa de OCR
                            else:
                                logger.warning(f'Texto OCR curto ou inválido ({current_ocr_solution}), tentativas OCR falhas: {tentativas_ocr}')
                                resolved_text = "" # Mantém resolved_text vazio se OCR também falhar em produzir algo útil
                            
                            tentativas_ocr += 1 # Incrementa para cada tentativa de OCR, bem-sucedida ou não em gerar texto útil

                            # Lógica de refresh da página se OCR falhar muito
                            if not resolved_text and tentativas_ocr > 5: # Se resolved_text ainda estiver vazio após esta tentativa de OCR
                                logger.info('Muitas falhas de OCR consecutivas, reiniciando a página...')
                                self.driver.refresh()
                                time.sleep(5) # Espera a página recarregar
                                tentativas_ocr = 0 # Reseta o contador de falhas de OCR
                                continue # Pula para a próxima iteração do loop while para pegar um novo CAPTCHA
                        else:
                            logger.warning("Não há imagem Pillow disponível para OCR.")
                            resolved_text = "" # Garante que resolved_text esteja vazio se não houver imagem para OCR
                            # Não precisa de 'continue' aqui, pois o próximo bloco verificará 'if not resolved_text'
                    
                    # Se resolved_text ainda estiver vazio após CNN e OCR, algo deu muito errado com este CAPTCHA
                    if not resolved_text:
                        logger.warning("Nenhuma solução encontrada (CNN e OCR falharam). Tentando novo captcha.")
                        # Adicionar um pequeno delay antes de tentar novamente pode ajudar
                        time.sleep(1) 
                        continue # Pula para a próxima iteração do loop while

                    # A partir daqui, 'resolved_text' deve ter a melhor solução encontrada (CNN ou OCR)
                    # O código para submeter o captcha e verificar o resultado continua abaixo.

                    # O 'continue' problemático da linha 253 foi removido.
                    # O 'else' abaixo está pareado com 'if not resolved_text:' (linha 243).
                    # Se chegamos aqui, resolved_text TEM um valor, então resetamos o contador de falhas de OCR.
                    tentativas_ocr = 0  # Reset do contador de falhas de OCR após uma solução ser encontrada (CNN ou OCR)

                    logger.info(f'Texto final para submissão: {resolved_text}')
                    
                    # A validação de comprimento e caracteres já foi feita. 
                    # O CAPTCHA do site parece ser de 6 caracteres. Se resolved_text for diferente, pode haver um problema.
                    # No entanto, vamos confiar nas validações anteriores por enquanto.
                    # Se for necessário forçar um comprimento específico (ex: 6), adicione aqui:
                    # if len(resolved_text) != 6:
                    #     logger.warning(f"Texto final '{resolved_text}' não tem 6 caracteres. Pulando submissão.")
                    #     continue

                except Exception as e: # Este except captura erros no bloco de processamento de imagem/CNN/OCR
                    logger.error(f'Erro no processamento/resolução do CAPTCHA: {e}')
                    continue # Tenta o próximo CAPTCHA
                
                # Bloco de submissão
                try:
                    input_box = self.driver.find_element(By.ID, 'captchaCode')
                    input_box.clear()
                    input_box.send_keys(resolved_text) # Usa resolved_text
                    botao = self.driver.find_element(By.ID, 'validateCaptchaButton')
                    botao.click()
                    logger.info(f'Enviado: {resolved_text}')
                except Exception as e:
                    logger.error(f'Erro ao enviar CAPTCHA: {e}')
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
    # Configuração básica do logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(sys.stdout) # Para imprimir no console
                            # Você pode adicionar logging.FileHandler("captcha_solver.log") aqui também
                        ])
    Desafio01().iniciar()
