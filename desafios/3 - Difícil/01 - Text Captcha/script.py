import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from selenium import webdriver
from config.config import Driver # Updated import
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import requests
from PIL import Image, ImageEnhance
import pytesseract
import numpy as np
import cv2
from io import BytesIO
import string
import re
import logging
import base64
from openai import OpenAI, APIStatusError, AuthenticationError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class Desafio01:
    def __init__(self):
        self.driver = Driver().driver # Use Driver from config
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY não encontrada nas variáveis de ambiente.")
            self.openai_client = None
        else:
            self.openai_client = OpenAI(api_key=self.openai_api_key)

        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

    def _solve_captcha_with_openai(self, image_bytes):
        if not self.openai_client:
            logger.error("Cliente OpenAI não inicializado. Verifique a OPENAI_API_KEY.")
            return None
        logger.info("Tentando resolver CAPTCHA com OpenAI API...")
        try:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert CAPTCHA solver. Your task is to identify the 6 alphanumeric characters in the provided image. Respond with ONLY these 6 characters. Do not include any other text, explanations, or apologies."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract the 6 alphanumeric characters from this image."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=10, # Reduced to prevent longer responses
                temperature=0.2
            )
            
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                solution = response.choices[0].message.content.strip()
                cleaned_solution = re.sub(r'[^a-zA-Z0-9]', '', solution)
                # logger.info(f"OpenAI API raw response: '{solution}', cleaned: '{cleaned_solution}'")
                return cleaned_solution
            else:
                logger.warning("OpenAI API não retornou conteúdo esperado na resposta.")
                return None
        except APIStatusError as api_err:
            logger.error(f"Erro na chamada da API OpenAI (APIStatusError): {api_err}")
            return None
        except AuthenticationError as auth_err:
            logger.error(f"Erro na chamada da API OpenAI: 401 Unauthorized. Verifique sua OPENAI_API_KEY e permissões. Detalhes: {auth_err}")
            return None
        except Exception as e:
            logger.error(f"Erro inesperado ao chamar a API OpenAI: {e}")
            return None
    
    def iniciar(self):
        try:
            logger.info('Iniciando navegador...')
            self.driver.get('https://captcha.com/demos/features/captcha-demo.aspx')
            time.sleep(2)
            self.driver.maximize_window()
            acerto = 0
            tentativas = 0
            tentativas_ocr = 0
            time.sleep(2)
            while True:
                # logger.info('Procurando imagem do captcha...')
                try:
                    img = WebDriverWait(self.driver, 20).until(EC.presence_of_element_located((By.ID, 'demoCaptcha_CaptchaImage')))
                except Exception as e:
                    logger.error(f'Não achou imagem do captcha: {e}')
                    if "invalid session id" in str(e).lower(): # Check for closed browser session
                        logger.info('Sessão do navegador fechada, reiniciando...')
                        if self.driver:
                            self.driver.quit()
                        # Re-initialize driver components
                        self.driver = Driver().driver # Use Driver from config for re-initialization
                        self.driver.get('https://captcha.com/demos/features/captcha-demo.aspx')
                        self.driver.maximize_window()
                        acerto = 0 # Reset counters
                        tentativas = 0
                        tentativas_ocr = 0
                        continue
                    logger.error('HTML da página para debug:')
                    logger.error(self.driver.page_source)
                    raise
                src = img.get_attribute('src')
                if src.startswith('/'):
                    src = 'https://captcha.com' + src
                # logger.info(f'Captcha encontrado: {src}')
                # logger.info(f'Baixando captcha: {src}')

                image_bytes = None
                imagem_para_ocr = None
                resolved_text = ""

                try:
                    if src.startswith('data:image'):
                        # logger.info("Imagem base64 detectada")
                        _, b64_data_str = src.split(',', 1)
                        b64_data_str_clean = re.sub(r'[^A-Za-z0-9+/=]', '', b64_data_str)
                        while len(b64_data_str_clean) % 4:
                            b64_data_str_clean += '=' 
                        image_bytes = base64.b64decode(b64_data_str_clean)
                        
                        # Tentativa de sanitização com OpenCV primeiro
                        np_arr = np.frombuffer(image_bytes, np.uint8)
                        img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if img_cv is not None:
                            # logger.info("Imagem base64 decodificada com OpenCV. Tentando re-encodar para PNG para sanitizar.")
                            success, encoded_cv_image_buffer = cv2.imencode('.png', img_cv)
                            if success:
                                image_bytes_sanitized = encoded_cv_image_buffer.tobytes()
                                try:
                                    pil_img = Image.open(BytesIO(image_bytes_sanitized))
                                    imagem_para_ocr = pil_img.copy()
                                    image_bytes = image_bytes_sanitized # Usar os bytes sanitizados daqui em diante
                                    # logger.info("Imagem base64 sanitizada com OpenCV e carregada com Pillow.")
                                except Exception as e_pil_after_cv:
                                    logger.warning(f"Pillow falhou ao processar imagem base64 sanitizada por OpenCV: {e_pil_after_cv}. Usando imagem OpenCV diretamente para OCR se possível.")
                                    try:
                                        imagem_para_ocr = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                                    except Exception as e_ocr_cv_direct:
                                        logger.error(f"Falha ao criar imagem OCR de OpenCV (base64, pós falha Pillow): {e_ocr_cv_direct}")
                            else:
                                logger.warning("OpenCV falhou ao re-encodar imagem base64 para PNG. Tentando usar Pillow com bytes originais.")
                                try:
                                    pil_img = Image.open(BytesIO(image_bytes))
                                    imagem_para_ocr = pil_img.copy()
                                    # logger.info("Imagem base64 processada com Pillow (OpenCV re-encode falhou).")
                                except Exception as e_pil_fallback:
                                    logger.error(f"Pillow também falhou ao processar imagem base64 original: {e_pil_fallback}")
                        else:
                            logger.error("OpenCV falhou ao decodificar imagem base64. Tentando Pillow com bytes originais.")
                            try:
                                pil_img = Image.open(BytesIO(image_bytes))
                                imagem_para_ocr = pil_img.copy()
                                # logger.info("Imagem base64 processada com Pillow (OpenCV decode falhou).")
                            except Exception as e_pil_fallback_cv_fail:
                                logger.error(f"Pillow também falhou ao processar imagem base64 original (OpenCV decode falhou): {e_pil_fallback_cv_fail}")
                    else: 
                        r = requests.get(src, timeout=10)
                        r.raise_for_status()
                        content_type = r.headers.get('Content-Type', '')
                        if not content_type.startswith('image/'):
                            logger.warning(f"Conteúdo inesperado: {content_type}, recarregando...")
                            continue
                        image_bytes = r.content
                        # Tentativa de sanitização com OpenCV primeiro
                        np_arr = np.frombuffer(image_bytes, np.uint8)
                        img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if img_cv is not None:
                            # logger.info("Imagem de URL decodificada com OpenCV. Tentando re-encodar para PNG para sanitizar.")
                            success, encoded_cv_image_buffer = cv2.imencode('.png', img_cv)
                            if success:
                                image_bytes_sanitized = encoded_cv_image_buffer.tobytes()
                                try:
                                    pil_img = Image.open(BytesIO(image_bytes_sanitized))
                                    imagem_para_ocr = pil_img.copy()
                                    image_bytes = image_bytes_sanitized # Usar os bytes sanitizados daqui em diante
                                    # logger.info("Imagem de URL sanitizada com OpenCV e carregada com Pillow.")
                                except Exception as e_pil_after_cv:
                                    logger.warning(f"Pillow falhou ao processar imagem de URL sanitizada por OpenCV: {e_pil_after_cv}. Usando imagem OpenCV diretamente para OCR se possível.")
                                    try:
                                        imagem_para_ocr = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                                    except Exception as e_ocr_cv_direct:
                                        logger.error(f"Falha ao criar imagem OCR de OpenCV (URL, pós falha Pillow): {e_ocr_cv_direct}")
                            else:
                                logger.warning("OpenCV falhou ao re-encodar imagem de URL para PNG. Tentando usar Pillow com bytes originais.")
                                try:
                                    pil_img = Image.open(BytesIO(image_bytes))
                                    imagem_para_ocr = pil_img.copy()
                                    # logger.info("Imagem de URL processada com Pillow (OpenCV re-encode falhou).")
                                except Exception as e_pil_fallback:
                                    logger.error(f"Pillow também falhou ao processar imagem de URL original: {e_pil_fallback}")
                        else:
                            logger.error("OpenCV falhou ao decodificar imagem de URL. Tentando Pillow com bytes originais.")
                            try:
                                pil_img = Image.open(BytesIO(image_bytes))
                                imagem_para_ocr = pil_img.copy()
                                # logger.info("Imagem de URL processada com Pillow (OpenCV decode falhou).")
                            except Exception as e_pil_fallback_cv_fail:
                                logger.error(f"Pillow também falhou ao processar imagem de URL original (OpenCV decode falhou): {e_pil_fallback_cv_fail}")

                    if image_bytes:
                        try:
                            nparr = np.frombuffer(image_bytes, np.uint8)
                            img_cv_preproc = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if img_cv_preproc is not None:
                                gray_cv = cv2.cvtColor(img_cv_preproc, cv2.COLOR_BGR2GRAY)
                                # Aplicar Median Blur para remoção de ruído
                                gray_cv_blurred = cv2.medianBlur(gray_cv, 3)
                                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                                contrast_cv = clahe.apply(gray_cv_blurred)
                                binary_cv = cv2.adaptiveThreshold(contrast_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                                _, processed_image_bytes_buffer = cv2.imencode('.png', binary_cv)
                                image_bytes = processed_image_bytes_buffer.tobytes()
                                # logger.info("Imagem adicionalmente pré-processada para API OpenAI.")
                            else:
                                logger.warning("Falha ao decodificar image_bytes com OpenCV para pré-processamento adicional.")
                        except Exception as e_proc:
                            logger.error(f"Erro durante pré-processamento adicional da imagem para API: {e_proc}")

                    if image_bytes:
                        try:
                            script_dir = os.path.dirname(__file__)
                            debug_image_folder = os.path.join(script_dir, "debug_api_images")
                            os.makedirs(debug_image_folder, exist_ok=True)
                            current_timestamp_for_api_debug = int(time.time() * 1000)
                            api_debug_filename = os.path.join(debug_image_folder, f"debug_captcha_for_api_{current_timestamp_for_api_debug}.png")
                            with open(api_debug_filename, 'wb') as f_api_debug:
                                f_api_debug.write(image_bytes)
                            # logger.info(f"Imagem SANITIZADA para API OpenAI salva em: {api_debug_filename}")
                        except Exception as e_save_api_debug:
                            logger.error(f"Falha ao salvar imagem de depuração para API OpenAI: {e_save_api_debug}")

                    if self.openai_client and image_bytes:
                        try:
                            captcha_solution_openai = self._solve_captcha_with_openai(image_bytes)
                            if captcha_solution_openai:
                                # logger.info(f"Solução via OpenAI API: '{captcha_solution_openai}'")
                                if len(captcha_solution_openai) == 6 and captcha_solution_openai.isalnum(): 
                                    resolved_text = captcha_solution_openai
                                else:
                                    logger.warning(f"Solução da OpenAI '{captcha_solution_openai}' (len: {len(captcha_solution_openai)}) inválida. Tentando OCR.")
                            else:
                                logger.warning("OpenAI API não retornou solução ou falhou. Tentando OCR.")
                        except Exception as e_openai:
                            logger.error(f"Erro ao tentar resolver com OpenAI API: {e_openai}")
                    elif not image_bytes:
                        logger.warning("Não há bytes de imagem para enviar à API OpenAI. Pulando.")
                    else:
                        logger.warning("Cliente OpenAI não configurado. Pulando API, indo para OCR.")
                    
                    if not resolved_text: 
                        # logger.info("Tentando OCR como fallback...")
                        if imagem_para_ocr:
                            ocr_image_L = imagem_para_ocr.convert('L')
                            ocr_image_resized = ocr_image_L.resize((ocr_image_L.width * 3, ocr_image_L.height * 3), Image.Resampling.LANCZOS)
                            ocr_np_array = np.array(ocr_image_resized)
                            ocr_np_array_binarized = cv2.adaptiveThreshold(ocr_np_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                            try:
                                script_dir_ocr = os.path.dirname(__file__)
                                debug_ocr_image_folder = os.path.join(script_dir_ocr, "debug_ocr_images")
                                os.makedirs(debug_ocr_image_folder, exist_ok=True)
                                current_timestamp_for_ocr_debug = int(time.time() * 1000)
                                ocr_debug_filename = os.path.join(debug_ocr_image_folder, f"debug_captcha_for_ocr_{current_timestamp_for_ocr_debug}.png")
                                Image.fromarray(ocr_np_array_binarized).save(ocr_debug_filename)
                                # logger.info(f"Imagem PRÉ-PROCESSADA para OCR salva em: {ocr_debug_filename}")
                            except Exception as e_save_ocr_debug:
                                logger.error(f"Falha ao salvar imagem de depuração para OCR: {e_save_ocr_debug}")
                            
                            whitelist_chars = string.ascii_letters + string.digits
                            ocr_text_raw = pytesseract.image_to_string(Image.fromarray(ocr_np_array_binarized), lang='eng', config=f'--psm 7 -c tessedit_char_whitelist={whitelist_chars}').strip()
                            current_ocr_solution = re.sub(r'[^a-zA-Z0-9]', '', ocr_text_raw)
                            # logger.info(f"Solução via OCR (raw: '{ocr_text_raw}', limpa: '{current_ocr_solution}')")
                            
                            if len(current_ocr_solution) >= 3:
                                resolved_text = current_ocr_solution
                            else:
                                logger.warning(f'Texto OCR curto ou inválido ({current_ocr_solution}), tentativas OCR falhas: {tentativas_ocr}')
                                resolved_text = ""
                            
                            tentativas_ocr += 1

                            if not resolved_text and tentativas_ocr > 5:
                                logger.info('Muitas falhas de OCR consecutivas, reiniciando a página...')
                                self.driver.refresh()
                                time.sleep(5)
                                tentativas_ocr = 0 
                                continue 
                        else:
                            logger.warning("Não há imagem Pillow disponível para OCR.")
                            resolved_text = ""
                    
                    if not resolved_text:
                        logger.warning("Nenhuma solução encontrada (API e OCR falharam). Tentando novo captcha.")
                        time.sleep(1) 
                        continue

                    tentativas_ocr = 0
                    logger.info(f'Texto final para submissão: {resolved_text}')
                    
                except Exception as e:
                    logger.error(f'Erro no processamento/resolução do CAPTCHA: {e}')
                    continue 
                
                try:
                    input_box = self.driver.find_element(By.ID, 'captchaCode')
                    input_box.clear()
                    input_box.send_keys(resolved_text)
                    botao = self.driver.find_element(By.ID, 'validateCaptchaButton')
                    botao.click()
                    logger.info(f'Enviado: {resolved_text}')
                except Exception as e:
                    logger.error(f'Erro ao enviar CAPTCHA: {e}')
                    continue
                time.sleep(1)
                tentativas += 1
                try:
                    validation_element = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.ID, 'validationResult'))
                    )
                    # Esperar um pouco para o texto estabilizar, se necessário, ou que contenha uma palavra chave
                    # WebDriverWait(self.driver, 3).until(
                    #     lambda driver: "correct" in driver.find_element(By.ID, 'validationResult').text.lower() or \
                    #                  "incorrect" in driver.find_element(By.ID, 'validationResult').text.lower()
                    # )
                    result_text = validation_element.text.strip().lower()
                    logger.info(f'Resultado: {result_text}')
                    
                    is_correct = "correct" in result_text
                    is_incorrect = "incorrect" in result_text

                    if is_correct and not is_incorrect:
                        acerto += 1
                    elif is_incorrect:
                        pass # Já é um erro, não incrementa acerto
                    else:
                        logger.warning(f"Resultado de validação incerto: '{result_text}'. Considerando como erro.")

                except Exception as e_val:
                    logger.warning(f"Não foi possível obter o resultado da validação ou resultado inesperado: {e_val}. Considerando como erro.")
                    result_text = "" # Garante que result_text exista
                erradas = tentativas - acerto
                taxa_acerto = (acerto / tentativas) * 100 if tentativas > 0 else 0
                logger.info(f'Tentativas: {tentativas}, Acertos: {acerto}, Erros: {erradas}, Taxa de Acerto: {taxa_acerto:.2f}%')

                logger.info("Aguardando 3.5 segundos antes da próxima tentativa...")
                time.sleep(3.5)

                if tentativas >= 4 and taxa_acerto > 75:
                    logger.info(f'Sucesso! Taxa de acerto de {taxa_acerto:.2f}% atingida após {tentativas} tentativas (mínimo de 4).')
                    break
            if self.driver:
                self.driver.quit()
        except Exception as e:
            logger.error(f'Erro geral no script: {e}')
            if self.driver:
                self.driver.quit()
        finally:
            logger.info("Fim do script.")

if __name__ == '__main__':
    # Configuração básica do logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(sys.stdout) # Para imprimir no console
                            # Você pode adicionar logging.FileHandler("captcha_solver.log") aqui também
                        ])
    Desafio01().iniciar()
