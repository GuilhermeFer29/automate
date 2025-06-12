import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config.config import Driver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

class Desafio03:

    def __init__(self):
        self.navegador = Driver().driver
        self.base = 'https://webscraper.io/test-sites/e-commerce/allinone'
        self.navegador.get(self.base)

    def iniciar(self):
        categorias = {
            'Laptops': 'laptops_expandido.xlsx',
            'Tablets': 'tablets_expandido.xlsx',
            'Phones': 'phones_expandido.xlsx'
        }
        
        for categoria, arquivo in categorias.items():
            print(f'Coletando categoria: {categoria}')
            self.navegador.get(self.base)
            
            if categoria in ['Laptops', 'Tablets']:
                comp = self.navegador.find_element(By.XPATH, "//a[normalize-space()='Computers']")
                self.navegador.execute_script("arguments[0].click();", comp)
                
                cat_link = self.navegador.find_element(By.XPATH, f"//a[normalize-space()='{categoria}']")
                self.navegador.execute_script("arguments[0].click();", cat_link)
            
            elif categoria == 'Phones':
                phones_link = self.navegador.find_element(By.XPATH, "//a[normalize-space()='Phones']")
                self.navegador.execute_script("arguments[0].click();", phones_link)
                
                touch_link = self.navegador.find_element(By.XPATH, "//a[normalize-space()='Touch']")
                self.navegador.execute_script("arguments[0].click();", touch_link)
            
            todos_produtos = []
            pagina = 1
            
            while True:
                print(f'Página {pagina}')
                
                WebDriverWait(self.navegador, 3).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'thumbnail'))
                )
                
                produtos = self.navegador.find_elements(By.CLASS_NAME, 'thumbnail')
                links_produtos = [p.find_element(By.CSS_SELECTOR, '.caption h4 a').get_attribute('href') for p in produtos]
                
                for link in links_produtos:
                    self.navegador.get(link)
                    
                    try:
                        nome = self.navegador.find_element(By.CSS_SELECTOR, '.caption h4:nth-child(2)').text
                        preco = float(self.navegador.find_element(By.CSS_SELECTOR, '.caption h4.pull-right.price').text.replace('$', ''))
                        descricao = self.navegador.find_element(By.CSS_SELECTOR, '.description').text
                        
                        caracteristicas = {}
                        divs = self.navegador.find_elements(By.CSS_SELECTOR, '.tab-pane > div')
                        for div in divs:
                            chave = div.find_element(By.CSS_SELECTOR, 'strong').text
                            valor = div.text.replace(chave, '').strip()
                            caracteristicas[chave] = valor
                        
                        estrelas = len(self.navegador.find_elements(By.CSS_SELECTOR, '.ratings .ws-icon-star, .ratings .glyphicon-star, .ratings i, .ratings svg'))
                        reviews = int(self.navegador.find_element(By.CSS_SELECTOR, '.ratings .review-count').text.split()[0])
                        
                        todos_produtos.append({
                            'nome': nome,
                            'preco': preco,
                            'descricao': descricao,
                            'estrelas': estrelas,
                            'reviews': reviews,
                            **caracteristicas
                        })
                    except Exception as e:
                        print(f' Erro: {e}')
                    
                    self.navegador.back()
                
                try:
                    proximo = self.navegador.find_element(By.CSS_SELECTOR, '.pagination .next')
                    if 'disabled' in proximo.get_attribute('class'):
                        break
                    proximo.click()
                    pagina += 1
                except Exception:
                    break
            
            df = pd.DataFrame(todos_produtos)
            df.to_excel(arquivo, index=False)
            print(f'Salvo {arquivo} com {len(df)} produtos')
        
        self.navegador.quit()
        print('Coleta expandida completa!')

if __name__ == '__main__':
    Desafio03().iniciar()
