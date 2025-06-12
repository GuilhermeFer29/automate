import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config.config import Driver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

class Desafio02:

    def __init__(self):
        self.navegador = Driver().driver
        self.base = 'https://webscraper.io/test-sites/e-commerce/allinone'
        self.navegador.get(self.base)

    def iniciar(self):
        categorias = {
            'Laptops': 'laptops.xlsx',
            'Tablets': 'tablets.xlsx',
            'Phones': 'phones.xlsx'
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
                
                WebDriverWait(self.navegador, 5).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'thumbnail'))
                )
                
                produtos = self.navegador.find_elements(By.CLASS_NAME, 'thumbnail')
                for p in produtos:
                    try:
                        self.navegador.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", p)
                        
                        nome = p.find_element(By.CSS_SELECTOR, '.caption h4 a').text
                        preco = float(p.find_element(By.CSS_SELECTOR, '.caption h4.pull-right.price').text.replace('$', ''))
                        descricao = p.find_elements(By.CSS_SELECTOR, '.caption p')[0].text
                        estrelas = len(p.find_elements(By.CSS_SELECTOR, '.ratings .ws-icon-star, .ratings .glyphicon-star, .ratings i, .ratings svg'))
                        reviews = int(p.find_element(By.CSS_SELECTOR, '.ratings .review-count').text.split()[0])
                        todos_produtos.append([nome, preco, descricao, estrelas, reviews])
                    except Exception as e:
                        print(f' Erro: {e}')
                
                try:
                    proximo = self.navegador.find_element(By.CSS_SELECTOR, '.pagination .next')
                    if 'disabled' in proximo.get_attribute('class'):
                        break
                    proximo.click()
                    pagina += 1
                except Exception:
                    break
            
            df = pd.DataFrame(todos_produtos, columns=['nome', 'preco', 'descricao', 'estrelas', 'reviews'])
            with pd.ExcelWriter(arquivo, engine="openpyxl") as writer:
                df.sort_values('preco').to_excel(writer, sheet_name='preco', index=False)
                df.sort_values('reviews', ascending=False).to_excel(writer, sheet_name='reviews', index=False)
                df.sort_values('estrelas', ascending=False).to_excel(writer, sheet_name='estrelas', index=False)
                df.sort_values(['preco', 'reviews', 'estrelas'], ascending=[True, False, False]).to_excel(writer, sheet_name='bonus', index=False)
            
            print(f'Salvo {arquivo} com {len(df)} produtos')
        
        self.navegador.quit()
        print('Coleta completa!')

if __name__ == '__main__':
    Desafio02().iniciar()
