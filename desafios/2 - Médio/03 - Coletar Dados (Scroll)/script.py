import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.config import Driver
import pandas as pd
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class Desafio03:
    def __init__(self):
        self.navegador = Driver().driver
        self.navegador.get('https://webscraper.io/test-sites/e-commerce/scroll')
        self.navegador.maximize_window()

    def iniciar(self):
        categorias = {
            'Laptops': ['Computers', 'Laptops'],
            'Tablets': ['Computers', 'Tablets'],
            'Phones': ['Phones']
        }
        dfs = {}
        for cat, path in categorias.items():
            self.navegador.get('https://webscraper.io/test-sites/e-commerce/scroll')
            for txt in path:
                el = WebDriverWait(self.navegador, 10).until(EC.element_to_be_clickable((By.XPATH, f"//a[normalize-space()='{txt}']")))
                self.navegador.execute_script("arguments[0].click();", el)
                time.sleep(1)
            ultimo = self.navegador.execute_script("return document.body.scrollHeight")
            while True:
                self.navegador.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                novo = self.navegador.execute_script("return document.body.scrollHeight")
                if novo == ultimo:
                    break
                ultimo = novo
            try:
                WebDriverWait(self.navegador, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'thumbnail')))
            except:
                pass
            produtos = self.navegador.find_elements(By.CLASS_NAME, 'thumbnail')
            linhas = []
            for p in produtos:
                try:
                    nome = p.find_element(By.CSS_SELECTOR, '.caption h4 a').text
                except:
                    nome = ''
                try:
                    preco = float(p.find_element(By.CSS_SELECTOR, '.caption h4.pull-right.price').text.replace('$',''))
                except:
                    preco = 0
                try:
                    descricao = p.find_elements(By.CSS_SELECTOR, '.caption p')[0].text
                except:
                    descricao = ''
                try:
                    estrelas = len(p.find_elements(By.CSS_SELECTOR, '.ratings .ws-icon-star, .ratings .glyphicon-star, .ratings i, .ratings svg'))
                except:
                    estrelas = 0
                try:
                    reviews = int(p.find_element(By.CSS_SELECTOR, '.ratings [itemprop="reviewCount"]').text.strip())
                except:
                    reviews = 0
                linhas.append([nome, preco, descricao, estrelas, reviews])
            df = pd.DataFrame(linhas, columns=['nome', 'preco', 'descricao', 'estrelas', 'reviews'])
            dfs[cat] = df
        with pd.ExcelWriter('produtos_scroll.xlsx', engine='openpyxl') as writer:
            for cat, df in dfs.items():
                df.sort_values('preco').to_excel(writer, sheet_name=f'{cat}_preco', index=False)
                df.sort_values('reviews', ascending=False).to_excel(writer, sheet_name=f'{cat}_reviews', index=False)
                df.sort_values('estrelas', ascending=False).to_excel(writer, sheet_name=f'{cat}_estrelas', index=False)
                df.sort_values(['preco', 'reviews', 'estrelas'], ascending=[True, False, False]).to_excel(writer, sheet_name=f'{cat}_bonus', index=False)
        self.navegador.quit()

if __name__ == '__main__':
    Desafio03().iniciar()