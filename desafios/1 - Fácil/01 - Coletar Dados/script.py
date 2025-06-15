import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.config import Driver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

try:
    import openpyxl
except ModuleNotFoundError:
    print('Biblioteca openpyxl não encontrada. Execute: pip install openpyxl')
    exit(1)


class Desafio01:

    def __init__(self):
        self.navegador = Driver().driver
        self.base = 'https://webscraper.io/test-sites/e-commerce/allinone'
        self.navegador.get(self.base)
        self.navegador.implicitly_wait(10)

    def iniciar(self):
        categorias = {'Laptops': 'laptops.xlsx', 'Tablets': 'tablets.xlsx', 'Phones': 'phones.xlsx'}
        for categoria, arquivo in categorias.items():
            self.navegador.get(self.base)
            if categoria in ['Laptops', 'Tablets']:
                comp = self.navegador.find_element(By.XPATH, "//a[normalize-space()='Computers']")
                self.navegador.execute_script("arguments[0].click();", comp)
                time.sleep(1)
            elif categoria == 'Phones':
                phones_link = self.navegador.find_element(By.XPATH, "//a[normalize-space()='Phones']")
                self.navegador.execute_script("arguments[0].click();", phones_link)
                time.sleep(1)
                touch_link = self.navegador.find_element(By.XPATH, "//a[normalize-space()='Touch']")
                self.navegador.execute_script("arguments[0].click();", touch_link)
                time.sleep(1)
            if categoria != 'Phones':
                cat_link = self.navegador.find_element(By.XPATH, f"//a[normalize-space()='{categoria}']")
                self.navegador.execute_script("arguments[0].click();", cat_link)
                time.sleep(1)
            try:
                WebDriverWait(self.navegador, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.ratings [itemprop="reviewCount"]')))
            except Exception:
                print(' Ratings não carregaram a tempo, continuando...')
            produtos = self.navegador.find_elements(By.CLASS_NAME, 'thumbnail')
            linhas = []
            for p in produtos:
                self.navegador.execute_script("arguments[0].scrollIntoView();", p)
                time.sleep(0.1)
                nome = p.find_element(By.CSS_SELECTOR, '.caption h4 a').text
                preco = float(p.find_element(By.CSS_SELECTOR, '.caption h4.pull-right.price').text.replace('$', ''))
                descricao = p.find_elements(By.CSS_SELECTOR, '.caption p')[0].text
                estrelas = len(p.find_elements(By.CSS_SELECTOR, '.ratings .ws-icon-star, .ratings .glyphicon-star, .ratings i, .ratings svg'))
                reviews_span = p.find_element(By.CSS_SELECTOR, '.ratings [itemprop="reviewCount"]')
                reviews = int(reviews_span.text.strip())
                linhas.append([nome, preco, descricao, estrelas, reviews])
            df = pd.DataFrame(linhas, columns=['nome', 'preco', 'descricao', 'estrelas', 'reviews'])
            with pd.ExcelWriter(arquivo, engine="openpyxl") as writer:
                df.sort_values('preco').to_excel(writer, sheet_name='preco', index=False)
                df.sort_values('reviews', ascending=False).to_excel(writer, sheet_name='reviews', index=False)
                df.sort_values('estrelas', ascending=False).to_excel(writer, sheet_name='estrelas', index=False)
                df.sort_values(['preco', 'reviews', 'estrelas'], ascending=[True, False, False]).to_excel(writer, sheet_name='bonus', index=False)
        self.navegador.quit()


if __name__ == '__main__':
    Desafio01().iniciar()
