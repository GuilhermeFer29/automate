import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.config import Driver
from selenium.webdriver.common.by import By
import pandas as pd
import time


class Desafio03:

    def __init__(self):
        self.navegador = Driver().driver
        self.base = 'https://webscraper.io/test-sites/e-commerce/more'
        self.navegador.get(self.base)
        self.navegador.maximize_window()
        self.navegador.implicitly_wait(10)

    def iniciar(self):
        categorias = {'Laptops': 'laptops.xlsx', 'Tablets': 'tablets.xlsx', 'Phones': 'phones.xlsx'}
        for categoria, arquivo in categorias.items():
            self.navegador.get(self.base)
            if categoria in ['Laptops', 'Tablets']:
                self.navegador.find_element(By.LINK_TEXT, 'Computers').click()
            self.navegador.find_element(By.LINK_TEXT, categoria).click()
            time.sleep(1)
            show_more = self.navegador.find_elements(By.ID, 'moreButton')
            if show_more:
                show_more[0].click()
                time.sleep(1)
            produtos = self.navegador.find_elements(By.CLASS_NAME, 'thumbnail')
            linhas = []
            for p in produtos:
                nome = p.find_element(By.CSS_SELECTOR, '.caption h4 a').text
                preco = float(p.find_element(By.CSS_SELECTOR, '.caption h4.pull-right.price').text.replace('$', ''))
                descricao = p.find_elements(By.CSS_SELECTOR, '.caption p')[0].text
                estrelas = len(p.find_elements(By.CSS_SELECTOR, '.glyphicon-star'))
                reviews = int(p.find_element(By.CSS_SELECTOR, '.ratings p.pull-right').text.split()[0])
                linhas.append([nome, preco, descricao, estrelas, reviews])
            df = pd.DataFrame(linhas, columns=['nome', 'preco', 'descricao', 'estrelas', 'reviews'])
            with pd.ExcelWriter(arquivo) as writer:
                df.sort_values('preco').to_excel(writer, sheet_name='preco', index=False)
                df.sort_values('reviews', ascending=False).to_excel(writer, sheet_name='reviews', index=False)
                df.sort_values('estrelas', ascending=False).to_excel(writer, sheet_name='estrelas', index=False)
                df.sort_values(['preco', 'reviews', 'estrelas'], ascending=[True, False, False]).to_excel(writer, sheet_name='bonus', index=False)
        self.navegador.quit()


if __name__ == '__main__':
    Desafio03().iniciar()
