import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config.config import Driver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import requests
import time

class RPAChallenge:

    def __init__(self):
        self.navegador = Driver().driver
        self.base = 'https://rpachallenge.com'
        self.navegador.get(self.base)

    def baixar_excel(self):
        url_excel = 'https://rpachallenge.com/assets/downloadFiles/challenge.xlsx'
        response = requests.get(url_excel)
        with open('challenge.xlsx', 'wb') as f:
            f.write(response.content)
        
        self.df = pd.read_excel('challenge.xlsx')

    def iniciar(self):
        self.baixar_excel()
        
        start = self.navegador.find_element(By.XPATH, "//button[text()='Start']")
        start.click()
        
        for i, linha in self.df.iterrows():
            print(f'Preenchendo linha {i+1}/10')
            
            campos = {
                'labelFirstName': linha['First Name'],
                'labelLastName': linha['Last Name '],
                'labelCompanyName': linha['Company Name'],
                'labelRole': linha['Role in Company'],
                'labelAddress': linha['Address'],
                'labelEmail': linha['Email'],
                'labelPhone': str(linha['Phone Number'])
            }
            
            for atributo, valor in campos.items():
                input_element = WebDriverWait(self.navegador, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, f"input[ng-reflect-name='{atributo}']"))
                )
                input_element.send_keys(valor)
            
            submit = self.navegador.find_element(By.CSS_SELECTOR, "input[value='Submit']")
            submit.click()
            
            time.sleep(1)
        
        resultado = self.navegador.find_element(By.CLASS_NAME, 'message1').text
        print(resultado)
        
        self.navegador.quit()

if __name__ == '__main__':
    RPAChallenge().iniciar()
