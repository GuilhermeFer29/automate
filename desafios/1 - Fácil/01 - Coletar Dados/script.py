from config.config import Driver


class Desafio01:

    def __init__(self) -> None:
        self.navegador = Driver().driver
        self.navegador.get('https://webscraper.io/test-sites/e-commerce/allinone')
        self.navegador.maximize_window()

    def iniciar(self) -> None:
        # Seu script aqui.
        input()


if __name__ == '__main__':
    Desafio01().iniciar()
