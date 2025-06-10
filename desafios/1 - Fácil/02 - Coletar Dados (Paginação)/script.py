from config.config import Driver


class Desafio02:

    def __init__(self) -> None:
        self.navegador = Driver().driver
        self.navegador.get('https://webscraper.io/test-sites/e-commerce/static')
        self.navegador.maximize_window()

    def iniciar(self) -> None:
        # Seu script aqui.
        input()


if __name__ == '__main__':
    Desafio02().iniciar()
