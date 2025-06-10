from config.config import Driver


class Desafio03:

    def __init__(self) -> None:
        self.navegador = Driver().driver
        self.navegador.get('https://webscraper.io/test-sites/e-commerce/more')
        self.navegador.maximize_window()

    def iniciar(self) -> None:
        # Seu script aqui.
        input()


if __name__ == '__main__':
    Desafio03().iniciar()
