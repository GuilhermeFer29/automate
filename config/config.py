from selenium import webdriver
from selenium.webdriver.chromium.webdriver import ChromiumDriver
from webdriver_manager.chrome import ChromeDriverManager


class Driver:

    def __init__(self) -> None:
        self.__driver_path = ChromeDriverManager().install()
        self.__service = webdriver.ChromeService(executable_path=self.__driver_path)
        self.__driver = webdriver.Chrome(service=self.__service)

    @property
    def driver(self) -> ChromiumDriver:
        return self.__driver


if __name__ == '__main__':
    b = Driver()
    b.driver.get('http://google.com')
