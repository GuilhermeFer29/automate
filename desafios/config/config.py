from selenium import webdriver
import shutil


class Driver:

    def __init__(self):
        options = webdriver.ChromeOptions()
        binary = shutil.which("chromium-browser") or shutil.which("chromium") or shutil.which("google-chrome") or shutil.which("brave-browser") or shutil.which("brave")
        if binary:
            options.binary_location = binary
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--remote-debugging-port=9222')
        options.add_argument('--user-data-dir=/tmp/chrome_profile')
        options.add_argument('--disable-crash-reporter')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-in-process-stack-traces')
        self.__driver = webdriver.Chrome(options=options)

    @property
    def driver(self):
        return self.__driver


if __name__ == '__main__':
    b = Driver()
    b.driver.get('https://www.google.com/')
    print(b.driver.title)
    b.driver.quit()
