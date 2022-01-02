from bs4 import BeautifulSoup
import requests
#https://www.theguardian.com/international

main_url = "https://www.theguardian.com/us-news/2022/jan/02/ghislaine-maxwell-the-demon-queen-is-behind-bars-does-she-have-a-secret-that-could-unlock-her-shackles"
req = requests.get(main_url)
soup = BeautifulSoup(req.text, "html.parser")

soup2 = soup.find_all("p", class_="dcr-o5gy41")[0]

#soup3 = soup2.getText()

print(soup2.getText())

def gotten(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")
    soup2 = soup.find_all("p", class_="dcr-o5gy41")[0]

    return soup2.getText()