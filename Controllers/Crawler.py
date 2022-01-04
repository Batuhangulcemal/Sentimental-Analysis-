from bs4 import BeautifulSoup
import requests
#https://www.theguardian.com/international

def Crawl(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")
    passage = soup.find_all("p")

    text = ""
    for i in passage:
        text += " "
        text += i.getText()
    print(text)
    return text
