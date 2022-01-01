from bs4 import BeautifulSoup
import requests
#https://www.theguardian.com/international

main_url = "https://www.theguardian.com/uk-news/2022/jan/01/prince-andrew-lawsuit-virginia-giuffre-effort-block-rejected"
req = requests.get(main_url)
soup = BeautifulSoup(req.text, "html.parser")

