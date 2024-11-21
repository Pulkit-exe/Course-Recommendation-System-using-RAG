import requests
from bs4 import BeautifulSoup

links = []
for i in range(1, 9):

    url = 'https://courses.analyticsvidhya.com/collections?page=' + str(i)

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    for a_tag in soup.find_all('a', class_='course-card course-card__public published'):

        href = a_tag.get('href')

        if href:
            links.append("https://courses.analyticsvidhya.com" + href)

with open('extracted_links.txt', 'w') as file:
    
    for link in links:
        file.write(link + '\n')


