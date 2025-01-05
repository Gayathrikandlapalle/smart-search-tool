import requests
from bs4 import BeautifulSoup

url = 'https://www.analyticsvidhya.com/courses/free-courses/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

courses = []
for course in soup.find_all('div', class_='course-card'):
    title = course.find('h3').text if course.find('h3') else 'No title found'
    description = course.find('p').text if course.find('p') else 'No description found'
    courses.append({'title': title, 'description': description})

print(courses)

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts([course['description'] for course in courses], embeddings)

def search_courses(query):
    results = vector_store.similarity_search(query)
    return results

query = "data science"
print(search_courses(query))

import gradio as gr

def search_interface(query):
    return search_courses(query)

iface = gr.Interface(fn=search_interface, inputs="text", outputs="text")
iface.launch()
