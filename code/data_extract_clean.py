import unicodedata
import contractions
import string
from nltk.stem.porter import PorterStemmer
import bs4 as bs
import re


def clean_text(text_to_clean):
    """
    Removes text punctuation, makes all lower case, expands contractions
    """
    clean_text = unicodedata.normalize('NFKD', text_to_clean).encode('ascii', 'ignore').decode('utf8')
    clean_text = clean_text.lower()
    clean_text = re.sub('\w*\d\w*', '', clean_text)
    clean_text = contractions.fix(clean_text)
    clean_text = clean_text.replace("’", "")
    clean_text = clean_text.translate(str.maketrans('', '', string.punctuation))
    return clean_text


def stem_tokenizer(text_to_stem):
    """
    Tokenizer including PorterStemmer to feed into Vectorizer
    """
    porter_stemmer = PorterStemmer()
    words = text_to_stem.split()
    words = [porter_stemmer.stem(word) for word in words]
    return words


def get_links(response):
    soup = bs.BeautifulSoup(response.text,'lxml')
    links = []
    for div in soup.find_all('div', {'class':'series-container archive-section subgroup series-book'}):
        for link in div.find_all('a', href=True):
            links.append(link['href'])
    return links


def find_text(links, link, soup_string):
    pattern = re.compile(r'Commentary')
    text_list = []
    current = soup_string.parent.parent
    title = current.find_previous('strong').text
    while not pattern.search(current.text):
        text_list.append(current.text)
        if link == links[26]:
            if current.parent.name == 'blockquote':
                current = current.parent.find_next_sibling()
            else:
                current = current.find_next_sibling()
        elif current.parent.name == 'span':
            current = current.find_parent('p').find_next_sibling()
        else:
            current = current.find_next_sibling()
    return [title, ' '.join(text_list).replace('\n','').replace('What Happens','')]