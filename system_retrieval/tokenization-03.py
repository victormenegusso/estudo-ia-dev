import os
import nltk
import shutil
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser

import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

#nltk.download('stopwords')

documents = [
    "Machine learning é um campo da inteligência artificial que permite que computadores aprendam padrões a partir de dados.",
    "O aprendizado de máquina dá aos sistemas a capacidade de melhorar seu desempenho sem serem explicitamente programados.",
    "Em vez de seguir apenas regras fixas, o machine learning descobre relações escondidas nos dados.",
    "Esse campo combina estatística, algoritmos e poder computacional para extrair conhecimento.",
    "O objetivo é criar modelos capazes de generalizar além dos exemplos vistos no treinamento.",
    "Aplicações de machine learning vão desde recomendações de filmes até diagnósticos médicos.",
    "Os algoritmos de aprendizado de máquina transformam dados brutos em previsões úteis.",
    "Diferente de um software tradicional, o ML adapta-se conforme novos dados chegam.",
    "O aprendizado pode ser supervisionado, não supervisionado ou por reforço, dependendo do tipo de problema.",
    "Na prática, machine learning é o motor que impulsiona muitos avanços em visão computacional e processamento de linguagem natural.",
    "Mais do que encontrar padrões, o machine learning ajuda a tomar decisões baseadas em evidências.",
]

print("\n--- Documentos ---\n")
print(documents)

def preprocess(text):
    print("\n--- Pré-processamento ---\n")
    print("Texto original:", text)
    text_lower = text.lower()
    tokens = nltk.word_tokenize(text_lower)
    print("Tokens antes da remoção de stop words:", tokens)
    tokens = [word for word in tokens if word.isalnum()]
    print("Tokens após remoção de pontuação:", tokens)
    stopwords = set(nltk.corpus.stopwords.words("portuguese")) - {"e", "ou", "não"}
    print("Stop words removidas:", stopwords)
    tokens = [word for word in tokens if word not in stopwords]
    print("Tokens após remoção de stop words:", tokens)
    return tokens


text = "Machine learning é um campo da inteligência artificial. que permite que computadores aprendam padrões a partir de dados."
preprocess(text)

if os.path.exists("index_dir"):
    shutil.rmtree("index_dir")
os.mkdir("index_dir")

schema = Schema(title=ID(stored=True, unique=True), content=TEXT(stored=True))

index = create_in("index_dir", schema)

writer = index.writer()
for i, doc in enumerate(documents):
    print(f"\n--- Adicionando documento {i} ---\n")
    print(doc)  
    writer.add_document(title=str(i), content=doc)
writer.commit()

query = "machine E learning"


def boolean_search(query, index):
    print(f"\n--- Consulta booleana: {query} ---\n")
    parser = QueryParser("content", schema=index.schema)
    parsed_query = parser.parse(query)
    print("Consulta parseada:", parsed_query)

    with index.searcher() as searcher:
        print("\n--- Resultados da busca ---\n")        
        results = searcher.search(parsed_query)
        print(f"Número de resultados encontrados: {len(results)}")
        for hit in results:
            print(f"Documento ID: {hit['title']}, Conteúdo: {hit['content']}")
        return [(hit["title"], hit["content"]) for hit in results]


boolean_search(query, index)
