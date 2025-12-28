import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


def preprocess(text):
    text_lower = text.lower()

    tokens = nltk.word_tokenize(text_lower)

    return [word for word in tokens if word.isalnum()]

print("\n--- Documentos ---\n")
print(documents)

preprocessed_docs = [" ".join(preprocess(doc)) for doc in documents]
preprocessed_docs

print("\n--- Documentos pré-processados ---\n")
print(preprocessed_docs)

# o mesmo vectorizer e usado no treinamento deve ser usado na consulta para manter a consistência
vectorizer = TfidfVectorizer() 

# o fit_transform ajusta o modelo e transforma os documentos em uma matriz TF-IDF
tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)
tfidf_matrix

print("\n--- Matriz TF-IDF ---\n")
print(tfidf_matrix) # <Compressed Sparse Row sparse matrix of dtype 'float64'
                    #with 139 stored elements and shape (11, 106)>
                    # 11 documentos e 106 termos únicos
                    # 139 elementos armazenados (não nulos), ele sao os termos que podem estar repetidos em varios documentos
                    # Coords sao (documento, termo)
                    # Values sao os pesos TF-IDF
                    # Exemplo: (0, 45)
                    # 0.31622776601683794 significa que no documento 0, o termo 45 tem peso TF-IDF 0.31622776601683794
print(tfidf_matrix.toarray())

# debug - documento X termo
print("\n--- Debug: Documento X Termo ---\n")
terms = vectorizer.get_feature_names_out()
cccc = 0
for doc_index, doc_vector in enumerate(tfidf_matrix.toarray()):
    print(f"\nDocumento {doc_index}:")
    for term_index, weight in enumerate(doc_vector):
        if weight > 0:
            print(f"  Termo: '{terms[term_index]}', Peso TF-IDF: {weight}")
            cccc += 1
print(f"\nTotal de termos com peso TF-IDF > 0: {cccc}\n")

#query = "machine learning modelos"
query = "machine learning"

def search_tfidf(query, vectorizer, tfidf_matrix):
    print(f"\n--- Consulta: {query} ---\n")
    query_vector = vectorizer.transform([query])
    print("\n--- Vetor da consulta ---\n")
    print(query_vector.toarray())
    print(query_vector)
    similarities = cosine_similarity(tfidf_matrix, query_vector).flatten()
    print("\n--- Similaridades calculadas ---\n")
    print(similarities)
    sorted_similarities = list(enumerate(similarities))
    print("\n--- Similaridades enumeradas ---\n")
    print(sorted_similarities)
    results = sorted(sorted_similarities, key=lambda x: x[1], reverse=True)

    return results


search_similarities = search_tfidf(query, vectorizer, tfidf_matrix)
search_similarities

print("\n--- search_similarities ---\n")
print(search_similarities)

print(f"top 10 documentos por score de similaridade {query}:")
for doc_index, score in search_similarities[:10]:
    print(f"documento {doc_index}: {documents[doc_index]}")


# %%
print("\n--- Detalhes do cálculo de similaridade ---\n")
# %%
