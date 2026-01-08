from groq import Groq

client = Groq()

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "Atue como um especialista em machine learning."},
        {"role": "user", "content": "De forma simples, o que é machine learning?"},
    ],
    temperature=0,
    top_p=1,
)

print("Resposta completa da API:")
print(response.choices)

print("Resposta do modelo:")
print(response.choices[0].message.content)


# Explicacoes:
# 1. from groq import Groq: Importa a classe Groq da biblioteca
# 2. client = Groq(): Cria uma instância do cliente Groq para interagir com a API.
# 3. response = client.chat.completions.create(...): Envia uma solicitação para criar uma conclusão de chat usando o modelo especificado.
# 4. model="llama-3.1-8b-instant": Especifica o modelo de linguagem a ser usado.
# 5. messages=[...]: Define a sequência de mensagens para a conversa, incluindo o papel do sistema e a mensagem do usuário.
# 6. temperature=0, top_p=1: Configura os parâmetros de geração de texto para controlar a criatividade e diversidade da resposta.

# Sobre as Roles
# - system: Define o comportamento do modelo. Aqui, instrui o modelo a atuar como um especialista em machine learning.
# - user: Representa a entrada do usuário.
# - assistant: Representa a resposta gerada pelo modelo.

# Porque preciso separar as mensagens em roles?
# Separar as mensagens em roles ajuda o modelo a entender o contexto e a dinâmica da conversa, permitindo respostas mais relevantes e coerentes.
# se colocar tudo como user, o modelo pode não entender que deve responder como um assistente.

# para executar export $(cat .env | xargs) && uv run main.py

# Entendo Temperature e Top_P
# - Temperature: Controla a aleatoriedade da saída do modelo. Valores mais baixos (próximos de 0) tornam a saída mais determinística e focada, enquanto valores mais altos (próximos de 1) aumentam a criatividade e diversidade das respostas.
# - Top_P: Implementa a amostragem de núcleo, limitando a seleção de palavras ao conjunto mais provável que compõe a probabilidade cumulativa p. Valores mais baixos restringem a seleção a palavras mais prováveis, enquanto valores mais altos permitem uma gama mais ampla de escolhas, promovendo diversidade na saída.
# Ajustar esses parâmetros permite controlar o equilíbrio entre coerência e criatividade nas respostas geradas pelo modelo.

# Em rag temperature alto ou baixo?
# Em RAG (Retrieval-Augmented Generation), geralmente é preferível usar um temperature mais baixo. Isso porque RAG depende de informações recuperadas de fontes externas para gerar respostas precisas e relevantes. Um temperature baixo ajuda a garantir que o modelo se concentre nas informações fornecidas, produz
# do respostas mais consistentes e alinhadas com os dados recuperados.


# max_completion_tokens -> serve para limitar o tamanho da resposta do modelo, garantindo que ele não gere respostas excessivamente longas.