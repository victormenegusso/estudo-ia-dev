from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer # tokenizer baseado na biblioteca HuggingFace
                                                                                       # ele permite usar qualquer modelo de tokenização disponível na HuggingFace, facilitando a integração com diversos modelos de linguagem
                                                                                       # além disso, ele conta os tokens de forma eficiente, o que é crucial para o chunking baseado em tokens
from transformers import AutoTokenizer # 

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # modelo leve e eficiente para criar embeddings de sentenças, mas e mais eficiente em inglês,

MAX_TOKENS = 300 # quantidade máxima de tokens por chunk ( o modelo all-MiniLM-L6-v2 suporta 512 tokens)
                 # temos que balancear entre chunks muito grandes (difícil para o modelo processar) e chunks muito pequenos (pode perder contexto)
                 # tambem tem o limite maximo de tokens do modelo que vamos usar para gerar respostas
                 
                 # ponto importante: o modelo ele aceita 512 tokens e responde um vetor de 384 dimensoes

converter = DocumentConverter()

result = converter.convert("./2408.09869v5.pdf")
document = result.document

tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL), max_tokens=MAX_TOKENS
)

print(f"Using tokenizer from model: {EMBED_MODEL}")
print(f"Max tokens per chunk: {MAX_TOKENS}")
print(f"Model max length: {tokenizer.tokenizer.model_max_length}") 

chunker = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS, merge_peers=True) # merge_peers=True vai tentar combinar chunks menores em chunks maiores, desde que nao ultrapassem o limite de tokens

chunks = list(chunker.chunk(document))
print(f"Number of chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"==={i}===\n")
    txt_tokens = tokenizer.count_tokens(chunk.text)
    print(f"chunk.text ({txt_tokens} tokens):\n{chunk.text!r}")
    print(chunk)

    
print(chunks[4].meta.doc_items[0].prov[0].page_no)
print(chunks[4].meta.headings)
