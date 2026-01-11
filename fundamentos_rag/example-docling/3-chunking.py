from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker

converter = DocumentConverter()

result = converter.convert("./2408.09869v5.pdf")
document = result.document

chunker = HierarchicalChunker() # esse é o chunker funciona dividindo o documento em pedaços menores através de uma abordagem hierárquica
                                # essa hierarquia pode ser baseada em seções, parágrafos, frases, etc.
chunks = list(chunker.chunk(document))

chunks
print(f"Total chunks created: {len(chunks)}")
print("Exemplo de chunks:")
print("-------------------")
print("Chunk 1:")
print(chunks[0].text)
print("Chunk 2:")
print(chunks[1].text)
print("Chunk 3:")
print(chunks[2].text)
print("Chunk 4:")
print(chunks[3].text)
print("Chunk 5:")
print(chunks[4].text)
