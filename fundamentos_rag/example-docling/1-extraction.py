from docling.document_converter import DocumentConverter
print("início")
converter = DocumentConverter()

print("convertendo PDF")
result = converter.convert("./2408.09869v5.pdf")
#result = converter.convert("https://arxiv.org/pdf/2408.09869")

document = result.document

markdown_output = document.export_to_markdown()
print("Markdown output from PDF:")
print(markdown_output)

# convertendo página HTML
result = converter.convert("https://docling-project.github.io/docling/")
document = result.document
markdown_output = document.export_to_markdown()
print("Markdown output from HTML:")
print(markdown_output)
print("fim")