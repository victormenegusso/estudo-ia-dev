# Fundamentos de Generative AI

## transformers

é composto principalmente por: 
- enconder -> processa a entrada
- decoder -> gera a saída

### input embeddings

Processo que converte tokens em vetores numéricos(Embeddings) que o modelo pode entender.

### Exemplo: your cat is a lovely cat
Tokens: [your, cat, is, a, lovely, cat]
Input IDs: [105, 4242, 6892, 1516, 72, 4242]
Embeddings: [[0.12, 0.45, ...], [0.67, 0.23, ...], ...]

Input IDs são mapeados para embeddings através de uma tabela de embeddings aprendida durante o treinamento, os numeros representam características semânticas dos tokens. ( position on vocabulary, context, etc )

Embeddings são vetores densos que capturam o significado semântico dos tokens, o tamanho do vetor depende do modelo (e.g., 768 para BERT base).

### Positional Encodings

Adiciona informações sobre a posição dos tokens na sequência, já que transformers não possuem uma noção intrínseca de ordem.
Serve para que o modelo entenda a ordem das palavras na frase.

### Multi-Head Self-Attention

Mecanismo que permite ao modelo focar em diferentes partes da sequência de entrada simultaneamente.
Cada "head" aprende a capturar diferentes aspectos das relações entre tokens.
Serve para entender o contexto e as dependências entre palavras, mesmo que estejam distantes na sequência.

### Add & Norm

Combina a saída do mecanismo de atenção com a entrada original através de uma operação de adição (residual connection) seguida por uma normalização.
Serve para estabilizar o treinamento e permitir que o modelo aprenda mais eficientemente.

### Feed-Forward Neural Network
Camada totalmente conectada que processa cada token individualmente após a atenção.
Consiste em duas camadas lineares com uma função de ativação não linear (e.g., ReLU) no meio.
Serve para transformar as representações dos tokens em um espaço mais rico

### Exemplo ludico

Imagine que você está lendo um livro (entrada) e quer entender o significado de uma frase.
1. **Input Embeddings**: Você converte cada palavra da frase em um código secreto (embedding) que representa seu significado.
2. **Positional Encodings**: Você anota a posição de cada palavra na frase para lembrar a ordem correta.
3. **Multi-Head Self-Attention**: Você lê a frase várias vezes, focando em diferentes palavras para entender como elas se relacionam entre si.
4. **Add & Norm**: Você compara o que entendeu com a frase original para garantir que não perdeu nada importante.
5. **Feed-Forward Neural Network**: Você refina seu entendimento, transformando o significado das palavras em algo mais profundo.
6. **Add & Norm**: Você faz mais uma verificação final para garantir que tudo pfaça sentido.
7. **Output Embeddings**: Finalmente, você traduz seu entendimento de volta para palavras que fazem sentido na língua que você está usando.

### Sites

- https://poloclub.github.io/transformer-explainer/ 
- https://www.youtube.com/watch?v=aCWm4eMQlQs


## Exemplos

### Groq

esta em fundamentos_gen_ai/groq-exemplo
