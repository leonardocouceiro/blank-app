# Laboratório RAG — Escopo 1

Protótipo de consulta a documentos com Streamlit, embeddings, recuperação semântica e LLM por API.

## Funcionalidades
- Upload de até três PDFs (texto selecionável), TXT ou Markdown, até 8 MB cada.
- Extração de texto por página usando PyMuPDF (sem OCR neste escopo).
- Divisão em trechos com sobreposição.
- Embeddings por Gemini ou OpenAI e busca por similaridade de cosseno em memória.
- Respostas por DeepSeek, Gemini, OpenAI ou OpenRouter (modelo configurável).
- Trechos e páginas recuperados disponíveis para inspeção.
- Índice temporário por sessão; não há persistência entre reinícios.

## Configurar Secrets no Streamlit Community Cloud
App settings → Secrets. Adicione apenas as chaves necessárias. Exemplo:
```toml
GEMINI_API_KEY = "SUA_CHAVE_GEMINI"
DEEPSEEK_API_KEY = "SUA_CHAVE_DEEPSEEK"
```
Também são aceitas OPENAI_API_KEY e OPENROUTER_API_KEY.
Não envie chaves para o GitHub. Para desenvolvimento local use .streamlit/secrets.toml (ignorado pelo Git).

## Executar localmente
```bash
python -m venv .venv
# Ative a venv conforme o sistema operacional
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Limites e segurança
- Não use documentos confidenciais, pessoais ou institucionais sem autorização.
- Textos são enviados ao serviço de embeddings; trechos recuperados são transmitidos ao serviço de LLM.
- PDFs digitalizados como imagens ainda exigem OCR (escopo 2).
- Respostas e referências devem ser revisadas; modelos podem alucinar citações.
- O índice não é persistente. Reinício exige reindexação.
- Requisições podem gerar custos de API. Se o aplicativo for público, visitantes podem gastar sua franquia.
- No momento não existe autenticação adicional integrada ao aplicativo.
- A recuperação usa os quatro trechos mais similares e pode omitir contexto relevante.
- Os nomes de modelos podem mudar e precisam ser ajustados no seletor.

## Próximos escopos
2. OCR de PDFs digitalizados.
3. Ferramentas, cálculos e chamadas de APIs.
4. PostgreSQL/pgvector para persistência.
5. LangGraph, estado, agentes e aprovação humana.
