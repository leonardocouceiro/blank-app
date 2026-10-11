"""MVP RAG - consulta a documentos por busca semântica."""
import hashlib

import streamlit as st

from rag_core import (
    CHAT_PROVIDERS,
    EMBEDDING_PROVIDERS,
    MAX_FILES,
    answer,
    build_index,
    retrieve,
)

st.set_page_config(page_title="Laboratório RAG", page_icon="📚", layout="wide")
st.title("📚 Laboratório RAG")
st.caption("Escopo 1 · PDF/TXT/MD → embeddings → recuperação → resposta com fontes")
st.info(
    "Protótipo experimental: use somente documentos públicos ou de teste. "
    "Os trechos são enviados ao serviço de embeddings e ao modelo escolhido."
)


def get_secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


with st.sidebar:
    st.header("Configurações")
    embedding_provider = st.selectbox(
        "Embeddings (busca semântica)", list(EMBEDDING_PROVIDERS)
    )
    chat_provider = st.selectbox("Modelo de resposta", list(CHAT_PROVIDERS))
    default_model = CHAT_PROVIDERS[chat_provider]["model"]
    model = st.text_input("Nome do modelo", value=default_model)
    st.caption("O modelo deve existir na sua conta do provedor.")
    embedding_secret_name = EMBEDDING_PROVIDERS[embedding_provider]
    chat_secret_name = CHAT_PROVIDERS[chat_provider]["secret"]
    embedding_key = get_secret(embedding_secret_name)
    chat_key = get_secret(chat_secret_name)

    st.subheader("Chaves de API")
    if embedding_key:
        st.success(f"{embedding_secret_name}: configurada")
    else:
        st.warning(f"{embedding_secret_name}: ausente")
    if chat_key:
        st.success(f"{chat_secret_name}: configurada")
    else:
        st.warning(f"{chat_secret_name}: ausente")
    st.caption("Configure em Streamlit Community Cloud → App settings → Secrets.")
    if st.button("Limpar documentos e conversa", use_container_width=True):
        for key in ("rag_index", "rag_fingerprint", "rag_history"):
            st.session_state.pop(key, None)
        st.rerun()

files = st.file_uploader(
    f"Envie até {MAX_FILES} documentos",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
    help="Até 8 MB por documento e 80 páginas PDF somadas. Ainda não há OCR.",
)
st.caption(
    "O índice permanece somente nesta sessão e pode desaparecer quando "
    "o aplicativo for reiniciado. Não há armazenamento permanente."
)

fingerprint = None
if files:
    fingerprint = (
        embedding_provider,
        tuple(
            (f.name, hashlib.sha256(f.getvalue()).hexdigest()) for f in files
        ),
    )

if st.session_state.get("rag_fingerprint") != fingerprint:
    st.session_state.pop("rag_index", None)
    st.session_state.pop("rag_history", None)
    st.session_state["rag_fingerprint"] = fingerprint

if st.button(
    "Indexar documentos",
    type="primary",
    disabled=not files or not embedding_key,
    help="A indexação chama a API de embeddings e pode consumir créditos.",
):
    try:
        with st.spinner("Extraindo texto e criando embeddings..."):
            st.session_state["rag_index"] = build_index(
                files, embedding_provider, embedding_key
            )
            st.session_state["rag_history"] = []
        st.success("Documentos indexados com sucesso.")
    except Exception as exc:
        st.error(f"Falha na indexação: {type(exc).__name__}: {exc}")

index = st.session_state.get("rag_index")
if index:
    st.success(
        f"Índice pronto: {len(index['chunks'])} trechos, "
        f"embeddings via {index['embedding_provider']}."
    )
    for warning in index["warnings"]:
        st.warning(warning)
else:
    st.info("Envie documentos, configure a chave de embeddings e clique em Indexar.")

for item in st.session_state.get("rag_history", []):
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.markdown(item["response"])
        with st.expander("Trechos recuperados e fontes"):
            for number, passage in enumerate(item["passages"], start=1):
                page = passage["page"] if passage["page"] is not None else "—"
                st.markdown(
                    f"**[{number}] {passage['filename']} · página {page}** "
                    f"(similaridade {passage['score']:.3f})"
                )
                st.write(passage["text"])

question = st.chat_input(
    "Faça uma pergunta sobre os documentos...",
    disabled=not index or not embedding_key or not chat_key,
)
if question and question.strip():
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Buscando trechos e consultando a IA..."):
                passages = retrieve(index, question, embedding_key)
                response = answer(question, passages, chat_provider, model, chat_key)
            st.markdown(response)
            with st.expander("Trechos recuperados e fontes"):
                for number, passage in enumerate(passages, start=1):
                    page = passage["page"] if passage["page"] is not None else "—"
                    st.markdown(
                        f"**[{number}] {passage['filename']} · página {page}** "
                        f"(similaridade {passage['score']:.3f})"
                    )
                    st.write(passage["text"])
            st.session_state.setdefault("rag_history", []).append(
                {"question": question, "response": response, "passages": passages}
            )
        except Exception as exc:
            st.error(f"Falha ao responder: {type(exc).__name__}: {exc}")
