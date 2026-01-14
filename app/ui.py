import streamlit as st
import requests

st.set_page_config(page_title="Notes RAG", layout="wide")
st.title("Notes RAG")

api_url = st.sidebar.text_input("API URL", "http://127.0.0.1:8000/ask")
strict = st.sidebar.checkbox("Строгий режим", value=True)
mode = st.sidebar.selectbox("Режим", ["llm", "extractive"], index=0)

q = st.text_input("Вопрос", placeholder="Например: Что такое serializable?")

if st.button("Спросить") and q.strip():
    resp = requests.post(api_url, json={"query": q, "strict": strict, "mode": mode}, timeout=120)
    # st.write(f"HTTP {resp.status_code}")
    # st.text(resp.text[:2000])

    try:
        data = resp.json()
    except Exception as e:
        st.error(f"Response is not JSON: {e}")
        st.stop()


    st.subheader("Ответ")
    st.write(data["answer"])

    st.subheader("Цитаты")
    for c in data.get("citations", []):
        st.markdown(f"- **{c['source_file']}** — `{c.get('header_path','')}`  \n  `{c.get('chunk_id','')}`")

    with st.expander("Показать найденные фрагменты (top-5)"):
        for r in data.get("retrieved", []):
            meta = r["meta"]
            st.markdown(f"**Score:** {r['score']:.3f} | **File:** {meta.get('source_file')} | **Header:** {' > '.join(meta.get('header_path', []))}")
            st.code(r["text"][:2000])
