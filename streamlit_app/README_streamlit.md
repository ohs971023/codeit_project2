# Streamlit RAG Demo (전처리 산출물 우선)

## 실행
```bash
python -m venv .venv_streamlit
source .venv_streamlit/bin/activate

pip install -r streamlit_app/requirements_streamlit.txt
python -m pip install faiss-cpu

streamlit run streamlit_app/app.py