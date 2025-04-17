import streamlit as st
import os
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
GET_URL = "http://localhost:8000"

def main():
    st.title("Конспектирование видео")

    if "file_id" not in st.session_state:
        st.session_state.file_id = None

    uploaded_file = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        st.info("Файл успешно загружен и обрабатывается...")
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        response = requests.post(f"{BACKEND_URL}/upload/", files=files)

        if response.status_code == 200:
            file_id = response.json()["file_id"]
            text_md = response.json()["text_md"]
            st.success("Видео успешно загружено!")
            st.session_state.file_id = file_id

            # Отображение видео
            video_url = f"{GET_URL}/files/{file_id}/video.mp4"
            st.video(video_url)

            with st.expander("Показать таймкоды"):
                st.code("#В разработке, остальное работает", language="markdown")
        else:
            st.error(f"Ошибка загрузки файла: загружен файл без корректной аудиодорожки!")
        with st.expander("Показать markdown"):
            st.code(text_md, language="markdown")


    if st.session_state.file_id:
        file_id = st.session_state.file_id

        st.subheader("Ссылки на готовые файлы")
        markdown_url = f"{GET_URL}/files/{file_id}/summary.md"
        word_url = f"{GET_URL}/files/{file_id}/summary.docx"
        # pdf_url = f"{GET_URL}/files/{file_id}/summary.pdf"

        st.markdown(f"- [Скачать docx](<{word_url}>)")
        st.markdown(f"- [Скачать markdown](<{markdown_url}>)")

if __name__ == "__main__":
    main()
