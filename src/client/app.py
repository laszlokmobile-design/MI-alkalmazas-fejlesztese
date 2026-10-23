import uuid
import streamlit as st
import requests
import base64

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="MI RAG Rendszer (Groq LPU)", page_icon="⚡", layout="centered")

st.title("⚡ Villámgyors RAG Rendszer Groq LPU-val")

# 5 fül definiálása, beleértve a Waldo Keresőt is
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📚 Dokumentum Feltöltés",
    "💬 Chat Memóriával",
    "🗂️ Adatbázis",
    "👁️ Gépi Látás (Vision)",
    "🎯 Waldo Kereső (Demo)"
])

with tab1:
    st.header("Új ismeret betáplálása")
    doc_text = st.text_area("Másolj be ide egy szakmai szöveget:", height=200)

    if st.button("Mentés a Tudásbázisba", type="primary"):
        if not doc_text.strip():
            st.warning("Kérlek, írj be valamilyen szöveget!")
        else:
            with st.spinner('Feldolgozás...'):
                try:
                    response = requests.post(f"{API_BASE_URL}/add_document", json={"text": doc_text})
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ {data.get('message')}")
                    else:
                        st.error(f"Hiba a szerveren: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Nem tudok csatlakozni a FastAPI backendhez.")

with tab2:
    st.header("Tudásalapú Kérdezz-Felelek & Chat")
    st.caption(f"Aktuális Session ID: `{st.session_state.get('session_id', 'Nincs')}`")

    user_query = st.text_input("Kérdezz a feltöltött dokumentumokból (RAG):")
    if st.button("RAG Válasz 🚀"):
        if user_query.strip():
            with st.spinner('Keresés és válasz generálása Groq-kal...'):
                try:
                    res = requests.post(f"{API_BASE_URL}/ask_rag", json={"prompt": user_query})
                    if res.status_code == 200:
                        data = res.json()
                        st.markdown("### 🤖 MI Válasza:")
                        st.write(data.get("llm_response"))
                        with st.expander("🔍 Felhasznált kontextus"):
                            st.write(data.get("retrieved_context"))
                    else:
                        st.error(res.text)
                except Exception as e:
                    st.error(f"Hiba: {e}")

    st.divider()

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Beszélgess a memóriával rendelkező asszisztenssel..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Groq generálás..."):
                try:
                    res = requests.post(
                        f"{API_BASE_URL}/chat",
                        json={"session_id": st.session_state.session_id, "prompt": prompt}
                    )
                    if res.status_code == 200:
                        reply = res.json().get("response")
                        st.markdown(reply)
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    else:
                        st.error(res.text)
                except Exception as e:
                    st.error(f"Hiba: {e}")

with tab3:
    st.header("Feltöltött dokumentumok")
    if st.button("Lista frissítése"):
        try:
            res = requests.get(f"{API_BASE_URL}/list_documents")
            if res.status_code == 200:
                data = res.json()
                st.metric("Összes dokumentum", data.get("total_documents"))
                for doc in data.get("documents", []):
                    st.text_area(f"ID: {doc['id']}", doc['text'], height=100, disabled=True)
        except Exception:
            st.error("Szerver hiba.")

with tab4:
    st.header("👁️ Llama 3.2 Vision Képelemzés")
    uploaded_file = st.file_uploader("Válassz egy képet...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Feltöltött kép", width='stretch')
        vision_prompt = st.text_input("Mit szeretnél tudni a képről?",
                                      value="Írd le részletesen, mi látható ezen a képen.")

        if st.button("Kép Elemzése Groq Vision-nel 🚀", type="primary"):
            with st.spinner('A Llama 3.2 Vision elemzi a képet...'):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {"session_id": st.session_state.session_id, "prompt": vision_prompt}

                    res = requests.post(f"{API_BASE_URL}/chat_with_image", files=files, data=data)

                    if res.status_code == 200:
                        st.success("Sikeres elemzés!")
                        st.info(res.json().get("response"))
                    else:
                        st.error(res.text)
                except Exception as e:
                    st.error(f"Kapcsolódási hiba: {e}")

with tab5:
    st.header("🎯 Speciális Waldo Kereső Modul")
    st.caption("Válassz az alábbi előre betöltött képek közül, hogy mit küldjünk el az MI-nek:")

    # Választás a három kép között
    image_choice = st.radio(
        "Válassz képet a küldéshez:",
        [
            "🔍 Rejtvénykép (waldo_puzzle.jpg)",
            "👤 Keresendő karakter (waldo_character.jpg)",
            "✅ Megoldás / Bekarikázott kép (waldo_solution.jpg)",
            "📁 Saját kép feltöltése"
        ]
    )

    waldo_file_bytes = None
    waldo_filename = "waldo_puzzle.jpg"

    # A kiválasztott kép betöltése
    if image_choice == "🔍 Rejtvénykép (waldo_puzzle.jpg)":
        waldo_filename = "waldo_puzzle.jpg"
        try:
            with open(waldo_filename, "rb") as f:
                waldo_file_bytes = f.read()
            st.image(waldo_filename, caption="Kiválasztva küldésre: Rejtvénykép", width='stretch')
        except FileNotFoundError:
            st.error(f"❌ A '{waldo_filename}' nem található a gyökérmappában!")

    elif image_choice == "👤 Keresendő karakter (waldo_character.jpg)":
        waldo_filename = "waldo_character.jpg"
        try:
            with open(waldo_filename, "rb") as f:
                waldo_file_bytes = f.read()
            st.image(waldo_filename, caption="Kiválasztva küldésre: Keresendő karakter", width='stretch')
        except FileNotFoundError:
            st.error(f"❌ A '{waldo_filename}' nem található a gyökérmappában!")

    elif image_choice == "✅ Megoldás / Bekarikázott kép (waldo_solution.jpg)":
        waldo_filename = "waldo_solution.jpg"
        try:
            with open(waldo_filename, "rb") as f:
                waldo_file_bytes = f.read()
            st.image(waldo_filename, caption="Kiválasztva küldésre: Megoldás kép", width='stretch')
        except FileNotFoundError:
            st.error(f"❌ A '{waldo_filename}' nem található a gyökérmappában!")

    else:
        uploaded_custom = st.file_uploader("Töltsd fel a saját képed:", type=["jpg", "jpeg", "png"])
        if uploaded_custom is not None:
            waldo_file_bytes = uploaded_custom.getvalue()
            waldo_filename = uploaded_custom.name
            st.image(uploaded_custom, caption="Feltöltött saját kép", width='stretch')

    # Ha van érvényes kép, jöhet a prompt és a küldés
    if waldo_file_bytes is not None:
        default_hint = "Waldonak fekete szemüvege van, piros-fehér csíkos sapkája bojttal, és egy bot a kezében."
        hint_input = st.text_area("Súgás vagy kérdés az MI-nek ehhez a képhez:", value=default_hint)

        if st.button("🚀 Kép Küldése az MI-nek", type="primary"):
            with st.spinner('A Groq LPU elemzi a kiválasztott képet...'):
                try:
                    waldo_prompt = (
                        f"Kérlek, válaszolj magyarul. "
                        f"A felhasználó ezt a képet küldte: {waldo_filename}. "
                        f"Feladat / Súgás: {hint_input}. "
                        f"Elemezd a képet ennek megfelelően!"
                    )

                    files = {"file": (waldo_filename, waldo_file_bytes, "image/jpeg")}
                    data = {"session_id": st.session_state.session_id, "prompt": waldo_prompt}

                    # Itt hívjuk a backend végpontot
                    res = requests.post(f"{API_BASE_URL}/waldo_search", files=files, data=data)

                    if res.status_code == 200:
                        res_data = res.json()
                        st.success("Sikeres elemzés!")
                        st.markdown("### 🤖 MI Válasza:")
                        st.info(res_data.get("response"))

                        if "annotated_image" in res_data:
                            annotated_bytes = base64.b64decode(res_data["annotated_image"])
                            st.image(annotated_bytes, caption="🎯 Eredmény (Backend által feldolgozva/bekarikázva)", width='stretch')
                    else:
                        st.error(res.text)
                except Exception as e:
                    st.error(f"Hiba történt: {e}")
