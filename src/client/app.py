import uuid
import streamlit as st
import requests
import base64
import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="MI RAG Rendszer (Groq LPU)", page_icon="⚡", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("⚡ Villámgyors RAG Rendszer Groq LPU-val")

# BAL OLDALI SÁV (Csak a Session ID és infók maradnak itt)
st.sidebar.title("🧭 Információ")
st.sidebar.divider()
st.sidebar.caption(f"Session ID:\n`{st.session_state.get('session_id', 'Nincs')}`")

# --- FŐOLDALI FÜLEK (5 fős struktúra) ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📚 Dokumentum Feltöltés",
    "💬 Chat Memóriával",
    "🗂️ Adatbázis",
    "👁️ Gépi Látás & Képvágás",
    "🎯 Waldo Kereső"
])

# --- TAB 1: Dokumentum Feltöltés (Szöveges + Képi RAG) ---
with tab1:
    st.header("Új ismeret betáplálása a Tudásbázisba")

    rag_sub_tab1, rag_sub_tab2 = st.tabs(["📝 Szöveges Dokumentum", "🖼️ Kép Alapú RAG"])

    with rag_sub_tab1:
        st.subheader("Szöveg mentése")
        doc_text = st.text_area("Másolj be ide egy szakmai szöveget:", height=200, key="doc_text_input")

        if st.button("Szöveg Mentése a Tudásbázisba", type="primary", key="save_doc_btn"):
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

    with rag_sub_tab2:
        st.subheader("Kép alapú RAG (Multimodal RAG)")
        uploaded_rag_image = st.file_uploader("Válassz ki egy JPG képet a tudásbázishoz:", type=["jpg", "jpeg"],
                                              key="rag_image_uploader")
        rag_image_description = st.text_area("Opcionális saját leírás/kulcsszavak:", height=100,
                                             key="rag_image_desc_input")

        if st.button("Kép Mentése a Tudásbázisba 🚀", type="primary", key="save_rag_image_btn"):
            if uploaded_rag_image is None:
                st.warning("Kérlek, tölts fel egy képet!")
            else:
                try:
                    files = {"file": (uploaded_rag_image.name, uploaded_rag_image.getvalue(),
                                      uploaded_rag_image.type or "image/jpeg")}
                    data = {"description": rag_image_description}
                    response = requests.post(f"{API_BASE_URL}/add_image_rag", files=files, data=data)
                    if response.status_code == 200:
                        res_data = response.json()
                        st.success(f"✅ {res_data.get('message')}")
                        st.image(uploaded_rag_image, caption=f"Feltöltve: {res_data.get('filename')}", width=300)
                        with st.expander("📝 Generált / Mentett kontextus"):
                            st.write(res_data.get("generated_description"))
                    else:
                        st.error(response.text)
                except Exception as e:
                    st.error(f"Hiba: {e}")

# --- TAB 2: Chat Memóriával ---
with tab2:
    st.header("Tudásalapú Kérdezz-Felelek & Chat")

    user_query = st.text_input("Kérdezz a feltöltött dokumentumokból (RAG):", key="chat_rag_query")
    if st.button("RAG Válasz 🚀", key="chat_rag_btn"):
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

# --- TAB 3: Adatbázis ---
with tab3:
    st.header("Feltöltött dokumentumok és képek")
    if st.button("Lista frissítése", key="refresh_db_btn"):
        try:
            res = requests.get(f"{API_BASE_URL}/list_documents")
            if res.status_code == 200:
                data = res.json()
                st.metric("Összes dokumentum/elem", data.get("total_documents"))
                for doc in data.get("documents", []):
                    st.text_area(f"ID: {doc['id']}", doc['text'], height=100, disabled=True)
        except Exception:
            st.error("Szerver hiba.")

# --- TAB 4: Gépi Látás és Képvágás (Összevont modul) ---
with tab4:
    st.header("👁️ Gépi Látás & Objektumdetektálás")

    vision_sub_tab1, vision_sub_tab2 = st.tabs(["👁️ Általános Képelemzés", "✂️ Objektumdetektálás & Kivágás"])

    with vision_sub_tab1:
        st.subheader("Qwen Vision Többképes Elemzés")
        uploaded_files = st.file_uploader(
            "Válassz ki egy vagy több képet...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="vision_multi_uploader"
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.image(uploaded_file, caption=f"Feltöltve: {uploaded_file.name}", width=300)

            vision_prompt = st.text_input(
                "Mit szeretnél tudni ezekről a képekről?",
                value="Hasonlítsd össze ezeket a képeket, vagy írd le, mik láthatók rajtuk.",
                key="multi_vision_prompt",
            )

            if st.button("Képek Elemzése Groq Vision-nel 🚀", type="primary", key="vision_analyze_btn"):
                with st.spinner("A multimodális modell elemzi a feltöltött képeket..."):
                    try:
                        files = [
                            ("files", (file.name, file.getvalue(), file.type or "image/jpeg"))
                            for file in uploaded_files
                        ]
                        data = {
                            "session_id": st.session_state.session_id,
                            "prompt": vision_prompt,
                        }

                        res = requests.post(f"{API_BASE_URL}/chat_with_image", files=files, data=data)

                        if res.status_code == 200:
                            st.success("Sikeres elemzés!")
                            st.info(res.json().get("response"))
                        else:
                            st.error(res.text)
                    except Exception as e:
                        st.error(f"Kapcsolódási hiba: {e}")

    with vision_sub_tab2:
        st.subheader("Interaktív Objektumdetektálás & Automatikus Kivágás")
        uploaded_crop_file = st.file_uploader("Válassz ki egy JPG képet elemzésre:", type=["jpg", "jpeg"],
                                              key="crop_image_uploader")
        detection_query = st.text_input("Mit keressek a képen?", value="Pl. az óra a falon",
                                        key="detection_query_input")

        if uploaded_crop_file is not None:
            st.image(uploaded_crop_file, caption="Feltöltött eredeti kép", width=400)
            if st.button("Keresés és Kivágás indítása 🚀", type="primary", key="run_detection_btn"):
                if not detection_query.strip():
                    st.warning("Add meg, mit keresel!")
                else:
                    try:
                        files = {"file": (uploaded_crop_file.name, uploaded_crop_file.getvalue(),
                                          uploaded_crop_file.type or "image/jpeg")}
                        data = {"session_id": st.session_state.session_id, "prompt": detection_query}
                        res = requests.post(f"{API_BASE_URL}/detect_and_crop", files=files, data=data)
                        if res.status_code == 200:
                            res_data = res.json()
                            st.info(res_data.get("response"))
                            if res_data.get("found", False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    if res_data.get("annotated_image"):
                                        st.image(base64.b64decode(res_data.get("annotated_image")),
                                                 caption="Berajzolva")
                                with col2:
                                    if res_data.get("cropped_image"):
                                        st.image(base64.b64decode(res_data.get("cropped_image")), caption="Kivágva")
                        else:
                            st.error(res.text)
                    except Exception as e:
                        st.error(f"Hiba: {e}")

# --- TAB 5: Waldo Kereső (Demo) ---
with tab5:
    st.header("🎯 Speciális Waldo Kereső Modul")
    with st.expander("💡 Tájékoztatás az MI modell működéséről és korlátairól", expanded=True):
        st.markdown("""
        ### 💡 Miért tévedhet az MI? (A multimodális modellek korlátai)

        Az olyan nagyméretű, multimodális modellek, mint a Qwen, általános feladatokra lettek optimalizálva, és **nincsenek feltétlenül feltanítva az apró, rejtett objektumok milliméter pontos (pixel-szintű) lokalizációjára.** Emiatt a következő jelenségek fordulhatnak elő:

        * 🔀 **A szöveg és a számok külön utakon járnak:** 
          A modell belső logikája szövegesen kitalálja a történetet (hogy hol van a figura), de a számítógépnek adandó koordinátákat egy teljesen más, statisztikai becslés alapján generálja. Így előfordulhat, hogy a válaszban okos dolgot ír, de a számok máshova mutatnak.

        * 🎨 **Szín- és minta-összezavarodás (Vizuális tévedés):** 
          Mivel a modell a zsúfolt képeken egyszerre próbálja figyelni a formákat és a színeket, gyakran előfordulhat az a hiba, hogy **a piros-fehér ruhás Waldo helyett egy kék ruhás vagy más színű figurát jelöl be**. Ennek az az oka, hogy a vizuális modul a keresett formát (pl. a sapkát vagy a csíkokat) összetéveszti a környezetében lévő más, kontrasztos figurákkal.
        """)

    image_choice = st.radio(
        "Válassz képet a küldéshez:",
        [
            "🔍 Rejtvénykép (waldo_puzzle.jpg)",
            "👤 Keresendő karakter (waldo_character.jpg)",
            "✅ Megoldás / Bekarikázott kép (waldo_solution.jpg)",
            "📁 Saját kép feltöltése"
        ],
        key="waldo_image_choice"
    )

    waldo_file_bytes = None
    waldo_filename = "waldo_puzzle.jpg"

    if image_choice == "🔍 Rejtvénykép (waldo_puzzle.jpg)":
        waldo_filename = "waldo_puzzle.jpg"
        try:
            with open(waldo_filename, "rb") as f:
                waldo_file_bytes = f.read()
            st.image(waldo_filename, caption="Kiválasztva küldésre: Rejtvénykép")
        except FileNotFoundError:
            st.error(f"❌ A '{waldo_filename}' nem található a gyökérmappában!")
    elif image_choice == "👤 Keresendő karakter (waldo_character.jpg)":
        waldo_filename = "waldo_character.jpg"
        try:
            with open(waldo_filename, "rb") as f:
                waldo_file_bytes = f.read()
            st.image(waldo_filename, caption="Kiválasztva küldésre: Keresendő karakter")
        except FileNotFoundError:
            st.error(f"❌ A '{waldo_filename}' nem található a gyökérmappában!")
    elif image_choice == "✅ Megoldás / Bekarikázott kép (waldo_solution.jpg)":
        waldo_filename = "waldo_solution.jpg"
        try:
            with open(waldo_filename, "rb") as f:
                waldo_file_bytes = f.read()
            st.image(waldo_filename, caption="Kiválasztva küldésre: Megoldás kép")
        except FileNotFoundError:
            st.error(f"❌ A '{waldo_filename}' nem található a gyökérmappában!")
    else:
        uploaded_custom = st.file_uploader("Töltsd fel a saját képed:", type=["jpg", "jpeg", "png"],
                                           key="waldo_custom_uploader")
        if uploaded_custom is not None:
            waldo_file_bytes = uploaded_custom.getvalue()
            waldo_filename = uploaded_custom.name
            st.image(uploaded_custom, caption="Feltöltött saját kép")

    if waldo_file_bytes is not None:
        default_hint = "Waldonak barna haja, fekete szemüvege van, piros-fehér sapkája piros bojttal. Illetve piros-fehér színű pulóvert hord."
        hint_input = st.text_area("Súgás vagy kérdés az MI-nek ehhez a képhez:", value=default_hint,
                                  key="waldo_hint_input")

        if st.button("🚀 Kép Küldése az MI-nek", type="primary", key="waldo_submit_btn"):
            waldo_prompt = f"Kérlek, válaszolj magyarul. A felhasználó ezt a képet küldte: {waldo_filename}. Feladat / Súgás: {hint_input}."
            mime_type, _ = mimetypes.guess_type(waldo_filename)
            mime_type = mime_type or "image/jpeg"

            files = {"file": (waldo_filename, waldo_file_bytes, mime_type)}
            data = {"session_id": st.session_state.session_id, "prompt": waldo_prompt}

            session_id = st.session_state.session_id
            status_box = st.status("A Waldo keresése folyamatban...", expanded=True)
            progress_bar = st.progress(0)
            progress_text = st.empty()

            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        requests.post,
                        f"{API_BASE_URL}/waldo_search",
                        files=files,
                        data=data,
                        timeout=300
                    )

                    while not future.done():
                        try:
                            progress_res = requests.get(f"{API_BASE_URL}/waldo_progress/{session_id}", timeout=5)
                            if progress_res.status_code == 200:
                                progress_data = progress_res.json()
                                current = progress_data.get("current", 0)
                                total = max(progress_data.get("total", 9), 1)
                                message = progress_data.get("message", "A keresés folyamatban...")
                                progress_bar.progress(min(current / total, 1.0))
                                progress_text.write(message)
                        except requests.RequestException:
                            progress_text.write("A keresés fut...")
                        time.sleep(0.5)

                    res = future.result()

                progress_bar.progress(1.0)
                if res.status_code == 200:
                    res_data = res.json()
                    status_box.update(label="A keresés befejeződött.", state="complete", expanded=False)
                    st.success("A modell lefutott.")
                    st.info(res_data.get("response", "Nincs szöveges válasz."))
                    annotated_image = res_data.get("annotated_image")
                    if annotated_image:
                        st.image(base64.b64decode(annotated_image), caption="🎯 Találat")
                else:
                    status_box.update(label="Hiba történt.", state="error")
                    st.error(res.text)
            except Exception as e:
                status_box.update(label="Hiba történt.", state="error")
                st.error(f"Hiba: {e}")
