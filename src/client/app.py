import uuid
import streamlit as st
import requests
import base64
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor
from streamlit_agraph import agraph, Node, Edge, Config

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="MI RAG Rendszer (Groq LPU)", page_icon="⚡", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("⚡ Villámgyors RAG Rendszer Groq LPU-val")

# BAL OLDALI SÁV
st.sidebar.title("🧭 Információ")
st.sidebar.divider()
st.sidebar.caption(f"Session ID:\n`{st.session_state.get('session_id', 'Nincs')}`")

# --- FŐOLDALI FÜLEK ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📚 Dokumentum Feltöltés",
    "💬 Chat Memóriával",
    "🗂️ Adatbázis",
    "👁️ Gépi Látás & Képvágás",
    "🎯 Waldo Kereső",
    "🕸️ Tudásgráf (GraphRAG)"
])

# --- TAB 1: Dokumentum Feltöltés ---
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
                            if data:
                                st.success(f"✅ {data.get('message', 'Siker')}")
                            else:
                                st.success("✅ Dokumentum sikeresen elmentve.")
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
                st.warning("Kérlek, tölts fel egy .jpg vagy .jpeg kiterjesztésű képet!")
            else:
                try:
                    files = {"files": (uploaded_rag_image.name, uploaded_rag_image.getvalue(),
                                       uploaded_rag_image.type or "image/jpeg")}
                    data = {"description": rag_image_description}
                    response = requests.post(f"{API_BASE_URL}/add_image_rag", files=files, data=data)
                    if response.status_code == 200:
                        res_data = response.json()
                        if res_data:
                            st.success(f"✅ {res_data.get('message', 'Siker')}")
                            st.image(uploaded_rag_image,
                                     caption=f"Feltöltve: {res_data.get('filename', uploaded_rag_image.name)}",
                                     width=300)
                            with st.expander("📝 Generált / Mentett kontextus"):
                                st.write(res_data.get("generated_description", ""))
                        else:
                            st.success("✅ Kép sikeresen rögzítve.")
                    else:
                        st.error(response.text)
                except Exception as e:
                    st.error(f"Hiba: {e}")

# --- TAB 2: Chat Memóriával & MongoDB ---
with tab2:
    st.header("💬 Chat, RAG és Adatbázis Napló")

    # Alfülek létrehozása a Chat fülön belül
    chat_sub_tab1, chat_sub_tab2 = st.tabs(["💬 Élő Chat & RAG", "🗄️ MongoDB Üzenetek (Admin)"])

    # --- 2.1 Alfüzet: Élő Chat & RAG ---
    with chat_sub_tab1:
        st.subheader("Tudásalapú Kérdezz-Felelek & Chat")

        user_query = st.text_input("Kérdezz a feltöltött dokumentumokból (RAG):", key="chat_rag_query")
        if st.button("RAG Válasz 🚀", key="chat_rag_btn"):
            if user_query.strip():
                with st.spinner('Keresés, képek betöltése és válasz generálása Groq-kal...'):
                    try:
                        res = requests.post(f"{API_BASE_URL}/ask_rag", json={"prompt": user_query})
                        if res.status_code == 200:
                            data = res.json()
                            if data:
                                st.markdown("### 🤖 MI Válasza:")
                                st.write(data.get("llm_response", "Nincs válasz."))

                                matched_images = data.get("matched_images", [])
                                if matched_images:
                                    st.markdown("##### 🖼️ A válaszhoz felhasznált kép(ek):")
                                    cols = st.columns(len(matched_images))
                                    for idx, img_filename in enumerate(matched_images):
                                        with cols[idx]:
                                            img_url = f"{API_BASE_URL}/get_image/{img_filename}"
                                            st.image(img_url, caption=img_filename, width=250)

                                with st.expander("🔍 Felhasznált szöveges kontextus"):
                                    st.write(data.get("retrieved_context", ""))
                            else:
                                st.warning("Üres választ adott a szerver.")
                        else:
                            st.error(res.text)
                    except Exception as e:
                        st.error(f"Hiba: {e}")

        st.divider()

        # 1. ELŐSZÖR kirajzoljuk az eddigi üzeneteket a history-ból
        for msg in st.session_state.chat_history:
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                st.markdown(msg["content"])

        # Hagyományos szövegmező és gomb a válaszok alatt
        with st.form(key="chat_form", clear_on_submit=True):
            prompt = st.text_input("Beszélgess a memóriával rendelkező asszisztenssel...",
                                   placeholder="Írd ide az üzenetet...")
            submit_button = st.form_submit_button("Küldés 🚀")

        if submit_button and prompt.strip():
            # Felhasználói üzenet megjelenítése
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.spinner("Groq generálás..."):
                try:
                    res = requests.post(
                        f"{API_BASE_URL}/chat",
                        json={"session_id": st.session_state.session_id, "prompt": prompt}
                    )
                    if res.status_code == 200:
                        data = res.json()
                        reply = data.get("response", "") if data else "Nincs válasz."
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                        st.rerun()  # Újratöltés, hogy látszódjon az új üzenet
                    else:
                        st.error(res.text)
                except Exception as e:
                    st.error(f"Hiba: {e}")

        # --- 2.2 Alfüzet: MongoDB Üzenetek (Admin Nézet) ---
        with chat_sub_tab2:
            st.subheader("🗄️ MongoDB Adatbázis - Chat Előzmények")
            st.markdown("Itt ellenőrizheted valós időben a MongoDB-be mentett üzeneteket.")

            # Opció 1: Összes üzenet lekérése az adatbázisból
            if st.button("🌐 Összes üzenet lekérdezése a DB-ből (Session független)", key="fetch_all_chats_btn"):
                try:
                    res = requests.get(f"{API_BASE_URL}/get_all_chats")
                    if res.status_code == 200:
                        data = res.json()
                        messages = data.get("messages", [])
                        if messages:
                            st.success(f"Összesen {len(messages)} üzenet található a MongoDB-ben!")
                            st.json(messages)
                        else:
                            st.warning("A MongoDB adatbázis teljesen üres.")
                    else:
                        st.error(res.text)
                except Exception as e:
                    st.error(f"Hiba: {e}")

            st.divider()

            # Opció 2: Lenyíló lista a meglévő Session ID-k alapján történő szűréshez
            st.subheader("🔍 Szűrés meglévő Session ID alapján")

            try:
                # Lekérjük az egyedi sessionöket a backendről
                sess_response = requests.get(f"{API_BASE_URL}/get_sessions")
                if sess_response.status_code == 200:
                    sessions = sess_response.json().get("sessions", [])

                    if sessions:
                        # Kiválasztó legördülő menü a véletlenszerű beírás helyett
                        selected_session_id = st.selectbox(
                            "Válassz egy létező Session ID-t:",
                            options=sessions,
                            key="inspect_session_selectbox"
                        )

                        if st.button("🔄 Üzenetek lekérdezése adott Session-höz", key="fetch_mongo_chats_btn"):
                            try:
                                res = requests.get(f"{API_BASE_URL}/get_chats/{selected_session_id}")
                                if res.status_code == 200:
                                    data = res.json()
                                    messages = data.get("messages", [])
                                    if messages:
                                        st.success(f"Találat: {len(messages)} üzenet ehhez a sessionhöz.")
                                        st.json(messages)
                                    else:
                                        st.warning("Nincsenek üzenetek mentve ehhez a sessionhöz.")
                                else:
                                    st.error(res.text)
                            except Exception as e:
                                st.error(f"Hiba: {e}")
                    else:
                        st.info("Még nincsenek rögzített sessionök a MongoDB adatbázisban.")
                else:
                    st.error("Nem sikerült lekérni a sessionök listáját a szervertől.")
            except Exception as e:
                st.error(f"❌ Kapcsolódási hiba a sessionök lekérdezésekor: {e}")

# --- TAB 3: Adatbázis (Vizuális kép- és szövegmegjelenítéssel) ---
with tab3:
    st.header("Feltöltött dokumentumok és képek")
    if st.button("Lista frissítése", key="refresh_db_btn"):
        try:
            res = requests.get(f"{API_BASE_URL}/list_documents")
            if res.status_code == 200:
                data = res.json()
                if data is not None:
                    st.metric("Összes dokumentum/elem", data.get("total_documents", 0))

                    for doc in data.get("documents", []):
                        metadata = doc.get("metadata", {}) or {}
                        is_image = metadata.get("type") == "image"

                        with st.container():
                            if is_image:
                                filename = metadata.get("filename", "")
                                filepath = metadata.get("filepath", "")
                                basename = os.path.basename(filepath) if filepath else ""

                                st.markdown(f"**🖼️ Kép dokumentum** (Fájlnév: `{filename}`)")
                                if basename:
                                    try:
                                        # Letöltjük a kép bájtjait a backendről, így elkerüljük az URL/szóköz problémákat
                                        img_response = requests.get(f"{API_BASE_URL}/get_image/{basename}")
                                        if img_response.status_code == 200:
                                            st.image(img_response.content, caption=filename, width=300)
                                        else:
                                            st.warning(f"A kép nem tölthető be (Státusz: {img_response.status_code})")
                                    except Exception as e:
                                        st.error(f"Hiba a kép letöltésekor: {e}")

                                st.text_area(f"Generált leírás (ID: {doc.get('id', '')})", doc.get('text', ''),
                                             height=80, disabled=True, key=f"area_{doc.get('id', '')}")
                            else:
                                st.markdown(f"**📝 Szöveges dokumentum**")
                                st.text_area(f"Tartalom (ID: {doc.get('id', '')})", doc.get('text', ''), height=100,
                                             disabled=True, key=f"area_{doc.get('id', '')}")
                            st.divider()
                else:
                    st.warning("Az adatbázis válasza üres volt.")
            else:
                st.error(f"Szerver hiba: {res.text}")
        except Exception as e:
            st.error(f"Szerver hiba: {e}")

# --- TAB 4: Gépi Látás és Képvágás ---
with tab4:
    st.header("👁️ Gépi Látás & Objektumdetektálás")

    vision_sub_tab1, vision_sub_tab2 = st.tabs(["👁️ Általános Képelemzés", "✂️ Objektumdetektálás & Kivágás"])

    with vision_sub_tab1:
        st.subheader("Qwen Vision Elemzés")
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
                "Mit szeretnél tudni a kép(ek)ről?",
                value="Írd le egy-egy mondatban mi van a képeken!",
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
                            res_data = res.json()
                            if res_data:
                                st.success("Sikeres elemzés!")
                                st.info(res_data.get("response", "Nincs válasz."))
                            else:
                                st.success("Sikeres elemzés, de üres válasz érkezett.")
                        else:
                            st.error(res.text)
                    except Exception as e:
                        st.error(f"Kapcsolódási hiba: {e}")

    with vision_sub_tab2:
        st.subheader("Interaktív Objektumdetektálás & Automatikus Kivágás")
        uploaded_crop_file = st.file_uploader("Válassz ki egy JPG képet elemzésre:", type=["jpg", "jpeg"],
                                              key="crop_image_uploader")
        detection_query = st.text_input("Mit keressek a képen?", value="Pl. a kép közepét emeld ki!",
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
                            if res_data:
                                st.info(res_data.get("response", ""))
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
                                st.warning("Üres válasz érkezett a detektáláshoz.")
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

        - **🔀 A szöveg és a számok külön utakon járnak:** 
          A modell belső logikája szövegesen kitalálja a történetet (hogy hol van a figura), de a számítógépnek adandó koordinátákat egy teljesen más, statisztikai becslés alapján generálja. Így előfordulhat, hogy a válaszban okos dolgot ír, de a számok máshova mutatnak.

        - **🎨 Szín- és minta-összezavarodás (Vizuális tévedés):** 
          Mivel a modell a zsúfolt képeken egyszerre próbálja figyelni a formákat és a színeket, gyakran előfordulhat az a hiba, hogy **a piros-fehér ruhás Waldo helyett egy kék ruhás vagy más színű figurát jelöl be**. Ennek az az oka, hogy a vizuális modul a keresett formát (pl. a sapkát vagy a csíkokat) összetéveszti a környezetében lévő más, kontrasztos figurákkal.

        Súgó: Waldo helyzete a nagy fesztiváli képen: [ymin:255, xmin:812, ymax:355, xmax:888] 
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
                                if progress_data:
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
                    if res_data:
                        status_box.update(label="A keresés befejeződött.", state="complete", expanded=False)
                        st.success("A modell lefutott.")
                        st.info(res_data.get("response", "Nincs szöveges válasz."))
                        annotated_image = res_data.get("annotated_image")
                        if annotated_image:
                            st.image(base64.b64decode(annotated_image), caption="🎯 Találat")
                    else:
                        status_box.update(label="Hiba történt.", state="error")
                        st.error("A szerver üres választ adott.")
                else:
                    status_box.update(label="Hiba történt.", state="error")
                    st.error(res.text)
            except Exception as e:
                status_box.update(label="Hiba történt.", state="error")
                st.error(f"Hiba: {e}")

with tab6:
    st.header("🕸️ Tudásgráf Építő és Megjelenítő (Groq + Neo4j)")

    # Alfülek létrehozása a Tudásgráf fülön belül
    graph_sub_tab1, graph_sub_tab2 = st.tabs(["✍️ Új Gráf Építése", "📊 Mentett Gráfok Megtekintése (DB)"])

    # --- 6.1 Alfül: Új Gráf Építése szövegből ---
    with graph_sub_tab1:
        st.subheader("Szöveg elemzése és mentése a Neo4j-be")
        graph_text = st.text_area("Másolj be egy szakmai szöveget a gráf építéséhez:", height=150,
                                  placeholder="Pl.: Harry Potter a Roxfortban tanul, amit Albus Dumbledore vezet...",
                                  key="graph_text_input")

        if st.button("Relációk Kinyerése és Mentés 🚀", type="primary", key="extract_graph_btn"):
            if graph_text.strip() == "":
                st.warning("Kérlek, írj be valamilyen szöveget!")
            else:
                with st.spinner("Szöveg elemzése Groq-kal és mentés a Neo4j adatbázisba..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/extract_graph",
                            json={"text": graph_text}
                        )
                        if response.status_code == 200:
                            data = response.json()
                            st.success(f"✅ {data.get('message')}")
                            extracted_data = data.get("extracted_data", [])

                            if extracted_data:
                                nodes = []
                                edges = []
                                added_nodes = set()

                                for rel in extracted_data:
                                    source = rel.get("source")
                                    target = rel.get("target")
                                    relation = rel.get("relationship")

                                    if source not in added_nodes:
                                        nodes.append(Node(id=source, label=source, size=25, color="#4CAF50"))
                                        added_nodes.add(source)
                                    if target not in added_nodes:
                                        nodes.append(Node(id=target, label=target, size=25, color="#2196F3"))
                                        added_nodes.add(target)

                                    edges.append(Edge(source=source, label=relation, target=target, color="#FF9800"))

                                config = Config(
                                    width="100%",
                                    height=500,
                                    directed=True,
                                    hierarchical=False,
                                    nodeHighlightBehavior=True,
                                    highlightColor="#F7A7A6",
                                    collapsible=False,
                                    link={'labelProperty': 'label', 'renderLabel': True},
                                    physics={
                                        "barnesHut": {
                                            "gravitationalConstant": -30000,
                                            "centralGravity": 0.3,
                                            "springLength": 250,
                                            "springConstant": 0.05,
                                            "avoidOverlap": 1
                                        }
                                    }
                                )

                                st.markdown("### 🕸️ Frissen kinyert gráf vizualizációja")
                                agraph(nodes=nodes, edges=edges, config=config)
                        else:
                            st.error(f"Hiba a szerver oldalon: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Kapcsolódási hiba: {e}")

        # --- 6.2 Alfül: Már elmentett adatbázis lekérdezése, olvasása és karbantartása ---
        with graph_sub_tab2:
            st.subheader("📊 Neo4j Adatbázis Tartalma & Menedzsment")
            st.write(
                "Itt tudod lekérdezni, vizualizálni és karbantartani a Neo4j-be eddig elmentett összes kapcsolatot.")

            if st.button("🔄 Adatbázis betöltése és kirajzolása", type="primary", key="load_db_graph_btn"):
                with st.spinner("Adatok lekérdezése a Neo4j adatbázisból..."):
                    try:
                        response = requests.get(f"{API_BASE_URL}/get_graph")
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.db_extracted_data = data.get("extracted_data", [])
                            st.success(
                                f"✅ Sikeresen betöltve! Összesen {len(st.session_state.db_extracted_data)} kapcsolat található.")
                        else:
                            st.error(f"Hiba a szerver oldalon: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Kapcsolódási hiba: {e}")

            if "db_extracted_data" in st.session_state and st.session_state.db_extracted_data:
                extracted_data = st.session_state.db_extracted_data

                nodes = []
                edges = []
                added_nodes = set()

                for rel in extracted_data:
                    source = rel.get("source")
                    target = rel.get("target")
                    relation = rel.get("relationship")

                    if source not in added_nodes:
                        nodes.append(Node(id=source, label=source, size=25, color="#4CAF50"))
                        added_nodes.add(source)
                    if target not in added_nodes:
                        nodes.append(Node(id=target, label=target, size=25, color="#2196F3"))
                        added_nodes.add(target)

                    edges.append(Edge(source=source, label=relation, target=target, color="#FF9800"))

                config = Config(
                    width="100%",
                    height=500,
                    directed=True,
                    hierarchical=False,
                    nodeHighlightBehavior=True,
                    highlightColor="#F7A7A6",
                    collapsible=False,
                    link={'labelProperty': 'label', 'renderLabel': True},
                    physics={
                        "barnesHut": {
                            "gravitationalConstant": -30000,
                            "centralGravity": 0.3,
                            "springLength": 250,
                            "springConstant": 0.05,
                            "avoidOverlap": 1
                        }
                    }
                )

                st.markdown("### 🕸️ Teljes Neo4j Vizuális Tudásgráf")
                agraph(nodes=nodes, edges=edges, config=config)

                with st.expander("📋 Nyers adatok (JSON / Táblázat) megtekintése"):
                    st.table(extracted_data)

                st.divider()
                st.subheader("🗑️ Hibás kapcsolat törlése")
                st.markdown("Ha egy kapcsolat helytelenül szerepel a gráfban, itt kiválaszthatod és eltávolíthatod:")

                rel_options = [
                    f"{r.get('source')} --[{r.get('relationship')}]--> {r.get('target')}"
                    for r in extracted_data
                ]

                selected_rel_str = st.selectbox("Válaszd ki a törölni kívánt kapcsolatot:", options=rel_options,
                                                key="select_del_rel")

                if st.button("Kiválasztott kapcsolat törlése ❌", type="secondary", key="delete_rel_btn"):
                    # Megkeressük a kiválasztott elem adatait
                    selected_idx = rel_options.index(selected_rel_str)
                    target_rel = extracted_data[selected_idx]

                    try:
                        del_response = requests.post(
                            f"{API_BASE_URL}/delete_graph_relation",
                            json={
                                "source": target_rel.get("source"),
                                "target": target_rel.get("target"),
                                "relationship": target_rel.get("relationship")
                            }
                        )
                        if del_response.status_code == 200:
                            st.success("✅ A kapcsolat sikeresen törölve a Neo4j adatbázisból!")
                            # Töröljük a session_state-ből, hogy újra lekérje a friss adatot
                            del st.session_state.db_extracted_data
                            st.rerun()
                        else:
                            st.error(f"Hiba a törlés során: {del_response.text}")
                    except Exception as e:
                        st.error(f"❌ Kapcsolódási hiba: {e}")
            else:
                st.info("Kattints a fenti gombra az adatbázis tartalmának betöltéséhez.")
