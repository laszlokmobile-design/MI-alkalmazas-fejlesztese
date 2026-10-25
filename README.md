![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=flat&logo=groq&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FC5E20?style=flat)
![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=flat&logo=mongodb&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat&logo=pydantic&logoColor=white)

# Projekt előrehaladása

### 📌 Összegzés a beadandó struktúrájához:
* **Patch (Minimum):** Groq LPU migráció (Gemini lecserélése, aszinkron kliens).
* **Minor (Közép):** Több kép egyidejű feltöltése és feldolgozása a Gépi Látás (Vision) modulban (*Multi-image support*).
* **Major (Haladó):** A **Waldo Kereső modul** (5. fül, PIL rajzolás, Regex koordináta-kinyerés, aszinkron `ThreadPoolExecutor` folyamatkövetés).

---

# Komplex MI alkalmazások fejlesztése

A projekt célja, hogy a hagyományos "Jupyter Notebook" szintű kísérletezésen túllépve, egy valós, skálázható szoftverarchitektúrát hozzunk létre, amely egy FastAPI backendből, egy Streamlit frontendből, valamint több adatbázisból áll.

---

##  RAG és a "Memória"

A modern Nagy Nyelvi Modellek (LLM-ek, mint a Llama vagy Qwen) lenyűgözőek, de van két komoly kihívás velük kapcsolatban:
1. **Nincs friss/privát tudásuk:** Nem látnak bele a vállalatod belső dokumentumaiba.
2. **"Amnéziásak":** Minden HTTP kérést tiszta lappal kezdenek, nem emlékszenek az egy perccel ezelőtt feltett kérdésedre.

Ezt a két problémát oldjuk meg az architektúránkkal:

### 1. RAG (Retrieval-Augmented Generation) és Vektorok
A saját dokumentumainkat vektorokká (számsorozatokká) alakítjuk az **Embedding** technológia segítségével, és egy **ChromaDB vektoradatbázisban** tároljuk. Amikor a felhasználó kérdez, a rendszer megkeresi a leginkább releváns bekezdéseket a térben, és ezt "puskaként" átadja a modellnek.

### 2. Állapottartó (Stateful) Beszélgetés és NoSQL
Ahhoz, hogy a gép emlékezzen a beszélgetés fonalára, bevezettünk egy **MongoDB (NoSQL) adatbázist**. 
* **Miért NoSQL?** Az MI-vel folytatott beszélgetések (kiküldött promptok, kapott válaszok, metaadatok) természetes módon JSON/dokumentum formátumúak, amire a MongoDB tökéletes választás. A relációs adatbázisok (SQL) ehhez túl merevek lennének.
* A rendszer generál egy `session_id`-t, és minden üzenetváltást letárol, így a következő kérdésnél a FastAPI a teljes beszélgetési előzményt fel tudja tölteni a modell memóriájába.

---

## Technológiai Stack és Architektúra

A projekt a legmodernebb ipari standardokra (Docker Compose) épül:
* **Környezet:** Docker DevContainer (Python App + MongoDB konténerek közös hálózaton).
* **Backend:** `FastAPI` (Aszinkron API), `Motor` (Aszinkron MongoDB driver) és `Pillow` (Memórián belüli képkezelés).
* **Frontend:** `Streamlit` (Modern, Python-alapú chat UI).
* **Adatbázisok:** `ChromaDB` (vektoroknak) és `MongoDB` (munkameneteknek és chat logoknak).
* **MI Integráció:** Hivatalos `google-genai` SDK.

---

## Fejlesztői Környezet Indítása

### 1. Előfeltételek és Beállítások
Győződj meg róla, hogy a VS Code-ban a DevContainer felépült (a Docker Compose alapján). 
Létre kell hoznod egy `.env` nevű fájlt a projekt gyökerében:
```env
GEMINI_API_KEY=ide_masold_a_sajat_kulcsod
```

## 2. Csatlakozás a MongoDB-hez (VS Code Plugin)
Nem kell vaktában kódolnod! A DevContainer tartalmaz egy hivatalos MongoDB kiterjesztést.

Kattints a VS Code bal oldali sávjában a MongoDB (falevél) ikonra.

Kattints az Add Connection gombra.

A fenti parancssorba írd be: mongodb://mongo:27017 majd nyomj Entert.
> (Megjegyzés: A Docker belső hálózata miatt a host nevünk mongo, nem pedig localhost!)

## 3. A Szolgáltatások Futtatása (start.py)

A programoz készült egy indító szkript is, amely egyszerre indítja a frontendet és a backendet az alkalmazás gyökérmappájában található "start.py" néven érhető el.

> Ez automatikusan megnyitja a böngészőben a Streamlit alkalmazást (jellemzően a http://localhost:8501 címen).

### 3. Multimodális Gépi Látás (Vision)
A rendszer képes képeket (számlákat, diagramokat, vizuális adatokat) fogadni és feldolgozni a Qwen modell segítségével, ami közvetlenül strukturált adatokat vagy elemzést generál a bináris fájlokból.

###  Az Adatfolyam Vizualizációja (Multimodális Látvány)
Az alábbi ábra bemutatja, hogyan utazik egy képes-szöveges kérés a kliensgéptől egészen az MI szerveréig és vissza:

```mermaid
sequenceDiagram
    actor User as Felhasználó (Streamlit)
    participant API as FastAPI Backend
    participant Mongo as MongoDB
    participant Groq as Groq API (Llama / Qwen)

    User->>API: 1. POST /chat_with_image (Kép bytes + Kérdés text)
    activate API
    API->>API: 2. Kép memóriába konvertálása (PIL / io.BytesIO)
    API->>Groq: 3. chat.completions.create([Kép, Kérdés])
    activate Groq
    Groq-->>API: 4. Elemzés / Válasz visszatérése
    deactivate Groq
    API->>Mongo: 5. Kérés és Válasz aszinkron mentése (Chat History)
    Mongo-->>API: Sikeres mentés (Ack)
    API-->>User: 6. JSON válasz (Szöveg + Képfájl metaadat)
    deactivate API
    User->>User: 7. Frontend frissíti a UI-t (Új üzenet a chaten)
```
## Project struktúra

```
/
├── start.py                # Egységes indító szkript (Backend + Frontend)
├── src/
│   ├── backend/          
│   │   └── api.py          # FastAPI végpontok (RAG, Chat és Vision logika)
│   └── client/
│       └── app.py          # Streamlit UI (Feltöltés, Chat és Waldo modul)
├── chroma_db/              # Lokális vektoradatbázis (automatikusan létrejön)
├── requirements.txt        # Függőségek (FastAPI, Streamlit, motor, chromadb, groq stb.)
└── .env                    # API kulcsok (NE KÜLDD BE GITHUB-RA!)
```
---

## ⚠️ Ismert jelenségek (Hibatűrés)

Mivel az alkalmazás élő felhőszolgáltatásokra (Google Gemini API) támaszkodik, előfordulhatnak terhelési tüskék a szervereiken. 
* Ha a felületen `503 UNAVAILABLE` (Leterheltség) hibát kapsz: a Google szerverei ideiglenesen túlterheltek. Várj pár másodpercet és próbáld újra elküldeni az üzenetet.
* Ha a feltöltött képnél a Streamlit figyelmeztetést dob, ellenőrizd, hogy a kiterjesztés szabványos `jpg`, `jpeg` vagy `png` formátumú-e, és mérete nem haladja meg a 20MB-ot.
