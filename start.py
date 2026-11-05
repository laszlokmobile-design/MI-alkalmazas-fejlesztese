import subprocess
import sys
import time
import signal
import os

def main():
    if not os.path.exists("src/backend/api.py") or not os.path.exists("src/client/app.py"):
        print("❌ Hiba: Nem találom a backend vagy a frontend fájlokat!")
        return

    print("🚀 FastAPI Backend indítása...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.backend.api:app", "--reload", "--host", "127.0.0.1", "--port", "8000"]
    )

    time.sleep(2)

    print("⚡ Streamlit Frontend indítása...")
    # Itt a -m streamlit parancsot használjuk a WinError 4551 elkerülésére
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/client/app.py"]
    )

    def signal_handler(sig, frame):
        print("\n🛑 Leállítás folyamatban...")
        backend_process.terminate()
        streamlit_process.terminate()
        backend_process.wait()
        streamlit_process.wait()
        print("👋 Minden szolgáltatás leállítva.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        backend_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()
