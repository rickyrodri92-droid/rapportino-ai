import streamlit as st
import pdfplumber
from transformers import pipeline

# ⚙️ Carica il classificatore una sola volta
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_classifier()

# 📄 Estrai testo da PDF
def estrai_testo_da_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])

# 📌 Estrai titolo
def estrai_titolo(testo):
    righe = testo.splitlines()
    for riga in righe[:15]:
        if (
            riga.isupper() and len(riga.split()) > 3
        ) or any(x in riga.lower() for x in ["manutenzione", "impianto", "rilevazione", "norma uni", "rapportino", "verbale", "scheda", "relazione"]):
            return riga.strip()
    return "(Titolo non trovato)"

# 📝 Estrai descrizione
def estrai_descrizione(testo):
    righe = testo.splitlines()
    descrizione = []
    attivo = False
    for riga in righe:
        if "descrizione" in riga.lower() or "note generali" in riga.lower():
            attivo = True
            continue
        if attivo:
            if any(x in riga.lower() for x in ["materiali", "durata", "firma", "responsabile"]):
                break
            descrizione.append(riga.strip())
    return "\n".join(descrizione)

# 🧠 Classifica
def classifica_testo(testo):
    etichette = ["Intervento tecnico", "Consegna materiali", "Richiesta pagamento", "Altro"]
    risultato = classifier(testo, etichette)
    return risultato["labels"][0]

# 🌐 Interfaccia Streamlit
st.set_page_config(page_title="Analizzatore Rapportini", layout="centered")
st.title("📄 Analizzatore Rapportini PDF")

uploaded_files = st.file_uploader("Trascina qui uno o più rapportini PDF", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"📄 {file.name}")
        testo = estrai_testo_da_pdf(file)
        titolo = estrai_titolo(testo)
        descrizione = estrai_descrizione(testo)
        tipo = classifica_testo(descrizione)

        st.markdown(f"**📌 Titolo:** {titolo}")
        st.markdown(f"**📝 Descrizione:**\n{descrizione if descrizione else '(Nessuna descrizione trovata)'}")
        st.markdown(f"**✅ Classificazione:** {tipo}")
        st.divider()
