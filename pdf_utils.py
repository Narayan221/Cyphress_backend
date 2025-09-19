from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from io import BytesIO

def extract_text_from_pdf(file_byte):
    reader = PdfReader(BytesIO(file_byte))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=800, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
