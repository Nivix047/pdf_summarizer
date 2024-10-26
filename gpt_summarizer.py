import nltk
import PyPDF2
import openai
import os
from dotenv import load_dotenv  # Ensure this line is included
from collections import deque
from sacremoses import MosesDetokenizer

# Check if the 'punkt' tokenizer models are available, download them if not
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' tokenizer models...")
    nltk.download('punkt')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as pdf_file_obj:
            pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page_obj = pdf_reader.pages[page_num]
                text += page_obj.extract_text() or ""
        if not text:
            print("Warning: No text was extracted from the PDF.")
        return text.strip()
    except Exception as e:
        print(f"An error occurred while extracting text from the PDF: {str(e)}")
        return None

# Function to summarize text using OpenAI GPT
def gpt_summarize(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a summarizing application. You are given a document and asked to summarize it"},
                {"role": "user", "content": f"{text}\n\nSummarize:"},
            ]
        )
        return response
    except openai.error.RateLimitError:
        print("Rate limit exceeded. Please try again later.")
        return None
    except openai.error.AuthenticationError:
        print("Invalid API key provided. Please check your API key.")
        return None
    except openai.error.APIConnectionError:
        print("Connection error. Please check your network settings.")
        return None
    except Exception as e:
        print(f"An error occurred during summarization: {str(e)}")
        return None

# Function to recursively summarize text
def recursively_summarize(text, section_size=3000, overlap=150):
    detokenizer = MosesDetokenizer()
    tokens = nltk.word_tokenize(text)
    summaries = []

    # Create overlapping sections
    sections = deque()
    start = 0
    while start < len(tokens):
        sections.append(tokens[start:start + section_size])
        start += section_size - overlap

    # Summarize each section and append to the summaries list
    while len(sections) > 0:
        section = sections.popleft()
        summary = gpt_summarize(detokenizer.detokenize(section))
        if summary and 'choices' in summary and summary['choices']:
            summaries.append(summary['choices'][0]['message']['content'])
        else:
            print("Error: Summarization failed.")
            return None

    return " ".join(summaries) if summaries else None

if __name__ == "__main__":
    # Set OpenAI API key
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # print(f"OpenAI API Key: {openai.api_key}") 
    if openai.api_key is None:
        print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        exit(1)
    
    print("API key loaded successfully.")  

    # Specify PDF path
    pdf_path = "/Users/nivix047/Desktop/Meeting_minutes3.pdf"
    print(f"Looking for PDF at: {pdf_path}")  
    text = extract_text_from_pdf(pdf_path)
    if text is None:
        print("Failed to extract text from PDF.")  
        exit(1)

    print("Text extracted successfully.")  

    # Summarize text
    response = recursively_summarize(text)
    if response is None:
        print("Summarization failed.")  # 
        exit(1)

    print("Summarization completed successfully.")  # Debugging line
    print("Summarized Text:")  # Print label for summarized text
    print(response)  # Print the actual summarized text
