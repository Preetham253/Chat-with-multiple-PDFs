# Chat with multiple PDFs

This project enables users to interact with multiple PDF documents through a chat interface, leveraging Google Generative AI. It allows for natural language querying across various PDFs, providing concise and accurate responses.

## Features

- Upload multiple PDFs and parse their content.
- Utilize Google Generative AI for generating embeddings and interacting with the data.
- Implement a chat interface for querying the PDFs.
- Store and retrieve embeddings using FAISS (Facebook AI Similarity Search).
- Simple and intuitive UI built with Streamlit.

## Requirements

Make sure you have the following installed:

- Python 3.8 or above
- `streamlit`
- `PyPDF2`
- `langchain`
- `langchain_google_genai`
- `faiss`
- `google-generativeai`
- `python-dotenv`

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/chat-with-multiple-pdfs.git
   cd chat-with-multiple-pdfs
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory.
   - Add your Google API key:
     ```env
     GOOGLE_API_KEY=your_google_api_key_here
     ```

## Usage

1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

2. **Upload PDFs**:
   - Use the interface to upload multiple PDFs.
   - The app will parse the PDFs and prepare them for querying.

3. **Chat with the PDFs**:
   - Enter your queries in the chat box.
   - The app will use Google Generative AI to understand your question and fetch relevant information from the PDFs.

## Libraries and Tools Used

- **Streamlit**: For building the web interface.
- **PyPDF2**: For reading and parsing PDF documents.
- **LangChain**: For managing and processing language models.
- **Google Generative AI**: To generate embeddings and interact with the text.
- **FAISS**: For efficient storage and retrieval of embeddings.
- **dotenv**: To manage environment variables securely.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgements

- Thanks to Google for providing the Generative AI API.
- Inspired by the LangChain community.
