# PDF Chat with Multilingual Support with SarvamAPi

This project is an interactive chat application that allows users to ask questions about the content of a PDF document. It features a user-friendly web interface, supports multiple languages, and utilizes advanced AI technologies for natural language processing and generation.
![ai1](https://github.com/user-attachments/assets/79cb7932-e532-47b3-9bce-e3499c94d416)



## Features

- **PDF Content Analysis**: Extract and analyze content from PDF documents.
- **Multilingual Support**: Interact with the system in multiple Indian languages.
- **Dual Query Modes**: 
  - Standard questions about the PDF content.
  - Agent-driven actions for more complex queries.
- **Speech Recognition**: Voice input support for both question types.
- **AI-Powered Responses**: Utilizes advanced language models for generating responses.
- **Translation**: Automatic translation of responses to the selected language.



## Technologies Used

- **Backend**: FastAPI, Python
- **Frontend**: HTML, JavaScript, CSS
- **AI/ML**: LangChain, Groq, Google Generative AI
- **PDF Processing**: PyPDF2
- **Vector Storage**: FAISS
- **Speech Recognition**: Web Speech API
- **Translation**: Sarvam AI API

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/pdf-chat-multilingual.git
   cd ragbot_sarvamai/ragchatbot
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   GROQ_API_KEY=your_groq_api_key
   GEMINI_KEY=your_gemini_api_key
   SARVAMAI_API_KEY=your_sarvamai_api_key
   ```

5. Place your PDF file in the `dataset` folder and update the `pdf_path` in `main.py` if necessary.



## Running the Application

1. Start the FastAPI server:
   ```
   python main.py
   ```

2. Open a web browser and navigate to `http://localhost:8000`

## Usage

1. Select your preferred language from the dropdown menu.
2. To ask a question about the PDF content:
   - Type your question in the "Ask a question..." input field and click "Ask", or
   - Click the microphone button and speak your question.
3. To use the agent for more complex queries:
   - Type your query in the "Ask an Agent..." input field and click "Ask Agent", or
   - Click the microphone button next to the agent input and speak your query.
4. View the AI's responses in the chat history above the input fields.



## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LangChain for providing the framework for building language model applications
- Groq and Google for their language models
- Sarvam AI for their translation API
- All other open-source libraries used in this project
