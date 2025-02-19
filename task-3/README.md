# Multi-Modal Chatbot with PaLM and Gemini

A sophisticated chatbot application that leverages Google's PaLM and Gemini APIs to process both text and images, providing intelligent responses and image generation capabilities.

## Features

- Text-based conversation using Google's PaLM API
- Image understanding and processing with Gemini API
- Image generation capabilities
- User-friendly Streamlit interface
- Conversation history management
- Support for multiple file formats

## Prerequisites

- Python 3.8 or higher
- Google Cloud account
- PaLM API access
- Gemini API access
- Streamlit account (for deployment)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multimodal-chatbot.git
cd multimodal-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## API Key Setup

Gemini API:
   - Visit the [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Generate an API key
   - Add to your `.env` file:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the chatbot interface at `http://localhost:8501`

3. Interact with the chatbot:
   - Type text messages in the input field
   - Upload images using the file uploader
   - View conversation history in the chat interface
   - Download generated images when available

## Troubleshooting

### Common Issues

1. API Key Errors:
   - Verify API keys are correctly set in the `.env` file
   - Ensure API services are enabled in Google Cloud Console
   - Check API quota limits

2. Image Processing Issues:
   - Supported formats: JPG, PNG, WebP
   - Maximum file size: 20MB
   - Ensure good image quality and resolution

3. Connection Problems:
   - Check internet connectivity
   - Verify firewall settings
   - Ensure required ports are open

4. Performance Issues:
   - Clear browser cache
   - Restart the application
   - Check system resources

### Error Messages

- "API key not found": Check `.env` file configuration
- "File format not supported": Convert image to supported format
- "Request timeout": Check internet connection or try again later

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For additional support or questions, please open an issue in the GitHub repository.