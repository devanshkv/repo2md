# GitHub Repository to LLM Markdown Converter

This Streamlit application fetches all text-based files from a given public GitHub repository, concatenates their content, and formats it into a single Markdown string. This output is suitable for pasting into Large Language Models (LLMs) to provide them with the context of an entire codebase.

## Features

- Fetches files recursively from a GitHub repository.
- Ignores common binary file types (images, archives, etc.) and irrelevant directories (e.g., `.git`, `node_modules`).
- Allows optional input of a GitHub Personal Access Token (PAT) to increase API rate limits and access private repositories (use with caution).
- Displays the combined Markdown output in a text area.
- Provides a download button for the generated Markdown file.
- Basic error handling for API requests and file downloads.

## How to Run

1.  **Clone the repository or download the files.**
    If you have this code as part of a repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
    If you only have `app.py` and `requirements.txt`, save them in a new directory and navigate into it.

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

5.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

6.  Enter a GitHub repository URL (e.g., `https://github.com/owner/repo`) into the input field and click "Convert to Markdown".

## Dependencies

-   Streamlit
-   Requests

These are listed in `requirements.txt`.
