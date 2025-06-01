import pytest
from unittest.mock import patch, MagicMock
import base64

# Assuming app.py is in the parent directory or PYTHONPATH is set correctly
# For local testing, you might need to adjust sys.path or run pytest with specific settings
# e.g., `python -m pytest` from the root directory
from app import get_repo_api_url, fetch_repo_contents
# IGNORED_EXTENSIONS is a global in app.py, useful for some tests
from app import IGNORED_EXTENSIONS # if needed directly, or test its effect

# --- Test get_repo_api_url ---
@pytest.mark.parametrize("input_url, expected_output", [
    ("https://github.com/owner/repo", "https://api.github.com/repos/owner/repo"),
    ("https://github.com/owner/repo/", "https://api.github.com/repos/owner/repo"),
    ("https://github.com/SomeUser/SomeProject", "https://api.github.com/repos/SomeUser/SomeProject"),
    ("https://github.com/owner/repo.git", "https://api.github.com/repos/owner/repo.git"),
])
def test_get_repo_api_url_valid(input_url, expected_output):
    assert get_repo_api_url(input_url) == expected_output

@pytest.mark.parametrize("input_url", [
    "http://github.com/owner/repo", "https://gitlab.com/owner/repo",
    "https://github.com/owner", "https://github.com/", "", "just_a_string"
])
def test_get_repo_api_url_invalid(input_url):
    assert get_repo_api_url(input_url) is None

# --- Fixtures for Mocking API responses ---
@pytest.fixture
def mock_streamlit_ui(mocker):
    mocker.patch('app.st.info', return_value=None)
    mocker.patch('app.st.warning', return_value=None)
    mocker.patch('app.st.error', return_value=None)
    mocker.patch('app.st.success', return_value=None)
    mocker.patch('app.st.spinner', MagicMock()) # If used as a context manager

@pytest.fixture
def mock_requests_get(mocker):
    return mocker.patch('app.requests.get')

def create_github_item(name, path, item_type="file", size=100, download_url=None, content=None, encoding=None):
    item = {
        "name": name,
        "path": path,
        "type": item_type,
        "size": size,
        "sha": f"sha_for_{name}", # Make SHA unique for easier debugging if needed
    }
    if item_type == "file":
        item["download_url"] = download_url if download_url else f"https://example.com/download/{path}"
        if content: # For .gitignore or direct content fetching (if API provided it, though usually it's download_url)
            item["content"] = base64.b64encode(content.encode('utf-8')).decode('utf-8') if encoding == "base64" else content
            item["encoding"] = encoding if encoding else "none" # 'base64' or 'none'
    return item

# --- Tests for Filtering Logic ---

# Helper to simulate calls to fetch_repo_contents
def run_fetch_repo_contents(mock_rg, repo_url_test="https://github.com/test/repo",
                            include_patterns="", exclude_patterns="", max_kb=0,
                            api_responses=None): # api_responses is a dict of url -> response_json_or_text

    # Default setup for requests.get mock
    def side_effect_func(url, headers=None):
        mock_resp = MagicMock()

        # Handle .gitignore first if it's defined in api_responses
        if ".gitignore" in url and api_responses and url in api_responses:
            content_data = api_responses[url]
            if isinstance(content_data, dict) and "content" in content_data : # JSON response for /contents API
                 mock_resp.status_code = 200
                 mock_resp.json.return_value = content_data
            elif isinstance(content_data, str): # Plain text from download_url
                 mock_resp.status_code = 200
                 mock_resp.text = content_data
            else: # Not found or other error for .gitignore
                mock_resp.status_code = content_data.get('status_code', 404) if isinstance(content_data, dict) else 404
                if mock_resp.status_code == 404:
                    mock_resp.json.side_effect = Exception("Not Found") # Simulate requests.exceptions.HTTPError
                else: # Other error
                    mock_resp.json.return_value = content_data
            return mock_resp

        # Handle directory listings (contents API)
        if "/contents/" in url and api_responses and url in api_responses:
            mock_resp.status_code = 200
            mock_resp.json.return_value = api_responses[url] # Should be a list of items
            return mock_resp

        # Handle file downloads from download_url
        if api_responses and url in api_responses: # URL is a download_url
            mock_resp.status_code = 200
            mock_resp.content = api_responses[url].encode('utf-8') # Content should be text
            mock_resp.text = api_responses[url]
            return mock_resp

        # Default for unhandled URLs (e.g. if a test doesn't mock a specific download)
        mock_resp.status_code = 404
        mock_resp.json.side_effect = Exception("Not Found for " + url)
        mock_resp.raise_for_status.side_effect = Exception("HTTP Error 404 for " + url)
        return mock_resp

    mock_rg.side_effect = side_effect_func

    # Call the actual function from app.py
    # Need to pass the global inputs that fetch_repo_contents reads
    with patch('app.include_patterns_input', include_patterns), \
         patch('app.exclude_patterns_input', exclude_patterns), \
         patch('app.max_file_size_kb', max_kb), \
         patch('app.github_pat', ""): # Assuming no PAT for these tests
        return fetch_repo_contents(repo_url_test)


def test_ignore_by_extension(mock_requests_get, mock_streamlit_ui):
    repo_files = [
        create_github_item("file.py", "file.py", content="print('hello')"),
        create_github_item("image.png", "image.png", content="dummy_png_content"), # Should be ignored by IGNORED_EXTENSIONS
        create_github_item("script.pyc", "script.pyc", content="compiled"), # Should be ignored
    ]
    api_url_root = "https://api.github.com/repos/test/repo"

    api_responses = {
        f"{api_url_root}/contents/": repo_files,
        f"{api_url_root}/contents/.gitignore": {"status_code": 404}, # No .gitignore
        repo_files[0]['download_url']: "print('hello')",
        # No need to mock download for .png or .pyc as they should be skipped before download based on name
    }

    # Max file size KB set to 0 (disabled)
    result = run_fetch_repo_contents(mock_requests_get, api_responses=api_responses, max_kb=0)

    assert "file.py" in result
    assert "image.png" not in result
    assert "script.pyc" not in result
    assert result["file.py"] == "print('hello')"

def test_ignore_git_directory(mock_requests_get, mock_streamlit_ui):
    repo_files_root = [
        create_github_item("file.py", "file.py", content="root file"),
        create_github_item(".git", ".git", item_type="dir"),
        create_github_item("src", "src", item_type="dir"),
    ]
    repo_files_src = [
        create_github_item("app.py", "src/app.py", content="app code"),
        create_github_item(".git", "src/.git", item_type="dir"), # A nested .git, should also be ignored
    ]
    api_url_root = "https://api.github.com/repos/test/repo"

    api_responses = {
        f"{api_url_root}/contents/": repo_files_root,
        f"{api_url_root}/contents/.git": [], # Should not be listed, but if it was, it should be skipped
        f"{api_url_root}/contents/src": repo_files_src,
        f"{api_url_root}/contents/.gitignore": {"status_code": 404},
        repo_files_root[0]['download_url']: "root file",
        repo_files_src[0]['download_url']: "app code",
    }

    result = run_fetch_repo_contents(mock_requests_get, api_responses=api_responses)

    assert "file.py" in result
    assert ".git" not in result # Should not be present as a key
    assert "src/app.py" in result
    assert "src/.git" not in result
    # Check that no files from within any .git directory are present
    for key in result.keys():
        assert ".git" not in key.split('/')


def test_gitignore_basic(mock_requests_get, mock_streamlit_ui):
    gitignore_content = "*.log\nbuild/\n"
    repo_files = [
        create_github_item("main.py", "main.py", content="code"),
        create_github_item("app.log", "app.log", content="log data"),
        create_github_item("build", "build", item_type="dir"),
        create_github_item("docs", "docs", item_type="dir"),
    ]
    repo_files_build = [create_github_item("output.txt", "build/output.txt", content="built")]
    repo_files_docs = [create_github_item("index.md", "docs/index.md", content="docs")]

    api_url_root = "https://api.github.com/repos/test/repo"
    api_responses = {
        f"{api_url_root}/contents/": repo_files,
        # Mock .gitignore fetching: can be either via contents API (base64) or download_url
        # Using download_url method for this test as it's simpler to mock plain text
        f"{api_url_root}/contents/.gitignore": create_github_item(".gitignore", ".gitignore", download_url=f"{api_url_root}/raw/.gitignore"),
        f"{api_url_root}/raw/.gitignore": gitignore_content,

        f"{api_url_root}/contents/build": repo_files_build, # Should not be accessed if 'build/' is ignored
        f"{api_url_root}/contents/docs": repo_files_docs,
        repo_files[0]['download_url']: "code",
        repo_files_docs[0]['download_url']: "docs",
    }

    result = run_fetch_repo_contents(mock_requests_get, api_responses=api_responses)

    assert "main.py" in result
    assert "app.log" not in result
    assert "build/output.txt" not in result # Files from ignored dirs should not be there
    assert "docs/index.md" in result
    # Verify that requests.get was not called for 'build' directory contents
    # This is tricky with current side_effect; would need more sophisticated call tracking on mock_rg
    # For now, relying on the output (result dict) being correct.


def test_user_glob_exclusion(mock_requests_get, mock_streamlit_ui):
    user_exclude_patterns = "*.txt\ntemp/"
    repo_files = [
        create_github_item("main.py", "main.py", content="code"),
        create_github_item("data.txt", "data.txt", content="text data"),
        create_github_item("temp", "temp", item_type="dir"),
    ]
    repo_files_temp = [create_github_item("temp_file.tmp", "temp/temp_file.tmp", content="temp")]
    api_url_root = "https://api.github.com/repos/test/repo"
    api_responses = {
        f"{api_url_root}/contents/": repo_files,
        f"{api_url_root}/contents/.gitignore": {"status_code": 404}, # No .gitignore
        f"{api_url_root}/contents/temp": repo_files_temp, # Should not be accessed
        repo_files[0]['download_url']: "code",
        # data.txt should not be downloaded
    }

    result = run_fetch_repo_contents(mock_requests_get, api_responses=api_responses, exclude_patterns=user_exclude_patterns)
    assert "main.py" in result
    assert "data.txt" not in result
    assert "temp/temp_file.tmp" not in result

def test_user_glob_inclusion(mock_requests_get, mock_streamlit_ui):
    user_include_patterns = "src/**/*.py\n*.md"
    repo_files = [
        create_github_item("main.py", "main.py", content="root python"), # Not in src/**
        create_github_item("README.md", "README.md", content="readme content"), # Matches *.md
        create_github_item("src", "src", item_type="dir"),
        create_github_item("docs", "docs", item_type="dir"), # Not in inclusion patterns
    ]
    repo_files_src = [
        create_github_item("app.py", "src/app.py", content="src python"), # Matches src/**/*.py
        create_github_item("utils.js", "src/utils.js", content="src js"), # No match
        create_github_item("sub", "src/sub", item_type="dir"),
    ]
    repo_files_src_sub = [
        create_github_item("helper.py", "src/sub/helper.py", content="src sub python"), # Matches src/**/*.py
    ]
    repo_files_docs = [create_github_item("info.txt", "docs/info.txt", content="docs text")]

    api_url_root = "https://api.github.com/repos/test/repo"
    api_responses = {
        f"{api_url_root}/contents/": repo_files,
        f"{api_url_root}/contents/.gitignore": {"status_code": 404},
        f"{api_url_root}/contents/src": repo_files_src,
        f"{api_url_root}/contents/src/sub": repo_files_src_sub,
        f"{api_url_root}/contents/docs": repo_files_docs, # Should not be accessed
        repo_files[1]['download_url']: "readme content",
        repo_files_src[0]['download_url']: "src python",
        repo_files_src_sub[0]['download_url']: "src sub python",
    }

    result = run_fetch_repo_contents(mock_requests_get, api_responses=api_responses, include_patterns=user_include_patterns)

    assert "main.py" not in result
    assert "README.md" in result
    assert "src/app.py" in result
    assert "src/utils.js" not in result
    assert "src/sub/helper.py" in result
    assert "docs/info.txt" not in result

def test_max_file_size_limit(mock_requests_get, mock_streamlit_ui):
    # item['size'] is in bytes. max_kb is in kilobytes.
    max_kb_limit = 1  # 1 KB limit
    repo_files = [
        create_github_item("small.txt", "small.txt", size=500, content="small file"), # 500 bytes
        create_github_item("large.txt", "large.txt", size=2048, content="large file"), # 2KB, should be skipped
        create_github_item("exact.txt", "exact.txt", size=1024, content="exact size"), # 1KB, should be included
    ]
    api_url_root = "https://api.github.com/repos/test/repo"
    api_responses = {
        f"{api_url_root}/contents/": repo_files,
        f"{api_url_root}/contents/.gitignore": {"status_code": 404},
        repo_files[0]['download_url']: "small file",
        # large.txt download_url should not be called
        repo_files[2]['download_url']: "exact size",
    }

    result = run_fetch_repo_contents(mock_requests_get, api_responses=api_responses, max_kb=max_kb_limit)

    assert "small.txt" in result
    assert "large.txt" not in result
    assert "exact.txt" in result
    assert result["small.txt"] == "small file"
    assert result["exact.txt"] == "exact size"


# --- Tests for output formatting and priority (would need to call the main button handler logic) ---
# These are more complex as they depend on the main Streamlit button handler structure.
# For now, focusing on fetch_repo_contents. If the formatting/priority logic
# is refactored into testable helpers in app.py, they can be tested directly.

# Example of what a test for priority files might look like if logic is extracted:
# from app import _format_and_order_contents # Hypothetical function
# def test_priority_file_ordering():
#     contents = {
#         "src/app.py": "content1",
#         "README.md": "readme_content",
#         "LICENSE": "license_content",
#         "docs/README.md": "docs_readme"
#     }
#     repo_url = "https://github.com/test/repo"
#     # Assume priority_filenames_lower is accessible or passed
#     # markdown_output = _format_and_order_contents(contents, repo_url, priority_filenames_lower)
#     # assert "README.md" appears before "src/app.py"
#     # assert "LICENSE" appears before "src/app.py"
#     # assert "docs/README.md" is not given top priority (appears with other_markdown_parts)


# Basic test for Markdown formatting (if a helper can be isolated)
# from app import _generate_file_markdown_part # Hypothetical
# def test_markdown_formatting():
#     path = "src/app.py"
#     content = "print('hello')"
#     expected_md = "## File: `src/app.py`\n\n```python\nprint('hello')\n```\n\n---\n"
#     # lang = ... (derive lang)
#     # md_part = _generate_file_markdown_part(path, content, lang)
#     # assert md_part == expected_md
#     pass # Placeholder, as this logic is currently inside the main button handler in app.py

# Test for precedence: .gitignore vs user exclude vs user include
# This requires careful setup of mock files and rules.
# E.g., a file ignored by .gitignore, but user includes it (should still be ignored by .gitignore).
# A file ignored by .gitignore, also by user exclude (still ignored).
# A file allowed by .gitignore, but user excludes it (should be excluded).
# A file allowed by .gitignore, allowed by user exclude (not matching), but user includes it (should be included).
# A file allowed by .gitignore, allowed by user exclude, but NOT in user include (if include is active) -> should be ignored.

def test_filtering_precedence(mock_requests_get, mock_streamlit_ui):
    gitignore_content = "ignored_by_git.txt\nexplicitly_excluded_by_git_but_user_includes.txt"
    user_exclude_patterns = "excluded_by_user.txt\nexplicitly_excluded_by_git_but_user_includes.txt" # User also excludes one of them
    user_include_patterns = "*.py\nincluded_by_user.txt\nexplicitly_excluded_by_git_but_user_includes.txt" # User tries to include the git-ignored one

    repo_files = [
        create_github_item("main.py", "main.py", content="python code"), # Included by user
        create_github_item("ignored_by_git.txt", "ignored_by_git.txt", content="git ignore only"),
        create_github_item("excluded_by_user.txt", "excluded_by_user.txt", content="user exclude only"),
        create_github_item("explicitly_excluded_by_git_but_user_includes.txt", "explicitly_excluded_by_git_but_user_includes.txt", content="git vs user include"),
        create_github_item("included_by_user.txt", "included_by_user.txt", content="user include only"),
        create_github_item("not_included.txt", "not_included.txt", content="passes other filters, but not in user include"),
        create_github_item("config.json", "config.json", content="no specific rule, not in include"), # Should be out if include is active
    ]
    api_url_root = "https://api.github.com/repos/test/repo"
    api_responses = {
        f"{api_url_root}/contents/": repo_files,
        f"{api_url_root}/contents/.gitignore": create_github_item(".gitignore", ".gitignore", download_url=f"{api_url_root}/raw/.gitignore"),
        f"{api_url_root}/raw/.gitignore": gitignore_content,
    }
    # Setup download_urls for files expected to be processed
    for item in repo_files:
        if item['name'] in ["main.py", "included_by_user.txt"]: # Only these should pass
             api_responses[item['download_url']] = item.get('content', "dummy content for " + item['name'])


    result = run_fetch_repo_contents(
        mock_requests_get,
        api_responses=api_responses,
        include_patterns=user_include_patterns,
        exclude_patterns=user_exclude_patterns
    )

    assert "main.py" in result                                  # Matches include (*.py)
    assert "ignored_by_git.txt" not in result                   # .gitignore takes precedence
    assert "excluded_by_user.txt" not in result                 # User exclude works
    assert "explicitly_excluded_by_git_but_user_includes.txt" not in result # .gitignore takes precedence over user include
    assert "included_by_user.txt" in result                     # Matches include
    assert "not_included.txt" not in result                     # No include match, and include is active
    assert "config.json" not in result                          # No include match, and include is active


# --- Placeholder Tests for Logic Embedded in Streamlit Handler ---

@pytest.mark.skip(reason="Requires refactoring formatting logic from Streamlit handler in app.py")
def test_markdown_formatting_logic():
    """
    Placeholder for testing the Markdown formatting of a single file.
    This test would ideally call a refactored helper function from app.py, e.g.:
    `_generate_file_markdown_part(path, content, lang)`
    """
    # from app import _generate_file_markdown_part # Hypothetical
    # path = "src/app.py"
    # content = "print('hello')"
    # lang = "python" # Derived language
    # expected_md = "## File: `src/app.py`\n\n```python\nprint('hello')\n```\n\n---\n"
    # md_part = _generate_file_markdown_part(path, content, lang)
    # assert md_part == expected_md
    assert False, "Test not implemented; requires app.py refactoring."

@pytest.mark.skip(reason="Requires refactoring ordering logic from Streamlit handler in app.py")
def test_priority_file_ordering_logic():
    """
    Placeholder for testing the priority file ordering logic.
    This test would ideally call a refactored helper function from app.py, e.g.:
    `_order_and_format_contents(contents, repo_url, priority_filenames_config)`
    """
    # from app import _order_and_format_contents # Hypothetical
    # contents_map = {
    #     "src/app.py": "content1",
    #     "README.md": "readme_content",
    #     "LICENSE": "license_content",
    #     "docs/README.md": "docs_readme"
    # }
    # repo_url = "https://github.com/test/repo"
    # # This would come from the main app logic
    # priority_filenames_lower = ["readme.md", "license", "license.txt", "contributing.md"]
    #
    # # The hypothetical function would return a list of formatted markdown strings in the correct order
    # ordered_markdown_sections = _order_and_format_contents(contents_map, repo_url, priority_filenames_lower)
    #
    # assert "README.md" in ordered_markdown_sections[0] # Check if README is among the first items
    # assert "LICENSE" in ordered_markdown_sections[1]   # Or similar logic depending on exact output
    # assert "src/app.py" in ordered_markdown_sections[-1] # Non-priority last
    # assert "docs/README.md" in ordered_markdown_sections[-1] # Non-priority (not root) also last
    assert False, "Test not implemented; requires app.py refactoring."
