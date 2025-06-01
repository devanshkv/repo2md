import pytest
from unittest.mock import patch, MagicMock
import base64

# Assuming app.py is in the parent directory or PYTHONPATH is set correctly
# For local testing, you might need to adjust sys.path or run pytest with specific settings
# e.g., `python -m pytest` from the root directory
from app import get_repo_api_url, fetch_repo_contents
# IGNORED_EXTENSIONS is a global in app.py, useful for some tests
from app import IGNORED_EXTENSIONS, cached_requests_get # Import new function
import app as streamlit_app # To mock streamlit internals like st.cache_data.clear
import streamlit as st # Required for st.cache_data decorator to be processed
import requests # For requests.exceptions.HTTPError

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

# Helper to simulate calls to fetch_repo_contents (now primarily for path fetching part)
def run_fetch_repo_paths_via_fetch_repo_contents(
        mock_cached_get, # Changed from mock_rg (raw requests.get) to mock_cached_get
        repo_url_test="https://github.com/test/repo",
        include_patterns="", exclude_patterns="", max_kb=0,
        api_responses=None): # api_responses is a dict of url -> response_json_or_item_list

    # Default setup for cached_requests_get mock
    def side_effect_func(url, headers=None): # headers is passed to cached_requests_get
        mock_resp = MagicMock()
        response_data = api_responses.get(url)

        if response_data is None: # URL not mocked
            mock_resp.status_code = 404
            mock_resp.json.side_effect = requests.exceptions.HTTPError(f"Mock Error: URL not found in api_responses: {url}")
            mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"Mock Error: 404 for {url}")
            return mock_resp

        if isinstance(response_data, list): # Directory listing from /contents/
            mock_resp.status_code = 200
            mock_resp.json.return_value = response_data
        elif isinstance(response_data, dict) and "content" in response_data: # .gitignore file from /contents/.gitignore
            mock_resp.status_code = 200
            mock_resp.json.return_value = response_data
        elif isinstance(response_data, str): # Plain text from a download_url (e.g. for .gitignore content)
            mock_resp.status_code = 200
            mock_resp.text = response_data
            # .content would be response_data.encode('utf-8'), not typically used by path fetcher for .gitignore
        elif isinstance(response_data, dict) and "status_code" in response_data: # Explicit error for a path (e.g. .gitignore not found)
            mock_resp.status_code = response_data["status_code"]
            if mock_resp.status_code != 200: # If it's an error status
                mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"Mock Error: Status {mock_resp.status_code}")
                # .json() might also raise or return error details depending on GitHub API
                mock_resp.json.return_value = response_data.get("json_error_payload", {})
        else: # Fallback for unexpected mock structure
            mock_resp.status_code = 500
            mock_resp.raise_for_status.side_effect = Exception(f"Mock Error: Bad api_response structure for {url}")
        return mock_resp

    mock_cached_get.side_effect = side_effect_func

    # Call fetch_repo_contents, which now returns the list of path dictionaries
    # It uses cached_requests_get internally.
    with patch('app.include_patterns_input', include_patterns), \
         patch('app.exclude_patterns_input', exclude_patterns), \
         patch('app.max_file_size_kb', max_kb), \
         patch('app.github_pat', ""): # Assuming no PAT
        # fetch_repo_contents now returns the flat list of paths
        return fetch_repo_contents(repo_url_test)


# Updated tests for path fetching logic (formerly content fetching tests)
def test_path_fetch_ignore_by_extension(mock_cached_requests_get, mock_streamlit_ui):
    # create_github_item now only needs to describe the item for path listing, not content fetching
    repo_files = [
        create_github_item("file.py", "file.py", size=10), # download_url not strictly needed for path test
        create_github_item("image.png", "image.png", size=20),
        create_github_item("script.pyc", "script.pyc", size=30),
    ]
    api_url_root = "https://api.github.com/repos/test/repo"
    api_responses = {
        f"{api_url_root}/contents/": repo_files,
        f"{api_url_root}/contents/.gitignore": {"status_code": 404},
    }

    # fetch_repo_contents now returns a list of path dicts
    result_paths = run_fetch_repo_paths_via_fetch_repo_contents(
        mock_cached_requests_get, api_responses=api_responses, max_kb=0
    )

    path_names = [item['path'] for item in result_paths]
    assert "file.py" in path_names
    assert "image.png" not in path_names # Filtered by IGNORED_EXTENSIONS
    assert "script.pyc" not in path_names # Filtered by IGNORED_EXTENSIONS

def test_path_fetch_ignore_git_directory(mock_cached_requests_get, mock_streamlit_ui):
    repo_files_root = [
        create_github_item("file.py", "file.py"),
        create_github_item(".git", ".git", item_type="dir"), # This dir itself should be ignored for listing
        create_github_item("src", "src", item_type="dir"),
    ]
    # If .git was listed by API (it shouldn't be if API is well-behaved), _fetch_repo_paths_recursive skips it.
    # If API lists contents of .git (e.g. .git/config), those should also be skipped.
    # For this test, assume .git dir itself is filtered out by name, so its contents aren't even requested.
    repo_files_src = [
        create_github_item("app.py", "src/app.py"),
        create_github_item(".git", "src/.git", item_type="dir"), # A nested .git dir name
    ]
    api_url_root = "https://api.github.com/repos/test/repo"
    api_responses = {
        f"{api_url_root}/contents/": repo_files_root,
        # No call to f"{api_url_root}/contents/.git" should happen.
        f"{api_url_root}/contents/src": repo_files_src,
        f"{api_url_root}/contents/.gitignore": {"status_code": 404},
    }

    result_paths = run_fetch_repo_paths_via_fetch_repo_contents(mock_cached_requests_get, api_responses=api_responses)
    path_names = [item['path'] for item in result_paths]

    assert "file.py" in path_names
    assert "src/app.py" in path_names
    # Check that no paths starting with .git or containing .git as a component are present
    for p_name in path_names:
        assert ".git" not in p_name.split('/') # Check each component
        assert not p_name.startswith(".git/") # Check if it's in root .git

def test_path_fetch_gitignore_basic(mock_cached_requests_get, mock_streamlit_ui):
    gitignore_content = "*.log\nbuild/\n"
    repo_files_root = [
        create_github_item("main.py", "main.py"),
        create_github_item("app.log", "app.log"), # Should be ignored by .gitignore
        create_github_item("build", "build", item_type="dir"), # Should be ignored by .gitignore
        create_github_item("docs", "docs", item_type="dir"),
    ]
    # Contents of 'build' should not be fetched if 'build/' is gitignored
    repo_files_build = [create_github_item("output.txt", "build/output.txt")]
    repo_files_docs = [create_github_item("index.md", "docs/index.md")]

    api_url_root = "https://api.github.com/repos/test/repo"
    api_responses = {
        f"{api_url_root}/contents/": repo_files_root,
        f"{api_url_root}/contents/.gitignore": create_github_item(".gitignore", ".gitignore", download_url=f"{api_url_root}/raw/.gitignore"), # download_url for gitignore content
        f"{api_url_root}/raw/.gitignore": gitignore_content, # Actual content of .gitignore
        f"{api_url_root}/contents/docs": repo_files_docs,
        # No entry for f"{api_url_root}/contents/build" as it should be skipped
    }

    result_paths = run_fetch_repo_paths_via_fetch_repo_contents(mock_cached_requests_get, api_responses=api_responses)
    path_names = [item['path'] for item in result_paths]

    assert "main.py" in path_names
    assert "docs/index.md" in path_names
    assert "app.log" not in path_names
    assert "build" not in path_names # The directory itself
    assert "build/output.txt" not in path_names
    # Verify that mock_cached_requests_get was not called for 'build' directory contents
    # Check call_args_list of the mock
    called_urls = [call_args[0][0] for call_args in mock_cached_requests_get.call_args_list]
    assert f"{api_url_root}/contents/build" not in called_urls


def test_path_fetch_user_glob_exclusion(mock_cached_requests_get, mock_streamlit_ui):
    user_exclude_patterns = "*.txt\ntemp/" # Exclude all .txt files and 'temp' directory
    repo_files_root = [
        create_github_item("main.py", "main.py"),
        create_github_item("data.txt", "data.txt"), # Should be excluded by *.txt
        create_github_item("temp", "temp", item_type="dir"), # Should be excluded by temp/
    ]
    repo_files_temp = [create_github_item("temp_file.tmp", "temp/temp_file.tmp")]
    api_url_root = "https://api.github.com/repos/test/repo"
    api_responses = {
        f"{api_url_root}/contents/": repo_files_root,
        f"{api_url_root}/contents/.gitignore": {"status_code": 404},
        # No f"{api_url_root}/contents/temp" as it should be skipped
    }

    result_paths = run_fetch_repo_paths_via_fetch_repo_contents(
        mock_cached_requests_get, api_responses=api_responses, exclude_patterns=user_exclude_patterns
    )
    path_names = [item['path'] for item in result_paths]

    assert "main.py" in path_names
    assert "data.txt" not in path_names
    assert "temp" not in path_names
    assert "temp/temp_file.tmp" not in path_names

def test_path_fetch_user_glob_inclusion(mock_cached_requests_get, mock_streamlit_ui):
    user_include_patterns = "src/**/*.py\n*.md" # Include only .py files in src and .md files anywhere
    repo_files_root = [
        create_github_item("main.py", "main.py"), # Not in src/**, not .md -> out
        create_github_item("README.md", "README.md"), # Matches *.md -> in
        create_github_item("src", "src", item_type="dir"), # Dir, traverse
        create_github_item("docs", "docs", item_type="dir"), # Dir, traverse (inclusion doesn't prune dirs directly)
    ]
    repo_files_src = [
        create_github_item("app.py", "src/app.py"), # Matches src/**/*.py -> in
        create_github_item("utils.js", "src/utils.js"), # Not .py -> out
        create_github_item("sub", "src/sub", item_type="dir"), # Dir, traverse
    ]
    repo_files_src_sub = [
        create_github_item("helper.py", "src/sub/helper.py"), # Matches src/**/*.py -> in
    ]
    repo_files_docs = [create_github_item("info.txt", "docs/info.txt")] # Not .md -> out

    api_url_root = "https://api.github.com/repos/test/repo"
    api_responses = {
        f"{api_url_root}/contents/": repo_files_root,
        f"{api_url_root}/contents/.gitignore": {"status_code": 404},
        f"{api_url_root}/contents/src": repo_files_src,
        f"{api_url_root}/contents/src/sub": repo_files_src_sub,
        f"{api_url_root}/contents/docs": repo_files_docs,
    }

    result_paths = run_fetch_repo_paths_via_fetch_repo_contents(
        mock_cached_requests_get, api_responses=api_responses, include_patterns=user_include_patterns
    )
    path_names = [item['path'] for item in result_paths]

    # Expected paths: README.md, src/app.py, src/sub/helper.py
    # Also directories that lead to these files: src, src/sub
    # And other traversed directories if not excluded: docs
    # The path fetcher returns dirs if they are traversed and not themselves excluded.
    # The actual content fetching is later based on 'file' type from these paths.

    assert "README.md" in path_names
    assert "src/app.py" in path_names
    assert "src/sub/helper.py" in path_names

    assert "main.py" not in path_names
    assert "src/utils.js" not in path_names
    assert "docs/info.txt" not in path_names

    # Check for presence of directories that were traversed
    assert "src" in path_names
    assert "src/sub" in path_names
    assert "docs" in path_names # 'docs' dir itself is not excluded by include pattern, so it's listed. Its files are.


def test_path_fetch_max_file_size_limit(mock_cached_requests_get, mock_streamlit_ui):
    max_kb_limit = 1  # 1 KB limit
    repo_files_root = [
        create_github_item("small.txt", "small.txt", size=500),   # 500 bytes -> in
        create_github_item("large.txt", "large.txt", size=2048),   # 2KB -> out (path itself shouldn't be returned)
        create_github_item("exact.txt", "exact.txt", size=1024),   # 1KB -> in
    ]
    api_url_root = "https://api.github.com/repos/test/repo"
    api_responses = {
        f"{api_url_root}/contents/": repo_files_root,
        f"{api_url_root}/contents/.gitignore": {"status_code": 404},
    }

    result_paths = run_fetch_repo_paths_via_fetch_repo_contents(
        mock_cached_requests_get, api_responses=api_responses, max_kb=max_kb_limit
    )
    path_names = [item['path'] for item in result_paths]

    assert "small.txt" in path_names
    assert "large.txt" not in path_names # Path of oversized file should not be in result
    assert "exact.txt" in path_names

# --- Tests for output formatting and priority (would need to call the main button handler logic) ---
# These were complex and depended on the old structure.
# The new structure separates path fetching, tree building, selection, content fetching, and map generation.
# Many of these are now tested by the newer, more granular tests.
# The old `test_filtering_precedence` needs to be adapted similarly.
def test_path_fetch_filtering_precedence(mock_cached_requests_get, mock_streamlit_ui):
    gitignore_content = "ignored_by_git.txt\nexplicitly_excluded_by_git_but_user_includes.txt"
    user_exclude_patterns = "excluded_by_user.txt\nexplicitly_excluded_by_git_but_user_includes.txt"
    user_include_patterns = "*.py\nincluded_by_user.txt\nexplicitly_excluded_by_git_but_user_includes.txt"

    # Define items for the root directory listing
    repo_files_root = [
        create_github_item("main.py", "main.py"),
        create_github_item("ignored_by_git.txt", "ignored_by_git.txt"),
        create_github_item("excluded_by_user.txt", "excluded_by_user.txt"),
        create_github_item("explicitly_excluded_by_git_but_user_includes.txt", "explicitly_excluded_by_git_but_user_includes.txt"),
        create_github_item("included_by_user.txt", "included_by_user.txt"),
        create_github_item("not_included.txt", "not_included.txt"), # Not in include, should be out
        create_github_item("config.json", "config.json"),       # Not in include, should be out
    ]
    api_url_root = "https://api.github.com/repos/test/repo"
    api_responses = {
        f"{api_url_root}/contents/": repo_files_root,
        f"{api_url_root}/contents/.gitignore": create_github_item(".gitignore", ".gitignore", download_url=f"{api_url_root}/raw/.gitignore"),
        f"{api_url_root}/raw/.gitignore": gitignore_content, # Content for .gitignore
    }
    # No download_urls needed for path fetching tests

    result_paths = run_fetch_repo_paths_via_fetch_repo_contents(
        mock_cached_requests_get,
        api_responses=api_responses,
        include_patterns=user_include_patterns,
        exclude_patterns=user_exclude_patterns
    )
    path_names = [item['path'] for item in result_paths]

    assert "main.py" in path_names
    assert "included_by_user.txt" in path_names

    assert "ignored_by_git.txt" not in path_names
    assert "excluded_by_user.txt" not in path_names
    assert "explicitly_excluded_by_git_but_user_includes.txt" not in path_names
    assert "not_included.txt" not in path_names
    assert "config.json" not in path_names


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


# --- Tests for Caching ---
# Note: Streamlit's caching (@st.cache_data) can be tricky to unit test perfectly
# without running in a Streamlit execution context. These tests will mock `requests.get`
# and observe its call count, which is a good proxy for cache behavior.
# Clearing st.cache_data might require more specific handling or integration-style tests.

def test_cached_requests_get_called_once_for_same_url(mocker):
    mock_get = mocker.patch('app.requests.get')
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": "test"}
    mock_response.text = "test data"
    mock_get.return_value = mock_response

    url = "https://api.example.com/data"
    headers = {"X-Test-Header": "value1"}

    # Call first time
    response1 = cached_requests_get(url, headers)
    # Call second time with same args
    response2 = cached_requests_get(url, headers)

    mock_get.assert_called_once_with(url, headers=headers)
    assert response1.json() == {"data": "test"}
    assert response2.json() == {"data": "test"}
    # Clear cache for other tests
    st.cache_data.clear()


def test_cached_requests_get_called_for_different_urls(mocker):
    mock_get = mocker.patch('app.requests.get')
    mock_response1 = MagicMock()
    mock_response1.status_code = 200
    mock_response1.json.return_value = {"data": "response1"}
    mock_response1.text = "response1 data"

    mock_response2 = MagicMock()
    mock_response2.status_code = 200
    mock_response2.json.return_value = {"data": "response2"}
    mock_response2.text = "response2 data"

    # Setup side_effect to return different responses for different URLs
    def get_side_effect(url, headers):
        if url == "https://api.example.com/data1":
            return mock_response1
        elif url == "https://api.example.com/data2":
            return mock_response2
        raise ValueError(f"Unexpected URL: {url}")
    mock_get.side_effect = get_side_effect

    url1 = "https://api.example.com/data1"
    url2 = "https://api.example.com/data2"
    headers = {"X-Test-Header": "value"}

    cached_requests_get(url1, headers)
    cached_requests_get(url2, headers)

    assert mock_get.call_count == 2
    mock_get.assert_any_call(url1, headers=headers)
    mock_get.assert_any_call(url2, headers=headers)
    st.cache_data.clear()


def test_cached_requests_get_called_for_different_headers(mocker):
    mock_get = mocker.patch('app.requests.get')
    # Return a new mock response object for each call to see different objects if cache misses
    mock_get.side_effect = lambda url, headers: MagicMock(status_code=200, json=lambda: {"url": url, "headers": headers})

    url = "https://api.example.com/data"
    headers1 = {"X-Test-Header": "value1"}
    headers2 = {"X-Test-Header": "value2"}

    cached_requests_get(url, headers1)
    cached_requests_get(url, headers2)

    assert mock_get.call_count == 2
    mock_get.assert_any_call(url, headers=headers1)
    mock_get.assert_any_call(url, headers=headers2)
    st.cache_data.clear()


def test_cached_requests_get_clear_cache(mocker):
    mock_get = mocker.patch('app.requests.get')
    mock_response = MagicMock(status_code=200)
    mock_get.return_value = mock_response

    url = "https://api.example.com/data-clear"
    headers = {"X-Test-Header": "value-clear"}

    # Call first time
    cached_requests_get(url, headers)
    # Clear the cache
    st.cache_data.clear() # This is the actual Streamlit clear function
    # Call second time
    cached_requests_get(url, headers)

    assert mock_get.call_count == 2 # Should be called twice because cache was cleared
    st.cache_data.clear() # Clean up for other tests


# Test that HTTP errors are raised by cached_requests_get
def test_cached_requests_get_raises_http_error(mocker):
    mock_get = mocker.patch('app.requests.get')
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
    mock_get.return_value = mock_response

    url = "https://api.example.com/nonexistent"
    headers = {}
    with pytest.raises(requests.exceptions.HTTPError):
        cached_requests_get(url, headers)
    st.cache_data.clear()


# --- Tests for Tree Generation (build_file_tree_from_paths) ---

@pytest.fixture
def sample_paths_list_flat():
    # Primarily files at the root, or simple structure
    return [
        {"path": "README.md", "type": "file", "name": "README.md", "sha": "readme_sha"},
        {"path": "app.py", "type": "file", "name": "app.py", "sha": "app_sha"},
        {"path": "src", "type": "dir", "name": "src", "sha": "src_dir_sha"},
        {"path": "src/utils.py", "type": "file", "name": "utils.py", "sha": "utils_sha"},
    ]

@pytest.fixture
def sample_paths_list_nested():
    return [
        {"path": "README.md", "type": "file", "name": "README.md", "sha": "readme_sha"},
        {"path": "project", "type": "dir", "name": "project", "sha": "project_dir_sha"},
        {"path": "project/main.py", "type": "file", "name": "main.py", "sha": "main_py_sha"},
        {"path": "project/core", "type": "dir", "name": "core", "sha": "core_dir_sha"},
        {"path": "project/core/api.py", "type": "file", "name": "api.py", "sha": "api_py_sha"},
        {"path": "project/core/impl", "type": "dir", "name": "impl", "sha": "impl_dir_sha"},
        {"path": "project/core/impl/logic.py", "type": "file", "name": "logic.py", "sha": "logic_py_sha"},
        {"path": "tests", "type": "dir", "name": "tests", "sha": "tests_dir_sha"},
        {"path": "tests/test_main.py", "type": "file", "name": "test_main.py", "sha": "test_main_sha"},
    ]

def test_build_file_tree_empty_input():
    from app import build_file_tree_from_paths
    assert build_file_tree_from_paths([]) == []

def test_build_file_tree_flat_structure(sample_paths_list_flat):
    from app import build_file_tree_from_paths
    tree = build_file_tree_from_paths(sample_paths_list_flat)

    assert len(tree) == 3 # README.md, app.py, src

    readme_node = next(n for n in tree if n['label'] == "README.md")
    app_node = next(n for n in tree if n['label'] == "app.py")
    src_node = next(n for n in tree if n['label'] == "src")

    assert readme_node['id'] == "readme_sha"
    assert readme_node['icon'] == "file-text"
    assert not readme_node.get('children')

    assert app_node['id'] == "app_sha"
    assert app_node['icon'] == "file-text"

    assert src_node['id'] == "src_dir_sha"
    assert src_node['icon'] == "folder"
    assert len(src_node.get('children', [])) == 1

    utils_node = src_node['children'][0]
    assert utils_node['label'] == "utils.py"
    assert utils_node['id'] == "utils_sha"
    assert utils_node['icon'] == "file-text"

def test_build_file_tree_nested_structure(sample_paths_list_nested):
    from app import build_file_tree_from_paths
    tree = build_file_tree_from_paths(sample_paths_list_nested)

    assert len(tree) == 3 # README.md, project, tests

    project_node = next(n for n in tree if n['label'] == "project")
    assert project_node['icon'] == "folder"
    assert len(project_node.get('children', [])) == 2 # main.py, core

    main_py_node = next(n for n in project_node['children'] if n['label'] == "main.py")
    core_node = next(n for n in project_node['children'] if n['label'] == "core")

    assert main_py_node['icon'] == "file-text"
    assert core_node['icon'] == "folder"
    assert len(core_node.get('children', [])) == 2 # api.py, impl

    api_py_node = next(n for n in core_node['children'] if n['label'] == "api.py")
    impl_node = next(n for n in core_node['children'] if n['label'] == "impl")

    assert api_py_node['icon'] == "file-text"
    assert impl_node['icon'] == "folder"
    assert len(impl_node.get('children', [])) == 1 # logic.py

    logic_py_node = impl_node['children'][0]
    assert logic_py_node['label'] == "logic.py"
    assert logic_py_node['icon'] == "file-text"

    tests_node = next(n for n in tree if n['label'] == "tests")
    assert tests_node['icon'] == "folder"
    assert len(tests_node.get('children', [])) == 1 # test_main.py
    test_main_node = tests_node['children'][0]
    assert test_main_node['label'] == "test_main.py"

# Test case for when a path might be missing its parent in the list (e.g. due to filtering)
# The current build_file_tree_from_paths is strict and might skip such orphans if parent_path is not ""
# or log a warning. If it adds to root, this test needs to check that.
# For now, assuming it skips if parent is not found and parent_path is not root.
def test_build_file_tree_with_orphaned_paths(mocker):
    from app import build_file_tree_from_paths
    mock_st_warning = mocker.patch('app.st.warning')

    paths_list_with_orphan = [
        {"path": "README.md", "type": "file", "name": "README.md", "sha": "readme_sha"},
        # "src" directory is missing from the list
        {"path": "src/deep/orphan.py", "type": "file", "name": "orphan.py", "sha": "orphan_sha"},
    ]
    tree = build_file_tree_from_paths(paths_list_with_orphan)

    assert len(tree) == 1 # Only README.md
    readme_node = next(n for n in tree if n['label'] == "README.md")
    assert readme_node is not None

    # Check that no node for orphan.py is present at the root
    assert not any(n['label'] == "orphan.py" for n in tree)
    # Check that src or deep were not accidentally created at root
    assert not any(n['label'] == "src" for n in tree)
    assert not any(n['label'] == "deep" for n in tree)

    # Check if st.warning was called for the orphan
    # The warning message is "Parent node for '{path}' (parent: '{parent_path}') not found in temp_map..."
    # For "src/deep/orphan.py", the first missing parent is "src".
    # Then, when trying to place "deep" under "src", "src" is not found.
    # When trying to place "orphan.py" under "src/deep", "src/deep" is not found.
    # The function creates parent dicts on the fly if they are part of a known path.
    # The actual warning might be for the first segment it can't place.
    # The current logic might create "src" and "deep" if it iterates path segments.
    # Let's re-verify build_file_tree_from_paths logic for parent creation.
    # It uses `parent_path = "/".join(path.split("/")[:-1])` and `nodes_map.get(parent_path)`.
    # If `parent_node` is None and `parent_path` is not "", it warns.
    # So "src/deep/orphan.py" -> parent_path "src/deep". If "src/deep" not in map, it warns.
    # It does NOT create intermediate parents if they are not explicitly listed as 'dir' type items.

    # In this specific case, "src/deep/orphan.py" has parent "src/deep".
    # "src/deep" is not in the map. So, it should warn for "src/deep/orphan.py".
    # The map creation for dirs happens if item_type == "dir".
    # The test paths_list_with_orphan does not list "src" or "src/deep" as dirs.
    # So the warning should be for "src/deep/orphan.py".

    # Let's refine the assertion for the warning.
    # The structure is built by iterating paths_list. "README.md" is fine.
    # For "src/deep/orphan.py": parent_path is "src/deep". temp_nodes_map does not contain "src/deep".
    # So it hits the `else` where `parent_node_in_temp_map` is None.
    # Then `parent_path` ("src/deep") is not `""`. So it calls `st.warning`.

    # We need to ensure `build_file_tree_from_paths` is imported from `app` where `st` is mocked by `mock_streamlit_ui`
    # For this test, we only need `st.warning`.

    # Expected warning for "src/deep/orphan.py" because parent "src/deep" is not in the map.
    # The path in the warning is "src/deep/orphan.py", parent is "src/deep".
    mock_st_warning.assert_any_call("Parent node for 'src/deep/orphan.py' (parent: 'src/deep') not found in temp_map. Item may be skipped or added to root if logic allows.")

    # If the code were to create intermediate stubs, the tree would be different.
    # Given the current code, this is the expected outcome.


# --- Tests for Selection Logic (get_effective_selected_files) ---

@pytest.fixture
def sample_sac_tree_data():
    # This tree structure should be what `build_file_tree_from_paths` outputs.
    # IDs are SHAs or paths. Labels are names. Icons indicate type.
    return [
        {"id": "readme_sha", "label": "README.md", "icon": "file-text", "path": "README.md"},
        {
            "id": "src_dir_sha", "label": "src", "icon": "folder", "path": "src",
            "children": [
                {"id": "app_py_sha", "label": "app.py", "icon": "file-text", "path": "src/app.py"},
                {
                    "id": "components_dir_sha", "label": "components", "icon": "folder", "path": "src/components",
                    "children": [
                        {"id": "button_py_sha", "label": "button.py", "icon": "file-text", "path": "src/components/button.py"}
                    ]
                },
                {"id": "empty_sub_dir_sha", "label": "empty_sub", "icon": "folder", "path": "src/empty_sub", "children": []}
            ]
        },
        {"id": "config_sha", "label": "config.json", "icon": "file-text", "path": "config.json"},
        {"id": "docs_dir_sha", "label": "docs", "icon": "folder", "path": "docs", "children": [
            {"id": "main_docs_sha", "label": "main.md", "icon": "file-text", "path": "docs/main.md"}
        ]}
    ]

def test_get_effective_selected_files_no_selection(sample_sac_tree_data):
    from app import get_effective_selected_files
    assert get_effective_selected_files(sample_sac_tree_data, []) == []

def test_get_effective_selected_files_only_files_selected(sample_sac_tree_data):
    from app import get_effective_selected_files
    selected_ids = ["readme_sha", "app_py_sha", "config_sha"]
    expected_files = ["README.md", "src/app.py", "config.json"]
    actual_files = get_effective_selected_files(sample_sac_tree_data, selected_ids)
    assert sorted(actual_files) == sorted(expected_files)


# --- Tests for Content Fetching (fetch_content_for_selected_files) ---

@pytest.fixture
def mock_all_paths_metadata():
    # This is the flat list of all items, similar to st.session_state['repo_file_paths']
    return [
        {"path": "README.md", "type": "file", "name": "README.md", "sha": "readme_sha", "size": 100, "download_url": "d_url_readme"},
        {"path": "src/app.py", "type": "file", "name": "app.py", "sha": "app_py_sha", "size": 500, "download_url": "d_url_app_py"},
        {"path": "src/components/button.py", "type": "file", "name": "button.py", "sha": "button_py_sha", "size": 200, "download_url": "d_url_button_py"},
        {"path": "config.json", "type": "file", "name": "config.json", "sha": "config_sha", "size": 80, "download_url": "d_url_config"},
        {"path": "large_file.txt", "type": "file", "name": "large_file.txt", "sha": "large_sha", "size": 2048, "download_url": "d_url_large"}, # 2KB
        {"path": "no_download.txt", "type": "file", "name": "no_download.txt", "sha": "no_dl_sha", "size": 50, "download_url": None},
        {"path": "src", "type": "dir", "name": "src", "sha": "src_dir_sha"}, # A directory, should be ignored by content fetcher
    ]

# We need to mock cached_requests_get for these tests
@pytest.fixture
def mock_cached_requests_get(mocker):
    return mocker.patch('app.cached_requests_get')

def test_fetch_content_selected_basic(mock_cached_requests_get, mock_all_paths_metadata, mock_streamlit_ui):
    from app import fetch_content_for_selected_files

    selected_paths = ["README.md", "src/app.py"]
    headers = {"Authorization": "token test_pat"}
    max_kb = 1024 # 1MB limit, all selected files are smaller

    # Mock responses from cached_requests_get
    def mock_get_side_effect(url, headers_arg):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        if url == "d_url_readme":
            mock_resp.content = b"Readme content"
        elif url == "d_url_app_py":
            mock_resp.content = b"App.py content"
        else:
            mock_resp.status_code = 404
            mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("Not found in mock_get_side_effect")
        return mock_resp
    mock_cached_requests_get.side_effect = mock_get_side_effect

    result_content = fetch_content_for_selected_files(selected_paths, mock_all_paths_metadata, headers, max_kb)

    assert len(result_content) == 2
    assert result_content["README.md"] == "Readme content"
    assert result_content["src/app.py"] == "App.py content"

    mock_cached_requests_get.assert_any_call("d_url_readme", headers=headers)
    mock_cached_requests_get.assert_any_call("d_url_app_py", headers=headers)
    assert mock_cached_requests_get.call_count == 2


def test_fetch_content_respects_max_file_size(mock_cached_requests_get, mock_all_paths_metadata, mock_streamlit_ui):
    from app import fetch_content_for_selected_files

    selected_paths = ["README.md", "large_file.txt"] # large_file.txt is 2KB
    headers = {}
    max_kb = 1 # 1KB limit

    mock_cached_requests_get.return_value = MagicMock(status_code=200, content=b"Readme content") # Only for README

    result_content = fetch_content_for_selected_files(selected_paths, mock_all_paths_metadata, headers, max_kb)

    assert len(result_content) == 1
    assert "README.md" in result_content
    assert "large_file.txt" not in result_content
    # Ensure cached_requests_get was only called for README.md
    mock_cached_requests_get.assert_called_once_with("d_url_readme", headers=headers)


def test_fetch_content_handles_missing_download_url(mock_cached_requests_get, mock_all_paths_metadata, mock_streamlit_ui):
    from app import fetch_content_for_selected_files
    selected_paths = ["no_download.txt", "README.md"]
    headers = {}
    max_kb = 1024

    # Setup mock for README.md only
    mock_readme_resp = MagicMock(status_code=200, content=b"Readme content")
    mock_cached_requests_get.return_value = mock_readme_resp

    result_content = fetch_content_for_selected_files(selected_paths, mock_all_paths_metadata, headers, max_kb)

    assert len(result_content) == 1
    assert "README.md" in result_content
    assert "no_download.txt" not in result_content
    mock_cached_requests_get.assert_called_once_with("d_url_readme", headers=headers)

def test_fetch_content_handles_download_error(mock_cached_requests_get, mock_all_paths_metadata, mock_streamlit_ui):
    from app import fetch_content_for_selected_files
    selected_paths = ["src/app.py", "README.md"] # app.py will fail
    headers = {}
    max_kb = 1024

    def mock_get_side_effect(url, headers_arg):
        if url == "d_url_app_py":
            raise requests.exceptions.RequestException("Simulated download error")
        elif url == "d_url_readme":
            return MagicMock(status_code=200, content=b"Readme ok")
        return MagicMock(status_code=404) # Should not happen for these paths
    mock_cached_requests_get.side_effect = mock_get_side_effect

    result_content = fetch_content_for_selected_files(selected_paths, mock_all_paths_metadata, headers, max_kb)

    assert len(result_content) == 1
    assert "README.md" in result_content
    assert "src/app.py" not in result_content
    # Check calls for both, even if one fails
    mock_cached_requests_get.assert_any_call("d_url_app_py", headers=headers)
    mock_cached_requests_get.assert_any_call("d_url_readme", headers=headers)

def test_fetch_content_skip_non_file_type(mock_cached_requests_get, mock_all_paths_metadata, mock_streamlit_ui):
    # Although get_effective_selected_files should only return files, test robustness here.
    from app import fetch_content_for_selected_files
    selected_paths = ["src"] # "src" is a directory in metadata
    headers = {}
    max_kb = 1024

    result_content = fetch_content_for_selected_files(selected_paths, mock_all_paths_metadata, headers, max_kb)

    assert len(result_content) == 0
    mock_cached_requests_get.assert_not_called()

def test_fetch_content_handles_decoding_error(mock_cached_requests_get, mock_all_paths_metadata, mock_streamlit_ui):
    from app import fetch_content_for_selected_files
    selected_paths = ["src/app.py"] # app.py will have bad content
    headers = {}
    max_kb = 1024

    # Content that fails both utf-8 and latin-1
    bad_content = b'\x80\xff'
    mock_response = MagicMock(status_code=200, content=bad_content)
    mock_cached_requests_get.return_value = mock_response

    result_content = fetch_content_for_selected_files(selected_paths, mock_all_paths_metadata, headers, max_kb)

    assert len(result_content) == 0
    assert "src/app.py" not in result_content
    mock_cached_requests_get.assert_called_once_with("d_url_app_py", headers=headers)
    # mock_streamlit_ui.st_warning should have been called
    # How to assert st.warning calls with pytest-mock and mocker fixture?
    # 'mock_streamlit_ui' fixture patches st.warning. We can check its call.
    # This requires `mock_streamlit_ui` to be active for the `app` module.
    streamlit_app.st.warning.assert_any_call("Could not decode file src/app.py with UTF-8 or Latin-1. Skipping.")

# Ensure mock_streamlit_ui is used if testing st.info/warning/error calls from the SUT
@pytest.mark.usefixtures("mock_streamlit_ui")
class TestFetchContentWithUIFeedback:
    def test_fetch_content_feedback_on_skip_large(self, mock_cached_requests_get, mock_all_paths_metadata):
        from app import fetch_content_for_selected_files
        selected_paths = ["large_file.txt"]
        fetch_content_for_selected_files(selected_paths, mock_all_paths_metadata, {}, 1)
        streamlit_app.st.warning.assert_any_call("Skipping 'large_file.txt' (size: 2.00 KB) as it exceeds the 1 KB limit for download.")

    def test_fetch_content_feedback_on_missing_url(self, mock_cached_requests_get, mock_all_paths_metadata):
        from app import fetch_content_for_selected_files
        selected_paths = ["no_download.txt"]
        fetch_content_for_selected_files(selected_paths, mock_all_paths_metadata, {}, 1024)
        streamlit_app.st.warning.assert_any_call("No download_url for file 'no_download.txt'. Skipping.")

    def test_fetch_content_feedback_on_download_error(self, mock_cached_requests_get, mock_all_paths_metadata):
        from app import fetch_content_for_selected_files
        mock_cached_requests_get.side_effect = requests.exceptions.RequestException("Simulated error")
        selected_paths = ["README.md"]
        fetch_content_for_selected_files(selected_paths, mock_all_paths_metadata, {}, 1024)
        streamlit_app.st.warning.assert_any_call("Error downloading file README.md: Simulated error. Skipping.")


# --- Tests for Repo Map Generation (generate_repo_map) ---
# Uses sample_sac_tree_data fixture from selection logic tests

def test_generate_repo_map_no_selection(sample_sac_tree_data):
    from app import generate_repo_map
    repo_map_str = generate_repo_map(sample_sac_tree_data, [])
    assert repo_map_str == "" # Or perhaps a root indicator if desired, current logic returns empty

def test_generate_repo_map_selected_files_only(sample_sac_tree_data):
    from app import generate_repo_map
    # Select README.md (root), src/app.py (under src), src/components/button.py (deeply nested)
    selected_ids = ["readme_sha", "app_py_sha", "button_py_sha"]

    # Expected: show selected files and their parent dirs to form the path
    # README.md
    # src/
    # ├── app.py
    # └── components/
    #     └── button.py
    # This requires careful checking of the logic in generate_repo_map regarding
    # how it includes parent directories. The current logic of generate_repo_map is:
    # `if node_is_selected or node_has_selected_descendant:`
    # This means a parent is only shown if it's selected OR has a selected descendant.

    repo_map_str = generate_repo_map(sample_sac_tree_data, selected_ids)
    expected_lines = [
        "├── README.md", # Root file selected
        "└── src/",      # Dir containing selected files
        "    ├── app.py", # Selected file
        "    └── components/", # Dir containing selected file
        "        └── button.py" # Selected file
    ]
    # Normalize by splitting lines and stripping, then comparing
    actual_lines_processed = [line.strip() for line in repo_map_str.splitlines() if line.strip()]
    expected_lines_processed = [line.strip() for line in expected_lines]

    # print("\nActual Map (selected_files_only):")
    # print(repo_map_str)
    # print("\nActual Processed:")
    # print(actual_lines_processed)
    # print("\nExpected Processed:")
    # print(expected_lines_processed)

    assert actual_lines_processed == expected_lines_processed

def test_generate_repo_map_selected_directory(sample_sac_tree_data):
    from app import generate_repo_map
    # Select 'src' directory. All items under it that are also in selected_ids should appear.
    # If only 'src_dir_sha' is selected, the map should show 'src/' and all its children
    # that were part of the original `selected_ids` passed to the function.
    # However, generate_repo_map takes selected_node_ids. If we only pass "src_dir_sha",
    # it will show "src/" and then recurse. If children are not in selected_node_ids, they won't be shown.
    # This means the `selected_node_ids` for `generate_repo_map` should ideally be the
    # *effective* selections, i.e., all items that should be visible.

    # Let's assume selected_ids for generate_repo_map means "these items are marked for display".
    # If "src_dir_sha" is selected, and its children "app_py_sha", "button_py_sha" are also implicitly selected
    # (e.g. by get_effective_selected_files, though that returns file paths not ids for map generation).
    # For this test, we assume selected_ids contains the IDs of items to be displayed in the map.

    selected_ids_for_map = ["src_dir_sha", "app_py_sha", "components_dir_sha", "button_py_sha"]
    repo_map_str = generate_repo_map(sample_sac_tree_data, selected_ids_for_map)

    # Expected:
    # src/  (because src_dir_sha is selected)
    # ├── app.py (because app_py_sha is selected)
    # └── components/ (because components_dir_sha is selected)
    #     └── button.py (because button_py_sha is selected)
    # The root `src` dir itself will be the first item if the tree traversal starts from root.
    # The sample_sac_tree_data is a list of root items. 'src' is one of them.

    expected_lines = [
        "└── src/",      # src_dir_sha selected
        "    ├── app.py", # app_py_sha selected
        "    └── components/", # components_dir_sha selected
        "        └── button.py" # button_py_sha selected
    ]
    actual_lines_processed = [line.strip() for line in repo_map_str.splitlines() if line.strip()]
    # print("\nActual Map (selected_directory):")
    # print(repo_map_str)
    # print("\nActual Processed:")
    # print(actual_lines_processed)
    # print("\nExpected Processed:")
    # print(expected_lines_processed)

    # This test needs to be precise about what `sample_sac_tree_data` node is passed.
    # `generate_repo_map` is called with the full tree and the selected IDs.
    # It iterates through the root nodes. If 'src' is found and it (or descendant) is selected, it's processed.

    # Filter sample_sac_tree_data to only process the 'src' branch for this test's expectation
    src_branch_tree_data = [node for node in sample_sac_tree_data if node['id'] == 'src_dir_sha']
    repo_map_for_src_branch = generate_repo_map(src_branch_tree_data, selected_ids_for_map)

    # print("\nActual Map (src branch only for map func):")
    # print(repo_map_for_src_branch)
    actual_lines_processed_src_branch = [line.strip() for line in repo_map_for_src_branch.splitlines() if line.strip()]
    assert actual_lines_processed_src_branch == expected_lines


def test_generate_repo_map_unselected_parents_shown_for_context(sample_sac_tree_data):
    from app import generate_repo_map
    # Only "button.py" is selected. Its parents "src" and "components" should be shown for context.
    selected_ids = ["button_py_sha"]
    repo_map_str = generate_repo_map(sample_sac_tree_data, selected_ids)

    expected_lines = [
        "└── src/",          # Not selected, but parent of selected
        "    └── components/", # Not selected, but parent of selected
        "        └── button.py"  # Selected
    ]
    actual_lines_processed = [line.strip() for line in repo_map_str.splitlines() if line.strip()]
    # print("\nActual Map (unselected_parents):")
    # print(repo_map_str)
    assert actual_lines_processed == expected_lines

def test_generate_repo_map_root_file_and_nested_file(sample_sac_tree_data):
    from app import generate_repo_map
    selected_ids = ["readme_sha", "button_py_sha"] # README.md at root, button.py nested
    repo_map_str = generate_repo_map(sample_sac_tree_data, selected_ids)

    expected_lines = [
        "├── README.md",     # Selected
        "└── src/",          # Parent of button.py
        "    └── components/", # Parent of button.py
        "        └── button.py"  # Selected
    ]
    actual_lines_processed = [line.strip() for line in repo_map_str.splitlines() if line.strip()]
    # print("\nActual Map (root_file_and_nested):")
    # print(repo_map_str)
    assert actual_lines_processed == expected_lines

def test_get_effective_selected_files_directory_selected(sample_sac_tree_data):
    from app import get_effective_selected_files
    # Select 'src' directory. All files under it should be included.
    selected_ids = ["src_dir_sha"]
    expected_files = ["src/app.py", "src/components/button.py"] # No files in src/empty_sub
    actual_files = get_effective_selected_files(sample_sac_tree_data, selected_ids)
    assert sorted(actual_files) == sorted(expected_files)

def test_get_effective_selected_files_mix_files_and_dirs(sample_sac_tree_data):
    from app import get_effective_selected_files
    selected_ids = ["readme_sha", "src_dir_sha", "config_sha", "docs_dir_sha"]
    expected_files = [
        "README.md",
        "src/app.py", "src/components/button.py", # from src_dir_sha
        "config.json",
        "docs/main.md" # from docs_dir_sha
    ]
    actual_files = get_effective_selected_files(sample_sac_tree_data, selected_ids)
    assert sorted(actual_files) == sorted(expected_files)

def test_get_effective_selected_files_empty_dir_selected(sample_sac_tree_data):
    from app import get_effective_selected_files
    # Select 'src/empty_sub' which is an empty directory
    selected_ids = ["empty_sub_dir_sha"]
    expected_files = [] # No files in this directory
    actual_files = get_effective_selected_files(sample_sac_tree_data, selected_ids)
    assert sorted(actual_files) == sorted(expected_files)

def test_get_effective_selected_files_file_within_unselected_dir(sample_sac_tree_data):
    from app import get_effective_selected_files
    # 'src' dir is NOT selected, but 'src/app.py' IS selected.
    selected_ids = ["app_py_sha"]
    expected_files = ["src/app.py"]
    actual_files = get_effective_selected_files(sample_sac_tree_data, selected_ids)
    assert sorted(actual_files) == sorted(expected_files)

def test_get_effective_selected_files_deeply_nested_file_selected(sample_sac_tree_data):
    from app import get_effective_selected_files
    # Select only 'src/components/button.py'
    selected_ids = ["button_py_sha"]
    expected_files = ["src/components/button.py"]
    actual_files = get_effective_selected_files(sample_sac_tree_data, selected_ids)
    assert sorted(actual_files) == sorted(expected_files)

def test_get_effective_selected_files_parent_and_child_file_selected(sample_sac_tree_data):
    from app import get_effective_selected_files
    # Select 'src' directory AND 'src/app.py' file explicitly.
    # The function should correctly handle potential duplicates if a file is covered by a dir selection
    # and also selected explicitly. The use of a set internally handles this.
    selected_ids = ["src_dir_sha", "app_py_sha"]
    expected_files = ["src/app.py", "src/components/button.py"]
    actual_files = get_effective_selected_files(sample_sac_tree_data, selected_ids)
    assert sorted(actual_files) == sorted(expected_files)
