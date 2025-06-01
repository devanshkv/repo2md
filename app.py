import streamlit as st
try:
    import requests
except ImportError:
    st.error("The 'requests' library is not installed. Please install it by running: pip install requests")
    st.stop()
try:
    import pathspec
except ImportError:
    st.error("The 'pathspec' library is not installed. Please install it by running: pip install pathspec")
    st.stop()
import base64

# Common non-text file extensions to ignore
IGNORED_EXTENSIONS = [
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.ico', # Images
    '.zip', '.tar', '.gz', '.rar', '.7z', # Archives
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', # Documents
    '.mp3', '.wav', '.ogg', '.mp4', '.avi', '.mov', '.flv', # Media
    '.exe', '.dll', '.so', '.o', '.a', # Binaries
    '.pyc', '.pyo', # Python compiled
    '.class', # Java compiled
    '.DS_Store', # macOS system file
    '.ipynb', # Jupyter notebooks (can be large and have complex JSON structure)
    # Font files
    '.woff', '.woff2', '.ttf', '.eot', '.otf',
    # Other common binary or non-primary source files
    '.svg', # Can be XML, but often complex for LLM context
    '.lock', # Dependency lock files (e.g. package-lock.json, yarn.lock, poetry.lock) - often very long
    '.log' # Log files
]

st.set_page_config(page_title="GitHub to Markdown", layout="wide")
st.title("GitHub Repository to LLM Markdown Converter")
st.markdown("""
Enter a public GitHub repository URL to fetch its text-based file contents and convert them into a single Markdown string,
suitable for input into Large Language Models. Provide optional glob patterns for fine-grained control over included/excluded files.
""")

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Settings")

st.sidebar.subheader("Repository Information")
repo_url = st.sidebar.text_input("GitHub Repository URL:", placeholder="e.g., https://github.com/streamlit/streamlit")
github_pat = st.sidebar.text_input("Optional: GitHub PAT", type="password", help="PAT increases API rate limits. Use with 'repo' scope for private repos.")

st.sidebar.subheader("Custom File Filtering")
col_glob1, col_glob2 = st.sidebar.columns(2)
with col_glob1:
    include_patterns_input = st.sidebar.text_area(
        "Inclusion Globs (per line):",
        placeholder="e.g.,\nsrc/**/*.py\n*.md\n!**/test_*.py",
        help="Files matching these patterns will be included. Supports .gitignore glob syntax. `!` prefix negates."
    )
with col_glob2:
    exclude_patterns_input = st.sidebar.text_area(
        "Exclusion Globs (per line):",
        placeholder="e.g.,\n*.log\ntemp/\n**/*.tmp\n!important.log",
        help="Files matching these patterns will be excluded. Supports .gitignore glob syntax. `!` prefix negates."
    )

st.sidebar.subheader("Content Size Management")
col_size1, col_size2 = st.sidebar.columns(2)
with col_size1:
    max_file_size_kb = st.sidebar.number_input(
        "Max File Size (KB):",
        min_value=0,
        value=1024,  # Default 1MB
        step=128,
        help="Files larger than this (KB) are skipped. 0 for no limit."
    )
with col_size2:
    total_output_threshold_mb = st.sidebar.number_input(
        "Total Output Threshold (MB):",
        min_value=0,
        value=5,  # Default 5MB
        step=1,
        help="Warn if total Markdown exceeds this (MB)."
    )

def get_repo_api_url(repo_url):
    # Example: https://github.com/owner/repo -> https://api.github.com/repos/owner/repo
    if not repo_url.startswith("https://github.com/"):
        return None
    parts = repo_url.strip("/").split("/")
    if len(parts) < 4:
        return None
    owner = parts[-2]
    repo = parts[-1]
    return f"https://api.github.com/repos/{owner}/{repo}"

def fetch_repo_contents(repo_url):
    api_url = get_repo_api_url(repo_url)
    if not api_url:
        st.error("Invalid GitHub repository URL format. Expected format: https://github.com/owner/repo")
        return {}

    contents_url = f"{api_url}/contents"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if github_pat:
        headers["Authorization"] = f"token {github_pat}"

    repo_files_content = {}
    processed_paths = set()

    # Parse user-defined glob patterns
    include_spec = None
    if include_patterns_input:
        try:
            include_spec = pathspec.PathSpec.from_lines('gitwildmatch', include_patterns_input.splitlines())
            st.info("Using custom inclusion patterns.")
        except Exception as e:
            st.error(f"Error parsing inclusion patterns: {e}")
            # Potentially stop or proceed without inclusion patterns

    exclude_spec = None
    if exclude_patterns_input:
        try:
            exclude_spec = pathspec.PathSpec.from_lines('gitwildmatch', exclude_patterns_input.splitlines())
            st.info("Using custom exclusion patterns.")
        except Exception as e:
            st.error(f"Error parsing exclusion patterns: {e}")
            # Potentially stop or proceed without exclusion patterns

    # Fetch and parse .gitignore
    gitignore_spec = None
    try:
        gitignore_url = f"{api_url}/contents/.gitignore"
        # ... (rest of .gitignore fetching logic remains the same) ...
        response = requests.get(gitignore_url, headers=headers)
        if response.status_code == 200:
            gitignore_data = response.json()
            if gitignore_data.get('encoding') == 'base64' and gitignore_data.get('content'):
                gitignore_content = base64.b64decode(gitignore_data['content']).decode('utf-8', errors='replace')
                standard_ignores = ["*.pyc", "*.pyo", "*.pyd", "__pycache__/", ".DS_Store"] # Keep these minimal as .gitignore is primary
                gitignore_lines = gitignore_content.splitlines() # Do not add standard_ignores here if they should be overridable by user patterns
                gitignore_spec = pathspec.PathSpec.from_lines('gitwildmatch', gitignore_lines)
                st.info("Successfully fetched and parsed .gitignore rules.")
            elif gitignore_data.get('download_url'): # Try download_url if content not directly available
                gitignore_content_resp = requests.get(gitignore_data['download_url'], headers=headers)
                gitignore_content_resp.raise_for_status()
                gitignore_content = gitignore_content_resp.text
                gitignore_spec = pathspec.PathSpec.from_lines('gitwildmatch', gitignore_content.splitlines())
                st.info("Successfully fetched and parsed .gitignore rules from download_url.")
            else:
                st.warning(".gitignore found but content is not in expected format or no download_url.")
        elif response.status_code == 404:
            st.info(".gitignore not found in the repository root.")
        else:
            st.warning(f"Failed to fetch .gitignore: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.warning(f"Error fetching .gitignore: {e}")
    except Exception as e:
        st.error(f"Error processing .gitignore: {e}")


    def _get_files_recursive(path=""):
        if path in processed_paths:
            return
        processed_paths.add(path)

        relative_path_for_specs = path.lstrip('./') # Used for all pathspec matching

        # Rule a: Always ignore .git directory (path component check)
        if '/.git/' in f'/{relative_path_for_specs}/' or relative_path_for_specs.startswith('.git/'):
            st.info(f"Ignoring (dot git dir): {path}")
            return

        # Rule b: IGNORED_EXTENSIONS and hardcoded directory names (primary for binaries and common unwanted folders)
        # This check is on individual item['name'] later, but for directories, we can check the path here.
        # For directories, this mainly applies to names like 'node_modules', 'venv' if the path itself matches.
        # Files are checked later.
        hardcoded_ignored_dir_names = ['node_modules', 'venv', '__pycache__', 'dist', 'build', '.vscode', '.idea']
        if any(f'/{ignored_name}/' in f'/{relative_path_for_specs}/' for ignored_name in hardcoded_ignored_dir_names) or \
           relative_path_for_specs in hardcoded_ignored_dir_names:
            # Check if any part of the path is a hardcoded ignored directory name
            # or if the path itself is one of these names (for root level ignored dirs)
            st.info(f"Ignoring (hardcoded dir name in path): {path}")
            return


        # Rule c: .gitignore rules
        if gitignore_spec and gitignore_spec.match_file(relative_path_for_specs):
            st.info(f"Ignoring ('.gitignore'): {path}")
            return

        # Rule d: User-defined EXCLUSION glob patterns
        if exclude_spec and exclude_spec.match_file(relative_path_for_specs):
            st.info(f"Ignoring (user exclusion pattern): {path}")
            return

        # Rule e: User-defined INCLUSION glob patterns (applies to files and affects directory traversal)
        # If include_spec exists, a file MUST match it. A directory MUST match it OR be a parent of a matched file.
        # For directories, if include_spec is active, we only traverse if the directory *could* contain included files.
        # pathspec's match_tree_files is good for this, but for simplicity, we can match the dir path itself.
        # If a dir like 'src/' is not matched by 'src/**/*.py' directly, match_file will be false.
        # However, we still need to traverse 'src/'.
        # So, for directories, this check is more about *not* pruning if it's a parent of an included item.
        # The actual file check happens later. Here, we decide whether to even list contents of 'path'.

        # This specific inclusion logic for directories is tricky if we don't use match_tree_files.
        # A simpler approach for directories: if include_patterns exist, and the current 'path' (a directory)
        # itself does not match any include pattern (e.g. 'src/' for a pattern like 'src/**'), then we might
        # be tempted to prune it. But we shouldn't if the pattern is 'src/sub/file.py'.
        # So, the inclusion check for directories is effectively "don't prune yet, check files inside".
        # The file-level inclusion check is definitive.

        current_url = f"{contents_url}/{path}"
        try:
            response = requests.get(current_url, headers=headers)
            response.raise_for_status() # Raise an exception for HTTP errors
            items = response.json()

            # Sort items to process files before directories (optional, but can be cleaner)
            items.sort(key=lambda x: x['type'], reverse=True)

            for item in items:
                # Construct item_path carefully to avoid leading/double slashes
                if not path or path.endswith("/"):
                    item_path = f"{path}{item['name']}"
                else:
                    item_path = f"{path}/{item['name']}"

                if item['type'] == 'file':
                    relative_item_path = item_path.lstrip('./') # Path relative to repo root

                    # Rule a: Always ignore .git directory (name check - defensive)
                    if item['name'] == '.git':
                        st.info(f"Ignoring (item name is .git): {item_path}")
                        continue

                    # Rule b: IGNORED_EXTENSIONS (for files) and hardcoded dir name for items inside path
                    # Check if item's name itself is a hardcoded ignored directory (e.g. path is 'src', item['name'] is 'node_modules')
                    if item['type'] == 'dir' and item['name'] in hardcoded_ignored_dir_names:
                        st.info(f"Ignoring (hardcoded item name): {item_path}")
                        continue
                    if item['type'] == 'file' and any(item['name'].lower().endswith(ext) for ext in IGNORED_EXTENSIONS):
                        st.info(f"Ignoring (extension): {item_path}")
                        continue

                    # Rule c: .gitignore rules (applied again here for items, path was checked before recursion)
                    if gitignore_spec and gitignore_spec.match_file(relative_item_path):
                        st.info(f"Ignoring ('.gitignore'): {item_path}")
                        continue

                    # Rule d: User-defined EXCLUSION glob patterns
                    if exclude_spec and exclude_spec.match_file(relative_item_path):
                        st.info(f"Ignoring (user exclusion): {item_path}")
                        continue

                    # Rule e: User-defined INCLUSION glob patterns (applies definitively to files)
                    if include_spec and not include_spec.match_file(relative_item_path):
                        st.info(f"Ignoring (not in user inclusion): {item_path}")
                        continue

                    # Max File Size Check (before download attempt)
                    # User input is in KB, item['size'] is in bytes. max_file_size_kb = 0 means no limit.
                    if max_file_size_kb > 0 and item.get('size', 0) > (max_file_size_kb * 1024):
                        st.warning(f"Skipping '{item_path}' (size: {item.get('size', 0)/1024:.2f} KB) as it exceeds the {max_file_size_kb} KB limit.")
                        continue

                    # If we reach here, the file is included and within size limits.
                    download_url = item.get('download_url')
                    if download_url:
                        try:
                            # Make sure to pass headers (e.g. for PAT on private repos for download_url)
                            file_response = requests.get(download_url, headers=headers)
                            file_response.raise_for_status() # Check for download errors
                            # Try decoding with UTF-8, then latin-1, then give up for this file
                            try:
                                content = file_response.content.decode('utf-8')
                            except UnicodeDecodeError:
                                try:
                                    content = file_response.content.decode('latin-1')
                                except UnicodeDecodeError:
                                    st.warning(f"Could not decode file {item_path} with UTF-8 or Latin-1. Skipping.")
                                    continue # Skip this file
                            repo_files_content[item_path] = content
                            st.success(f"Fetched: {item_path}")
                        except requests.exceptions.RequestException as e_file:
                            st.warning(f"Error downloading file {item_path}: {e_file}. Skipping.")
                            # repo_files_content[item_path] = f"Error downloading file: {e_file}" # Don't add error as content
                    else:
                        st.info(f"Skipping {item_path} (no download URL, possibly a submodule, symlink or too large).")

                elif item['type'] == 'dir': # For directories, we decide whether to recurse
                    # The path-level checks at the start of _get_files_recursive handle most directory exclusions.
                    # If we are here, the directory 'path' itself wasn't ignored by those top-level checks.
                    # Now we check the specific 'item' (which is a directory).
                    # If include_spec is active, we should only recurse if this directory *could* contain included files.
                    # This is where match_tree_files from pathspec would be ideal.
                    # A simpler heuristic: if include_spec exists and the dir itself doesn't match, AND no files inside it could match, then prune.
                    # For now, let's assume if a directory isn't EXCLUDED, we should look inside it,
                    # and the file-level inclusion patterns will filter the files found.
                    # This means we might traverse more than necessary if specific include patterns are like `src/a/b/file.py`
                    # but it's safer than pruning too aggressively without `match_tree_files`.
                    _get_files_recursive(item_path)

                elif item['type'] == 'symlink':
                    st.info(f"Ignoring (symlink): {item_path}")

        except requests.exceptions.HTTPError as e:
            # Handle 404 for path not found within the repo (e.g. a path component was a file)
            if e.response.status_code == 404 and path: # only if path is not empty (root check is done before)
                st.warning(f"Path component of '{path}' not found or is a file, cannot list contents. Skipping this path.")
                return # Stop recursion for this path
            if e.response.status_code == 404:
                st.error(f"Repository or path not found: {current_url}. If it's a private repository, a PAT with 'repo' scope is required.")
            elif e.response.status_code == 403:
                st.error(f"Access forbidden for {current_url}. This could be due to API rate limits or insufficient permissions. Try adding a GitHub PAT.")
                st.error(f"Rate limit info: {e.response.headers.get('X-RateLimit-Remaining')}/{e.response.headers.get('X-RateLimit-Limit')}")
            else:
                st.error(f"Error fetching repository structure from {current_url}: {e}")
        except requests.exceptions.RequestException as e: # Catch other request errors (network, timeout)
            st.error(f"Network error while fetching {current_url}: {e}")
        # Do not clear repo_files_content here, might have partial data

    with st.spinner("Fetching repository contents... This may take a while for large repositories."):
        _get_files_recursive()

    if not repo_files_content and get_repo_api_url(repo_url):
        st.warning("No text files found that match the criteria, or the repository is empty/inaccessible with current settings.")
        return {}

    return repo_files_content


if st.sidebar.button("Convert to Markdown", key="convert_button"):
    if not repo_url: # Check if repo_url is empty before proceeding
        st.error("Please enter a GitHub repository URL.")
        st.stop()

    # Validate URL format early
    api_url_check = get_repo_api_url(repo_url)
    if not api_url_check:
        st.error("Invalid GitHub repository URL format. Expected: https://github.com/owner/repo")
        st.stop()

    contents = fetch_repo_contents(repo_url)

    if contents:
        markdown_output = f"# Repository: {repo_url}\n\n"
        total_files = len(contents)
        st.info(f"Formatting {total_files} file(s) into Markdown...")

        # Define priority files (lowercase for case-insensitive comparison)
        priority_filenames_lower = [
            "readme.md", "readme", "readme.txt",  # Common README variations
            "license", "license.md", "license.txt", "copying", "copying.md", # Common LICENSE variations
            "contributing.md", "contributing", "contributing.txt", # Common CONTRIBUTING variations
            "code_of_conduct.md", "code_of_conduct", # Common Code of Conduct variations
            "changelog.md", "changelog", "news.md", # Common Changelog variations
            "security.md" # Security policy
        ]

        priority_markdown_parts = []
        other_markdown_parts = []

        # Sort content items by path for consistent ordering of non-priority files
        sorted_content_items = sorted(contents.items())

        for path, content in sorted_content_items:
            lang = path.split('.')[-1].lower() if '.' in path else ""
            # Sanitize lang for common cases not directly supported or to improve highlighting
            if lang == "py": lang = "python"
            if lang == "js": lang = "javascript"
            if lang == "md": lang = "markdown"
            if lang == "rb": lang = "ruby"
            if lang == "yml": lang = "yaml"
            if lang == "sh": lang = "bash"
            if lang == "txt": lang = "text"
            if not lang: lang = "text" # Default for files without extension

            file_markdown = f"## File: `{path}`\n\n```{lang}\n{content}\n```\n\n---\n"

            # Check if it's a top-level priority file
            is_top_level = '/' not in path
            file_basename_lower = path.split('/')[-1].lower() # In case path is just 'README.md', this gives 'readme.md'

            if is_top_level and file_basename_lower in priority_filenames_lower:
                priority_markdown_parts.append(file_markdown)
            else:
                other_markdown_parts.append(file_markdown)

        # Concatenate priority files first, then others
        markdown_output += "".join(priority_markdown_parts)
        markdown_output += "".join(other_markdown_parts)

        st.markdown("### Converted Markdown Output:")

        # Total Output Size Warning
        if total_output_threshold_mb > 0:
            output_bytes = len(markdown_output.encode('utf-8'))
            output_mb = output_bytes / (1024 * 1024)
            threshold_bytes = total_output_threshold_mb * 1024 * 1024
            if output_bytes > threshold_bytes:
                st.warning(
                    f"Warning: The total generated Markdown output is approximately {output_mb:.2f} MB. "
                    f"This exceeds your configured warning threshold of {total_output_threshold_mb} MB and "
                    f"may be too large for some LLM context windows or API requests."
                )

        st.text_area("Combined Markdown for LLM", markdown_output, height=600, key="md_output_area")

        # Add a download button for the generated markdown
        download_filename = f"{repo_url.split('/')[-1]}_converted.md" if repo_url else "converted_repo.md"
        st.download_button(
            label="Download Markdown",
            data=markdown_output,
            file_name=download_filename,
            mime="text/markdown",
        )
    elif get_repo_api_url(repo_url): # If URL was valid but contents is empty (and not due to invalid URL)
        st.warning("No processable files were found in the repository, or all files were filtered out by the current settings.")
    else:
        st.error("Please enter a GitHub repository URL.")
