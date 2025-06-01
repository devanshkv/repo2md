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
try:
    import streamlit_antd_components as sac
except ImportError:
    st.error("The 'streamlit-antd-components' library is not installed. Please install it by running: pip install streamlit-antd-components")
    # No st.stop() here, will try to provide a fallback later if sac is None or tree display fails
    sac = None


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

st.sidebar.subheader("Output Options")
st.sidebar.checkbox("Include repository map at the top of Markdown", key="repo_map_option", value=False)

# --- Cache Clear Button ---
if st.sidebar.button("Clear Cache & Re-fetch", key="clear_cache_button"):
    st.cache_data.clear()
    st.success("Cache cleared! Click 'Convert to Markdown' again to re-fetch with latest data.")
    # Optionally, could trigger a re-run if complex state management is involved,
    # but for now, user can click the main button again.

# --- Cached API Call Functions ---
@st.cache_data(show_spinner=False) # Disable spinner for individual cached calls; main spinner is in fetch_repo_contents
def cached_requests_get(url, headers):
    """Wraps requests.get for caching.
    We want the cache to depend on url and headers (especially PAT).
    """
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response
    except requests.exceptions.RequestException as e:
        # Instead of st.error here, which might be too disruptive for a cached function,
        # let the caller handle it or return a specific error object.
        # For now, re-raise to be handled by the caller.
        raise

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

    # repo_files_content = {} # This will be populated later by a different function
    all_repo_paths = [] # Stores list of dicts: {'path': str, 'type': 'file'/'dir', 'name': str, 'sha': str}
    processed_paths_for_recursion = set() # Used by the recursive path fetching function

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
        # Use cached request for .gitignore
        response = cached_requests_get(gitignore_url, headers)
        # Since cached_requests_get now raises for status, we don't need to check response.status_code == 200 here
        gitignore_data = response.json() # If no error, proceed
        if gitignore_data.get('encoding') == 'base64' and gitignore_data.get('content'):
            gitignore_content = base64.b64decode(gitignore_data['content']).decode('utf-8', errors='replace')
            # standard_ignores = ["*.pyc", "*.pyo", "*.pyd", "__pycache__/", ".DS_Store"] # Keep these minimal
            gitignore_lines = gitignore_content.splitlines()
            gitignore_spec = pathspec.PathSpec.from_lines('gitwildmatch', gitignore_lines)
            st.info("Successfully fetched and parsed .gitignore rules.")
        elif gitignore_data.get('download_url'): # Try download_url if content not directly available
            # Use cached request for download_url content
            gitignore_content_resp = cached_requests_get(gitignore_data['download_url'], headers)
            gitignore_content = gitignore_content_resp.text
            gitignore_spec = pathspec.PathSpec.from_lines('gitwildmatch', gitignore_content.splitlines())
            st.info("Successfully fetched and parsed .gitignore rules from download_url.")
        else:
            st.warning(".gitignore found but content is not in expected format or no download_url.")
    # Handle specific exceptions from cached_requests_get or general ones
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.info(".gitignore not found in the repository root.")
        else:
            st.warning(f"Failed to fetch .gitignore: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        st.warning(f"Error fetching .gitignore: {e}")
    except Exception as e: # Catch other potential errors during .gitignore processing
        st.error(f"Error processing .gitignore: {e}")

    # Renamed and refactored function to only fetch paths and types
    def _fetch_repo_paths_recursive(current_api_path=""):
        # current_api_path is the path used for API calls, e.g., "src/components"
        # relative_path_for_specs is the path used for matching against .gitignore, user globs, etc.
        # It should be relative to the repo root and not have a leading "./"

        # Memoization: if we've processed this API path, skip.
        if current_api_path in processed_paths_for_recursion:
            return
        processed_paths_for_recursion.add(current_api_path)

        relative_path_for_specs = current_api_path.lstrip('./')

        # Rule a: Always ignore .git directory (path component check) - applies to current_api_path as a whole
        if '/.git/' in f'/{relative_path_for_specs}/' or relative_path_for_specs.startswith('.git/'):
            st.info(f"Ignoring (dot git dir path): {current_api_path}")
            return

        # Rule b: Hardcoded ignored directory names (applies to current_api_path as a whole)
        hardcoded_ignored_dir_names = ['node_modules', 'venv', '__pycache__', 'dist', 'build', '.vscode', '.idea']
        # Check if any component of the current_api_path is a hardcoded ignored directory name
        path_components = set(relative_path_for_specs.split('/'))
        if any(ignored_name in path_components for ignored_name in hardcoded_ignored_dir_names):
            # More precise check: if current_api_path itself is 'node_modules' or 'src/node_modules' etc.
            is_hardcoded_ignored_path = False
            temp_path = relative_path_for_specs
            for part in temp_path.split('/'):
                if part in hardcoded_ignored_dir_names:
                    is_hardcoded_ignored_path = True
                    break
            if is_hardcoded_ignored_path:
                st.info(f"Ignoring (hardcoded ignored directory in path): {current_api_path}")
                return

        # Rule c: .gitignore rules (applies to current_api_path if it's a directory being considered for traversal)
        # If current_api_path itself is matched by .gitignore, we don't traverse it.
        if gitignore_spec and gitignore_spec.match_file(relative_path_for_specs) and relative_path_for_specs: # Non-empty path
            st.info(f"Ignoring (directory matched by .gitignore): {current_api_path}")
            return

        # Rule d: User-defined EXCLUSION glob patterns (applies to current_api_path for directory traversal)
        if exclude_spec and exclude_spec.match_file(relative_path_for_specs) and relative_path_for_specs: # Non-empty path
            st.info(f"Ignoring (directory matched by user exclusion): {current_api_path}")
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
        # The file-level inclusion check is definitive for files.
        # For directories, if include_patterns exist, we traverse if the dir *might* contain included files.
        # This means a dir not matching an include pattern (e.g. 'docs/') when pattern is 'src/**/*.py'
        # should still be traversed if it's not explicitly excluded.
        # The definitive file filtering happens when an item is identified as a file.

        api_call_url = f"{contents_url}/{current_api_path}"
        try:
            response = cached_requests_get(api_call_url, headers)
            items = response.json()
            items.sort(key=lambda x: x['type'], reverse=True) # Process files first or dirs first? Dirs for deeper traversal.

            for item in items:
                # item_api_path is the full path for the next API call if it's a dir, or the full path of the file.
                if not current_api_path or current_api_path.endswith("/"):
                    item_api_path = f"{current_api_path}{item['name']}"
                else:
                    item_api_path = f"{current_api_path}/{item['name']}"

                item_relative_path_for_specs = item_api_path.lstrip('./')

                # --- Apply filtering rules to each item ---
                # Rule a.1: .git directory by name (defensive, path check should catch it)
                if item['name'] == '.git':
                    st.info(f"Ignoring (item name .git): {item_api_path}")
                    continue

                # Rule b.1: Hardcoded ignored item names (e.g. if item['name'] is 'node_modules')
                # This is mostly for directories. Files are by extension.
                if item['type'] == 'dir' and item['name'] in hardcoded_ignored_dir_names:
                    st.info(f"Ignoring (hardcoded item name): {item_api_path}")
                    continue

                # Rule for files: IGNORED_EXTENSIONS
                if item['type'] == 'file' and any(item['name'].lower().endswith(ext) for ext in IGNORED_EXTENSIONS):
                    st.info(f"Ignoring (file extension): {item_api_path}")
                    continue

                # Rule c.1: .gitignore (applied to each specific item path)
                if gitignore_spec and gitignore_spec.match_file(item_relative_path_for_specs):
                    st.info(f"Ignoring (item matched by .gitignore): {item_api_path}")
                    continue

                # Rule d.1: User-defined EXCLUSION glob patterns (applied to each specific item path)
                if exclude_spec and exclude_spec.match_file(item_relative_path_for_specs):
                    st.info(f"Ignoring (item matched by user exclusion): {item_api_path}")
                    continue

                # Rule e.1: User-defined INCLUSION glob patterns (applies definitively to files)
                # For directories, if include_spec is active, we only recurse if the directory *could* contain included files.
                # This means if a directory itself doesn't match an include pattern, but its sub-path could, we still go in.
                # Example: include_spec = "src/utils/*.py". Current item_api_path = "src". We should recurse.
                # If item_api_path = "docs" and type is dir, we might not recurse if "docs" itself doesn't match any part of include_spec.
                # This is complex without full tree matching. A simpler rule: if include_spec exists,
                # a file must match it. A directory is traversed if it's not excluded.
                if item['type'] == 'file' and include_spec and not include_spec.match_file(item_relative_path_for_specs):
                    st.info(f"Ignoring (file not in user inclusion): {item_api_path}")
                    continue

                # Max File Size Check (only for files, and only if we were to download content, here just for info)
                if item['type'] == 'file' and max_file_size_kb > 0 and item.get('size', 0) > (max_file_size_kb * 1024):
                    st.warning(f"Skipping path (file size limit): {item_api_path} (size: {item.get('size',0)/1024:.2f}KB)")
                    continue # Don't even add its path if it's too large

                # If all checks pass, add the item's path and type
                path_info = {
                    "path": item_api_path, # Full path from repo root
                    "type": item['type'],   # 'file' or 'dir'
                    "name": item['name'],
                    "sha": item.get('sha') # Useful for tree key or future operations
                }
                all_repo_paths.append(path_info)
                # st.success(f"Path added: {item_api_path} (type: {item['type']})")


                if item['type'] == 'dir':
                    # Recurse into subdirectories that haven't been filtered out by path-level or item-level checks
                    _fetch_repo_paths_recursive(item_api_path)

                elif item['type'] == 'symlink':
                    st.info(f"Ignoring (symlink): {item_api_path}")
                    # Optionally, could try to resolve symlinks if necessary, but GitHub API might not directly support it for contents

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404 and current_api_path:
                st.warning(f"Path '{current_api_path}' not found or is a file; cannot list contents. Skipping.")
                return
            if e.response.status_code == 404: # Root path not found
                 st.error(f"Repository or root path not found: {api_call_url}. Private repo? PAT with 'repo' scope needed.")
            elif e.response.status_code == 403:
                st.error(f"Access forbidden for {api_call_url}. Rate limit or insufficient permissions. PAT may help.")
                st.error(f"Rate limit: {e.response.headers.get('X-RateLimit-Remaining')}/{e.response.headers.get('X-RateLimit-Limit')}")
            else:
                st.error(f"Error fetching structure from {api_call_url}: {e}")
        except requests.exceptions.RequestException as e:
            st.error(f"Network error for {api_call_url}: {e}")
        except Exception as e:
            st.error(f"Unexpected error processing path '{current_api_path}': {e}")

    # Initial call to start fetching all paths
    with st.spinner("Fetching repository file and directory paths..."):
        _fetch_repo_paths_recursive() # Start with empty path for repo root

    if not all_repo_paths and get_repo_api_url(repo_url): # Check if repo_url itself is valid
        st.warning("No files or directories found that match criteria, or repo is empty/inaccessible.")
        return [] # Return empty list if nothing found or error occurred that prevented path collection

    # For now, just return the flat list of paths. Tree building will be separate.
    # st.write("Fetched paths:", all_repo_paths) # For debugging
    return all_repo_paths

def get_effective_selected_files(tree_nodes_data, selected_node_ids):
    """
    Traverses the SAC tree data and compiles a list of file paths to fetch based on selected node IDs.
    If a directory is selected, all files within it (that were part of the original path fetching) are included.
    """
    files_to_fetch = set() # Use a set to avoid duplicate paths

    # Helper recursive function to traverse the tree
    def traverse_and_collect(nodes, current_path_prefix=""):
        for node in nodes:
            node_id = node['id'] # 'id' in sac.tree maps to 'key' from our internal tree ('sha' or full path)
            node_path = node['path'] # The full path to the item, stored during tree conversion

            # Check if the node itself or any of its parents were selected.
            # For simplicity in this step, we'll assume sac.tree returns all individually checked items.
            # If a parent dir is checked, sac.tree might return only the parent, or parent + all children
            # depending on its configuration. We assume `selected_node_ids` contains all items that should
            # be considered "checked", whether directly or by parent selection.
            # The `sac.tree` with `return_checked=True` should ideally give us a list of all explicitly
            # and implicitly checked items. Let's work with that assumption.

            if node_id in selected_node_ids:
                if node['icon'] == 'file-text': # Check based on icon, assuming it maps to file type
                    files_to_fetch.add(node_path)
                elif node['icon'] == 'folder' and node.get('children'):
                    # If a folder is selected, recursively add all files from its children
                    # The children here are already part of the filtered tree.
                    collect_all_files_from_subtree(node['children'])

            # Even if a parent isn't explicitly in selected_node_ids, its children might be.
            # This is important if sac.tree only returns explicitly checked items.
            # However, if `checked` implies children are also returned by sac.tree, this explicit recursion might be redundant.
            # For safety, let's ensure we check children if the parent is selected OR if we need to find explicitly selected children.
            # The current logic: if a node (file/dir) is in selected_node_ids, process it.
            # Then, regardless, recurse for its children to find other selected items.
            # This seems slightly off. Let's refine:
            # 1. Create a set of selected_node_ids for quick lookup.
            # 2. Traverse the *entire* tree.
            # 3. If a node is a file and its ID is in selected_node_ids, add its path.
            # 4. If a node is a directory and its ID is in selected_node_ids, add all files in its subtree.
            # 5. If a node (dir) is NOT in selected_node_ids, still recurse into its children to find selected items deeper in the tree.

    # Refined traversal logic:
    selected_ids_set = set(selected_node_ids)

    def collect_all_files_from_subtree(nodes_subtree):
        # Helper to add all files from a given list of nodes (typically children of a selected folder)
        for sub_node in nodes_subtree:
            if sub_node['icon'] == 'file-text':
                files_to_fetch.add(sub_node['path'])
            elif sub_node['icon'] == 'folder' and sub_node.get('children'):
                collect_all_files_from_subtree(sub_node['children'])

    def find_selected_recursive(nodes_current_level):
        for node in nodes_current_level:
            node_id = node['id']
            node_path = node['path']
            is_selected = node_id in selected_ids_set

            if node['icon'] == 'file-text':
                if is_selected:
                    files_to_fetch.add(node_path)
            elif node['icon'] == 'folder':
                if is_selected: # If folder is selected, grab all files under it
                    collect_all_files_from_subtree(node.get('children', []))
                else: # If folder is not selected, still need to check its children
                    find_selected_recursive(node.get('children', []))

    find_selected_recursive(tree_nodes_data)
    return list(files_to_fetch)

def generate_repo_map(tree_nodes_data, selected_node_ids):
    """
    Generates a string representation of the repository map, showing only selected items
    and their necessary parent directories.
    - tree_nodes_data: The SAC tree data (list of nodes).
    - selected_node_ids: A list or set of selected node IDs.
    """
    map_lines = []
    selected_set = set(selected_node_ids)

    def build_map_recursive(nodes, indent_level, parent_selected=False):
        for i, node in enumerate(nodes):
            is_last_child = (i == len(nodes) - 1)
            prefix = "    " * indent_level
            connector = "└── " if is_last_child else "├── "

            node_id = node['id']
            node_label = node['label']
            node_is_selected = node_id in selected_set
            node_has_selected_descendant = any(sid.startswith(node_id) and sid != node_id for sid in selected_set if node['type'] == 'dir') # A bit simplistic if IDs are not path-like

            # More robust check for selected descendants by searching the subtree
            # This is important if node IDs are not hierarchical by string prefix.
            # For now, we assume our selected_ids are SHAs or full paths used as keys.
            # The key thing is to know if ANY child path is selected.

            # We need to check if this node OR any of its children are selected to decide to print it.
            # Let's refine `node_has_selected_descendant`

            def has_selected_descendant_in_subtree(subtree_nodes):
                for sub_node in subtree_nodes:
                    if sub_node['id'] in selected_set:
                        return True
                    if sub_node.get('children') and has_selected_descendant_in_subtree(sub_node['children']):
                        return True
                return False

            if node.get('children'): # It's a directory
                node_has_selected_descendant = has_selected_descendant_in_subtree(node['children'])
            else: # It's a file
                node_has_selected_descendant = False


            if node_is_selected or node_has_selected_descendant:
                map_lines.append(f"{prefix}{connector}{node_label}{'/' if node.get('children') else ''}")
                if node.get('children'):
                    build_map_recursive(node['children'], indent_level + 1, node_is_selected or parent_selected)

    build_map_recursive(tree_nodes_data, 0)
    return "\n".join(map_lines)


def build_file_tree_from_paths(paths_list):
    """
    Constructs a hierarchical tree data structure from a flat list of path dictionaries.
    Each path dictionary should have 'path', 'type', 'name', and 'sha'.
    """
    tree = []
    # Use a dictionary to keep track of nodes and their children for easy access
    # The root node's children will be the actual top-level items of the tree.
    # Format for sac.tree: list of dicts with 'label', 'id', 'children', 'icon' (optional)
    # Our internal format: 'title', 'key', 'type', 'path', 'children'

    # Helper to convert our node format to sac.tree format
    def convert_to_sac_node_format(node_list):
        sac_nodes = []
        for node in node_list:
            sac_node = {
                "label": node['title'],
                "id": node['key'], # 'key' is already unique (SHA or path)
                "icon": 'folder' if node['type'] == 'dir' else 'file-text', # Simple icons
                "path": node['path'] # Keep path for easy reference if needed
            }
            if node.get('children'): # If it's a directory with children
                sac_node['children'] = convert_to_sac_node_format(node['children'])
            sac_nodes.append(sac_node)
        return sac_nodes

    # Temporary map to build the hierarchy. We'll convert it at the end.
    temp_nodes_map = {"": {"title": "<root>", "key": "", "type": "dir", "path": "", "children": []}}


    # Sort paths to ensure parent directories are processed before their children
    # This might not be strictly necessary if paths are already somewhat ordered or parents are created on demand.
    # However, sorting by path length or by path string can help.
    paths_list.sort(key=lambda x: x['path'])

    for item in paths_list:
        path = item['path']
        name = item['name']
        item_type = item['type']
        item_sha = item.get('sha', path) # Use SHA if available, else path as key for uniqueness

        # Find the parent node in our temporary map
        parent_path = "/".join(path.split("/")[:-1])
        parent_node_in_temp_map = temp_nodes_map.get(parent_path)

        # If parent doesn't exist, create it (can happen if paths are not perfectly ordered or structure is sparse)
        # This part might need to be more robust, creating intermediate parent stubs if necessary.
        # For simplicity, we assume GitHub API gives us items in a way that parent dirs are implicitly defined
        # or can be inferred correctly by iterating and creating nodes.
        # A more robust approach would be to create all parent directory nodes explicitly if not found.

        current_node_internal_format = {
            "title": name,
            "key": item_sha,
            "type": item_type,
            "path": path,
        }

        if item_type == "dir":
            current_node_internal_format["children"] = []
            # Add this new directory node to our temporary map so it can become a parent itself
            temp_nodes_map[path] = current_node_internal_format

        if parent_node_in_temp_map:
            parent_node_in_temp_map["children"].append(current_node_internal_format)
        else:
            if parent_path == "": # Top-level item
                 temp_nodes_map[""]["children"].append(current_node_internal_format)
            else:
                # This logic remains similar: handle cases where parent might be missing due to filtering or API issues
                st.warning(f"Parent node for '{path}' (parent: '{parent_path}') not found in temp_map. Item may be skipped or added to root if logic allows.")
                # Fallback or more robust parent creation could be added here if needed.
                # For now, strict parent existence (or being a root item) is enforced.

    # Convert the constructed tree (from temp_nodes_map root's children) to sac.tree format
    final_sac_tree = convert_to_sac_node_format(temp_nodes_map[""]["children"])
    return final_sac_tree


# Function to fetch content for selected files
def fetch_content_for_selected_files(selected_file_paths, all_paths_metadata, headers, max_kb):
    """
    Fetches content for a list of selected file paths.
    - selected_file_paths: List of unique file paths (strings) from repo root.
    - all_paths_metadata: The flat list of dicts (from _fetch_repo_paths_recursive)
                          containing metadata like 'path', 'download_url', 'size'.
    - headers: Headers for API requests (including PAT if any).
    - max_kb: Max file size in KB.
    """
    repo_files_content = {}

    # Create a quick lookup map for metadata from the flat list
    metadata_map = {item['path']: item for item in all_paths_metadata}

    for file_path in selected_file_paths:
        item_metadata = metadata_map.get(file_path)

        if not item_metadata:
            st.warning(f"Metadata for '{file_path}' not found. Skipping.")
            continue

        if item_metadata['type'] != 'file': # Should not happen if get_effective_selected_files works correctly
            st.warning(f"'{file_path}' is not a file. Skipping content fetch.")
            continue

        # Max File Size Check
        file_size_bytes = item_metadata.get('size', 0)
        if max_kb > 0 and file_size_bytes > (max_kb * 1024):
            st.warning(f"Skipping '{file_path}' (size: {file_size_bytes / 1024:.2f} KB) as it exceeds the {max_kb} KB limit for download.")
            continue

        download_url = item_metadata.get('download_url')
        if not download_url: # Should have been populated by path fetcher
            st.warning(f"No download_url for file '{file_path}'. Skipping.")
            continue

        try:
            st.info(f"Fetching content for: {file_path}")
            file_response = cached_requests_get(download_url, headers) # Assuming headers are correctly passed
            # Try decoding with UTF-8, then latin-1
            try:
                content = file_response.content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    content = file_response.content.decode('latin-1')
                except UnicodeDecodeError:
                    st.warning(f"Could not decode file {file_path} with UTF-8 or Latin-1. Skipping.")
                    continue
            repo_files_content[file_path] = content
            st.success(f"Fetched and decoded: {file_path}")
        except requests.exceptions.RequestException as e_file:
            st.warning(f"Error downloading file {file_path}: {e_file}. Skipping.")
        except Exception as e_gen: # Catch any other unexpected error during fetch/decode
             st.error(f"Unexpected error processing file {file_path}: {e_gen}. Skipping.")

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

    # --- Step 1: Fetch all paths ---
    # Note: fetch_repo_contents has been refactored to _fetch_all_repo_paths
    # and the main function fetch_repo_contents now primarily calls the path fetching
    # and will later be modified to handle content fetching based on tree selection.

    # For clarity, let's assume fetch_repo_contents is now the function that fetches paths
    # This is a bit of a misnomer now, it should be renamed to e.g. fetch_repo_paths_and_build_tree
    # For now, it returns the flat list of paths as per the refactoring.
    all_paths_list = fetch_repo_contents(repo_url) # This now returns list of path dicts

    if not all_paths_list:
        st.warning("No paths were fetched. Cannot build file tree or Markdown.")
        st.stop()

    # --- Step 2: Build File Tree Data Structure ---
    st.session_state['repo_file_paths'] = all_paths_list  # Store flat list as well

    if all_paths_list:
        st.info(f"Total paths fetched: {len(all_paths_list)}. Now building file tree...")
        # build_file_tree_from_paths now returns sac.tree compatible format
        repo_tree_for_sac = build_file_tree_from_paths(all_paths_list)
        st.session_state['repo_file_tree_sac_format'] = repo_tree_for_sac # Store the sac-compatible tree

        if 'repo_file_tree_sac_format' in st.session_state and st.session_state['repo_file_tree_sac_format']:
            st.subheader("Select Files and Folders to Include:")

            # --- Initial Selection State ---
            # Pre-select all items initially. Keys are 'id' in sac_format tree.
            def get_all_keys(nodes):
                keys = []
                for node in nodes:
                    keys.append(node['id'])
                    if node.get('children'):
                        keys.extend(get_all_keys(node['children']))
                return keys

            all_node_keys = get_all_keys(st.session_state['repo_file_tree_sac_format'])
            # If 'selected_tree_nodes' is not in session_state, initialize with all keys.
            # Otherwise, respect existing selections (e.g., if user unselected some).
            if 'selected_tree_nodes' not in st.session_state:
                 st.session_state.selected_tree_nodes = all_node_keys

            if sac: # Check if streamlit_antd_components was imported successfully
                try:
                    selected_keys = sac.tree(
                        items=st.session_state['repo_file_tree_sac_format'],
                        checkbox=True,
                        show_line=True,
                        height=400, # Adjust height as needed
                        return_checked=True, # Ensure it returns only checked items' keys
                        checked=st.session_state.selected_tree_nodes # Pass current selections
                    )
                    st.session_state.selected_tree_nodes = selected_keys
                    st.info(f"Selected items: {len(selected_keys)} (Keys: {selected_keys[:10]}...)") # Show some selected keys
                except Exception as e:
                    st.error(f"Error displaying sac.tree: {e}. Falling back to basic display if possible.")
                    # Here you could implement the fallback st.checkbox tree if sac.tree fails at runtime
                    # For now, just show an error. The subtask mentions fallback if it cannot be installed.
                    # If it's installed but fails at runtime, that's a different scenario.
                    # Fallback logic is not implemented in this step for runtime failure.
            else:
                st.warning("streamlit-antd-components (sac) is not available. File tree selection is disabled.")
                # Basic fallback display (not interactive selection for this step, just showing structure)
                def display_basic_tree(nodes, indent=0):
                    for node in nodes:
                        prefix = "    " * indent
                        icon = "📁" if node.get('children') else "📄"
                        st.text(f"{prefix}{icon} {node['label']}")
                        if node.get('children'):
                            display_basic_tree(node['children'], indent + 1)
                if 'repo_file_tree_sac_format' in st.session_state:
                    display_basic_tree(st.session_state['repo_file_tree_sac_format'])

        else:
            st.info("File tree could not be generated or is empty.")
    else:
        st.info("No paths available to build the tree.")

    # --- Step 3: Fetch content for selected files and generate Markdown ---

    contents_to_convert_to_markdown = {}
    if 'selected_tree_nodes' in st.session_state and st.session_state.selected_tree_nodes and \
       'repo_file_tree_sac_format' in st.session_state and 'repo_file_paths' in st.session_state:

        effective_files_to_fetch = get_effective_selected_files(
            st.session_state.repo_file_tree_sac_format,
            st.session_state.selected_tree_nodes
        )
        st.info(f"Effective files selected for download: {len(effective_files_to_fetch)}")
        # st.write("Files to fetch:", effective_files_to_fetch) # Debugging

        if effective_files_to_fetch:
            # Need headers for fetch_content_for_selected_files
            current_headers = {"Accept": "application/vnd.github.v3.raw"} # Or application/octet-stream
            if github_pat:
                current_headers["Authorization"] = f"token {github_pat}"

            contents_to_convert_to_markdown = fetch_content_for_selected_files(
                effective_files_to_fetch,
                st.session_state.repo_file_paths, # Pass the flat list with all metadata
                current_headers,
                max_file_size_kb # User's preference for max file size
            )
        else:
            st.info("No files are effectively selected to fetch content for Markdown.")
    else:
        st.info("No selections made in the tree, or tree data not available.")


    if contents_to_convert_to_markdown:
        markdown_output = f"# Repository: {repo_url}\n\n"

        # --- Add Repository Map if option is selected ---
        if st.session_state.get("repo_map_option", False) and 'repo_file_tree_sac_format' in st.session_state:
            if st.session_state.selected_tree_nodes: # Ensure there are selections
                st.info("Generating repository map...")
                repo_map_str = generate_repo_map(
                    st.session_state.repo_file_tree_sac_format,
                    st.session_state.selected_tree_nodes
                )
                if repo_map_str:
                    markdown_output += "## Repository Map\n```\n"
                    markdown_output += repo_map_str
                    markdown_output += "\n```\n\n---\n\n"
                else:
                    st.info("Map generated but was empty (no selected items to show in map).")
            else:
                st.info("No items selected in the tree, skipping repository map.")

        total_files = len(contents_to_convert_to_markdown)
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
    # elif get_repo_api_url(repo_url) and not contents_to_convert_to_markdown: # If URL valid but no content to convert
    #    st.info("No files selected or processed for Markdown conversion yet. Select files from the tree (once implemented).")
    elif not all_paths_list and get_repo_api_url(repo_url): # If paths list is empty but URL was valid
         st.warning("No processable files or directories were found in the repository based on current filters.")
    # else: # This case should be caught by initial repo_url check
    #    st.error("Please enter a GitHub repository URL.")
