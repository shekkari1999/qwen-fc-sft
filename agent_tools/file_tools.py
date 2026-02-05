"""File operation tools for the agent framework."""

import zipfile
import sys
import os
import base64
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent_framework import tool

# Load environment variables
load_dotenv()

# Import optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
except Exception as e:
    print(f"Warning: pandas import failed with: {e}")
    PANDAS_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
except Exception as e:
    print(f"Warning: openpyxl import failed with: {e}")
    OPENPYXL_AVAILABLE = False


try:
    import fitz  # pymupdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@tool
def unzip_file(zip_path: str, extract_to: str = None) -> str:
    """Extract a zip file to the specified directory.
    
    Args:
        zip_path: Path to the zip file to extract
        extract_to: Directory to extract to. If None, creates a folder with the zip filename.
    
    Returns:
        String describing the extraction results, including file count and contents list.
    
    Example:
        result = unzip_file("archive.zip", "extracted/")
    """
    zip_path = Path(zip_path)
    
    if not zip_path.exists():
        return f"Error: File not found: {zip_path}"
    
    # Default extraction path: create folder with zip filename
    if extract_to is None:
        extract_to = zip_path.parent / zip_path.stem
    else:
        extract_to = Path(extract_to)
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            zip_ref.extractall(extract_to)
        
        # Format results
        result = f"Successfully extracted {len(file_list)} files to {extract_to}/\n\n"
        result += "Contents:\n"
        for f in file_list[:20]:
            result += f"  - {f}\n"
        if len(file_list) > 20:
            result += f"  ... and {len(file_list) - 20} more files\n"
        
        return result
    except Exception as e:
        return f"Error extracting zip file: {str(e)}"

@tool
def list_files(path: str = ".") -> str:
    """List files and directories in the given path."""
    path = Path(path)
    
    if not path.exists():
        return f"Path not found: {path}"
    
    if not path.is_dir():
        return f"Not a directory: {path}"
    
    items = []
    for item in sorted(path.iterdir()):
        if item.name.startswith('.'):
            continue
        
        if item.is_dir():
            items.append(f"{item.name}/")
        else:
            items.append(f"{item.name}")
    
    # Sort directories first
    dirs = [i for i in items if i.endswith('/')]
    files = [i for i in items if not i.endswith('/')]
    
    result = f"Directory: {path}\n"
    for item in dirs + files:
        result += f"  {item}\n"
    
    return result
# Helper function - not exposed as tool (starts with _)
def _read_text_file(file_path: str, start_line: int, end_line: int) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Adjust line numbers (1-indexed to 0-indexed)
    start_idx = max(0, start_line - 1)
    end_idx = len(lines) if end_line == -1 else min(end_line, len(lines))
    
    selected_lines = lines[start_idx:end_idx]
    
    result = []
    for i, line in enumerate(selected_lines, start=start_line):
        result.append(f"{i:4d} | {line.rstrip()}")
    return '\n'.join(result)
# Helper function - not exposed as tool
def _read_csv(file_path: str) -> str:
    if not PANDAS_AVAILABLE:
        return "Error: pandas is required for CSV reading. Install with: pip install pandas"
    try:
        df = pd.read_csv(file_path)
        result = f"CSV file: {file_path}\n"
        result += f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
        result += df.to_string(index=False)
        return result
    except Exception as e:
        return f"Error reading CSV file: {str(e)}"
 
# Helper function - not exposed as tool
def _read_excel(file_path: str) -> str:
    if not PANDAS_AVAILABLE:
        return "Error: pandas is required for Excel reading. Install with: pip install pandas openpyxl"
    
    # Check for openpyxl specifically for .xlsx files
    if file_path.endswith('.xlsx') and not OPENPYXL_AVAILABLE:
        return ("Error: openpyxl package is not installed. "
                "To read .xlsx files, install it with: pip install openpyxl or uv pip install openpyxl. "
                "The package is listed in pyproject.toml but may not be installed in the current environment.")
    
    try:
        # Explicitly use openpyxl for .xlsx files
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            df = pd.read_excel(file_path)
        
        # Use to_string() instead of to_markdown() to avoid tabulate dependency
        # Format as a clean table
        result = f"Excel file: {file_path}\n"
        result += f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
        result += df.to_string(index=False)
        return result
    except ImportError as e:
        error_msg = str(e).lower()
        if 'openpyxl' in error_msg:
            return ("Error: openpyxl is required for .xlsx files. "
                    "Install with: pip install openpyxl or uv pip install openpyxl. "
                    "Then restart the Python environment.")
        if 'tabulate' in error_msg:
            # Fallback if somehow to_string fails too
            return f"Error: tabulate dependency issue. {str(e)}"
        return f"Error: Missing dependency. {str(e)}. Install required packages: pip install pandas openpyxl"
    except Exception as e:
        return f"Error reading Excel file: {str(e)}"
 
TEXT_EXTENSIONS = ['.txt', '.py', '.js', '.json', '.md', '.html', 
                   '.css', '.xml', '.yaml', '.yml', '.log', '.sh']
SPREADSHEET_EXTENSIONS = ['.xlsx', '.xls', '.csv']

@tool
def read_file(file_path: str, start_line: int = 1, end_line: int = -1) -> str:
    """Read file content. Supports txt, py, json, md, csv, xlsx."""
    path = Path(file_path)
    
    if not path.exists():
        return f"File not found: {file_path}"
    
    ext = path.suffix.lower()
    
    if ext in TEXT_EXTENSIONS:
        return _read_text_file(file_path, start_line, end_line)
    elif ext == '.csv':
        return _read_csv(file_path)
    elif ext in SPREADSHEET_EXTENSIONS:
        return _read_excel(file_path)
    else:
        return _read_text_file(file_path, start_line, end_line)

IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']
AUDIO_EXTENSIONS = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm']
PDF_EXTENSIONS = ['.pdf']

@tool
def read_media_file(file_path: str, query: str) -> str:
    """Analyze an image, audio, or PDF file using LLM."""
    ext = Path(file_path).suffix.lower()
    
    if ext in IMAGE_EXTENSIONS:
        return _analyze_image(file_path, query)
    elif ext in AUDIO_EXTENSIONS:
        return _analyze_audio(file_path, query)
    elif ext in PDF_EXTENSIONS:
        return _analyze_pdf(file_path, query)
    else:
        return f"Unsupported media format: {ext}"
 
# Helper function - not exposed as tool
def _analyze_image(file_path: str, query: str) -> str:
    if not OPENAI_AVAILABLE:
        return "Error: openai is required for image analysis. Install with: pip install openai"
    
    with open(file_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    ext = Path(file_path).suffix.lower().lstrip('.')
    media_type = "image/jpeg" if ext == "jpg" else f"image/{ext}"
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {
                    "url": f"data:{media_type};base64,{image_data}"
                }}
            ]
        }]
    )
    return response.choices[0].message.content
 
# Helper function - not exposed as tool
def _analyze_audio(file_path: str, query: str) -> str:
    if not OPENAI_AVAILABLE:
        return "Error: openai is required for audio analysis. Install with: pip install openai"
    
    with open(file_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode("utf-8")
    
    ext = Path(file_path).suffix.lower().lstrip('.')
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "input_audio", "input_audio": {
                    "data": audio_data,
                    "format": ext
                }}
            ]
        }]
    )
    return response.choices[0].message.content
 
# Helper function - not exposed as tool
def _analyze_pdf(file_path: str, query: str) -> str:
    if not PYPDF_AVAILABLE:
        return "Error: pymupdf is required for PDF analysis. Install with: pip install pymupdf"
    if not OPENAI_AVAILABLE:
        return "Error: openai is required for PDF analysis. Install with: pip install openai"
    
    doc = fitz.open(file_path)
    
    # Extract text for context
    text_content = ""
    for page in doc:
        text_content += page.get_text()
    
    # Convert pages to images
    images = []
    for page in doc[:5]:  # First 5 pages
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        images.append(base64.b64encode(img_bytes).decode('utf-8'))
    
    # Build content with text and images
    content = [{
        "type": "text", 
        "text": f"{query}\n\nExtracted text:\n{text_content[:3000]}"
    }]
    
    for img_b64 in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}]
    )
    return response.choices[0].message.content

@tool(
    name="delete_file",
    description="Delete a file from the filesystem",
    requires_confirmation=True,
    confirmation_message="The agent wants to delete a file. Arguments: {arguments}. "
                        "This action cannot be undone. Do you approve?"
)
def delete_file(filename: str) -> str:
    """Delete the specified file."""
    import os
    os.remove(filename)
    return f"Successfully deleted {filename}"