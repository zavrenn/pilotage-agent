"""Which files are not text, decided from the name alone.

Copied from Hermes (tools/binary_extensions.py), unchanged. It is a pure
string check on purpose: knowing whether to read a path as text must not cost
a stat call, and the answer must be the same on a path that does not exist yet.

Two lists, and the difference between them matters. A binary file cannot be
shown to the model at all. An opaque document — .docx and its relatives — can
be shown, because we extract text out of it, but must never be written back
as text: the model would read the extracted text, edit it, write it, and
destroy the document while reporting success.

.pdf is deliberately in neither list. Its syntax is text, so writing a new one
is legitimate; only overwriting an existing one is dangerous, and that is the
write guard's job.
"""

BINARY_EXTENSIONS = frozenset({
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".tif",
    # Videos
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".m4v", ".mpeg", ".mpg",
    # Audio
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".aiff", ".opus",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz", ".z", ".tgz", ".iso",
    # Executables/binaries
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".obj", ".lib",
    ".app", ".msi", ".deb", ".rpm",
    # Documents (exclude .pdf — text-based, agents may want to inspect)
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".odp",
    # Fonts
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    # Bytecode / VM artifacts
    ".pyc", ".pyo", ".class", ".jar", ".war", ".ear", ".node", ".wasm", ".rlib",
    # Database files
    ".sqlite", ".sqlite3", ".db", ".mdb", ".idx",
    # Design / 3D
    ".psd", ".ai", ".eps", ".sketch", ".fig", ".xd", ".blend", ".3ds", ".max",
    # Flash
    ".swf", ".fla",
    # Lock/profiling data
    ".lockb", ".dat", ".data",
})


def has_binary_extension(path: str) -> bool:
    """Check if a file path has a binary extension. Pure string check, no I/O."""
    dot = path.rfind(".")
    if dot == -1:
        return False
    return path[dot:].lower() in BINARY_EXTENSIONS


# Container document formats (OOXML zip / OLE compound / ODF zip / EPUB zip /
# RTF) that a plain-text write can NEVER produce validly.
OPAQUE_DOCUMENT_EXTENSIONS = frozenset({
    ".doc", ".docx", ".docm",
    ".xls", ".xlsx", ".xlsm", ".xlsb",
    ".ppt", ".pps", ".pot", ".pptx", ".pptm", ".ppsx", ".ppsm",
    ".odt", ".ods", ".odp",
    ".rtf", ".epub",
})


def has_opaque_document_extension(path: str) -> bool:
    """True when the path names an opaque container document (.docx etc.).

    Pure string check, no I/O.
    """
    dot = path.rfind(".")
    if dot == -1:
        return False
    return path[dot:].lower() in OPAQUE_DOCUMENT_EXTENSIONS


def is_pdf_path(path: str) -> bool:
    """True when the path has a .pdf extension. Pure string check, no I/O."""
    return path.lower().endswith(".pdf")


__all__ = [
    "BINARY_EXTENSIONS",
    "OPAQUE_DOCUMENT_EXTENSIONS",
    "has_binary_extension",
    "has_opaque_document_extension",
    "is_pdf_path",
]
