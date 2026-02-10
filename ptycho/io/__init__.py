"""I/O helpers and adapters.

This package will host adapters for saving/loading artifacts that interoperate
with external tools such as Ptychodus. See `ptychodus_product_io.py` for the
Ptychodus product format scaffolding.
"""

__all__ = [
    "ExportMeta",
    "export_product_from_rawdata",
    "import_product_to_rawdata",
]

