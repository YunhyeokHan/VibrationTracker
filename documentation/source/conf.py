import os
from PIL import Image

def convert_images_in_latex_folder(app, exception):
    """
    Convert .bmp and .gif images to .png in the LaTeX build folder,
    and update .tex files to use the .png files.
    """
    if app.builder.name != "latex":  # Only process for LaTeX builds
        return

    # LaTeX build directory
    latex_build_dir = os.path.join(app.outdir)

    # Supported extensions for conversion
    image_extensions = [".bmp", ".gif"]

    # Iterate over all files in the LaTeX build directory
    for root, _, files in os.walk(latex_build_dir):
        # Process image files
        for file in files:
            if any(file.endswith(ext) for ext in image_extensions):
                source_image_path = os.path.join(root, file)
                converted_image_path = os.path.splitext(source_image_path)[0] + ".png"

                # Convert the image to .png if not already converted
                if not os.path.exists(converted_image_path):
                    try:
                        with Image.open(source_image_path) as img:
                            img.convert("RGB").save(converted_image_path)
                    except Exception as e:
                        print(f"Failed to convert {source_image_path}: {e}")

                # Delay file removal to ensure no conflicts
                try:
                    os.remove(source_image_path)
                except Exception as e:
                    print(f"Failed to delete {source_image_path}: {e}")

        # Process LaTeX files within the same directory
        for file in files:
            if file.endswith(".tex"):
                tex_file_path = os.path.join(root, file)
                with open(tex_file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Replace references to .bmp and .gif with .png
                for ext in image_extensions:
                    content = content.replace(ext, ".png")

                with open(tex_file_path, "w", encoding="utf-8") as f:
                    f.write(content)

def setup(app):
    """
    Connect Sphinx events to process images for LaTeX builds.
    """
    app.connect("build-finished", convert_images_in_latex_folder)



# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'VibrationTracker'
copyright = '2024, Yunhyeok Han'
author = 'Yunhyeok Han'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',     
    'sphinx.ext.napoleon',     
    'sphinx.ext.viewcode',
    'myst_parser',
]

source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = []

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

latex_documents = [
    ('index', 'documentation.tex', 'Documentation', 'Yunhyeok Han', 'manual'),
]