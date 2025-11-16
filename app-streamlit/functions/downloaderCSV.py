import streamlit as st
import pickle
import pandas as pd
import json
import base64
import uuid
import re

import importlib.util


def import_from_file(module_name: str, filepath: str):
    """
    Imports a module from file.
    Args:
        module_name (str): Assigned to the module's __name__ parameter (does not
            influence how the module is named outside of this function)
        filepath (str): Path to the .py file
    Returns:
        The module
    """
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def notebook_header(text):
    """
    Insert section header into a jinja file, formatted as notebook cell.
    Leave 2 blank lines before the header.
    """
    return f"""# # {text}
"""


def code_header(text):
    """
    Insert section header into a jinja file, formatted as Python comment.
    Leave 2 blank lines before the header.
    """
    seperator_len = (75 - len(text)) / 2
    seperator_len_left = math.floor(seperator_len)
    seperator_len_right = math.ceil(seperator_len)
    return f"# {'-' * seperator_len_left} {text} {'-' * seperator_len_right}"


def to_notebook(code):
    """Converts Python code to Jupyter notebook format."""
    notebook = jupytext.reads(code, fmt="py")
    return jupytext.writes(notebook, fmt="ipynb")


def open_link(url, new_tab=True):
    """Dirty hack to open a new web page with a streamlit button."""
    # From: https://discuss.streamlit.io/t/how-to-link-a-button-to-a-webpage/1661/3
    if new_tab:
        js = f"window.open('{url}')"  # New tab or window
    else:
        js = f"window.location.href = '{url}'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)


def download_button(object_to_download, download_filename, button_text):
    """
    Generates a link to download the given object_to_download.
    From: https://discuss.streamlit.io/t/a-download-button-with-custom-css/4220
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    # if pickle_it:
    #    try:
    #        object_to_download = pickle.dumps(object_to_download)
    #    except pickle.PicklingError as e:
    #        st.write(e)
    #        return None

    # if:
    if isinstance(object_to_download, bytes):
        pass

    elif isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    # Try JSON encode for everything else
    else:
        object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()
    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)

    custom_css = f"""
        <style>
            /* Download button styling aligned with primary theme */
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background: linear-gradient(135deg, #0a3552 0%, #0a3552 100%);
                color: #ffffff !important;
                padding: 0.6rem 1rem;
                text-decoration: none !important;
                border-radius: 8px;
                border: none;
                font-weight: 600;
                font-size: 0.95rem;
                box-shadow: 0 4px 12px rgba(10, 53, 82, 0.3);
                transition: transform 0.2s ease, box-shadow 0.2s ease, opacity 0.2s ease;
                will-change: transform;
            }}
            #{button_id}:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(10, 53, 82, 0.4);
                opacity: 0.95;
            }}
            #{button_id}:active {{
                transform: translateY(0);
                box-shadow: 0 3px 8px rgba(10, 53, 82, 0.25);
            }}
            #{button_id} .icon {{
                display: inline-flex;
                width: 18px;
                height: 18px;
            }}
            #{button_id} .label {{
                line-height: 1;
            }}
        </style>
    """

    # Inline SVG download icon followed by the label
    svg_icon = (
        "<svg class=\"icon\" xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\">"
        "<path d=\"M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4\"></path>"
        "<polyline points=\"7 10 12 15 17 10\"></polyline>"
        "<line x1=\"12\" y1=\"15\" x2=\"12\" y2=\"3\"></line>"
        "</svg>"
    )

    dl_link = (
        custom_css
        + (
            f'<div style="display:flex; justify-content:flex-end; width:100%;">'
            f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{svg_icon}<span class="label">{button_text}</span></a>'
            f'</div><br><br>'
        )
    )
    # dl_link = f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}"><input type="button" kind="primary" value="{button_text}"></a><br></br>'

    st.markdown(dl_link, unsafe_allow_html=True)


# def download_link(
#     content, label="Download", filename="file.txt", mimetype="text/plain"
# ):
#     """Create a HTML link to download a string as a file."""
#     # From: https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/9
#     b64 = base64.b64encode(
#         content.encode()
#     ).decode()  # some strings <-> bytes conversions necessary here
#     href = (
#         f'<a href="data:{mimetype};base64,{b64}" download="{filename}">{label}</a>'
#     )
#     return href