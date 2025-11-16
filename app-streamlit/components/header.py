import streamlit as st


def render_header(active: str | None = None) -> None:
    """Render a fixed, full-width header bar with logo. Menu is handled by st.navigation."""
    # Full-width header bar
    st.markdown('<div class="app-header"><div class="app-header-inner">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #1f77b4; margin: 0;">🏥 Wound Detection Platform</h2>', unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

    # Spacer so body content doesn't hide under fixed header
    st.markdown('<div class="header-spacer"></div>', unsafe_allow_html=True)


