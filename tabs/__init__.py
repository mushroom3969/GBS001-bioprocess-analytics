# tabs/__init__.py
# Empty — tabs are imported directly in app.py via:
#   from tabs import tab0_overview as tab_overview
#   from tabs import tab1_trends   as tab_trend
#   ...etc
# Do NOT import tab modules here — each tab has heavy UI code that
# must only run inside its own st.tabs() context block.
