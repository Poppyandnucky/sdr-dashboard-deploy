# SDR primitive HTML frontend

From the repository root, start the frontend and Streamlit together:

```bash
python3 run_html_dev.py
```

Then open `http://localhost:8501`.

The frontend uses plain HTML, CSS, and JavaScript modules. It has no npm or
build dependency. Changes to frontend files appear after refreshing the page.

For direct component loading without a separate development server:

```bash
SDR_HTML_UI_BUILD_DIR="$PWD/frontend" streamlit run SDR_Dash_TI.py
```
