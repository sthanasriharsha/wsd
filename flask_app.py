# ============================================================
# flask_app.py  —  Web UI for WSD Pipeline
# ============================================================
#
# HOW THIS WORKS (plain English):
#
#   Your existing app.py has all the AI logic (ConSeC, WordNet,
#   scoring). This file wraps it with a web server so you can:
#     1. Type a single sentence in a text box
#     2. Upload a .txt file  (one sentence per line)
#     3. Upload a .csv file  (one column of sentences)
#   ...and get back results as a web page + downloadable files.
#
# FLASK EXPLAINED:
#   Flask is a tiny Python web framework. It listens on a port
#   (like 5000), receives requests from your browser, calls your
#   Python functions, and sends back HTML pages.
#
#   Three URL routes are defined:
#     GET  /          → shows the upload form (home page)
#     POST /run       → receives form data, runs the pipeline,
#                       returns results page
#     GET  /download/<filename>  → lets user download output files
#
# USAGE:
#   set PYTHONPATH=%CD%
#   python flask_app.py
#   Then open:  http://127.0.0.1:5000
#
# REQUIREMENTS (already installed if you have the pipeline):
#   pip install flask
# ============================================================

import os
import io
import csv
import json
import uuid
import traceback
from pathlib import Path

from flask import (
    Flask,
    render_template_string,
    request,
    jsonify,
    send_from_directory,
    url_for,
)

# ── Import ALL your pipeline logic from app.py ──────────────
# We import the functions directly so we never duplicate code.
# The models (ConSeC, spaCy) are loaded ONCE when Flask starts.
from score import (
    process_sentence,
    format_output_dict,
    save_json,
    save_csv,
    tokenize_and_tag,
)

# ============================================================
# FLASK SETUP
# ============================================================

app = Flask(__name__)

# Where uploaded files and results are saved temporarily
UPLOAD_FOLDER = "flask_uploads"
RESULTS_FOLDER = "flask_results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# HELPER: read sentences from different input types
# ============================================================

def sentences_from_text(text):
    """Single sentence typed into the text box."""
    sentence = text.strip()
    return [sentence] if sentence else []


def sentences_from_txt(file_bytes):
    """
    .txt file: one sentence per line.
    Blank lines and lines starting with # are skipped.
    """
    content = file_bytes.decode("utf-8", errors="replace")
    sentences = []
    for line in content.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            sentences.append(line)
    return sentences


def sentences_from_csv(file_bytes):
    """
    .csv file: reads the FIRST column that looks like a sentence.
    Skips the header row if it contains words like 'sentence' or 'text'.
    Also handles CSV files where there is only one column of sentences.
    """
    content   = file_bytes.decode("utf-8", errors="replace")
    reader    = csv.reader(io.StringIO(content))
    sentences = []
    header_skipped = False

    for row in reader:
        if not row:
            continue
        cell = row[0].strip()

        # Skip header row
        if not header_skipped:
            header_skipped = True
            if cell.lower() in ("sentence", "sentences", "text",
                                 "input", "sent", "s"):
                continue

        if cell:
            sentences.append(cell)

    return sentences


# ============================================================
# HELPER: run pipeline on a list of sentences
# ============================================================

def run_pipeline(sentences):
    """
    Calls process_sentence() from app.py on each sentence.
    Returns (all_results, error_message)
    all_results is a list of per-sentence dicts.
    """
    all_results = []
    for idx, sentence in enumerate(sentences, start=1):
        try:
            result = process_sentence(idx, sentence)
            all_results.append(result)
        except Exception as e:
            traceback.print_exc()
            all_results.append({
                "sentence_id": idx,
                "sentence":    sentence,
                "words":       [],
                "error":       str(e),
            })
    return all_results


# ============================================================
# HELPER: save output files and return their filenames
# ============================================================

def save_outputs(all_results, run_id):
    """
    Saves JSON and CSV output files with a unique run_id prefix
    so multiple users don't overwrite each other's results.
    Returns dict of {type: filename}.
    """
    json_file = os.path.join(RESULTS_FOLDER, f"{run_id}_output.json")
    csv_file  = os.path.join(RESULTS_FOLDER, f"{run_id}_output.csv")
    clean_json = os.path.join(RESULTS_FOLDER, f"{run_id}_output_clean.json")

    save_json(all_results, output_file=json_file)
    save_csv(all_results,  output_file=csv_file)

    return {
        "json":       f"{run_id}_output.json",
        "csv":        f"{run_id}_output.csv",
        "clean_json": f"{run_id}_output_clean.json",
    }


# ============================================================
# HTML TEMPLATE (home page + results in one file)
# ============================================================
# We use render_template_string() so everything is in one file.
# In a bigger project you would put this in templates/index.html.

HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WSD Pipeline</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #f5f5f0;
      color: #1a1a1a;
      min-height: 100vh;
    }

    .header {
      background: #1a1a2e;
      color: white;
      padding: 1.5rem 2rem;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .header h1 { font-size: 1.4rem; font-weight: 500; }
    .header p  { font-size: 0.85rem; opacity: 0.7; margin-top: 3px; }

    .container { max-width: 900px; margin: 2rem auto; padding: 0 1.5rem; }

    /* ── Input card ── */
    .card {
      background: white;
      border-radius: 12px;
      border: 0.5px solid #e0e0e0;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }
    .card h2 {
      font-size: 1rem;
      font-weight: 500;
      color: #333;
      margin-bottom: 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 0.5px solid #eee;
    }

    /* ── Tabs (single / file) ── */
    .tab-row {
      display: flex;
      gap: 0;
      border: 0.5px solid #ddd;
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 1.25rem;
    }
    .tab-btn {
      flex: 1;
      padding: 9px;
      background: #f9f9f9;
      border: none;
      font-size: 0.875rem;
      cursor: pointer;
      color: #666;
      transition: all .15s;
      border-right: 0.5px solid #ddd;
    }
    .tab-btn:last-child { border-right: none; }
    .tab-btn.active {
      background: white;
      color: #1a1a2e;
      font-weight: 500;
    }

    .tab-panel { display: none; }
    .tab-panel.active { display: block; }

    textarea {
      width: 100%;
      padding: 0.75rem 1rem;
      border: 0.5px solid #ddd;
      border-radius: 8px;
      font-size: 0.9rem;
      font-family: inherit;
      resize: vertical;
      min-height: 80px;
      line-height: 1.5;
      outline: none;
    }
    textarea:focus { border-color: #4a90d9; }

    .upload-zone {
      border: 2px dashed #ccc;
      border-radius: 8px;
      padding: 2rem;
      text-align: center;
      cursor: pointer;
      transition: border-color .15s;
      position: relative;
    }
    .upload-zone:hover { border-color: #4a90d9; }
    .upload-zone input[type=file] {
      position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%;
    }
    .upload-zone .icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .upload-zone p    { font-size: 0.875rem; color: #666; }
    .upload-zone .fname { color: #4a90d9; font-weight: 500; margin-top: 6px; }

    .hint {
      font-size: 0.8rem;
      color: #888;
      margin-top: 0.5rem;
      line-height: 1.5;
    }

    .run-btn {
      width: 100%;
      padding: 12px;
      background: #1a1a2e;
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 500;
      cursor: pointer;
      margin-top: 1rem;
      transition: background .15s;
    }
    .run-btn:hover    { background: #2d2d4a; }
    .run-btn:disabled { background: #999; cursor: not-allowed; }

    /* ── Loading spinner ── */
    #loading {
      display: none;
      text-align: center;
      padding: 2rem;
    }
    .spinner {
      width: 36px; height: 36px;
      border: 3px solid #eee;
      border-top-color: #1a1a2e;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin: 0 auto 1rem;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    #loading p { color: #666; font-size: 0.9rem; }

    /* ── Results ── */
    #results { display: none; }

    .results-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
      flex-wrap: wrap;
      gap: 8px;
    }
    .results-header h2 { font-size: 1rem; font-weight: 500; }

    .download-row { display: flex; gap: 8px; flex-wrap: wrap; }
    .dl-btn {
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 500;
      text-decoration: none;
      border: 0.5px solid;
      display: inline-block;
      cursor: pointer;
    }
    .dl-csv  { color: #1a7a4a; border-color: #1a7a4a; background: #f0faf5; }
    .dl-json { color: #1a4a7a; border-color: #1a4a7a; background: #f0f5fa; }

    /* ── Sentence result block ── */
    .sentence-block {
      background: white;
      border: 0.5px solid #e0e0e0;
      border-radius: 10px;
      margin-bottom: 1rem;
      overflow: hidden;
    }
    .sentence-header {
      background: #f8f8f8;
      padding: 0.75rem 1rem;
      font-size: 0.85rem;
      color: #444;
      border-bottom: 0.5px solid #eee;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .sentence-header .sid {
      background: #1a1a2e;
      color: white;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 0.75rem;
    }
    .sentence-text { font-weight: 500; color: #222; }

    .word-grid {
      display: grid;
      gap: 0;
    }
    .word-row {
      display: grid;
      grid-template-columns: 180px 1fr;
      border-bottom: 0.5px solid #f0f0f0;
      align-items: start;
    }
    .word-row:last-child { border-bottom: none; }

    .word-label {
      padding: 0.6rem 1rem;
      font-size: 0.8rem;
      font-family: 'SF Mono', 'Consolas', monospace;
      border-right: 0.5px solid #f0f0f0;
      background: #fafafa;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }
    .word-orig  { font-weight: 600; color: #1a1a2e; }
    .word-arrow { color: #aaa; font-size: 0.7rem; }
    .word-best  { color: #1a7a4a; font-size: 0.75rem; }

    .word-specifics {
      padding: 0.5rem 0.75rem;
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      align-items: center;
    }
    .spec-pill {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 3px 8px;
      border-radius: 12px;
      font-size: 0.75rem;
      border: 0.5px solid;
    }
    .spec-pill.top {
      background: #e8f5ee;
      border-color: #1a7a4a;
      color: #1a5a34;
    }
    .spec-pill.mid {
      background: #f0f5fa;
      border-color: #4a80b4;
      color: #2a5a8a;
    }
    .spec-pill.low {
      background: #f8f8f8;
      border-color: #ccc;
      color: #666;
    }
    .spec-score {
      opacity: 0.75;
      font-size: 0.7rem;
    }
    .no-spec {
      font-size: 0.8rem;
      color: #aaa;
      font-style: italic;
    }

    .error-block {
      background: #fff5f5;
      border: 0.5px solid #f0a0a0;
      border-radius: 8px;
      padding: 0.75rem 1rem;
      font-size: 0.85rem;
      color: #c00;
      margin-bottom: 0.75rem;
    }

    .summary-bar {
      display: flex;
      gap: 1.5rem;
      padding: 0.75rem 0;
      font-size: 0.85rem;
      color: #555;
      border-top: 0.5px solid #eee;
      margin-top: 0.5rem;
    }
    .summary-bar span b { color: #1a1a2e; }

    @media (max-width: 600px) {
      .word-row { grid-template-columns: 1fr; }
      .word-label { border-right: none; border-bottom: 0.5px solid #f0f0f0; }
    }
  </style>
</head>
<body>

<div class="header">
  <div>
    <h1>WSD Pipeline</h1>
    <p>Word Sense Disambiguation — specific word scorer</p>
  </div>
</div>

<div class="container">

  <!-- ── Input card ── -->
  <div class="card">
    <h2>Input</h2>

    <div class="tab-row">
      <button class="tab-btn active" onclick="switchTab('single')">
        Single sentence
      </button>
      <button class="tab-btn" onclick="switchTab('txt')">
        Upload .txt file
      </button>
      <button class="tab-btn" onclick="switchTab('csv')">
        Upload .csv file
      </button>
    </div>

    <!-- Tab 1: single sentence -->
    <div id="tab-single" class="tab-panel active">
      <textarea id="single-input"
                placeholder="Type a sentence here...  e.g.  I deposited money in the bank"
                rows="3"></textarea>
      <p class="hint">The system will find specific words for every word in this sentence.</p>
    </div>

    <!-- Tab 2: .txt file -->
    <div id="tab-txt" class="tab-panel">
      <div class="upload-zone" id="zone-txt">
        <input type="file" accept=".txt" id="file-txt"
               onchange="showFilename('file-txt','fname-txt')">
        <div class="icon">📄</div>
        <p>Click or drag a <strong>.txt</strong> file here</p>
        <p class="hint">One sentence per line. Lines starting with # are skipped.</p>
        <p class="fname" id="fname-txt"></p>
      </div>
    </div>

    <!-- Tab 3: .csv file -->
    <div id="tab-csv" class="tab-panel">
      <div class="upload-zone" id="zone-csv">
        <input type="file" accept=".csv" id="file-csv"
               onchange="showFilename('file-csv','fname-csv')">
        <div class="icon">📊</div>
        <p>Click or drag a <strong>.csv</strong> file here</p>
        <p class="hint">
          First column should contain sentences.<br>
          Header row is automatically skipped.
        </p>
        <p class="fname" id="fname-csv"></p>
      </div>
    </div>

    <button class="run-btn" id="run-btn" onclick="runPipeline()">
      Run WSD Pipeline
    </button>
  </div>

  <!-- ── Loading ── -->
  <div id="loading">
    <div class="spinner"></div>
    <p>Running ConSeC + WordNet scoring...</p>
    <p style="margin-top:6px;font-size:0.8rem;color:#aaa">
      This takes 20–60 seconds per sentence depending on GPU speed.
    </p>
  </div>

  <!-- ── Results ── -->
  <div id="results">
    <div class="results-header">
      <h2>Results</h2>
      <div class="download-row" id="download-row"></div>
    </div>
    <div id="results-body"></div>
    <div class="summary-bar" id="summary-bar"></div>
  </div>

</div><!-- /container -->

<script>
// ── Tab switching ──────────────────────────────────────────
let activeTab = 'single';

function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab-btn').forEach((b, i) => {
    b.classList.toggle('active', ['single','txt','csv'][i] === tab);
  });
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
}

// ── Show uploaded filename ─────────────────────────────────
function showFilename(inputId, labelId) {
  const f = document.getElementById(inputId).files[0];
  document.getElementById(labelId).textContent = f ? f.name : '';
}

// ── Run pipeline ───────────────────────────────────────────
async function runPipeline() {
  const btn = document.getElementById('run-btn');

  // Build FormData depending on active tab
  const fd = new FormData();
  fd.append('input_type', activeTab);

  if (activeTab === 'single') {
    const txt = document.getElementById('single-input').value.trim();
    if (!txt) { alert('Please type a sentence first.'); return; }
    fd.append('sentence', txt);

  } else if (activeTab === 'txt') {
    const f = document.getElementById('file-txt').files[0];
    if (!f) { alert('Please select a .txt file first.'); return; }
    fd.append('file', f);

  } else if (activeTab === 'csv') {
    const f = document.getElementById('file-csv').files[0];
    if (!f) { alert('Please select a .csv file first.'); return; }
    fd.append('file', f);
  }

  // Show loading, hide results
  btn.disabled = true;
  btn.textContent = 'Running...';
  document.getElementById('loading').style.display  = 'block';
  document.getElementById('results').style.display  = 'none';
  document.getElementById('results-body').innerHTML = '';

  try {
    const resp = await fetch('/run', { method: 'POST', body: fd });
    const data = await resp.json();

    if (data.error) {
      document.getElementById('results-body').innerHTML =
        `<div class="error-block">Error: ${data.error}</div>`;
    } else {
      renderResults(data);
    }
    document.getElementById('results').style.display = 'block';

  } catch (err) {
    document.getElementById('results-body').innerHTML =
      `<div class="error-block">Network error: ${err}</div>`;
    document.getElementById('results').style.display = 'block';
  }

  document.getElementById('loading').style.display = 'none';
  btn.disabled = false;
  btn.textContent = 'Run WSD Pipeline';
}

// ── Render results into the page ───────────────────────────
function renderResults(data) {
  // Download buttons
  const dlRow = document.getElementById('download-row');
  dlRow.innerHTML = `
    <a class="dl-btn dl-csv"  href="/download/${data.files.csv}"  target="_blank">
      Download CSV
    </a>
    <a class="dl-btn dl-json" href="/download/${data.files.clean_json}" target="_blank">
      Download JSON
    </a>
  `;

  const body = document.getElementById('results-body');
  body.innerHTML = '';

  let totalWords = 0;
  let coveredWords = 0;

  data.results.forEach(sentResult => {
    if (sentResult.error) {
      body.innerHTML += `
        <div class="error-block">
          Sentence ${sentResult.sentence_id}: ${sentResult.error}
        </div>`;
      return;
    }

    // Build word rows
    let wordRowsHTML = '';
    (sentResult.words || []).forEach(w => {
      totalWords++;

      const specifics = w.specifics_scored || [];
      if (specifics.length > 0) coveredWords++;

      // Pill colour: top (green) / mid (blue) / low (gray)
      const pillsHTML = specifics.length === 0
        ? '<span class="no-spec">no specific words</span>'
        : specifics.map((s, i) => {
            const cls = i === 0 ? 'top' : (i <= 2 ? 'mid' : 'low');
            const scoreStr = (s.score).toFixed(2);
            return `<span class="spec-pill ${cls}">
                      ${s.specific}
                      <span class="spec-score">(${scoreStr})</span>
                    </span>`;
          }).join('');

      // Left side label:  original word → best specific
      const arrow    = w.best_specific ? '↓' : '';
      const bestLabel = w.best_specific
        ? `<span class="word-best">${w.best_specific}</span>` : '';

      wordRowsHTML += `
        <div class="word-row">
          <div class="word-label">
            <span class="word-orig">${w.original_word}</span>
            <span class="word-arrow">${arrow}</span>
            ${bestLabel}
          </div>
          <div class="word-specifics">${pillsHTML}</div>
        </div>`;
    });

    body.innerHTML += `
      <div class="sentence-block">
        <div class="sentence-header">
          <span class="sid">#${sentResult.sentence_id}</span>
          <span class="sentence-text">${sentResult.sentence}</span>
        </div>
        <div class="word-grid">${wordRowsHTML}</div>
      </div>`;
  });

  // Summary bar
  document.getElementById('summary-bar').innerHTML = `
    <span>Sentences: <b>${data.results.length}</b></span>
    <span>Words processed: <b>${totalWords}</b></span>
    <span>Words with specifics: <b>${coveredWords}</b></span>
  `;
}
</script>
</body>
</html>
"""


# ============================================================
# ROUTE 1 — Home page  (GET /)
# ============================================================
# This is what you see when you open http://127.0.0.1:5000
# It simply returns the HTML form above.

@app.route("/")
def home():
    return render_template_string(HOME_TEMPLATE)


# ============================================================
# ROUTE 2 — Run pipeline  (POST /run)
# ============================================================
# Called by the browser's fetch() when you click "Run".
# Reads the form data, extracts sentences, runs the pipeline,
# saves output files, and returns JSON to the browser.

@app.route("/run", methods=["POST"])
def run():
    input_type = request.form.get("input_type", "single")

    # ── Extract sentences ──────────────────────────────────
    sentences = []

    if input_type == "single":
        text = request.form.get("sentence", "").strip()
        if not text:
            return jsonify({"error": "No sentence provided."})
        sentences = sentences_from_text(text)

    elif input_type in ("txt", "csv"):
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded."})
        f = request.files["file"]
        if f.filename == "":
            return jsonify({"error": "Empty filename."})

        file_bytes = f.read()

        if input_type == "txt":
            sentences = sentences_from_txt(file_bytes)
        else:
            sentences = sentences_from_csv(file_bytes)

        if not sentences:
            return jsonify({"error": "No sentences found in the file."})

    else:
        return jsonify({"error": f"Unknown input type: {input_type}"})

    # ── Run pipeline ───────────────────────────────────────
    all_results = run_pipeline(sentences)

    # ── Save output files ──────────────────────────────────
    # Use a unique run ID so parallel requests don't overwrite
    run_id = uuid.uuid4().hex[:8]
    files  = save_outputs(all_results, run_id)

    # ── Return JSON to the browser ─────────────────────────
    return jsonify({
        "results": all_results,
        "files":   files,
    })


# ============================================================
# ROUTE 3 — Download files  (GET /download/<filename>)
# ============================================================
# Lets the user download the CSV or JSON result file.

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(
        RESULTS_FOLDER,
        filename,
        as_attachment=True,
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  WSD Flask UI")
    print("  Hugging Face Live Link starting...")
    print("="*55 + "\n")
    
    # IMPORTANT: Change port to 7860 for Hugging Face
    app.run(host="0.0.0.0", port=7860, debug=False)

