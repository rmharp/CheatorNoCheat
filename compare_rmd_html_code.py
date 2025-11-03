#!/usr/bin/env python3
"""
compare_rmd_html_code.py

Usage:
    CLI:
        python compare_rmd_html_code.py /path/to/html_dir [--seq-threshold 0.82] [--jaccard-threshold 0.70] [--top-n 50]

    GUI:
        python compare_rmd_html_code.py --gui

What it does:
- reads all .html files in a directory (assume 1 per student)
- extracts code blocks produced by knit Rmd
- normalizes code
- computes pairwise similarity
- writes suspicious_pairs.csv and suspicious_pairs.txt
"""

import os
import sys
import re
import json
import itertools
import difflib
import argparse
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Dict, Tuple, Any

# Silence macOS system Tk deprecation noise; does not affect functionality
os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

# ====== CONFIG ======
# similarity thresholds – adjust if you get too many / too few hits
SEQUENCEMATCHER_THRESHOLD = 0.82   # 0..1, higher = stricter
JACCARD_THRESHOLD = 0.70           # 0..1, higher = stricter
TOP_N_TO_PRINT = 50                # show top N in console
# =====================

class CodeBlockHTMLParser(HTMLParser):
    """
    Fallback HTML parser to grab <pre>, <code>, and RMarkdown-like blocks
    without requiring BeautifulSoup.
    """
    def __init__(self):
        super().__init__()
        self.in_pre = False
        self.in_code = False
        self.current_classes = set()
        self.current_block = []
        self.blocks = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        classes = set(attrs_dict.get("class", "").split())
        if tag == "pre":
            self.in_pre = True
            self.current_classes = classes
            self.current_block = []
        elif tag == "code":
            self.in_code = True
            self.current_classes = classes
            self.current_block = []

    def handle_endtag(self, tag):
        if tag == "pre" and self.in_pre:
            text = "".join(self.current_block).strip()
            if text:
                self.blocks.append(text)
            self.in_pre = False
            self.current_classes = set()
            self.current_block = []
        elif tag == "code" and self.in_code:
            text = "".join(self.current_block).strip()
            if text:
                self.blocks.append(text)
            self.in_code = False
            self.current_classes = set()
            self.current_block = []

    def handle_data(self, data):
        if self.in_pre or self.in_code:
            self.current_block.append(data)


def read_html_code_blocks(path: Path) -> List[str]:
    """
    Read HTML and extract likely R/code chunks from knitted Rmd.
    We try:
      - with BeautifulSoup (if installed)
      - otherwise fallback to HTMLParser
    """
    text = path.read_text(encoding="utf-8", errors="ignore")

    # 1) Try BeautifulSoup if available – more robust for rmarkdown HTML
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(text, "html.parser")
        code_blocks = []

        # rmarkdown often uses <pre class="sourceCode r"><code>...</code></pre>
        for pre in soup.find_all("pre"):
            # sometimes inside <pre><code>
            code = pre.get_text("\n", strip=True)
            if code:
                code_blocks.append(code)

        # also catch <code class="r">...</code> not inside pre
        for code_tag in soup.find_all("code"):
            code = code_tag.get_text("\n", strip=True)
            if code and code not in code_blocks:
                code_blocks.append(code)

        return code_blocks
    except ImportError:
        # 2) Fallback: basic HTML parser
        parser = CodeBlockHTMLParser()
        parser.feed(text)
        return parser.blocks


def normalize_code_block(block: str) -> str:
    """
    Make code comparable:
    - drop R prompts like '>' or '+'
    - remove comments (# ...)
    - collapse whitespace
    - lowercase
    """
    lines = block.splitlines()
    norm_lines = []
    for ln in lines:
        ln = ln.strip()

        # remove leading R console prompt if present
        if ln.startswith("> "):
            ln = ln[2:].strip()
        elif ln.startswith("+ "):
            ln = ln[2:].strip()

        # remove comments (R / many langs)
        # be careful not to kill '#' inside strings, but for plagiarism check
        # this is usually fine.
        ln = re.sub(r"#.*$", "", ln).strip()
        if not ln:
            continue
        norm_lines.append(ln)

    # collapse to one string
    joined = " ".join(norm_lines)
    # normalize whitespace
    joined = re.sub(r"\s+", " ", joined)
    # lowercase to avoid case-based evasion
    joined = joined.lower()
    return joined


def jaccard_similarity(a: str, b: str) -> float:
    """
    Token-based Jaccard on words.
    """
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens and not b_tokens:
        return 0.0
    inter = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(inter) / len(union)


def aggregate_student_code(code_blocks: List[str]) -> str:
    """
    Combine all normalized blocks into one big string per student.
    This way small differences across blocks don't kill the signal.
    """
    return "\n".join(code_blocks)


def analyze_directory(
    html_dir: Path,
    seq_threshold: float = SEQUENCEMATCHER_THRESHOLD,
    jaccard_threshold: float = JACCARD_THRESHOLD,
    top_n_to_print: int = TOP_N_TO_PRINT,
) -> Dict[str, Any]:
    """
    Run analysis over all .html files in a directory. Writes CSV/TXT/JSON next to the data.
    Returns a dictionary with results and output paths. Does not print to stdout.
    """
    if not html_dir.is_dir():
        raise ValueError(f"{html_dir} is not a directory")

    # Recursively find all .html files (handles nested student folders)
    html_files = sorted([p for p in html_dir.rglob("*.html") if p.is_file()])
    if not html_files:
        return {
            "html_files": [],
            "results": [],
            "flagged_pairs": [],
            "csv_path": html_dir / "suspicious_pairs.csv",
            "txt_path": html_dir / "suspicious_pairs.txt",
            "json_path": html_dir / "suspicious_pairs.json",
            "top_n": [],
            "seq_threshold": seq_threshold,
            "jaccard_threshold": jaccard_threshold,
        }

    student_code: Dict[str, List[str]] = {}
    name_to_path: Dict[str, Path] = {}
    for f in html_files:
        blocks = read_html_code_blocks(f)
        norm_blocks = [normalize_code_block(b) for b in blocks if normalize_code_block(b)]
        rel_name = str(f.relative_to(html_dir))
        student_code[rel_name] = norm_blocks
        name_to_path[rel_name] = f

    aggregated: Dict[str, str] = {
        name: aggregate_student_code(blocks)
        for name, blocks in student_code.items()
    }

    results: List[Tuple[str, str, float, float]] = []  # (fileA, fileB, seq, jaccard)
    for (name_a, code_a), (name_b, code_b) in itertools.combinations(aggregated.items(), 2):
        seq_sim = difflib.SequenceMatcher(None, code_a, code_b).ratio()
        jac_sim = jaccard_similarity(code_a, code_b)
        results.append((name_a, name_b, seq_sim, jac_sim))

    results.sort(key=lambda x: (x[2] + x[3]) / 2, reverse=True)

    csv_path = html_dir / "suspicious_pairs.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("file_a,file_b,sequence_similarity,jaccard_similarity,flagged\n")
        for a, b, ss, js in results:
            flagged = (ss >= seq_threshold) or (js >= jaccard_threshold)
            f.write(f"{a},{b},{ss:.4f},{js:.4f},{int(flagged)}\n")

    txt_path = html_dir / "suspicious_pairs.txt"
    flagged_pairs = [
        (a, b, ss, js)
        for (a, b, ss, js) in results
        if ss >= seq_threshold or js >= jaccard_threshold
    ]
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Pairs of students to look into (possible copying):\n")
        f.write(f"(thresholds: seq>={seq_threshold}, jaccard>={jaccard_threshold})\n\n")
        for a, b, ss, js in flagged_pairs:
            f.write(f"{a}  <-->  {b}   (seq={ss:.3f}, jaccard={js:.3f})\n")

    json_path = html_dir / "suspicious_pairs.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "file_a": a,
                    "file_b": b,
                    "sequence_similarity": ss,
                    "jaccard_similarity": js,
                    "flagged": bool(ss >= seq_threshold or js >= jaccard_threshold),
                }
                for (a, b, ss, js) in results
            ],
            f,
            indent=2,
        )

    top_n = results[:top_n_to_print]

    return {
        "html_files": html_files,
        "results": results,
        "flagged_pairs": flagged_pairs,
        "csv_path": csv_path,
        "txt_path": txt_path,
        "json_path": json_path,
        "top_n": top_n,
        "seq_threshold": seq_threshold,
        "jaccard_threshold": jaccard_threshold,
        "name_to_path": name_to_path,
    }


def launch_gui() -> None:
    """Launch a simple Tkinter GUI for running the analysis."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, scrolledtext
        import tkinter.font as tkfont
    except Exception as e:
        print("Failed to initialize GUI components:", e)
        sys.exit(1)

    root = tk.Tk()
    root.title("Rmd/HTML Code Similarity Checker")
    root.geometry("820x640")
    # Explicit background to avoid dark/blank rendering issues
    root.configure(background="white")

    # Fonts
    ui_font = tkfont.Font(family="Helvetica", size=12)
    mono_font = tkfont.Font(family="Menlo", size=11) if sys.platform == "darwin" else tkfont.Font(family="Courier", size=11)

    # State variables
    dir_var = tk.StringVar(value=str(Path.cwd()))
    seq_var = tk.StringVar(value=str(SEQUENCEMATCHER_THRESHOLD))
    jac_var = tk.StringVar(value=str(JACCARD_THRESHOLD))
    topn_var = tk.StringVar(value=str(TOP_N_TO_PRINT))
    status_var = tk.StringVar(value="Idle")
    # Container with pack (robust on macOS)
    container = tk.Frame(root, bg="white")
    container.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

    # Header
    header = tk.Label(container, text="Rmd/HTML Code Similarity Checker", font=("Helvetica", 16, "bold"), bg="white", fg="black")
    header.pack(anchor="w", pady=(0, 8))

    # Dataset row
    ds_row = tk.Frame(container, bg="white")
    ds_row.pack(fill=tk.X, pady=4)
    tk.Label(ds_row, text="Directory:", bg="white", fg="#222", font=ui_font).pack(side=tk.LEFT)
    dir_entry = tk.Entry(ds_row, textvariable=dir_var, font=ui_font)
    dir_entry.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
    def browse_dir():
        chosen = filedialog.askdirectory(initialdir=dir_var.get() or str(Path.cwd()))
        if chosen:
            dir_var.set(chosen)
    tk.Button(ds_row, text="Browse", command=browse_dir, bg="#1976d2", fg="white", activebackground="#1565c0", padx=10, pady=4).pack(side=tk.LEFT, padx=(6,0))

    # Thresholds row
    thr_row = tk.Frame(container, bg="white")
    thr_row.pack(fill=tk.X, pady=2)
    tk.Label(thr_row, text="Sequence:", bg="white", fg="#222", font=ui_font).pack(side=tk.LEFT)
    seq_spin = tk.Spinbox(thr_row, from_=0.0, to=1.0, increment=0.01, textvariable=seq_var, width=6, font=ui_font)
    seq_spin.pack(side=tk.LEFT, padx=(4, 16))
    tk.Label(thr_row, text="Jaccard:", bg="white", fg="#222", font=ui_font).pack(side=tk.LEFT)
    jac_spin = tk.Spinbox(thr_row, from_=0.0, to=1.0, increment=0.01, textvariable=jac_var, width=6, font=ui_font)
    jac_spin.pack(side=tk.LEFT, padx=(4, 16))
    tk.Label(thr_row, text="Top N:", bg="white", fg="#222", font=ui_font).pack(side=tk.LEFT)
    topn_spin = tk.Spinbox(thr_row, from_=1, to=500, increment=1, textvariable=topn_var, width=6, font=ui_font)
    topn_spin.pack(side=tk.LEFT, padx=(4, 0))

    # Actions row
    act_row = tk.Frame(container, bg="white")
    act_row.pack(fill=tk.X, pady=6)
    analyze_btn = tk.Button(act_row, text="Analyze", command=lambda: run_analysis_clicked(), bg="#2e7d32", fg="white", activebackground="#1b5e20", padx=12, pady=6)
    analyze_btn.pack(side=tk.LEFT)
    def open_selected_folder():
        path = Path(dir_var.get().strip()).expanduser()
        if path.exists():
            try:
                if sys.platform == "darwin":
                    os.system(f'open "{path}"')
                elif sys.platform.startswith("win"):
                    os.startfile(str(path))  # type: ignore[attr-defined]
                else:
                    os.system(f'xdg-open "{path}"')
            except Exception:
                pass
    tk.Button(act_row, text="Open Folder", command=open_selected_folder, padx=10, pady=6).pack(side=tk.LEFT, padx=8)
    tk.Label(act_row, textvariable=status_var, bg="white", fg="#333").pack(side=tk.RIGHT)

    # Review button (enabled after analysis)
    review_btn_state = {"enabled": False}
    def noop():
        pass
    review_btn = tk.Button(act_row, text="Review Flagged...", command=noop, state=tk.DISABLED, padx=10, pady=6)
    review_btn.pack(side=tk.LEFT, padx=8)

    # Results label
    tk.Label(container, text="Results", bg="white", fg="#222", font=("Helvetica", 13, "bold")).pack(anchor="w", pady=(6, 2))
    # Results area (reduced height so controls never get occluded)
    text = scrolledtext.ScrolledText(container, height=16, wrap=tk.WORD, font=mono_font)
    text.configure(background="white", foreground="#111", insertbackground="#111")
    text.pack(fill=tk.BOTH, expand=True)
    text.insert("1.0", "Select a directory and click Analyze to begin.\n")

    try:
        root.lift(); root.focus_force()
    except Exception:
        pass

    # Dedicated results window (fallback) to guarantee visibility on quirky macOS Tk themes
    results_win_ref = {"win": None, "txt": None}
    def open_results_window():
        if results_win_ref["win"] and results_win_ref["win"].winfo_exists():
            return
        win = tk.Toplevel(root)
        win.title("Analysis Results")
        win.geometry("900x700")
        win.configure(background="white")
        txt = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=mono_font)
        txt.configure(background="white", foreground="#111", insertbackground="#111")
        txt.pack(fill=tk.BOTH, expand=True)
        results_win_ref["win"] = win
        results_win_ref["txt"] = txt
        try:
            win.lift(); win.focus_force()
        except Exception:
            pass

    def write_results_line(line: str):
        try:
            text.insert(tk.END, line)
            text.see(tk.END)
            text.update_idletasks()
        except Exception as e:
            print("UI write (main) failed:", e, file=sys.stderr)
        try:
            open_results_window()
            results_win_ref["txt"].insert(tk.END, line)
            results_win_ref["txt"].see(tk.END)
            results_win_ref["txt"].update_idletasks()
        except Exception as e:
            print("UI write (results window) failed:", e, file=sys.stderr)

    # Review state
    review_state = {
        "pairs": [],               # list of (a,b,ss,js)
        "idx": 0,
        "labels_path": None,       # Path
        "labels": {},              # key -> label
        "name_to_path": {},        # name -> Path
        "html_dir": None,          # Path
    }

    def pair_key(a: str, b: str) -> str:
        return "||".join(sorted([a, b]))

    def load_labels_csv(labels_csv: Path) -> Dict[str, str]:
        if not labels_csv.exists():
            return {}
        data: Dict[str, str] = {}
        try:
            for line in labels_csv.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
                if not line.strip():
                    continue
                parts = line.split(",")
                if len(parts) < 5:
                    continue
                a, b, _ss, _js, label = parts[:5]
                data[pair_key(a, b)] = label.strip()
        except Exception:
            pass
        return data

    def append_label(labels_csv: Path, a: str, b: str, ss: float, js: float, label: str) -> None:
        try:
            header_needed = not labels_csv.exists()
            with labels_csv.open("a", encoding="utf-8") as f:
                if header_needed:
                    f.write("file_a,file_b,sequence_similarity,jaccard_similarity,label\n")
                f.write(f"{a},{b},{ss:.4f},{js:.4f},{label}\n")
        except Exception as e:
            write_results_line(f"Failed to record label: {e}\n")

    def open_pair(a: str, b: str):
        name_to_path: Dict[str, Path] = review_state["name_to_path"]  # type: ignore[assignment]
        p1 = name_to_path.get(a)
        p2 = name_to_path.get(b)
        if not p1 and not p2:
            return
        try:
            if sys.platform == "darwin":
                paths = [str(p) for p in [p1, p2] if p]
                if paths:
                    subprocess.run(["open", *paths], check=False)
            elif sys.platform.startswith("win"):
                for p in [p1, p2]:
                    if p:
                        os.startfile(str(p))  # type: ignore[attr-defined]
            else:
                for p in [p1, p2]:
                    if p:
                        subprocess.run(["xdg-open", str(p)], check=False)
        except Exception as e:
            write_results_line(f"Failed to open pair: {e}\n")

    def start_review():
        if not review_state["pairs"]:
            messagebox.showinfo("No pairs", "No flagged pairs to review.")
            return
        # Build review window
        win = tk.Toplevel(root)
        win.title("Review Flagged Pairs")
        win.geometry("1100x720")
        win.configure(background="white")

        # Header and progress
        header = tk.Label(win, text="Review Flagged Pairs", font=("Helvetica", 14, "bold"), bg="white")
        header.pack(anchor="w", padx=10, pady=(8, 4))

        progress_var = tk.StringVar(value="")
        progress_lbl = tk.Label(win, textvariable=progress_var, bg="white", fg="#444")
        progress_lbl.pack(anchor="w", padx=10)

        # View mode
        mode_row = tk.Frame(win, bg="white")
        mode_row.pack(fill=tk.X, padx=10, pady=(6, 4))
        tk.Label(mode_row, text="View:", bg="white").pack(side=tk.LEFT)
        view_mode = tk.StringVar(value="normalized")
        tk.Radiobutton(mode_row, text="Normalized code", variable=view_mode, value="normalized", bg="white", command=lambda: refresh()).pack(side=tk.LEFT, padx=6)
        tk.Radiobutton(mode_row, text="Raw HTML", variable=view_mode, value="raw", bg="white", command=lambda: refresh()).pack(side=tk.LEFT)

        # Side-by-side panes
        panes = tk.Frame(win, bg="white")
        panes.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        left_panel = tk.Frame(panes, bg="white")
        right_panel = tk.Frame(panes, bg="white")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5,0))

        left_name_var = tk.StringVar(value="")
        right_name_var = tk.StringVar(value="")
        tk.Label(left_panel, textvariable=left_name_var, bg="white", fg="#000", anchor="w").pack(fill=tk.X)
        tk.Label(right_panel, textvariable=right_name_var, bg="white", fg="#000", anchor="w").pack(fill=tk.X)

        left_text = scrolledtext.ScrolledText(left_panel, wrap=tk.WORD, font=mono_font)
        right_text = scrolledtext.ScrolledText(right_panel, wrap=tk.WORD, font=mono_font)
        for txt in (left_text, right_text):
            txt.configure(background="white", foreground="#111", insertbackground="#111")
            txt.pack(fill=tk.BOTH, expand=True)

        # Buttons
        btn_row = tk.Frame(win, bg="white")
        btn_row.pack(fill=tk.X, padx=10, pady=10)
        tk.Button(btn_row, text="Open Pair", command=lambda: open_current_pair(), padx=10, pady=6).pack(side=tk.LEFT)
        tk.Button(btn_row, text="Cheated", command=lambda: vote("cheated"), bg="#c62828", fg="white", padx=12, pady=6).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_row, text="No Cheating", command=lambda: vote("no_cheating"), bg="#2e7d32", fg="white", padx=12, pady=6).pack(side=tk.LEFT)
        tk.Button(btn_row, text="Prev", command=lambda: move(-1), padx=10, pady=6).pack(side=tk.RIGHT)
        tk.Button(btn_row, text="Next", command=lambda: move(1), padx=10, pady=6).pack(side=tk.RIGHT, padx=8)

        def refresh():
            i = review_state["idx"]
            pairs = review_state["pairs"]
            if not pairs:
                progress_var.set("No pairs to review.")
                return
            a, b, ss, js = pairs[i]
            labels = review_state["labels"]
            lab = labels.get(pair_key(a, b), "(unlabeled)")
            progress_var.set(f"Pair {i+1}/{len(pairs)} | seq={ss:.3f}, jaccard={js:.3f} | Label: {lab}")
            left_name_var.set(a)
            right_name_var.set(b)

            # Load content for display
            name_to_path: Dict[str, Path] = review_state["name_to_path"]  # type: ignore[assignment]
            p1 = name_to_path.get(a)
            p2 = name_to_path.get(b)
            mode = view_mode.get()

            def render(path: Path) -> str:
                if not path:
                    return "(missing file)"
                try:
                    if mode == "raw":
                        return path.read_text(encoding="utf-8", errors="ignore")
                    # normalized code view (from extracted code blocks)
                    blocks = read_html_code_blocks(path)
                    normed = [normalize_code_block(b) for b in blocks if normalize_code_block(b)]
                    return "\n".join(normed) if normed else "(no code blocks found)"
                except Exception as e:
                    return f"(failed to load: {e})"

            left_text.configure(state="normal")
            right_text.configure(state="normal")
            left_text.delete("1.0", tk.END)
            right_text.delete("1.0", tk.END)
            left_text.insert(tk.END, render(p1) if p1 else "(missing)")
            right_text.insert(tk.END, render(p2) if p2 else "(missing)")
            left_text.see(tk.END); right_text.see(tk.END)
            left_text.configure(state="normal"); right_text.configure(state="normal")

        def open_current_pair():
            a, b, _ss, _js = review_state["pairs"][review_state["idx"]]
            open_pair(a, b)

        def vote(label: str):
            i = review_state["idx"]
            a, b, ss, js = review_state["pairs"][i]
            labels_csv: Path = review_state["labels_path"]  # type: ignore[assignment]
            append_label(labels_csv, a, b, ss, js, label)
            review_state["labels"][pair_key(a, b)] = label
            move(1)

        def move(delta: int):
            n = len(review_state["pairs"]) or 1
            review_state["idx"] = max(0, min(n - 1, review_state["idx"] + delta))
            refresh()

        refresh()
        try:
            win.lift(); win.focus_force()
        except Exception:
            pass

    def enable_review(result: Dict[str, Any], html_dir: Path):
        review_state["pairs"] = result.get("flagged_pairs", [])
        review_state["idx"] = 0
        review_state["labels_path"] = html_dir / "manual_labels.csv"
        review_state["labels"] = load_labels_csv(review_state["labels_path"])  # type: ignore[arg-type]
        review_state["name_to_path"] = result.get("name_to_path", {})
        review_state["html_dir"] = html_dir
        review_btn.configure(state=tk.NORMAL, command=start_review)
        review_btn_state["enabled"] = True

    def run_analysis_clicked():
        directory = dir_var.get().strip()
        try:
            seq_t = float(seq_var.get())
            jac_t = float(jac_var.get())
            top_n = int(topn_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter numeric thresholds and top N.")
            return

        if not directory:
            messagebox.showerror("Missing directory", "Please choose a directory containing .html files.")
            return

        html_dir = Path(directory).expanduser().resolve()
        if not html_dir.is_dir():
            messagebox.showerror("Invalid directory", f"{html_dir} is not a directory.")
            return

        status_var.set("Running...")
        root.config(cursor="watch")
        # Show immediate feedback in output area
        try:
            text.config(state="normal")
            text.delete("1.0", tk.END)
        except Exception:
            pass
        open_results_window()
        try:
            results_win_ref["txt"].delete("1.0", tk.END)
        except Exception:
            pass
        write_results_line(f"Analyzing directory: {html_dir}\n")
        write_results_line(f"Seq threshold: {seq_t}, Jaccard threshold: {jac_t}, Top N: {top_n}\n\n")
        try:
            result = analyze_directory(html_dir, seq_t, jac_t, top_n)
        except Exception as e:
            write_results_line(f"Error: {e}\n")
            messagebox.showerror("Error", str(e))
            status_var.set("Error")
            return

        try:
            text.delete("1.0", tk.END)
            results_win_ref["txt"].delete("1.0", tk.END)
        except Exception:
            pass
        html_count = len(result.get("html_files", []))
        if html_count == 0:
            write_results_line("No .html files found. If your student work is in subfolders, selecting the parent folder is fine (the tool now searches recursively).\n")
            status_var.set("Done")
            root.config(cursor="")
            return
        write_results_line(f"Analyzed {html_count} HTML files.\n")
        write_results_line(f"Full results written to: {result['csv_path']}\n")
        write_results_line(f"Flagged pairs written to: {result['txt_path']}\n\n")

        write_results_line("Top pairs by similarity:\n")
        for i, (a, b, ss, js) in enumerate(result.get("top_n", []), start=1):
            mark = ""
            if ss >= result["seq_threshold"] or js >= result["jaccard_threshold"]:
                mark = "  <-- CHECK"
            write_results_line(f"{i:2d}. {a}  <-->  {b} | seq={ss:.3f}, jaccard={js:.3f}{mark}\n")
        if not result.get("top_n"):
            write_results_line("(No pairs to display.)\n")

        # Enable review workflow
        enable_review(result, html_dir)

        status_var.set("Done")
        root.config(cursor="")

    # Bind real handler for Analyze (pack-based UI already wired)
    def run_analysis_clicked_bound():
        run_analysis_clicked()
    # keep a reference to avoid GC in some Tk builds
    root._analyze_cb = run_analysis_clicked_bound

    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Compare code blocks in knitted Rmd HTML exports.")
    parser.add_argument("html_dir", nargs="?", help="Directory containing .html files (one per student)")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--seq-threshold", type=float, default=SEQUENCEMATCHER_THRESHOLD, help="SequenceMatcher threshold (0-1)")
    parser.add_argument("--jaccard-threshold", type=float, default=JACCARD_THRESHOLD, help="Jaccard threshold (0-1)")
    parser.add_argument("--top-n", type=int, default=TOP_N_TO_PRINT, help="Top N pairs to print")
    parser.add_argument("--open-flagged", action="store_true", help="Open flagged HTML pairs after analysis")
    parser.add_argument("--open-limit", type=int, default=3, help="Maximum number of flagged pairs to open")

    args = parser.parse_args()

    if args.gui:
        launch_gui()
        return

    if not args.html_dir:
        parser.print_usage()
        sys.exit(1)

    html_dir = Path(args.html_dir).expanduser().resolve()
    try:
        result = analyze_directory(
            html_dir,
            seq_threshold=args.__dict__["seq_threshold"],
            jaccard_threshold=args.__dict__["jaccard_threshold"],
            top_n_to_print=args.__dict__["top_n"],
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not result.get("html_files"):
        print("No .html files found in directory.")
        sys.exit(0)

    print(f"\nAnalyzed {len(result['html_files'])} HTML files.")
    print(f"Full results written to: {result['csv_path']}")
    print(f"Flagged pairs written to: {result['txt_path']}\n")

    print("Top pairs by similarity:")
    for i, (a, b, ss, js) in enumerate(result.get("top_n", []), start=1):
        mark = ""
        if ss >= result["seq_threshold"] or js >= result["jaccard_threshold"]:
            mark = "  <-- CHECK"
        print(f"{i:2d}. {a}  <-->  {b} | seq={ss:.3f}, jaccard={js:.3f}{mark}")

    # Optionally open flagged pairs
    if args.__dict__.get("open_flagged"):
        flagged_pairs = result.get("flagged_pairs", [])
        name_to_path: Dict[str, Path] = result.get("name_to_path", {})  # type: ignore[assignment]
        to_open = flagged_pairs[: max(0, args.__dict__.get("open_limit", 0))]
        def platform_open(paths: List[Path]) -> None:
            try:
                if sys.platform == "darwin":
                    subprocess.run(["open", *[str(p) for p in paths]], check=False)
                elif sys.platform.startswith("win"):
                    for p in paths:
                        os.startfile(str(p))  # type: ignore[attr-defined]
                else:
                    subprocess.run(["xdg-open", str(paths[0])], check=False)
                    for p in paths[1:]:
                        subprocess.run(["xdg-open", str(p)], check=False)
            except Exception as e:
                print(f"Failed to open files: {e}")

        if not to_open:
            print("No flagged pairs to open.")
        else:
            print(f"Opening up to {len(to_open)} flagged pairs...")
            for a, b, _ss, _js in to_open:
                p1 = name_to_path.get(a)
                p2 = name_to_path.get(b)
                if p1 and p2:
                    platform_open([p1, p2])
                elif p1:
                    platform_open([p1])
                elif p2:
                    platform_open([p2])


if __name__ == "__main__":
    main()
