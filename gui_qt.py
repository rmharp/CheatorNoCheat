#!/usr/bin/env python3
"""
Qt GUI for Cheat or No Cheat - side-by-side HTML comparison with voting.

Run:
    python gui_qt.py

Requires:
    pip install PySide6 beautifulsoup4
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWebEngineWidgets import QWebEngineView
from datetime import datetime
import re
import json
import shutil
import random
import networkx as nx
import math
import io
import base64
import statistics
import numpy as np
import webbrowser

# Reuse the analyzer from the existing script
from compare_rmd_html_code import analyze_directory

# Matplotlib embedding for statistics
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self, width: float = 5, height: float = 3, dpi: int = 100) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


class StatsWindow(QtWidgets.QMainWindow):
    def __init__(self, assignments_root: Path) -> None:
        super().__init__()
        self.setWindowTitle("Cheat or No Cheat - Statistics")
        self.resize(1200, 800)
        # Fallback to legacy folder name if needed
        if not assignments_root.exists():
            legacy = assignments_root.parent.parent / ".youcheated" / "assignments"
            if legacy.exists():
                assignments_root = legacy
        self.root = assignments_root

        # Load assignment list
        self.assignments = self.discover_assignments()

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        # Heatmap tab
        heat_tab = QtWidgets.QWidget()
        h_layout = QtWidgets.QVBoxLayout(heat_tab)
        top = QtWidgets.QHBoxLayout()
        self.heat_assign = QtWidgets.QComboBox()
        self.heat_assign.addItems(sorted(self.assignments.keys()))
        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.addItems(["jaccard", "sequence"])
        draw_btn = QtWidgets.QPushButton("Draw Heatmap")
        draw_btn.clicked.connect(self.draw_heatmap)
        top.addWidget(QtWidgets.QLabel("Assignment:"))
        top.addWidget(self.heat_assign)
        top.addWidget(QtWidgets.QLabel("Metric:"))
        top.addWidget(self.metric_combo)
        top.addStretch(1)
        top.addWidget(draw_btn)
        h_layout.addLayout(top)
        self.heat_canvas = MplCanvas(width=8, height=6, dpi=100)
        self.heat_cbar = None  # track the heatmap colorbar to avoid duplicates
        h_layout.addWidget(self.heat_canvas, 1)
        tabs.addTab(heat_tab, "Heatmap")

        # Trends tab
        trends_tab = QtWidgets.QWidget()
        t_layout = QtWidgets.QVBoxLayout(trends_tab)
        ctl = QtWidgets.QHBoxLayout()
        self.category_combo = QtWidgets.QComboBox()
        self.category_combo.setEditable(False)
        self.category_combo.currentIndexChanged.connect(self.on_category_changed)
        self.student_combo = QtWidgets.QComboBox()
        self.all_students_chk = QtWidgets.QCheckBox("All students")
        self.all_students_chk.stateChanged.connect(self.draw_trends)
        self.only_cheaters_chk = QtWidgets.QCheckBox("Cheaters only")
        self.only_cheaters_chk.stateChanged.connect(self.refresh_students)
        plot_btn = QtWidgets.QPushButton("Plot Trends")
        plot_btn.clicked.connect(self.draw_trends)
        ctl.addWidget(QtWidgets.QLabel("Categories:"))
        ctl.addWidget(self.category_combo, 1)
        ctl.addWidget(QtWidgets.QLabel("Student:"))
        ctl.addWidget(self.student_combo)
        ctl.addWidget(self.all_students_chk)
        ctl.addWidget(self.only_cheaters_chk)
        ctl.addStretch(1)
        ctl.addWidget(plot_btn)
        t_layout.addLayout(ctl)

        # Assignment checklist (filtered by category)
        assign_row = QtWidgets.QHBoxLayout()
        assign_row.addWidget(QtWidgets.QLabel("Assignments:"))
        self.assign_select_all = QtWidgets.QCheckBox("Select all")
        self.assign_select_all.setChecked(True)
        self.assign_select_all.stateChanged.connect(self.on_assign_select_all)
        assign_row.addWidget(self.assign_select_all)
        assign_row.addStretch(1)
        t_layout.addLayout(assign_row)

        self.assign_list = QtWidgets.QListWidget()
        self.assign_list.setAlternatingRowColors(True)
        self.assign_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.assign_list.itemChanged.connect(self.on_assign_item_changed)
        self.assign_list.setMaximumHeight(120)
        t_layout.addWidget(self.assign_list)
        self.trend_canvas_j = MplCanvas(width=8, height=3, dpi=100)
        self.trend_canvas_s = MplCanvas(width=8, height=3, dpi=100)
        t_layout.addWidget(QtWidgets.QLabel("Jaccard (max vs others per assignment)"))
        t_layout.addWidget(self.trend_canvas_j)
        t_layout.addWidget(QtWidgets.QLabel("Sequence (max vs others per assignment)"))
        t_layout.addWidget(self.trend_canvas_s)
        tabs.addTab(trends_tab, "Trends")

        # Network tab
        net_tab = QtWidgets.QWidget()
        n_layout = QtWidgets.QVBoxLayout(net_tab)

        n_ctl = QtWidgets.QHBoxLayout()
        self.net_category = QtWidgets.QComboBox()
        self.net_category.addItems(["All"] + sorted({(m.get("category", "") or "Uncategorized").strip() for m in self.assignments.values()}))
        self.net_category.currentIndexChanged.connect(self.on_net_category_changed)
        self.net_metric = QtWidgets.QComboBox(); self.net_metric.addItems(["count", "max_jaccard", "max_sequence"])
        self.net_min_count = QtWidgets.QSpinBox(); self.net_min_count.setRange(1, 99); self.net_min_count.setValue(1)
        self.net_jac_thr = QtWidgets.QDoubleSpinBox(); self.net_jac_thr.setRange(0,1); self.net_jac_thr.setSingleStep(0.01); self.net_jac_thr.setValue(0.75)
        self.net_seq_thr = QtWidgets.QDoubleSpinBox(); self.net_seq_thr.setRange(0,1); self.net_seq_thr.setSingleStep(0.01); self.net_seq_thr.setValue(0.80)
        net_btn = QtWidgets.QPushButton("Build Graph"); net_btn.clicked.connect(self.draw_cheater_network)
        n_ctl.addWidget(QtWidgets.QLabel("Category:")); n_ctl.addWidget(self.net_category)
        n_ctl.addWidget(QtWidgets.QLabel("Edge metric:")); n_ctl.addWidget(self.net_metric)
        n_ctl.addWidget(QtWidgets.QLabel("Min co-flag count:")); n_ctl.addWidget(self.net_min_count)
        n_ctl.addWidget(QtWidgets.QLabel("Jaccard ≥")); n_ctl.addWidget(self.net_jac_thr)
        n_ctl.addWidget(QtWidgets.QLabel("Seq ≥")); n_ctl.addWidget(self.net_seq_thr)
        n_ctl.addStretch(1); n_ctl.addWidget(net_btn)
        n_layout.addLayout(n_ctl)
        # Explanatory note
        note = QtWidgets.QLabel("Edges are included only if a pair was manually labeled 'cheated' together at least 'Min co-flag count' times. Edge opacity visualizes max Jaccard/Sequence (for context only); edge metric controls edge width.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#555")
        n_layout.addWidget(note)

        # Network assignment checklist (independent of Trends)
        net_assign_row = QtWidgets.QHBoxLayout()
        net_assign_row.addWidget(QtWidgets.QLabel("Assignments:"))
        self.net_assign_select_all = QtWidgets.QCheckBox("Select all")
        self.net_assign_select_all.setChecked(True)
        self.net_assign_select_all.stateChanged.connect(self.on_net_assign_select_all)
        net_assign_row.addWidget(self.net_assign_select_all)
        net_assign_row.addStretch(1)
        n_layout.addLayout(net_assign_row)

        self.net_assign_list = QtWidgets.QListWidget()
        self.net_assign_list.setAlternatingRowColors(True)
        self.net_assign_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.net_assign_list.itemChanged.connect(self.on_net_assign_item_changed)
        self.net_assign_list.setMaximumHeight(120)
        n_layout.addWidget(self.net_assign_list)
        # Interactive Plotly view in a web engine
        self.net_web = QWebEngineView()
        n_layout.addWidget(self.net_web, 1)
        tabs.addTab(net_tab, "Network")

        # Report tab
        report_tab = QtWidgets.QWidget()
        r_layout = QtWidgets.QVBoxLayout(report_tab)
        r_ctl = QtWidgets.QHBoxLayout()
        self.rep_category = QtWidgets.QComboBox()
        self.rep_category.addItems(["All"] + sorted({(m.get("category", "") or "Uncategorized").strip() for m in self.assignments.values()}))
        self.rep_category.currentIndexChanged.connect(self.on_report_category_changed)
        r_ctl.addWidget(QtWidgets.QLabel("Category:")); r_ctl.addWidget(self.rep_category)
        r_ctl.addStretch(1)
        self.rep_btn = QtWidgets.QPushButton("Generate Report…")
        self.rep_btn.clicked.connect(self.generate_report)
        r_ctl.addWidget(self.rep_btn)
        r_layout.addLayout(r_ctl)

        rep_assign_row = QtWidgets.QHBoxLayout()
        rep_assign_row.addWidget(QtWidgets.QLabel("Assignments:"))
        self.rep_assign_select_all = QtWidgets.QCheckBox("Select all")
        self.rep_assign_select_all.setChecked(True)
        self.rep_assign_select_all.stateChanged.connect(self.on_report_assign_select_all)
        rep_assign_row.addWidget(self.rep_assign_select_all)
        rep_assign_row.addStretch(1)
        r_layout.addLayout(rep_assign_row)

        self.rep_assign_list = QtWidgets.QListWidget()
        self.rep_assign_list.setAlternatingRowColors(True)
        self.rep_assign_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.rep_assign_list.itemChanged.connect(self.on_report_assign_item_changed)
        self.rep_assign_list.setMaximumHeight(140)
        r_layout.addWidget(self.rep_assign_list)

        tabs.addTab(report_tab, "Report")

        self.refresh_categories()
        self.refresh_students()
        self.populate_assignment_checks()
        self.populate_network_assignment_checks()
        self.populate_report_assignment_checks()
    def on_category_changed(self) -> None:
        self.populate_assignment_checks()
        self.refresh_students()
        self.draw_trends()

    def refresh_categories(self) -> None:
        cats = { (m.get("category", "") or "Uncategorized").strip() for m in self.assignments.values() }
        ordered = ["All"] + sorted(cats)
        current = self.category_combo.currentText()
        self.category_combo.blockSignals(True)
        self.category_combo.clear()
        self.category_combo.addItems(ordered)
        if current and current in ordered:
            self.category_combo.setCurrentText(current)
        self.category_combo.blockSignals(False)
        self.populate_assignment_checks()

    def populate_assignment_checks(self) -> None:
        sel_cat = (self.category_combo.currentText() or "All").strip()
        names = [name for name, m in self.assignments.items() if sel_cat == "All" or (m.get("category", "") or "Uncategorized").strip() == sel_cat]
        names.sort()
        self.assign_list.blockSignals(True)
        self.assign_list.clear()
        for n in names:
            it = QtWidgets.QListWidgetItem(n)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked)
            self.assign_list.addItem(it)
        self.assign_list.blockSignals(False)
        self.assign_select_all.blockSignals(True)
        self.assign_select_all.setChecked(True)
        self.assign_select_all.blockSignals(False)

    # --- Network assignment checklist (independent from Trends) ---
    def on_net_category_changed(self) -> None:
        self.populate_network_assignment_checks()

    def populate_network_assignment_checks(self) -> None:
        # Build list for currently selected Network category
        sel_cat = (self.net_category.currentText() or "All").strip()
        names = [name for name, m in self.assignments.items() if sel_cat == "All" or (m.get("category", "") or "Uncategorized").strip() == sel_cat]
        names.sort()
        if not hasattr(self, 'net_assign_list'):
            return
        self.net_assign_list.blockSignals(True)
        self.net_assign_list.clear()
        for n in names:
            it = QtWidgets.QListWidgetItem(n)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked)
            self.net_assign_list.addItem(it)
        self.net_assign_list.blockSignals(False)
        if hasattr(self, 'net_assign_select_all'):
            self.net_assign_select_all.blockSignals(True)
            self.net_assign_select_all.setChecked(True)
            self.net_assign_select_all.blockSignals(False)

    def get_selected_network_assignments(self) -> set:
        if not hasattr(self, 'net_assign_list'):
            return set()
        selected = set()
        for i in range(self.net_assign_list.count()):
            it = self.net_assign_list.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                selected.add(it.text())
        return selected

    def on_net_assign_select_all(self) -> None:
        if not hasattr(self, 'net_assign_list'):
            return
        check = self.net_assign_select_all.isChecked()
        self.net_assign_list.blockSignals(True)
        for i in range(self.net_assign_list.count()):
            self.net_assign_list.item(i).setCheckState(QtCore.Qt.Checked if check else QtCore.Qt.Unchecked)
        self.net_assign_list.blockSignals(False)

    def on_net_assign_item_changed(self, _item: QtWidgets.QListWidgetItem) -> None:
        if not hasattr(self, 'net_assign_list'):
            return
        total = self.net_assign_list.count()
        checked = sum(1 for i in range(total) if self.net_assign_list.item(i).checkState() == QtCore.Qt.Checked)
        self.net_assign_select_all.blockSignals(True)
        self.net_assign_select_all.setChecked(checked == total and total > 0)
        self.net_assign_select_all.blockSignals(False)

    def get_selected_assignments(self) -> set:
        selected = set()
        for i in range(self.assign_list.count()):
            it = self.assign_list.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                selected.add(it.text())
        return selected

    def on_assign_select_all(self) -> None:
        check = self.assign_select_all.isChecked()
        self.assign_list.blockSignals(True)
        for i in range(self.assign_list.count()):
            self.assign_list.item(i).setCheckState(QtCore.Qt.Checked if check else QtCore.Qt.Unchecked)
        self.assign_list.blockSignals(False)
        self.refresh_students()
        self.draw_trends()

    def on_assign_item_changed(self, _item: QtWidgets.QListWidgetItem) -> None:
        # update select-all state
        total = self.assign_list.count()
        checked = sum(1 for i in range(total) if self.assign_list.item(i).checkState() == QtCore.Qt.Checked)
        self.assign_select_all.blockSignals(True)
        self.assign_select_all.setChecked(checked == total and total > 0)
        self.assign_select_all.blockSignals(False)
        self.refresh_students()

    # --- Report assignment checklist ---
    def on_report_category_changed(self) -> None:
        self.populate_report_assignment_checks()

    def populate_report_assignment_checks(self) -> None:
        sel_cat = (self.rep_category.currentText() or "All").strip()
        names = [name for name, m in self.assignments.items() if sel_cat == "All" or (m.get("category", "") or "Uncategorized").strip() == sel_cat]
        names.sort()
        self.rep_assign_list.blockSignals(True)
        self.rep_assign_list.clear()
        for n in names:
            it = QtWidgets.QListWidgetItem(n)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked)
            self.rep_assign_list.addItem(it)
        self.rep_assign_list.blockSignals(False)
        self.rep_assign_select_all.blockSignals(True)
        self.rep_assign_select_all.setChecked(True)
        self.rep_assign_select_all.blockSignals(False)

    def get_selected_report_assignments(self) -> set:
        selected = set()
        for i in range(self.rep_assign_list.count()):
            it = self.rep_assign_list.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                selected.add(it.text())
        return selected

    def on_report_assign_select_all(self) -> None:
        check = self.rep_assign_select_all.isChecked()
        self.rep_assign_list.blockSignals(True)
        for i in range(self.rep_assign_list.count()):
            self.rep_assign_list.item(i).setCheckState(QtCore.Qt.Checked if check else QtCore.Qt.Unchecked)
        self.rep_assign_list.blockSignals(False)

    def on_report_assign_item_changed(self, _item: QtWidgets.QListWidgetItem) -> None:
        total = self.rep_assign_list.count()
        checked = sum(1 for i in range(total) if self.rep_assign_list.item(i).checkState() == QtCore.Qt.Checked)
        self.rep_assign_select_all.blockSignals(True)
        self.rep_assign_select_all.setChecked(checked == total and total > 0)
        self.rep_assign_select_all.blockSignals(False)

    def discover_assignments(self) -> dict:
        out = {}
        if not self.root.exists():
            return out
        for d in sorted(self.root.iterdir()):
            meta = d / "results.json"
            if meta.exists():
                try:
                    out[d.name] = json.loads(meta.read_text(encoding="utf-8"))
                except Exception:
                    pass
        return out

    @staticmethod
    def extract_student(name_or_path: str) -> str:
        """Get stable student identifier: filename stem before first underscore.
        Accepts relative paths like "subdir/lastnamefirstname_123_hw1.html".
        """
        base = Path(name_or_path).name
        stem = Path(base).stem
        idx = stem.find("_")
        return stem[:idx] if idx != -1 else stem

    def get_all_students(self) -> set:
        # restrict to current category selection if any
        sel = (self.category_combo.currentText() or "All").strip().lower()
        sel_assign = self.get_selected_assignments()
        names = set()
        for name, meta in self.assignments.items():
            if sel != "all" and (meta.get("category", "").strip().lower() != sel):
                continue
            if sel_assign and name not in sel_assign:
                continue
            for pair in meta.get("results", []):
                names.add(self.extract_student(pair["a"]))
                names.add(self.extract_student(pair["b"]))  # type: ignore[index]
        return names

    def get_cheater_students(self) -> set:
        names = set()
        sel = (self.category_combo.currentText() or "All").strip()
        sel_assign = self.get_selected_assignments()
        for assign_name, meta in self.assignments.items():
            if sel != "All" and (meta.get("category", "") or "Uncategorized").strip() != sel:
                continue
            if sel_assign and assign_name not in sel_assign:
                continue
            labels_path = self.root / assign_name / "manual_labels.csv"
            if not labels_path.exists():
                continue
            try:
                lines = labels_path.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]
                for line in lines:
                    if not line.strip():
                        continue
                    a, b, _s, _j, lab = line.split(",", 4)
                    if lab.strip() == "cheated":
                        names.add(self.extract_student(a))
                        names.add(self.extract_student(b))
            except Exception:
                pass
        return names

    def refresh_students(self) -> None:
        current = self.student_combo.currentText()
        if self.only_cheaters_chk.isChecked():
            names = self.get_cheater_students()
        else:
            names = self.get_all_students()
        self.student_combo.blockSignals(True)
        self.student_combo.clear()
        self.student_combo.addItems(sorted(names))
        # restore selection if possible
        if current and current in names:
            idx = self.student_combo.findText(current)
            if idx >= 0:
                self.student_combo.setCurrentIndex(idx)
        self.student_combo.blockSignals(False)

    # --- Network helpers ---
    def _norm(self, name: str) -> str:
        return self.extract_student(name)

    def _load_labels(self, path: Path) -> List[Tuple[str, str, str]]:
        if not path.exists():
            return []
        out: List[Tuple[str, str, str]] = []
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
                if not line.strip():
                    continue
                a, b, ss, js, label = line.split(",", 4)
                out.append((a, b, label.strip()))
        except Exception:
            pass
        return out

    def _pair_lookup(self, meta: dict) -> Dict[Tuple[str, str], Tuple[float, float]]:
        lookup: Dict[Tuple[str, str], Tuple[float, float]] = {}
        for p in meta.get("results", []):
            a = self._norm(p["a"]) ; b = self._norm(p["b"])  # type: ignore[index]
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            lookup[key] = (float(p["seq"]), float(p["jac"]))  # type: ignore[index]
        return lookup

    def draw_cheater_network(self) -> None:
        cat = (self.net_category.currentText() or "All").strip().lower()
        sel_assign = self.get_selected_network_assignments() if hasattr(self, 'get_selected_network_assignments') else set()
        metas = [(name, m) for name, m in self.assignments.items()
                 if (cat == "all" or (m.get("category", "").strip().lower() == cat))
                 and ((not sel_assign) or (name in sel_assign))]

        # aggregate pair info
        agg: Dict[Tuple[str, str], Dict[str, float]] = {}
        for name, meta in metas:
            labels_path = self.root / name / "manual_labels.csv"
            labeled = {(self._norm(a), self._norm(b)) for a, b, lab in self._load_labels(labels_path) if lab == "cheated"}
            lookup = self._pair_lookup(meta)
            for a, b in list(labeled):
                if a == b:
                    continue
                key = tuple(sorted((a, b)))
                seq, jac = lookup.get(key, (0.0, 0.0))
                rec = agg.setdefault(key, {"count": 0.0, "max_jaccard": 0.0, "max_sequence": 0.0})
                rec["count"] += 1.0
                rec["max_jaccard"] = max(rec["max_jaccard"], jac)
                rec["max_sequence"] = max(rec["max_sequence"], seq)

        min_count = float(self.net_min_count.value())
        # Build network solely from manual labels; do not filter by similarity thresholds
        jac_thr = float(self.net_jac_thr.value())
        seq_thr = float(self.net_seq_thr.value())
        metric = self.net_metric.currentText()

        G = nx.Graph()
        for (a, b), rec in agg.items():
            if rec["count"] < min_count:
                continue
            G.add_node(a); G.add_node(b)
            w = rec[metric]
            G.add_edge(a, b, **rec, weight=max(1e-6, w))

        if G.number_of_edges() == 0:
            self.net_web.setHtml("<html><body style='font-family:system-ui;color:#333'><h3>No edges with current filters</h3></body></html>")
            return

        # communities
        try:
            comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="count"))
            cmap = {}
            for i, group in enumerate(comms):
                for n in group:
                    cmap[n] = i
        except Exception:
            cmap = {n: 0 for n in G.nodes()}

        # layout: separate communities, then layout within each community and offset
        communities: Dict[int, List[str]] = {}
        for n, cid in cmap.items():
            communities.setdefault(cid, []).append(n)
        num_c = max(1, len(communities))
        # Radius grows slightly with largest community to reduce cross-community overlap
        largest = max((len(v) for v in communities.values()), default=1)
        radius = 5.0 + 0.6 * math.log1p(largest)
        centers: Dict[int, Tuple[float, float]] = {}
        for i in range(num_c):
            ang = 2 * math.pi * (i / num_c)
            centers[i] = (radius * math.cos(ang), radius * math.sin(ang))

        pos: Dict[str, Tuple[float, float]] = {}
        rng = random.Random(42)
        for cid, nodes in communities.items():
            H = G.subgraph(nodes)
            n_h = max(1, len(H.nodes()))
            # More nodes -> larger k (more repulsion).
            k_in = 1.2 + 0.45 * math.log1p(n_h)
            sub = nx.spring_layout(H, weight="weight", k=k_in, seed=42, iterations=800)
            cx, cy = centers.get(cid, (0.0, 0.0))
            # Scale area based on community size; big groups get more area, small stay tighter
            scale = min(4.0, 0.9 + 0.28 * math.log1p(n_h))
            # Jitter inversely with size (prevents stacking while keeping large groups tidy)
            jitter = 0.12 + 0.22 / (1.0 + math.log1p(n_h))
            for n in nodes:
                x, y = sub.get(n, (0.0, 0.0))
                jx = (rng.random() - 0.5) * jitter
                jy = (rng.random() - 0.5) * jitter
                pos[n] = (cx + scale * x + jx, cy + scale * y + jy)
        palette = [(random.random(), random.random(), random.random()) for _ in range(max(1, max(cmap.values()) + 1))]

        fig = go.Figure()
        # edges as individual traces to control opacity/width
        edge_ann = []
        for (u, v) in G.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            c = int(G[u][v]["count"])
            alpha = 0.2 + 0.6 * float(G[u][v]["max_jaccard"])  # visual only
            width = 1 + 2 * c
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                     line=dict(color=f"rgba(68,68,68,{alpha:.2f})", width=width),
                                     hoverinfo="skip", showlegend=False))
            # midpoint label with co-flag count, offset slightly off the edge
            mx = (x0 + x1) / 2.0
            my = (y0 + y1) / 2.0
            dx = x1 - x0; dy = y1 - y0
            L = (dx*dx + dy*dy) ** 0.5
            off = 0.035
            ox = (-dy / L) * off if L > 1e-6 else 0.0
            oy = ( dx / L) * off if L > 1e-6 else 0.0
            edge_ann.append(dict(x=mx+ox, y=my+oy, xref='x', yref='y', text=str(c),
                                  showarrow=False, font=dict(size=10, color='#111'),
                                  bgcolor='rgba(255,255,255,0.85)', bordercolor='#666',
                                  borderwidth=0.5, borderpad=2, align='center'))

        node_x = [] ; node_y = [] ; texts = [] ; sizes = [] ; colors = [] ; hovers = []
        # Use a qualitative palette suited for light backgrounds
        qual_palette = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1
        for n in G.nodes():
            x, y = pos[n]
            node_x.append(x); node_y.append(y)
            label = n[:10]
            texts.append(label)
            deg = G.degree(n)
            # Node size grows sublinearly with degree and must fit text, kept very compact
            size_deg = 16 + 10 * math.sqrt(deg + 1)
            size_text = 5.0 * len(label) + 10  # approx px needed for label inside bubble
            sizes.append(min(60, max(size_deg, size_text)))
            hex_color = qual_palette[cmap[n] % len(qual_palette)]
            # Convert hex to rgba for consistent opacity
            if hex_color.startswith('#') and len(hex_color) == 7:
                r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
                colors.append(f"rgba({r},{g},{b},1)")
            else:
                colors.append(hex_color)
            hovers.append(f"{n}<br>degree={deg}")

        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=texts,
                                 textposition="middle center", textfont=dict(size=8, color="white"),
                                 hovertext=hovers, hoverinfo="text",
                                 marker=dict(size=sizes, color=colors, line=dict(color="#222", width=1)),
                                 showlegend=False))

        fig.update_layout(template="plotly_white",
                          title=f"Cheater network — {cat if cat != 'all' else 'All categories'} (nodes={G.number_of_nodes()}, edges={G.number_of_edges()})",
                          xaxis=dict(showgrid=False, zeroline=False, visible=False),
                          yaxis=dict(showgrid=False, zeroline=False, visible=False),
                          margin=dict(l=10, r=10, t=40, b=10),
                          dragmode="pan",
                          annotations=edge_ann)

        self.net_web.setHtml(fig.to_html(include_plotlyjs="cdn", full_html=False))

    def draw_heatmap(self) -> None:
        assign = self.heat_assign.currentText()
        metric = self.metric_combo.currentText()
        meta = self.assignments.get(assign, {})
        pairs = meta.get("results", [])
        # Build list of names and index
        names = sorted({self.extract_student(p["a"]) for p in pairs} | {self.extract_student(p["b"]) for p in pairs})
        idx = {n: i for i, n in enumerate(names)}
        import numpy as np
        mat = np.zeros((len(names), len(names)))
        for p in pairs:
            i = idx[self.extract_student(p["a"])]; j = idx[self.extract_student(p["b"])]
            val = float(p["jac"]) if metric == "jaccard" else float(p["seq"])  # type: ignore[index]
            mat[i, j] = mat[j, i] = val
        # Reset figure to avoid cumulative layout shifts when replacing colorbar
        fig = self.heat_canvas.fig
        fig.clf()
        ax = fig.add_subplot(111)
        self.heat_canvas.ax = ax
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{assign} — {metric.title()} similarity")
        ax.set_xticks(range(len(names)))
        # Dynamic tick font size based on number of students
        nlabels = len(names)
        tick_fs = max(5, min(12, int(12 - (nlabels / 15))))
        ax.set_xticklabels([n[:20] for n in names], rotation=90, fontsize=tick_fs)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n[:20] for n in names], fontsize=tick_fs)
        # Replace previous colorbar (if any) to avoid accumulation
        # Remove previous colorbar if any, then create a new one on the fresh figure
        self.heat_cbar = None
        self.heat_cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self.heat_canvas.draw()

    def draw_trends(self) -> None:
        # Filter assignments by selected category (normalized) and checklist selection
        sel_cat = (self.category_combo.currentText() or "All").strip().lower()
        sel_assign = self.get_selected_assignments()
        metas = [(name, m) for name, m in self.assignments.items()
                 if (sel_cat == "all" or (m.get("category", "").strip().lower() == sel_cat))
                 and ((not sel_assign) or (name in sel_assign))]

        def numeric_suffix_key(name: str):
            # Extract trailing digits to determine ordering (e.g., HW1, HW_12)
            m = re.search(r"(\d+)\D*$", name)
            if m:
                return (0, int(m.group(1)))
            return (1, name.lower())

        if sel_cat == "all":
            metas.sort(key=lambda x: x[0])
        else:
            metas.sort(key=lambda x: numeric_suffix_key(x[0]))
        xs = [name for name, _ in metas]
        axj = self.trend_canvas_j.ax ; axs = self.trend_canvas_s.ax
        axj.clear(); axs.clear()

        def compute_series(student: str):
            mj = [] ; ms = [] ; ch = []
            for name, meta in metas:
                pairs = meta.get("results", [])
                sj = 0.0 ; ss = 0.0 ; flagged = False
                labels_path = self.root / name / "manual_labels.csv"
                if labels_path.exists():
                    try:
                        lines = labels_path.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]
                        for line in lines:
                            a, b, _s, _j, lab = line.split(",", 4)
                            if lab.strip() == "cheated" and (self.extract_student(a) == student or self.extract_student(b) == student):
                                flagged = True
                    except Exception:
                        pass
                for p in pairs:
                    a = self.extract_student(p["a"]) ; b = self.extract_student(p["b"]) ; j = float(p["jac"]) ; s = float(p["seq"])  # type: ignore[index]
                    if a == student or b == student:
                        sj = max(sj, j)
                        ss = max(ss, s)
                mj.append(sj); ms.append(ss); ch.append(flagged)
            return mj, ms, ch

        if self.all_students_chk.isChecked():
            # overlay all students
            all_names = sorted({self.extract_student(p["a"]) for _, m in metas for p in m.get("results", [])} | {self.extract_student(p["b"]) for _, m in metas for p in m.get("results", [])})
            plotted = 0
            for name in all_names:
                mj, ms, ch = compute_series(name)
                if len(xs) == 1:
                    # single assignment – draw visible dots
                    axj.scatter([0], [mj[0] if mj else 0.0], color="#1f77b4", alpha=0.30, s=10)
                    axs.scatter([0], [ms[0] if ms else 0.0], color="#ff7f0e", alpha=0.30, s=10)
                    # light cheating marker if applicable
                    if ch and ch[0]:
                        axj.plot(0, mj[0], marker="x", color="red", alpha=0.35, markersize=7)
                        axs.plot(0, ms[0], marker="x", color="red", alpha=0.35, markersize=7)
                else:
                    axj.plot(xs, mj, color="#1f77b4", alpha=0.20, linewidth=1, marker=".", markersize=2)
                    axs.plot(xs, ms, color="#ff7f0e", alpha=0.20, linewidth=1, marker=".", markersize=2)
                    # light cheating markers on top
                    for i, flag in enumerate(ch):
                        if flag:
                            if i < len(mj):
                                axj.plot(i, mj[i], marker="x", color="red", alpha=0.25, markersize=6)
                            if i < len(ms):
                                axs.plot(i, ms[i], marker="x", color="red", alpha=0.25, markersize=6)
                plotted += 1
            # highlight selected student if any
            if self.student_combo.count() > 0:
                sel = self.student_combo.currentText()
                mj, ms, ch = compute_series(sel)
                axj.plot(xs, mj, color="#1f77b4", marker="o", linewidth=2, label=f"{sel}")
                axs.plot(xs, ms, color="#ff7f0e", marker="o", linewidth=2, label=f"{sel}")
                for i, flag in enumerate(ch):
                    if flag:
                        axj.plot(i, mj[i], marker="x", color="red", markersize=9)
                        axs.plot(i, ms[i], marker="x", color="red", markersize=9)
            axj.set_title(f"Max Jaccard — All students (n={plotted})")
            axs.set_title(f"Max Sequence — All students (n={plotted})")
        else:
            student = self.student_combo.currentText()
            mj, ms, ch = compute_series(student)
            axj.plot(xs, mj, marker="o", label=student)
            axs.plot(xs, ms, marker="o", label=student)
            for i, flag in enumerate(ch):
                if flag:
                    axj.plot(i, mj[i], marker="x", color="red", markersize=10)
                    axs.plot(i, ms[i], marker="x", color="red", markersize=10)
            axj.set_title("Max Jaccard")
            axs.set_title("Max Sequence")

        axj.set_ylim(0, 1); axs.set_ylim(0, 1)
        axj.set_ylabel("Max Jaccard"); axs.set_ylabel("Max Sequence")
        tick_fs = max(7, min(11, int(12 - (len(xs)/10))))
        axj.set_xticks(range(len(xs))); axj.set_xticklabels(xs, rotation=45, ha="right", fontsize=tick_fs)
        axs.set_xticks(range(len(xs))); axs.set_xticklabels(xs, rotation=45, ha="right", fontsize=tick_fs)
        axj.grid(True, alpha=0.3); axs.grid(True, alpha=0.3)
        # Optional small legend when highlighting selection
        try:
            axj.legend(loc="upper left", fontsize=8)
            axs.legend(loc="upper left", fontsize=8)
        except Exception:
            pass
        self.trend_canvas_j.draw(); self.trend_canvas_s.draw()

    # --- Report generation ---
    def generate_report(self) -> None:
        # Determine selection
        sel_cat = (self.rep_category.currentText() or "All").strip()
        selected = self.get_selected_report_assignments()
        metas = {name: self.assignments[name] for name in self.assignments.keys()
                 if (sel_cat == "All" or (self.assignments[name].get("category", "") or "Uncategorized").strip() == sel_cat)
                 and ((not selected) or (name in selected))}
        if not metas:
            QtWidgets.QMessageBox.information(self, "No assignments", "No assignments selected for the report.")
            return

        # Build per-student and per-assignment cheating data from manual_labels.csv
        per_student: Dict[str, List[Tuple[str, str, float, float]]] = {}
        per_assignment_pairs: Dict[str, List[Tuple[str, str, float, float]]] = {}
        name_maps: Dict[str, Dict[str, Path]] = {}
        for assign_name, meta in metas.items():
            # map pair -> (seq, jac)
            lookup = self._pair_lookup(meta)
            labels_path = self.root / assign_name / "manual_labels.csv"
            for a, b, lab in self._load_labels(labels_path):
                if lab != "cheated":
                    continue
                na = self.extract_student(a); nb = self.extract_student(b)
                seq, jac = lookup.get(tuple(sorted((na, nb))), (0.0, 0.0))
                per_student.setdefault(na, []).append((assign_name, nb, seq, jac))
                per_student.setdefault(nb, []).append((assign_name, na, seq, jac))
                per_assignment_pairs.setdefault(assign_name, []).append((na, nb, seq, jac))
            # keep path mapping for possible future links
            name_maps[assign_name] = {k: Path(v) for k, v in meta.get("name_to_path", {}).items()}

        # Build HTML content
        html_parts: List[str] = []
        html_parts.append("<html><head><meta charset='utf-8'><title>Cheat or No Cheat Report</title>"
                          "<style>body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;}"
                          "h1,h2{margin-top:24px;} details{margin:8px 0;} summary{font-weight:600;cursor:pointer;}"
                          "code,badge{font-family:monospace;} table{border-collapse:collapse;} td,th{border:1px solid #ddd;padding:4px 8px;}"
                          "</style></head><body>")
        html_parts.append("<h1>Cheat or No Cheat Report</h1>")
        html_parts.append("<h3>Table of contents</h3><ul>"
                          "<li><a href='#by-student'>By student (cheaters)</a></li>"
                          "<li><a href='#by-assignment'>By assignment</a></li>"
                          "<li><a href='#network'>Network</a></li></ul>")

        # By Student
        html_parts.append("<h2 id='by-student'>By student (cheaters)</h2>")
        for student in sorted(per_student.keys()):
            html_parts.append(f"<details><summary>{student}</summary>")
            html_parts.append("<ul>")
            for (assign_name, partner, seq, jac) in sorted(per_student[student], key=lambda x: x[0]):
                html_parts.append(f"<li><b>{assign_name}</b>: with {partner} (seq={seq:.3f}, jaccard={jac:.3f})</li>")
            html_parts.append("</ul></details>")

        # By Assignment with stats and heatmap
        html_parts.append("<h2 id='by-assignment'>By assignment</h2>")
        for assign_name in sorted(metas.keys()):
            meta = metas[assign_name]
            pairs = per_assignment_pairs.get(assign_name, [])
            j_list = [p[3] for p in pairs]
            s_list = [p[2] for p in pairs]
            def safe_stats(vals: List[float]):
                if not vals:
                    return (0.0, 0.0, 0.0)
                return (statistics.mean(vals), statistics.median(vals), statistics.pstdev(vals) if len(vals)>1 else 0.0)
            j_mean, j_med, j_std = safe_stats(j_list)
            s_mean, s_med, s_std = safe_stats(s_list)

            # Heatmap image (overall similarity across all students for this assignment)
            hm_img = self._render_heatmap_image(meta)
            cheaters = sorted({x for tup in pairs for x in (tup[0], tup[1])})
            html_parts.append(f"<details><summary>{assign_name}</summary>")
            html_parts.append("<p><b>Cheaters:</b> " + (", ".join(cheaters) if cheaters else "(none)") + "</p>")
            html_parts.append(f"<p><b>Jaccard</b> mean={j_mean:.3f}, median={j_med:.3f}, std={j_std:.3f} &nbsp; | &nbsp; "
                              f"<b>Sequence</b> mean={s_mean:.3f}, median={s_med:.3f}, std={s_std:.3f}</p>")
            if hm_img:
                html_parts.append(f"<img alt='heatmap {assign_name}' src='data:image/png;base64,{hm_img}' style='max-width:100%;height:auto;border:1px solid #ddd' />")
            html_parts.append("</details>")

        # Network section (reuse current selection for category/assignments)
        html_parts.append("<h2 id='network'>Network</h2>")
        net_html = self._render_network_html_for(metas)
        html_parts.append(net_html)

        html_parts.append("</body></html>")

        # Save
        default_path = str((self.root / "report.html").resolve())
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save report", default_path, "HTML Files (*.html);;All Files (*)")
        if not out_path:
            return
        Path(out_path).write_text("\n".join(html_parts), encoding="utf-8")
        try:
            webbrowser.open(f"file://{out_path}")
        except Exception:
            pass

    def _render_heatmap_image(self, meta: dict) -> str:
        try:
            pairs = meta.get("results", [])
            names = sorted({self.extract_student(p["a"]) for p in pairs} | {self.extract_student(p["b"]) for p in pairs})
            idx = {n: i for i, n in enumerate(names)}
            mat = np.zeros((len(names), len(names)))
            for p in pairs:
                i = idx[self.extract_student(p["a"])]; j = idx[self.extract_student(p["b"])];
                mat[i, j] = mat[j, i] = float(p["jac"])  # use Jaccard for heatmap context
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
            im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            bio = io.BytesIO()
            fig.tight_layout()
            fig.savefig(bio, format="png")
            plt.close(fig)
            return base64.b64encode(bio.getvalue()).decode("ascii")
        except Exception:
            return ""

    def _render_network_html_for(self, metas: Dict[str, dict]) -> str:
        # Build graph from selected metas (manual labels only)
        agg: Dict[Tuple[str, str], Dict[str, float]] = {}
        for assign_name, meta in metas.items():
            lookup = self._pair_lookup(meta)
            labels_path = self.root / assign_name / "manual_labels.csv"
            for a, b, lab in self._load_labels(labels_path):
                if lab != "cheated":
                    continue
                na = self.extract_student(a); nb = self.extract_student(b)
                seq, jac = lookup.get(tuple(sorted((na, nb))), (0.0, 0.0))
                rec = agg.setdefault(tuple(sorted((na, nb))), {"count": 0.0, "max_jaccard": 0.0, "max_sequence": 0.0})
                rec["count"] += 1.0
                rec["max_jaccard"] = max(rec["max_jaccard"], jac)
                rec["max_sequence"] = max(rec["max_sequence"], seq)
        G = nx.Graph()
        for (a, b), rec in agg.items():
            G.add_node(a); G.add_node(b); G.add_edge(a, b, **rec, weight=max(1e-6, rec["count"]))
        if G.number_of_edges() == 0:
            return "<p>(No edges for selected assignments)</p>"
        try:
            comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="count"))
            cmap = {}
            for i, group in enumerate(comms):
                for n in group: cmap[n] = i
        except Exception:
            cmap = {n:0 for n in G.nodes()}
        # layout
        communities: Dict[int, List[str]] = {}
        for n, cid in cmap.items(): communities.setdefault(cid, []).append(n)
        num_c = max(1, len(communities))
        radius = 5.0
        centers = {i:(radius*math.cos(2*math.pi*i/num_c), radius*math.sin(2*math.pi*i/num_c)) for i in range(num_c)}
        pos = {}
        rng = random.Random(42)
        for cid, nodes in communities.items():
            H = G.subgraph(nodes)
            k_in = 1.2 + 0.4*math.log1p(len(nodes))
            sub = nx.spring_layout(H, weight="weight", k=k_in, seed=42, iterations=600)
            cx, cy = centers[cid]
            for n in nodes:
                x,y=sub.get(n,(0.0,0.0)); jx=(rng.random()-0.5)*0.2; jy=(rng.random()-0.5)*0.2
                pos[n]=(cx+1.2*x+jx, cy+1.2*y+jy)
        # plotly fig
        fig = go.Figure()
        for (u,v) in G.edges():
            x0,y0=pos[u]; x1,y1=pos[v]
            c=int(G[u][v]["count"]); width=1+2*c; alpha=0.2+0.6*float(G[u][v]["max_jaccard"]) 
            fig.add_trace(go.Scatter(x=[x0,x1],y=[y0,y1],mode="lines",line=dict(color=f"rgba(68,68,68,{alpha:.2f})",width=width),hoverinfo="skip",showlegend=False))
        qual_palette = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1
        node_x=[]; node_y=[]; texts=[]; sizes=[]; colors=[]
        for n in G.nodes():
            x,y=pos[n]; node_x.append(x); node_y.append(y); label=n[:10]; texts.append(label)
            deg=G.degree(n); size=min(60, max(16+10*math.sqrt(deg+1), 5*len(label)+10))
            sizes.append(size); hex_color=qual_palette[cmap[n]%len(qual_palette)]
            if hex_color.startswith('#') and len(hex_color)==7:
                r=int(hex_color[1:3],16); g=int(hex_color[3:5],16); b=int(hex_color[5:7],16)
                colors.append(f"rgba({r},{g},{b},1)")
            else:
                colors.append(hex_color)
        fig.add_trace(go.Scatter(x=node_x,y=node_y,mode="markers+text",text=texts,textposition="middle center",textfont=dict(size=8,color="white"),marker=dict(size=sizes,color=colors,line=dict(color="#222",width=1)),showlegend=False))
        return fig.to_html(include_plotlyjs="cdn", full_html=False)

class AnalysisWorker(QtCore.QThread):
    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, directory: Path, seq: float, jac: float, topn: int) -> None:
        super().__init__()
        self.directory = directory
        self.seq = seq
        self.jac = jac
        self.topn = topn

    def run(self) -> None:
        try:
            result = analyze_directory(self.directory, self.seq, self.jac, self.topn)
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Cheat or No Cheat - Reviewer")
        self.resize(1280, 840)

        self.directory_edit = QtWidgets.QLineEdit(str(Path.cwd()))
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self.on_browse)

        self.seq_spin = QtWidgets.QDoubleSpinBox()
        self.seq_spin.setRange(0.0, 1.0)
        self.seq_spin.setSingleStep(0.01)
        self.seq_spin.setValue(0.82)

        self.jac_spin = QtWidgets.QDoubleSpinBox()
        self.jac_spin.setRange(0.0, 1.0)
        self.jac_spin.setSingleStep(0.01)
        self.jac_spin.setValue(0.70)

        self.topn_spin = QtWidgets.QSpinBox()
        self.topn_spin.setRange(1, 1000)
        self.topn_spin.setValue(50)

        analyze_btn = QtWidgets.QPushButton("Analyze")
        analyze_btn.clicked.connect(self.on_analyze)
        self.save_btn = QtWidgets.QPushButton("Save Assignment…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.on_save_assignment)
        self.stats_btn = QtWidgets.QPushButton("Statistics…")
        self.stats_btn.setEnabled(False)
        self.stats_btn.clicked.connect(self.on_open_statistics)

        toolbar = QtWidgets.QWidget()
        tlayout = QtWidgets.QHBoxLayout(toolbar)
        tlayout.setContentsMargins(8, 8, 8, 8)
        tlayout.addWidget(QtWidgets.QLabel("Directory:"))
        tlayout.addWidget(self.directory_edit, 1)
        tlayout.addWidget(browse_btn)
        tlayout.addSpacing(16)
        tlayout.addWidget(QtWidgets.QLabel("Seq:"))
        tlayout.addWidget(self.seq_spin)
        tlayout.addWidget(QtWidgets.QLabel("Jac:"))
        tlayout.addWidget(self.jac_spin)
        tlayout.addWidget(QtWidgets.QLabel("Top N:"))
        tlayout.addWidget(self.topn_spin)
        tlayout.addSpacing(16)
        tlayout.addWidget(analyze_btn)
        tlayout.addWidget(self.save_btn)
        tlayout.addWidget(self.stats_btn)

        # Splitter for side-by-side HTML views (build containers first)
        splitter = QtWidgets.QSplitter()

        # Left side
        left_container = QtWidgets.QWidget()
        left_box = QtWidgets.QVBoxLayout(left_container)
        left_box.setContentsMargins(0, 0, 0, 0)
        self.left_label = QtWidgets.QLabel("")
        self.left_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.left_view = QWebEngineView(left_container)
        left_box.addWidget(self.left_label)
        left_box.addWidget(self.left_view, 1)
        self.left_view.setHtml("<html><body style='font-family:system-ui;color:#333'>Select a pair to preview here.</body></html>")

        # Right side
        right_container = QtWidgets.QWidget()
        right_box = QtWidgets.QVBoxLayout(right_container)
        right_box.setContentsMargins(0, 0, 0, 0)
        self.right_label = QtWidgets.QLabel("")
        self.right_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.right_view = QWebEngineView(right_container)
        right_box.addWidget(self.right_label)
        right_box.addWidget(self.right_view, 1)
        self.right_view.setHtml("<html><body style='font-family:system-ui;color:#333'>Select a pair to preview here.</body></html>")

        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        # Controls for which pairs to display in the table
        table_ctl = QtWidgets.QWidget()
        tcl = QtWidgets.QHBoxLayout(table_ctl)
        tcl.setContentsMargins(8, 0, 8, 0)
        tcl.addWidget(QtWidgets.QLabel("Table view:"))
        self.table_view_mode = QtWidgets.QComboBox()
        self.table_view_mode.addItems(["Flagged only", "Top N by similarity"])  # Top N uses the value from Analyze
        # Default to Top N to align with user expectation when setting N
        self.table_view_mode.setCurrentIndex(1)
        self.table_view_mode.currentIndexChanged.connect(self.rebuild_pairs_table)
        tcl.addWidget(self.table_view_mode)
        tcl.addStretch(1)

        # Bottom table for pairs
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["A", "B", "Seq", "Jac", "Label"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        # Use itemSelectionChanged for a stable signal; selectionModel can be recreated
        self.table.itemSelectionChanged.connect(self.on_row_changed)

        # Buttons for voting and navigation
        self.progress = QtWidgets.QLabel("")
        cheated_btn = QtWidgets.QPushButton("Cheated")
        cheated_btn.setStyleSheet("background:#c62828;color:white")
        cheated_btn.clicked.connect(lambda: self.vote("cheated"))
        no_btn = QtWidgets.QPushButton("No Cheating")
        no_btn.setStyleSheet("background:#2e7d32;color:white")
        no_btn.clicked.connect(lambda: self.vote("no_cheating"))
        prev_btn = QtWidgets.QPushButton("Prev")
        next_btn = QtWidgets.QPushButton("Next")
        prev_btn.clicked.connect(lambda: self.move(-1))
        next_btn.clicked.connect(lambda: self.move(1))

        action_bar = QtWidgets.QWidget()
        ab_layout = QtWidgets.QHBoxLayout(action_bar)
        ab_layout.setContentsMargins(8, 4, 8, 4)
        ab_layout.addWidget(self.progress)
        ab_layout.addStretch(1)
        ab_layout.addWidget(prev_btn)
        ab_layout.addWidget(next_btn)
        ab_layout.addSpacing(12)
        ab_layout.addWidget(cheated_btn)
        ab_layout.addWidget(no_btn)

        # Central layout
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(toolbar)
        layout.addWidget(splitter, 2)
        layout.addWidget(table_ctl)
        layout.addWidget(self.table, 1)
        layout.addWidget(action_bar)
        self.setCentralWidget(central)

        self.result: dict = {}
        self.labels_path: Path | None = None
        self.labels: Dict[str, str] = {}
        self.assignment_root: Path | None = None  # where assignments will be stored

    # --- Helpers ---
    def on_browse(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select submissions directory", self.directory_edit.text())
        if d:
            self.directory_edit.setText(d)

    def on_analyze(self) -> None:
        directory = Path(self.directory_edit.text()).expanduser().resolve()
        if not directory.is_dir():
            QtWidgets.QMessageBox.warning(self, "Invalid directory", f"{directory} is not a directory")
            return
        self.statusBar().showMessage("Analyzing…")
        worker = AnalysisWorker(directory, self.seq_spin.value(), self.jac_spin.value(), int(self.topn_spin.value()))
        worker.finished.connect(self.on_analysis_done)
        worker.failed.connect(lambda e: QtWidgets.QMessageBox.critical(self, "Error", e))
        worker.finished.connect(lambda _: self.statusBar().clearMessage())
        worker.start()
        self._worker = worker  # keep ref

    def on_analysis_done(self, result: dict) -> None:
        self.result = result
        current_dir = Path(self.directory_edit.text()).expanduser().resolve()
        self.labels_path = current_dir / "manual_labels.csv"
        self.labels = self.load_labels(self.labels_path)
        # Enable saving and statistics
        # Store under the parent of the selected directory so runs from sibling
        # assignment folders (e.g., HW1, HW2) share a common history.
        # Migrate legacy storage folder if encountered
        new_root_base = current_dir.parent / ".cheatornocheat"
        old_root_base = current_dir.parent / ".youcheated"
        try:
            if (not new_root_base.exists()) and old_root_base.exists():
                shutil.move(str(old_root_base), str(new_root_base))
        except Exception:
            pass
        self.assignment_root = new_root_base / "assignments"
        self.assignment_root.mkdir(parents=True, exist_ok=True)
        self.save_btn.setEnabled(True)
        self.stats_btn.setEnabled(True)

        # Build table based on selected view mode
        self.rebuild_pairs_table()
        self.update_progress()

    # --- Save Assignment & Statistics ---
    def on_save_assignment(self) -> None:
        if not self.result:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Save Assignment")
        form = QtWidgets.QFormLayout(dialog)
        name_edit = QtWidgets.QLineEdit()
        category_edit = QtWidgets.QLineEdit()
        form.addRow("Assignment name:", name_edit)
        form.addRow("Category (e.g., Homework/Lab):", category_edit)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        form.addRow(btns)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        name = name_edit.text().strip()
        category = category_edit.text().strip() or "Uncategorized"
        if not name:
            QtWidgets.QMessageBox.warning(self, "Missing name", "Please provide an assignment name.")
            return
        assert self.assignment_root is not None
        out_dir = self.assignment_root / name
        if out_dir.exists():
            overwrite = QtWidgets.QMessageBox.question(self, "Overwrite?", f"Assignment '{name}' exists. Overwrite?")
            if overwrite != QtWidgets.QMessageBox.Yes:
                return
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Serialize minimal result
        serial = {
            "created_at": datetime.utcnow().isoformat(),
            "category": category,
            "directory": str(Path(self.directory_edit.text()).expanduser().resolve()),
            "seq_threshold": float(self.seq_spin.value()),
            "jaccard_threshold": float(self.jac_spin.value()),
            "results": [
                {"a": a, "b": b, "seq": float(ss), "jac": float(js)}
                for (a, b, ss, js) in self.result.get("results", [])
            ],
            "name_to_path": {k: str(v) for k, v in self.result.get("name_to_path", {}).items()},
        }
        (out_dir / "results.json").write_text(json.dumps(serial, indent=2), encoding="utf-8")

        # Copy CSVs if present
        for fn in ("suspicious_pairs.csv", "suspicious_pairs.txt", "suspicious_pairs.json", "manual_labels.csv"):
            src = Path(self.directory_edit.text()).expanduser().resolve() / fn
            if src.exists():
                shutil.copy2(src, out_dir / fn)

        QtWidgets.QMessageBox.information(self, "Saved", f"Assignment saved to\n{out_dir}")

    def on_open_statistics(self) -> None:
        assert self.assignment_root is not None
        win = StatsWindow(self.assignment_root)
        win.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        win.show()
        # Keep reference
        self._stats_win = win

    def rebuild_pairs_table(self) -> None:
        # Decide which list to show based on combo selection
        if not self.result:
            return
        view = self.table_view_mode.currentText() if hasattr(self, 'table_view_mode') else "Flagged only"
        # Always derive from flagged pairs; sort by similarity and cap to Top N
        flagged = list(self.result.get("flagged_pairs", []))
        # Sort by (seq+jac)/2 descending for consistent ordering
        flagged.sort(key=lambda x: (x[2] + x[3]) / 2, reverse=True)
        # Enforce Top N limit
        try:
            n = int(self.topn_spin.value())
            if n > 0:
                flagged = flagged[:n]
        except Exception:
            pass
        pairs: List[Tuple[str, str, float, float]] = flagged

        self.table.setRowCount(0)
        for (a, b, ss, js) in pairs:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(a))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(b))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{ss:.3f}"))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{js:.3f}"))
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(self.labels.get(self.key(a, b), "")))

        if pairs:
            self.table.selectRow(0)
            QtCore.QTimer.singleShot(0, self.on_row_changed)


    def on_row_changed(self, *_) -> None:
        idxs = self.table.selectionModel().selectedRows()
        if not idxs:
            return
        row = idxs[0].row()
        a = self.table.item(row, 0).text()
        b = self.table.item(row, 1).text()
        mapping: Dict[str, Path] = self.result.get("name_to_path", {})
        p1 = mapping.get(a)
        p2 = mapping.get(b)
        if p1:
            self.left_label.setText(a)
            self.left_view.setUrl(QtCore.QUrl.fromLocalFile(str(p1)))
        if p2:
            self.right_label.setText(b)
            self.right_view.setUrl(QtCore.QUrl.fromLocalFile(str(p2)))
        self.update_progress()

    def move(self, delta: int) -> None:
        row = 0
        idxs = self.table.selectionModel().selectedRows()
        if idxs:
            row = idxs[0].row()
        row = max(0, min(self.table.rowCount() - 1, row + delta))
        self.table.selectRow(row)

    def vote(self, label: str) -> None:
        idxs = self.table.selectionModel().selectedRows()
        if not idxs or self.labels_path is None:
            return
        row = idxs[0].row()
        a = self.table.item(row, 0).text()
        b = self.table.item(row, 1).text()
        ss = float(self.table.item(row, 2).text())
        js = float(self.table.item(row, 3).text())
        self.append_label(self.labels_path, a, b, ss, js, label)
        self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(label))
        self.labels[self.key(a, b)] = label
        self.move(1)

    def update_progress(self) -> None:
        total = self.table.rowCount()
        row = 0
        idxs = self.table.selectionModel().selectedRows()
        if idxs:
            row = idxs[0].row()
        self.progress.setText(f"Pair {row+1 if total else 0}/{total}")

    # --- labels persistence ---
    @staticmethod
    def key(a: str, b: str) -> str:
        return "||".join(sorted([a, b]))

    @staticmethod
    def load_labels(path: Path) -> Dict[str, str]:
        if not path.exists():
            return {}
        out: Dict[str, str] = {}
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines[1:]:
                if not line.strip():
                    continue
                a, b, _ss, _js, lab = line.split(",", 4)
                out[MainWindow.key(a, b)] = lab.strip()
        except Exception:
            pass
        return out

    @staticmethod
    def append_label(path: Path, a: str, b: str, ss: float, js: float, label: str) -> None:
        try:
            header_needed = not path.exists()
            with path.open("a", encoding="utf-8") as f:
                if header_needed:
                    f.write("file_a,file_b,sequence_similarity,jaccard_similarity,label\n")
                f.write(f"{a},{b},{ss:.4f},{js:.4f},{label}\n")
        except Exception as e:
            QtWidgets.QMessageBox.warning(None, "Write failed", str(e))


def main() -> None:
    # Reduce QtWebEngine console noise and provide a valid dictionaries path outside the repo
    try:
        import tempfile
        if sys.platform == "darwin":
            dictionaries_dir = Path.home() / "Library" / "Caches" / "CheatOrNoCheat" / "qtwebengine_dictionaries"
        elif sys.platform.startswith("win"):
            dictionaries_dir = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "CheatOrNoCheat" / "qtwebengine_dictionaries"
        else:
            dictionaries_dir = Path(os.environ.get("XDG_CACHE_HOME", tempfile.gettempdir())) / "cheatornocheat" / "qtwebengine_dictionaries"
        dictionaries_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("QTWEBENGINE_DICTIONARIES_PATH", str(dictionaries_dir))
    except Exception:
        pass
    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-logging --log-level=3")
    # Only disable sandbox by default on Linux; avoid noise on macOS where it's often not required
    if sys.platform.startswith("linux"):
        os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


