#!/usr/bin/env python3
"""Desktop GUI for the pill bug light/shelter tracker.  Run:  python3 pillbug_gui.py"""
import base64
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

import pillbug_core as core

GEOM = [  # (attr, label, from, to, increment)
    ("cx", "Centre x (px)", 0, 3000, 1),
    ("cy", "Centre y (px)", 0, 3000, 1),
    ("r_in", "Centre-cup radius (px)", 5, 1500, 1),
    ("r_out", "Arena inner radius (px)", 5, 1500, 1),
    ("r_ab", "A/B boundary = roof edge (px)", 5, 1500, 1),
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pill bug tracker - light / shelter")
        self.geometry("1280x800")
        self.p = core.Params()
        self.bg = self.first = None
        self.q = queue.Queue()
        self.worker = None
        self.cancel_flag = False
        self.vars = {}
        self._build()
        self.after(100, self._poll)

    # ------------------------------------------------------------- layout
    def _build(self):
        left = ttk.Frame(self, padding=8)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=8)
        right.pack(side="right", fill="both", expand=True)

        def row(parent, text, var, browse=None, width=34):
            f = ttk.Frame(parent)
            f.pack(fill="x", pady=1)
            ttk.Label(f, text=text, width=22).pack(side="left")
            ttk.Entry(f, textvariable=var, width=width).pack(side="left", fill="x", expand=True)
            if browse:
                ttk.Button(f, text="...", width=3, command=browse).pack(side="left")

        files = ttk.LabelFrame(left, text="1. Files", padding=6)
        files.pack(fill="x", pady=4)
        self.video = tk.StringVar()
        self.empty = tk.StringVar()
        self.out = tk.StringVar()
        row(files, "Video", self.video, self._pick_video)
        row(files, "Empty arena (optional)", self.empty, self._pick_empty)
        row(files, "Output folder", self.out, self._pick_out)
        ttk.Label(files, text="Empty arena = a short image/video of the arena with no bugs.\n"
                              "Strongly recommended: a bug that sits still is otherwise lost.",
                  foreground="#555").pack(anchor="w")

        geo = ttk.LabelFrame(left, text="2. Arena geometry (check the preview)", padding=6)
        geo.pack(fill="x", pady=4)
        f = ttk.Frame(geo)
        f.pack(fill="x")
        ttk.Label(f, text="Chambers", width=22).pack(side="left")
        self.vars["n_sectors"] = tk.IntVar(value=self.p.n_sectors)
        ttk.Spinbox(f, from_=2, to=16, textvariable=self.vars["n_sectors"], width=6,
                    command=self._redraw).pack(side="left")
        for attr, label, lo, hi, inc in GEOM:
            f = ttk.Frame(geo)
            f.pack(fill="x", pady=1)
            ttk.Label(f, text=label, width=22).pack(side="left")
            v = tk.DoubleVar(value=getattr(self.p, attr) or 0)
            self.vars[attr] = v
            ttk.Spinbox(f, from_=lo, to=hi, increment=inc, textvariable=v, width=8,
                        command=self._redraw).pack(side="left")
            if attr == "r_ab":
                ttk.Button(f, text="= arena radius", command=self._equal_area).pack(side="left", padx=4)
        f = ttk.Frame(geo)
        f.pack(fill="x", pady=1)
        ttk.Label(f, text="Rotate chamber numbers", width=22).pack(side="left")
        self.vars["number_shift"] = tk.IntVar(value=0)
        ttk.Spinbox(f, from_=0, to=15, textvariable=self.vars["number_shift"], width=6,
                    command=self._redraw).pack(side="left")
        f = ttk.Frame(geo)
        f.pack(fill="x", pady=1)
        ttk.Label(f, text="Centre cup diameter (cm)", width=22).pack(side="left")
        self.vars["cup_diam_cm"] = tk.DoubleVar(value=self.p.cup_diam_cm)
        ttk.Spinbox(f, from_=1, to=200, textvariable=self.vars["cup_diam_cm"], width=8).pack(side="left")
        for w in geo.winfo_children():
            for c in w.winfo_children():
                if isinstance(c, ttk.Spinbox):
                    c.bind("<Return>", lambda e: self._redraw())
                    c.bind("<FocusOut>", lambda e: self._redraw())
        ttk.Button(geo, text="Reload preview", command=self._load_preview).pack(anchor="w", pady=3)

        opt = ttk.LabelFrame(left, text="3. Analysis settings", padding=6)
        opt.pack(fill="x", pady=4)
        for attr, label, default in [
            ("step", "Analyse every Nth frame", self.p.step),
            ("thresh", "Detection threshold", self.p.thresh),
            ("min_area", "Min blob area (px)", self.p.min_area),
            ("entry_zone_px", "Shelter entry zone (px)", self.p.entry_zone_px),
            ("min_hidden_s", "Min hidden time (s)", self.p.min_hidden_s),
            ("start_s", "Start at (s)", self.p.start_s),
            ("empty_first_s", "First N s are empty", self.p.empty_first_s),
            ("max_minutes", "Analyse only first N min", 0),
        ]:
            f = ttk.Frame(opt)
            f.pack(fill="x", pady=1)
            ttk.Label(f, text=label, width=22).pack(side="left")
            self.vars[attr] = tk.DoubleVar(value=default)
            ttk.Entry(f, textvariable=self.vars[attr], width=8).pack(side="left")
        self.vars["save_video"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Write annotated check video", variable=self.vars["save_video"]).pack(anchor="w")
        ttk.Label(opt, text="A bug that vanishes deep inside A (divider shadow) stays in A.\n"
                            "It counts as in B only if last seen within the entry zone of the\n"
                            "A/B boundary (or in B) and then disappears under the roof.",
                  foreground="#555").pack(anchor="w")

        run = ttk.Frame(left)
        run.pack(fill="x", pady=6)
        self.btn_run = ttk.Button(run, text="Run analysis", command=self._run)
        self.btn_run.pack(side="left")
        self.btn_cancel = ttk.Button(run, text="Cancel", command=self._cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=6)
        self.btn_open = ttk.Button(run, text="Open output folder", command=self._open_out)
        self.btn_open.pack(side="left")
        self.prog = ttk.Progressbar(left, mode="determinate", maximum=100)
        self.prog.pack(fill="x")

        self.canvas = tk.Label(right, text="Choose a video to see the arena preview", background="#222",
                               foreground="#ccc")
        self.canvas.pack(fill="both", expand=True)
        self.logbox = tk.Text(right, height=12, wrap="word")
        self.logbox.pack(fill="x", pady=(6, 0))

    # ------------------------------------------------------------- actions
    def _pick_video(self):
        f = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All", "*.*")])
        if f:
            self.video.set(f)
            if not self.out.get():
                self.out.set(os.path.splitext(f)[0] + "_tracking")
            self._load_preview()

    def _pick_empty(self):
        f = filedialog.askopenfilename(filetypes=[("Image or video", "*.png *.jpg *.jpeg *.mp4 *.mov *.avi *.mkv"),
                                                  ("All", "*.*")])
        if f:
            self.empty.set(f)

    def _pick_out(self):
        d = filedialog.askdirectory()
        if d:
            self.out.set(d)

    def _open_out(self):
        d = self.out.get()
        if d and os.path.isdir(d):
            os.system(f'open "{d}"' if sys.platform == "darwin" else f'xdg-open "{d}"')

    def log(self, msg):
        self.q.put(("log", msg))

    def _read_params(self):
        p = core.Params()
        g = lambda k: self.vars[k].get()
        p.n_sectors = int(g("n_sectors"))
        for attr in ("cx", "cy", "r_in", "r_out", "r_ab"):
            setattr(p, attr, float(g(attr)))
        p.number_shift = int(g("number_shift"))
        p.cup_diam_cm = float(g("cup_diam_cm"))
        p.step = max(1, int(g("step")))
        p.thresh = int(g("thresh"))
        p.min_area = int(g("min_area"))
        p.entry_zone_px = float(g("entry_zone_px"))
        p.min_hidden_s = float(g("min_hidden_s"))
        p.start_s = float(g("start_s"))
        p.empty_first_s = float(g("empty_first_s"))
        mm = float(g("max_minutes"))
        p.max_minutes = mm if mm > 0 else None
        p.save_video = bool(g("save_video"))
        p.empty_bg = self.empty.get().strip()
        return p

    def _equal_area(self):
        try:
            p = self._read_params()
            p.r_ab = None
            self.vars["r_ab"].set(round(p.boundary(), 1))
        except tk.TclError:
            pass
        self._redraw()

    def _load_preview(self):
        v = self.video.get().strip()
        if not os.path.isfile(v):
            return
        self.canvas.config(text="Loading preview...", image="")
        self.update_idletasks()
        try:
            self.bg, self.first = core.quick_background(v)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Preview", str(e))
            return
        h, w = self.bg.shape
        # first-time defaults: centre of the picture, radius ~ 1/3 of the height
        if self.vars["cx"].get() == core.Params().cx and self.vars["cy"].get() == core.Params().cy:
            pass
        self._equal_area()

    def _redraw(self, *_):
        if self.bg is None:
            return
        try:
            p = self._read_params()
            spokes = core.find_spoke_angles(self.bg, p)
            sec = core.build_sector_map(self.bg.shape, p, spokes)
            img = core.draw_overlay(self.first if self.first is not None else self.bg, p, spokes, sec)
        except (tk.TclError, ValueError, ZeroDivisionError):
            return
        scale = min(820 / img.shape[1], 560 / img.shape[0], 1.0)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        ok, buf = cv2.imencode(".png", img)
        self._photo = tk.PhotoImage(data=base64.b64encode(buf.tobytes()))
        self.canvas.config(image=self._photo, text="")

    def _run(self):
        v, o = self.video.get().strip(), self.out.get().strip()
        if not os.path.isfile(v):
            messagebox.showwarning("Video", "Choose a video file first.")
            return
        if not o:
            o = os.path.splitext(v)[0] + "_tracking"
            self.out.set(o)
        try:
            p = self._read_params()
        except (tk.TclError, ValueError) as e:
            messagebox.showwarning("Settings", f"Check the numeric fields: {e}")
            return
        self.cancel_flag = False
        self.btn_run.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.prog["value"] = 0
        self.logbox.delete("1.0", "end")

        def work():
            try:
                rows = core.run_analysis(v, p, o, progress=lambda x: self.q.put(("prog", x)),
                                         cancel=lambda: self.cancel_flag, log=self.log)
                self.q.put(("done", rows))
            except core.Cancelled:
                self.q.put(("log", "cancelled"))
                self.q.put(("done", None))
            except Exception as e:  # noqa: BLE001
                self.q.put(("log", f"ERROR: {e}"))
                self.q.put(("done", None))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _cancel(self):
        self.cancel_flag = True

    def _poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    self.logbox.insert("end", val + "\n")
                    self.logbox.see("end")
                elif kind == "prog":
                    self.prog["value"] = val * 100
                elif kind == "done":
                    self.btn_run.config(state="normal")
                    self.btn_cancel.config(state="disabled")
                    if val:
                        self.logbox.insert("end", "\nchamber  time_A_s  time_B_s  prop_A  latency_A_s  crossings\n")
                        for r in val:
                            self.logbox.insert("end", f"{r['chamber']:>7}  {r['time_A_s']:>8}  {r['time_B_s']:>8}  "
                                                      f"{r['prop_A']:>6}  {r['latency_A_s']:>11}  {r['n_crossings']:>9}\n")
                        self.logbox.see("end")
        except queue.Empty:
            pass
        self.after(100, self._poll)


def main():
    app = App()
    if "--selftest" in sys.argv:
        app.update()
        app.destroy()
        print("gui ok")
        return
    app.mainloop()


if __name__ == "__main__":
    main()
