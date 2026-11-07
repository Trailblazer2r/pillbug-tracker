#!/usr/bin/env python3
"""Desktop GUI for the pill bug light/shelter tracker.  Run:  python3 pillbug_gui.py

Open a video -> the arena circles and the number of chambers are detected automatically.
Fix them by dragging in the preview if needed, then press "Run analysis".
"""
import base64
import math
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

import pillbug_core as core

PREVIEW_W, PREVIEW_H = 860, 600


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pill bug tracker - light / shelter")
        self.geometry("1260x820")
        self.d = core.Params()
        self.bg = self.first = None
        self.scale = 1.0
        self.q = queue.Queue()
        self.cancel_flag = False
        self.drag = None
        self._redraw_pending = False
        self.vars = {}
        self._build()
        self.after(100, self._poll)

    # ------------------------------------------------------------------ UI
    def _num(self, key, value):
        self.vars[key] = tk.DoubleVar(value=value)
        return self.vars[key]

    def _build(self):
        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=10)
        right.pack(side="right", fill="both", expand=True)

        # --- main controls
        self.video = tk.StringVar()
        self.empty = tk.StringVar()
        self.out = tk.StringVar()
        ttk.Label(left, text="1. Video", font=("", 12, "bold")).pack(anchor="w")
        f = ttk.Frame(left)
        f.pack(fill="x", pady=2)
        ttk.Entry(f, textvariable=self.video, width=34).pack(side="left", fill="x", expand=True)
        ttk.Button(f, text="Choose...", command=self._pick_video).pack(side="left", padx=3)

        ttk.Label(left, text="2. Empty arena (optional)", font=("", 12, "bold")).pack(anchor="w", pady=(10, 0))
        f = ttk.Frame(left)
        f.pack(fill="x", pady=2)
        ttk.Entry(f, textvariable=self.empty, width=34).pack(side="left", fill="x", expand=True)
        ttk.Button(f, text="Choose...", command=self._pick_empty).pack(side="left", padx=3)
        ttk.Label(left, text="A short image/video of the arena with no bugs. Recommended:\n"
                             "a bug that sits still is otherwise easy to lose.",
                  foreground="#666").pack(anchor="w")

        ttk.Label(left, text="3. Check the circles", font=("", 12, "bold")).pack(anchor="w", pady=(10, 0))
        ttk.Label(left, justify="left", foreground="#444",
                  text="They are found automatically. To fix them, drag in the picture:\n"
                       "  - inner red circle edge = edge of the centre cup\n"
                       "  - outer red circle edge = inner edge of the black roof\n"
                       "  - anywhere else = move both circles\n"
                       "Yellow = A/B boundary. The blue band is B (under the roof)\n"
                       "and is not tracked: a bug that goes there just disappears.").pack(anchor="w")
        ttk.Button(left, text="Auto-detect again", command=self._auto).pack(anchor="w", pady=4)

        run = ttk.Frame(left)
        run.pack(fill="x", pady=(14, 4))
        self.btn_run = ttk.Button(run, text="Run analysis", command=self._run)
        self.btn_run.pack(side="left")
        self.btn_cancel = ttk.Button(run, text="Cancel", command=self._cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=6)
        ttk.Button(run, text="Open results", command=self._open_out).pack(side="left")
        self.prog = ttk.Progressbar(left, mode="determinate", maximum=100)
        self.prog.pack(fill="x")
        f = ttk.Frame(left)
        f.pack(fill="x", pady=2)
        ttk.Label(f, text="Results go to:", foreground="#666").pack(side="left")
        ttk.Label(f, textvariable=self.out, foreground="#666", wraplength=300, justify="left").pack(side="left", padx=4)
        ttk.Button(left, text="Change results folder...", command=self._pick_out).pack(anchor="w")

        # --- advanced (hidden by default)
        self.adv_open = False
        self.adv_btn = ttk.Button(left, text="Advanced settings  >", command=self._toggle_adv)
        self.adv_btn.pack(anchor="w", pady=(14, 2))
        self.adv = ttk.Frame(left)
        def arow(label, key, value, width=8):
            r = ttk.Frame(self.adv)
            r.pack(fill="x", pady=1)
            ttk.Label(r, text=label, width=28).pack(side="left")
            e = ttk.Entry(r, textvariable=self._num(key, value), width=width)
            e.pack(side="left")
            e.bind("<Return>", lambda ev: self._redraw())
            e.bind("<FocusOut>", lambda ev: self._redraw())
        arow("Chambers", "n_sectors", self.d.n_sectors)
        arow("Rotate chamber numbers", "number_shift", 0)
        arow("Centre x / y (px)", "cx", self.d.cx)
        arow("", "cy", self.d.cy)
        arow("Cup radius (px)", "r_in", self.d.r_in)
        arow("Roof inner-edge radius (px)", "r_out", self.d.r_out)
        arow("Centre cup diameter (cm)", "cup_diam_cm", self.d.cup_diam_cm)
        arow("Analyse every Nth frame", "step", self.d.step)
        arow("Detection threshold", "thresh", self.d.thresh)
        arow("Min blob area (px)", "min_area", self.d.min_area)
        arow("Shelter entry zone (px)", "entry_zone_px", self.d.entry_zone_px)
        arow("Min hidden time (s)", "min_hidden_s", self.d.min_hidden_s)
        arow("Start at (s)", "start_s", 0)
        arow("First N s of video are empty", "empty_first_s", 0)
        arow("Only analyse first N min (0=all)", "max_minutes", 0)
        self.vars["save_video"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.adv, text="Write annotated check video", variable=self.vars["save_video"]).pack(anchor="w")

        # --- preview + log
        self.canvas = tk.Canvas(right, width=PREVIEW_W, height=PREVIEW_H, background="#222", highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_text(PREVIEW_W // 2, PREVIEW_H // 2, text="Choose a video to see the arena",
                                fill="#ccc", font=("", 14), tags="hint")
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.logbox = tk.Text(right, height=10, wrap="word")
        self.logbox.pack(fill="both", expand=True, pady=(8, 0))

    def _toggle_adv(self):
        self.adv_open = not self.adv_open
        if self.adv_open:
            self.adv.pack(anchor="w", after=self.adv_btn)
            self.adv_btn.config(text="Advanced settings  v")
        else:
            self.adv.pack_forget()
            self.adv_btn.config(text="Advanced settings  >")

    # ---------------------------------------------------------------- files
    def _pick_video(self):
        f = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All", "*.*")])
        if f:
            self.video.set(f)
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

    # -------------------------------------------------------------- geometry
    def _params(self):
        p = core.Params()
        g = lambda k: self.vars[k].get()
        p.n_sectors = max(2, int(round(g("n_sectors"))))
        p.number_shift = int(round(g("number_shift")))
        for k in ("cx", "cy", "r_in", "r_out", "cup_diam_cm", "entry_zone_px", "min_hidden_s", "start_s", "empty_first_s"):
            setattr(p, k, float(g(k)))
        p.r_ab = None   # boundary = roof edge
        p.step = max(1, int(round(g("step"))))
        p.thresh = int(round(g("thresh")))
        p.min_area = int(round(g("min_area")))
        mm = float(g("max_minutes"))
        p.max_minutes = mm if mm > 0 else None
        p.save_video = bool(g("save_video"))
        p.empty_bg = self.empty.get().strip()
        return p

    def _load_preview(self):
        v = self.video.get().strip()
        if not os.path.isfile(v):
            return
        self.canvas.delete("hint")
        self.canvas.create_text(PREVIEW_W // 2, PREVIEW_H // 2, text="Loading...", fill="#ccc", font=("", 14), tags="hint")
        self.update_idletasks()
        try:
            self.bg, self.first = core.quick_background(v)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Preview", str(e))
            return
        self._auto()

    def _auto(self):
        if self.bg is None:
            return
        geo = core.auto_geometry(self.bg)
        if geo is None:
            h, w = self.bg.shape
            geo = dict(self.vars_default())
            self.log("Could not detect the circles automatically - drag them into place.")
        for k, v in geo.items():
            self.vars[k].set(round(float(v), 1))
        try:
            self.vars["n_sectors"].set(core.guess_n_sectors(self.bg, self._params()))
        except Exception:  # noqa: BLE001
            pass
        self._redraw()

    def vars_default(self):
        return dict(cx=self.d.cx, cy=self.d.cy, r_in=self.d.r_in, r_out=self.d.r_out)

    def _redraw(self, *_):
        self._redraw_pending = False
        if self.bg is None:
            return
        try:
            p = self._params()
            spokes = core.find_spoke_angles(self.bg, p)
            sec = core.build_sector_map(self.bg.shape, p, spokes)
            img = core.draw_overlay(self.first if self.first is not None else self.bg, p, spokes, sec)
        except (tk.TclError, ValueError, ZeroDivisionError):
            return
        self.scale = min(PREVIEW_W / img.shape[1], PREVIEW_H / img.shape[0], 1.0)
        img = cv2.resize(img, None, fx=self.scale, fy=self.scale)
        ok, buf = cv2.imencode(".png", img)
        self._photo = tk.PhotoImage(data=base64.b64encode(buf.tobytes()))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")

    def _schedule_redraw(self):
        if not self._redraw_pending:
            self._redraw_pending = True
            self.after(60, self._redraw)

    # ----------------------------------------------------------------- drag
    def _press(self, ev):
        if self.bg is None:
            return
        x, y = ev.x / self.scale, ev.y / self.scale
        cx, cy = self.vars["cx"].get(), self.vars["cy"].get()
        r_in, r_out = self.vars["r_in"].get(), self.vars["r_out"].get()
        d = math.hypot(x - cx, y - cy)
        tol = max(10.0, 14.0 / self.scale)
        if abs(d - r_in) <= tol and abs(d - r_in) <= abs(d - r_out):
            self.drag = ("cup", None)
        elif abs(d - r_out) <= tol:
            self.drag = ("roof", None)
        else:
            self.drag = ("move", (x - cx, y - cy))

    def _motion(self, ev):
        if not self.drag or self.bg is None:
            return
        x, y = ev.x / self.scale, ev.y / self.scale
        cx, cy = self.vars["cx"].get(), self.vars["cy"].get()
        mode, off = self.drag
        if mode == "cup":
            self.vars["r_in"].set(round(max(10.0, min(math.hypot(x - cx, y - cy), self.vars["r_out"].get() - 20)), 1))
        elif mode == "roof":
            self.vars["r_out"].set(round(max(self.vars["r_in"].get() + 20, math.hypot(x - cx, y - cy)), 1))
        else:
            self.vars["cx"].set(round(x - off[0], 1))
            self.vars["cy"].set(round(y - off[1], 1))
        self._schedule_redraw()

    def _release(self, ev):
        if self.drag:
            self.drag = None
            self._redraw()

    # ------------------------------------------------------------------ run
    def _run(self):
        v, o = self.video.get().strip(), self.out.get().strip()
        if not os.path.isfile(v):
            messagebox.showwarning("Video", "Choose a video file first.")
            return
        if not o:
            o = os.path.splitext(v)[0] + "_tracking"
            self.out.set(o)
        try:
            p = self._params()
        except (tk.TclError, ValueError) as e:
            messagebox.showwarning("Settings", f"Check the numeric fields in Advanced settings: {e}")
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

        threading.Thread(target=work, daemon=True).start()

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
                        self.logbox.insert("end", "\nFull results are in the results folder (summary_per_chamber.csv).\n")
                        self.logbox.see("end")
        except queue.Empty:
            pass
        self.after(100, self._poll)


def main():
    app = App()
    if "--selftest" in sys.argv:
        i = sys.argv.index("--selftest")
        if len(sys.argv) > i + 1:   # optional video path: exercise preview + auto-detect
            app.video.set(sys.argv[i + 1])
            app._load_preview()
            print({k: app.vars[k].get() for k in ("n_sectors", "cx", "cy", "r_in", "r_out")})
        app.update()
        app.destroy()
        print("gui ok")
        return
    app.mainloop()


if __name__ == "__main__":
    main()
