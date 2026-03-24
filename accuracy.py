import argparse
import sys
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from math import sqrt
from statistics import mean, stdev
from scipy.stats import norm, chi2, binom, shapiro, ttest_1samp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

###############################################################################
# 1) ShotsData: a logic class that holds shot data & performs all calculations
###############################################################################

class ShotsData:
    def __init__(self, radius=15.0, seed=42):
        np.random.seed(seed=seed)
        self.shots = []  # list of (x, y) floats
        self.valid_radius = radius
        
        # Derived metrics
        self.avg_coords = None  # (mean_x, mean_y)
        self.distances = []     # radial distances from the mean
        self.stdX_dev = None
        self.stdY_dev = None
        self.X_mean = None
        self.Y_mean = None
        self.stdX_error = None
        self.stdY_error = None
        self.total_shots = 0
        
        # Gaussianity scores (Shapiro-Wilk)
        self.gaussianity_x_w = None
        self.gaussianity_x_p = None
        self.gaussianity_y_w = None
        self.gaussianity_y_p = None

        # Probability inputs
        self.trials = None
        self.hits = None
        
        # Probability results (95% etc.)
        self.prob_hit_one_shot = None
        self.prob_binomial = None
        self.prob_binomial_lower_95 = None
        self.prob_binomial_higher_95 = None
        self.prob_binomial_lower_50 = None
        self.prob_binomial_higher_50 = None

    def add_shot(self, x, y):
        self.shots.append((float(x), float(y)))

    def remove_shot(self, index):
        """Remove shot by list index (0-based)."""
        if 0 <= index < len(self.shots):
            del self.shots[index]

    def list_shots(self):
        """Return list of all shots as (index, x, y)."""
        return [(i, s[0], s[1]) for i, s in enumerate(self.shots)]

    def import_from_excel(self, path):
        """Load shots from an Excel file.

        Supported formats:
        - 2D: columns 'X (cm)' and 'Y (cm)'
        - 1D: column 'distance_m' (values in meters, converted to cm)
        """
        with open(path, 'rb') as f:
            data = f.read()
        import io
        df_shots = pd.read_excel(io.BytesIO(data))
        self.shots.clear()
        if 'X (cm)' in df_shots.columns and 'Y (cm)' in df_shots.columns:
            for _, row in df_shots.iterrows():
                x = float(row['X (cm)'])
                y = float(row['Y (cm)'])
                self.shots.append((x, y))
        elif 'distance_m' in df_shots.columns:
            # Drop summary/header rows: keep only rows with integer index
            idx = pd.to_numeric(df_shots['index'], errors='coerce')
            df_shots = df_shots[idx.notna() & (idx % 1 == 0)]
            df_shots['distance_m'] = pd.to_numeric(df_shots['distance_m'], errors='coerce')
            df_shots = df_shots.dropna(subset=['distance_m'])
            for _, row in df_shots.iterrows():
                d = float(row['distance_m']) * 100  # meters to cm
                self.shots.append((d, 0.0))
        else:
            raise ValueError(
                "Excel must have columns 'X (cm)' and 'Y (cm)', "
                "or 'distance_m'"
            )

    def export_to_excel(self, path):
        """Save current shots to an Excel file."""
        if not self.shots:
            raise ValueError("No shots data to export.")
        all_y_zero = all(s[1] == 0.0 for s in self.shots)
        if all_y_zero:
            df_shots = pd.DataFrame(
                [s[0] / 100 for s in self.shots], columns=['distance_m']
            )
        else:
            df_shots = pd.DataFrame(self.shots, columns=['X (cm)', 'Y (cm)'])
        with pd.ExcelWriter(path) as writer:
            df_shots.to_excel(writer, sheet_name='Shots', index=False)

    def set_radius(self, radius):
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        self.valid_radius = float(radius)

    def set_trials(self, trials):
        if trials < 0:
            raise ValueError("Trials must be >= 0.")
        self.trials = trials

    def set_hits(self, hits):
        if hits < 0:
            raise ValueError("Hits must be >= 0.")
        self.hits = hits

    def calculate_metrics(self):
        """Compute all the basic metrics: average, std dev, accuracy, etc."""
        if not self.shots:
            # Clear out any old metrics
            self.avg_coords = None
            self.distances = []
            self.stdX_dev = None
            self.stdY_dev = None
            self.X_mean = None
            self.Y_mean = None
            self.stdX_error = None
            self.stdY_error = None
            self.gaussianity_x_w = None
            self.gaussianity_x_p = None
            self.gaussianity_y_w = None
            self.gaussianity_y_p = None
            self.total_shots = 0
            return
        
        x_vals = [s[0] for s in self.shots]
        y_vals = [s[1] for s in self.shots]
        
        avg_x = mean(x_vals)
        avg_y = mean(y_vals)
        self.avg_coords = (avg_x, avg_y)
        
        distances = [sqrt((sx - avg_x)**2 + (sy - avg_y)**2) for sx, sy in self.shots]
        self.distances = distances
        self.total_shots = len(self.shots)
        
        # Standard deviation (stdev) only valid if 2+ points
        if len(self.shots) > 1:
            self.stdX_dev = stdev(x_vals)
            self.stdY_dev = stdev(y_vals)
            self.X_mean = avg_x
            self.Y_mean = avg_y
            n = len(self.shots)
            self.stdX_error = self.stdX_dev / sqrt(2 * (n - 1))
            self.stdY_error = self.stdY_dev / sqrt(2 * (n - 1))
        else:
            self.stdX_dev = None
            self.stdY_dev = None
            self.X_mean = avg_x
            self.Y_mean = avg_y
            self.stdX_error = None
            self.stdY_error = None

        # Gaussianity (Shapiro-Wilk requires 3+ samples)
        if len(self.shots) >= 3:
            self.gaussianity_x_w, self.gaussianity_x_p = shapiro(x_vals)
            self.gaussianity_y_w, self.gaussianity_y_p = shapiro(y_vals)
        else:
            self.gaussianity_x_w = None
            self.gaussianity_x_p = None
            self.gaussianity_y_w = None
            self.gaussianity_y_p = None

    def compute_metric_pvalues(self, n_boot=2000):
        """Compute p-values and bootstrap standard errors for all metrics.

        Returns a dict with keys like 'mean_x_p', 'stdX_se', 'rms_x_se', etc.
        For means: t-test p-value (H0: mean = 0, i.e. no systematic bias).
        For all others: bootstrap standard error.
        """
        result = {}
        if len(self.shots) < 2:
            return result

        x_arr = np.array([s[0] for s in self.shots])
        y_arr = np.array([s[1] for s in self.shots])
        n = len(self.shots)

        # T-test p-values for mean X and mean Y (H0: mean = 0)
        _, result['mean_x_p'] = ttest_1samp(x_arr, 0.0)
        _, result['mean_y_p'] = ttest_1samp(y_arr, 0.0)

        if n < 3:
            return result

        # Bootstrap standard errors for all other metrics
        shots_arr = np.array(self.shots)
        boot_metrics = {k: [] for k in [
            'stdX', 'stdY', 'rms_x', 'rms_y', 'rms_radial',
            'range_x_50', 'range_y_50', 'cep_50', 'cep_95',
            'avg_abs_x', 'avg_abs_y', 'cep_100',
            'range_x_100', 'range_y_100',
        ]}

        for _ in range(n_boot):
            idx = np.random.randint(0, n, size=n)
            bx = shots_arr[idx, 0]
            by = shots_arr[idx, 1]
            bmx, bmy = bx.mean(), by.mean()
            bdists = np.sqrt((bx - bmx)**2 + (by - bmy)**2)

            boot_metrics['stdX'].append(bx.std(ddof=1))
            boot_metrics['stdY'].append(by.std(ddof=1))
            boot_metrics['rms_x'].append(np.sqrt(np.mean((bx - bmx)**2)))
            boot_metrics['rms_y'].append(np.sqrt(np.mean((by - bmy)**2)))
            boot_metrics['rms_radial'].append(np.sqrt(np.mean(bdists**2)))
            boot_metrics['range_x_50'].append(2 * np.percentile(np.abs(bx - bmx), 50))
            boot_metrics['range_y_50'].append(2 * np.percentile(np.abs(by - bmy), 50))
            boot_metrics['cep_50'].append(np.percentile(bdists, 50))
            boot_metrics['cep_95'].append(np.percentile(bdists, 95))
            boot_metrics['avg_abs_x'].append(np.mean(np.abs(bx - bmx)))
            boot_metrics['avg_abs_y'].append(np.mean(np.abs(by - bmy)))
            boot_metrics['cep_100'].append(bdists.max())
            boot_metrics['range_x_100'].append(bx.max() - bx.min())
            boot_metrics['range_y_100'].append(by.max() - by.min())

        for key, vals in boot_metrics.items():
            result[f'{key}_se'] = np.std(vals)

        return result

    def rolling_disparity(self, window_size):
        """Compute disparity for each rolling window of *window_size* shots.

        Disparity is the maximum pairwise Euclidean distance between any two
        shots in the window (the "diameter" of the group).  For 1-D data
        (all y == 0) this reduces to max(x) - min(x).

        Returns a list of dicts sorted by disparity descending::

            [{'start': int,          # 0-based index of first shot in window
              'end': int,            # 0-based index of last shot in window
              'disparity': float,    # max pairwise distance (cm)
              'shots': [(x,y), ...]  # the shots in this window
             }, ...]
        """
        n = len(self.shots)
        if window_size < 2 or n < window_size:
            return []

        results = []
        shots_arr = np.array(self.shots)
        global_mean = shots_arr.mean(axis=0)
        for i in range(n - window_size + 1):
            window = shots_arr[i:i + window_size]
            # Max pairwise distance = max over all pairs ||p_i - p_j||
            diffs = window[:, np.newaxis, :] - window[np.newaxis, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))
            disparity = float(dists.max())
            # Distance from window center to full-data mean
            window_mean = window.mean(axis=0)
            dist_to_mean = float(np.sqrt(((window_mean - global_mean) ** 2).sum()))
            results.append({
                'start': i,
                'end': i + window_size - 1,
                'disparity': disparity,
                'dist_to_mean': dist_to_mean,
                'shots': [tuple(s) for s in window],
            })

        results.sort(key=lambda r: r['disparity'], reverse=True)
        return results

    def reset_probability_results(self):
        self.prob_hit_one_shot = None
        self.prob_binomial = None
        self.prob_binomial_lower_95 = None
        self.prob_binomial_higher_95 = None
        self.prob_binomial_lower_50 = None
        self.prob_binomial_higher_50 = None

    def can_compute_probabilities(self):
        # We need at least 2 shots for a meaningful stdev, and trials/hits must be set.
        if not self.shots or (self.stdX_dev is None or self.stdY_dev is None):
            return False
        if self.trials is None or self.hits is None:
            return False
        return True

    def compute_single_monte_carlo_estimate(self, n_mc=10000):
        # This returns the approximate probability of hitting within self.valid_radius.
        sim_x = np.random.normal(self.X_mean, self.stdX_dev, size=n_mc)
        sim_y = np.random.normal(self.Y_mean, self.stdY_dev, size=n_mc)
        dists = np.sqrt((sim_x - self.X_mean)**2 + (sim_y - self.Y_mean)**2)
        prob_hit = np.mean(dists <= self.valid_radius)
        return prob_hit

    def compute_parametric_bootstrap_ci(self, prob_hit_one_shot, n_boot=10000):
        # Produces arrays of binomial probabilities for bootstrapping.
        shots_arr = np.array(self.shots)
        n_data = len(shots_arr)
        boot_binom_probs = []
        
        for _ in range(n_boot):
            chi2_x = chi2.rvs(df=n_data - 1)
            sigma_x_star_sq = (n_data - 1)*(self.stdX_dev**2)/chi2_x
            sigma_x_star = np.sqrt(sigma_x_star_sq)
            mu_x_star = np.random.normal(loc=self.X_mean,
                                         scale=sigma_x_star / np.sqrt(n_data))

            chi2_y = chi2.rvs(df=n_data - 1)
            sigma_y_star_sq = (n_data - 1)*(self.stdY_dev**2)/chi2_y
            sigma_y_star = np.sqrt(sigma_y_star_sq)
            mu_y_star = np.random.normal(loc=self.Y_mean,
                                         scale=sigma_y_star / np.sqrt(n_data))

            sim_x_star = np.random.normal(mu_x_star, sigma_x_star, size=10000)
            sim_y_star = np.random.normal(mu_y_star, sigma_y_star, size=10000)
            dist_star = np.sqrt((sim_x_star - mu_x_star)**2 + (sim_y_star - mu_y_star)**2)
            p_star = np.mean(dist_star <= self.valid_radius)

            prob_binom_star = 1 - binom.cdf(self.hits - 1, self.trials, p_star)
            boot_binom_probs.append(prob_binom_star)

        boot_binom_probs = np.array(boot_binom_probs)
        return (
            np.percentile(boot_binom_probs, 2.5),   # lower_95
            np.percentile(boot_binom_probs, 97.5),  # higher_95
            np.percentile(boot_binom_probs, 25),    # lower_50
            np.percentile(boot_binom_probs, 75)     # higher_50
        )

    def calculate_probabilities(self):
        """
        Compute all probability & confidence interval metrics
        based on self.trials and self.hits (if set).
        """
        self.reset_probability_results()
        
        # Check if we have enough data to compute.
        if not self.can_compute_probabilities():
            return
        
        # Single Monte Carlo estimate
        prob_hit_one_shot = self.compute_single_monte_carlo_estimate()
        self.prob_hit_one_shot = prob_hit_one_shot

        # Probability of >= hits in binomial
        prob_binom_center = 1 - binom.cdf(self.hits - 1, self.trials, prob_hit_one_shot)
        self.prob_binomial = prob_binom_center

        # Parametric bootstrap for confidence intervals
        (low_95, high_95, low_50, high_50) = self.compute_parametric_bootstrap_ci(prob_hit_one_shot)
        self.prob_binomial_lower_95 = low_95
        self.prob_binomial_higher_95 = high_95
        self.prob_binomial_lower_50 = low_50
        self.prob_binomial_higher_50 = high_50


###############################################################################
# 1.5) Compare two datasets subtool (logic)
###############################################################################

def compare_datasets(dataA, dataB, n_boot=10000, mc_size=10000):
    """
    Compare two ShotsData-like objects (dataA, dataB),
    returning the probability that A has a higher 'hit probability'
    than B by parametric bootstrap.
    """

    # Check each dataset has at least 2 shots => std dev
    if dataA.stdX_dev is None or dataA.stdY_dev is None:
        raise ValueError("Dataset A does not have enough shots for stdev (need >= 2).")
    if dataB.stdX_dev is None or dataB.stdY_dev is None:
        raise ValueError("Dataset B does not have enough shots for stdev (need >= 2).")

    shotsA = np.array(dataA.shots)
    nA = len(shotsA)
    shotsB = np.array(dataB.shots)
    nB = len(shotsB)

    countA_better = 0

    for _ in range(n_boot):
        # Param sample for A
        chi2_xA = chi2.rvs(df=nA - 1)
        sigma_xA_star_sq = (nA - 1)*(dataA.stdX_dev**2)/chi2_xA
        sigma_xA_star = np.sqrt(sigma_xA_star_sq)
        mu_xA_star = np.random.normal(
            loc=dataA.X_mean,
            scale=sigma_xA_star / np.sqrt(nA)
        )

        chi2_yA = chi2.rvs(df=nA - 1)
        sigma_yA_star_sq = (nA - 1)*(dataA.stdY_dev**2)/chi2_yA
        sigma_yA_star = np.sqrt(sigma_yA_star_sq)
        mu_yA_star = np.random.normal(
            loc=dataA.Y_mean,
            scale=sigma_yA_star / np.sqrt(nA)
        )

        sim_xA = np.random.normal(mu_xA_star, sigma_xA_star, size=mc_size)
        sim_yA = np.random.normal(mu_yA_star, sigma_yA_star, size=mc_size)
        distA = np.sqrt((sim_xA - mu_xA_star)**2 + (sim_yA - mu_yA_star)**2)
        pA_star = np.mean(distA <= dataA.valid_radius)

        # Param sample for B
        chi2_xB = chi2.rvs(df=nB - 1)
        sigma_xB_star_sq = (nB - 1)*(dataB.stdX_dev**2)/chi2_xB
        sigma_xB_star = np.sqrt(sigma_xB_star_sq)
        mu_xB_star = np.random.normal(
            loc=dataB.X_mean,
            scale=sigma_xB_star / np.sqrt(nB)
        )

        chi2_yB = chi2.rvs(df=nB - 1)
        sigma_yB_star_sq = (nB - 1)*(dataB.stdY_dev**2)/chi2_yB
        sigma_yB_star = np.sqrt(sigma_yB_star_sq)
        mu_yB_star = np.random.normal(
            loc=dataB.Y_mean,
            scale=sigma_yB_star / np.sqrt(nB)
        )

        sim_xB = np.random.normal(mu_xB_star, sigma_xB_star, size=mc_size)
        sim_yB = np.random.normal(mu_yB_star, sigma_yB_star, size=mc_size)
        distB = np.sqrt((sim_xB - mu_xB_star)**2 + (sim_yB - mu_yB_star)**2)
        pB_star = np.mean(distB <= dataB.valid_radius)

        if pA_star > pB_star:
            countA_better += 1

    fraction = countA_better / n_boot
    return fraction


###############################################################################
# 2) The GUI class (with a second tab for comparing two datasets)
###############################################################################

class ShotAccuracyApp:
    def __init__(self, master):
        self.master = master
        master.title("Shot Accuracy Calculator")
        master.geometry("1600x800")

        # Set default font to something thicker and bigger
        import tkinter.font as tkfont
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="DejaVu Sans", size=11, weight="bold")
        master.option_add("*Font", default_font)

        # Store background color for selectable labels
        self._label_bg = master.cget('bg')

        # Create a Notebook for multiple tabs
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill='both', expand=True)

        # -----------------
        # MAIN CALCULATOR TAB
        # -----------------
        self.main_tab = tk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Main Calculator")

        self.data = ShotsData(radius=15.0)
        self.metric_pvalues = {}

        # Tkinter Variables
        self.trials_var = tk.StringVar()
        self.hits_var = tk.StringVar()

        # Scrollable canvas inside main tab
        _main_canvas = tk.Canvas(self.main_tab)
        _main_scrollbar = ttk.Scrollbar(self.main_tab, orient=tk.VERTICAL, command=_main_canvas.yview)
        _main_canvas.configure(yscrollcommand=_main_scrollbar.set)
        _main_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        _main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        _main_content = tk.Frame(_main_canvas)
        _main_canvas_window = _main_canvas.create_window((0, 0), window=_main_content, anchor='nw')

        def _on_main_content_configure(event):
            _main_canvas.configure(scrollregion=_main_canvas.bbox('all'))

        def _on_main_canvas_configure(event):
            _main_canvas.itemconfig(_main_canvas_window, width=event.width)

        _main_content.bind('<Configure>', _on_main_content_configure)
        _main_canvas.bind('<Configure>', _on_main_canvas_configure)

        def _on_mousewheel(event):
            _main_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

        _main_canvas.bind_all('<MouseWheel>', _on_mousewheel)

        # Frames inside main tab (placed in scrollable content frame)
        self.input_frame = tk.Frame(_main_content)
        self.input_frame.pack(pady=10, padx=10, fill=tk.X)

        self.controls_frame = tk.Frame(_main_content)
        self.controls_frame.pack(pady=10, padx=10, fill=tk.X)

        self.metrics_frame = tk.Frame(_main_content)
        self.metrics_frame.pack(pady=10, padx=10, fill=tk.X)

        self.prob_frame = tk.Frame(_main_content)
        self.prob_frame.pack(pady=10, padx=10, fill=tk.X)

        self.plots_frame = tk.Frame(_main_content)
        self.plots_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.create_shot_entries()
        self.create_main_controls()
        self.setup_visualization_plot()

        # -----------------
        # ROLLING DISPARITY TAB
        # -----------------
        self.disparity_tab = tk.Frame(self.notebook)
        self.notebook.add(self.disparity_tab, text="Rolling Disparity")
        self.build_disparity_ui(self.disparity_tab)

        # -----------------
        # COMPARISON TAB
        # -----------------
        self.compare_tab = tk.Frame(self.notebook)
        self.notebook.add(self.compare_tab, text="Compare Datasets")

        # We'll store two ShotsData objects
        self.compare_dataA = ShotsData(radius=15.0)
        self.compare_dataB = ShotsData(radius=15.0)
        self.build_compare_ui(self.compare_tab)

    # ----------------------------------------------------------------
    # MAIN TAB
    # ----------------------------------------------------------------
    def create_shot_entries(self):
        columns = ('order', 'x', 'y')
        self.tree = ttk.Treeview(self.input_frame, columns=columns, show='headings', height=10)
        self.tree.heading('order', text='#', command=lambda: self.sort_tree('order'))
        self.tree.heading('x', text='X (cm)', command=lambda: self.sort_tree('x'))
        self.tree.heading('y', text='Y (cm)', command=lambda: self.sort_tree('y'))
        self.tree.column('order', width=50, anchor='center')
        self.tree.column('x', width=100, anchor='center')
        self.tree.column('y', width=100, anchor='center')
        self.tree.pack(side='left', fill=tk.BOTH, expand=True)

        self.sort_column = 'order'
        self.sort_reverse = False

        scrollbar = ttk.Scrollbar(self.input_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

        self.tree.bind('<ButtonRelease-1>', self.on_shot_change)
        self.tree.bind('<KeyRelease>', self.on_shot_change)

    def sort_tree(self, col):
        if self.sort_column == col:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = col
            self.sort_reverse = False
        items = [(self.tree.set(k, col), k) for k in self.tree.get_children()]
        items.sort(key=lambda t: float(t[0]), reverse=self.sort_reverse)
        for index, (_, k) in enumerate(items):
            self.tree.move(k, '', index)
        # Update heading to show sort direction
        for c in ('order', 'x', 'y'):
            base = {'order': '#', 'x': 'X (cm)', 'y': 'Y (cm)'}[c]
            arrow = ''
            if c == col:
                arrow = ' \u25bc' if self.sort_reverse else ' \u25b2'
            self.tree.heading(c, text=base + arrow)

    def create_main_controls(self):
        # Add & Remove
        self.add_button = tk.Button(self.controls_frame, text="Add Shot", command=self.add_shot)
        self.add_button.grid(row=0, column=0, padx=5)

        self.remove_button = tk.Button(self.controls_frame, text="Remove Selected Shot", command=self.remove_shot)
        self.remove_button.grid(row=0, column=1, padx=5)

        # Import/Export
        self.import_button = tk.Button(self.controls_frame, text="Import from Excel", command=self.import_from_excel)
        self.import_button.grid(row=0, column=2, padx=5)

        self.export_button = tk.Button(self.controls_frame, text="Export to Excel", command=self.export_to_excel)
        self.export_button.grid(row=0, column=3, padx=5)

        self.clear_button = tk.Button(self.controls_frame, text="Clear Data", command=self.clear_data)
        self.clear_button.grid(row=0, column=4, padx=5)

        # Metrics (selectable labels) — column 0: values, column 1: p-values / ± SE
        self.avg_label = self._selectable_label(self.metrics_frame, "Average Coordinates: N/A",
                                                row=0, column=0, sticky='w', padx=10)
        self.avg_p_label = self._selectable_label(self.metrics_frame, "",
                                                  row=0, column=1, sticky='w', padx=10)
        self.stdX_label = self._selectable_label(self.metrics_frame, "Standard Deviation X: N/A",
                                                 row=1, column=0, sticky='w', padx=10)
        self.stdX_p_label = self._selectable_label(self.metrics_frame, "",
                                                   row=1, column=1, sticky='w', padx=10)
        self.stdY_label = self._selectable_label(self.metrics_frame, "Standard Deviation Y: N/A",
                                                 row=3, column=0, sticky='w', padx=10)
        self.stdY_p_label = self._selectable_label(self.metrics_frame, "",
                                                   row=3, column=1, sticky='w', padx=10)
        self.rms_x_label = self._selectable_label(self.metrics_frame, "RMS X: N/A",
                                                  row=4, column=0, sticky='w', padx=10)
        self.rms_x_p_label = self._selectable_label(self.metrics_frame, "",
                                                    row=4, column=1, sticky='w', padx=10)
        self.rms_y_label = self._selectable_label(self.metrics_frame, "RMS Y: N/A",
                                                  row=5, column=0, sticky='w', padx=10)
        self.rms_y_p_label = self._selectable_label(self.metrics_frame, "",
                                                    row=5, column=1, sticky='w', padx=10)
        self.rms_radial_label = self._selectable_label(self.metrics_frame, "RMS Radial: N/A",
                                                       row=6, column=0, sticky='w', padx=10)
        self.rms_radial_p_label = self._selectable_label(self.metrics_frame, "",
                                                         row=6, column=1, sticky='w', padx=10)
        self.range_x_50_label = self._selectable_label(self.metrics_frame, "50% X Range: N/A",
                                                       row=7, column=0, sticky='w', padx=10)
        self.range_x_50_p_label = self._selectable_label(self.metrics_frame, "",
                                                         row=7, column=1, sticky='w', padx=10)
        self.range_y_50_label = self._selectable_label(self.metrics_frame, "50% Y Range: N/A",
                                                       row=8, column=0, sticky='w', padx=10)
        self.range_y_50_p_label = self._selectable_label(self.metrics_frame, "",
                                                         row=8, column=1, sticky='w', padx=10)
        self.cep_50_label = self._selectable_label(self.metrics_frame, "CEP50 (radius): N/A",
                                                   row=9, column=0, sticky='w', padx=10)
        self.cep_50_p_label = self._selectable_label(self.metrics_frame, "",
                                                     row=9, column=1, sticky='w', padx=10)
        self.cumulative_x_error_label = self._selectable_label(self.metrics_frame, "Average absolute X: N/A",
                                                               row=10, column=0, sticky='w', padx=10)
        self.avg_abs_x_p_label = self._selectable_label(self.metrics_frame, "",
                                                        row=10, column=1, sticky='w', padx=10)
        self.cumulative_y_error_label = self._selectable_label(self.metrics_frame, "Average absolute Y: N/A",
                                                               row=11, column=0, sticky='w', padx=10)
        self.avg_abs_y_p_label = self._selectable_label(self.metrics_frame, "",
                                                        row=11, column=1, sticky='w', padx=10)
        self.cep_95_label = self._selectable_label(self.metrics_frame, "CEP95 (radius): N/A",
                                                   row=12, column=0, sticky='w', padx=10)
        self.cep_95_p_label = self._selectable_label(self.metrics_frame, "",
                                                     row=12, column=1, sticky='w', padx=10)
        self.cep_100_label = self._selectable_label(self.metrics_frame, "CEP100 (radius): N/A",
                                                    row=13, column=0, sticky='w', padx=10)
        self.cep_100_p_label = self._selectable_label(self.metrics_frame, "",
                                                      row=13, column=1, sticky='w', padx=10)
        self.range_x_100_label = self._selectable_label(self.metrics_frame, "100% X Range: N/A",
                                                        row=14, column=0, sticky='w', padx=10)
        self.range_x_100_p_label = self._selectable_label(self.metrics_frame, "",
                                                          row=14, column=1, sticky='w', padx=10)
        self.range_y_100_label = self._selectable_label(self.metrics_frame, "100% Y Range: N/A",
                                                        row=15, column=0, sticky='w', padx=10)
        self.range_y_100_p_label = self._selectable_label(self.metrics_frame, "",
                                                          row=15, column=1, sticky='w', padx=10)
        self.gauss_x_label = self._selectable_label(self.metrics_frame, "Gaussianity X: N/A",
                                                    row=16, column=0, sticky='w', padx=10)
        self.gauss_x_p_label = self._selectable_label(self.metrics_frame, "",
                                                      row=16, column=1, sticky='w', padx=10)
        self.gauss_y_label = self._selectable_label(self.metrics_frame, "Gaussianity Y: N/A",
                                                    row=17, column=0, sticky='w', padx=10)
        self.gauss_y_p_label = self._selectable_label(self.metrics_frame, "",
                                                      row=17, column=1, sticky='w', padx=10)

        # Probability area
        self.trials_label = tk.Label(self.prob_frame, text="Number of Trials:")
        self.trials_label.grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.trials_entry = tk.Entry(self.prob_frame, textvariable=self.trials_var)
        self.trials_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        self.hits_label = tk.Label(self.prob_frame, text="Number of Hits Within Radius:")
        self.hits_label.grid(row=1, column=0, sticky='e', padx=5, pady=5)
        self.hits_entry = tk.Entry(self.prob_frame, textvariable=self.hits_var)
        self.hits_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        self.prob_xy_label = self._selectable_label(self.prob_frame, "Probability of one shot hitting: N/A",
                                                    row=6, column=0, columnspan=2, sticky='w', padx=10)
        self.prob_binomial_label = self._selectable_label(self.prob_frame, "Probability of reaching desired result: N/A",
                                                          row=7, column=0, columnspan=2, sticky='w', padx=10)
        self.prob_lower_label = self._selectable_label(self.prob_frame, "Lower Probability (95% confidence): N/A",
                                                       row=8, column=0, columnspan=2, sticky='w', padx=10)
        self.prob_higher_label = self._selectable_label(self.prob_frame, "Higher Probability (95% confidence): N/A",
                                                        row=9, column=0, columnspan=2, sticky='w', padx=10)
        self.prob_lower_50_label = self._selectable_label(self.prob_frame, "Lower Probability (50% confidence): N/A",
                                                          row=10, column=0, columnspan=2, sticky='w', padx=10)
        self.prob_higher_50_label = self._selectable_label(self.prob_frame, "Higher Probability (50% confidence): N/A",
                                                           row=11, column=0, columnspan=2, sticky='w', padx=10)

        # Event bindings
        self.trials_var.trace_add('write', self.on_prob_input_change)
        self.hits_var.trace_add('write', self.on_prob_input_change)

    def _selectable_label(self, parent, text, **grid_kwargs):
        """Create a read-only Entry widget that looks like a label but allows text selection."""
        var = tk.StringVar(value=text)
        entry = tk.Entry(parent, textvariable=var, state='readonly',
                         readonlybackground=self._label_bg, relief='flat',
                         borderwidth=0, highlightthickness=0, width=60)
        entry.grid(**grid_kwargs)
        entry._var = var  # keep reference for updates
        return entry

    @staticmethod
    def _set_label_text(entry, text):
        """Update text of a selectable label."""
        entry._var.set(text)

    def on_shot_change(self, event):
        # Rebuild shots from the tree
        new_shots = []
        for item in self.tree.get_children():
            _, x, y = self.tree.item(item)['values']
            new_shots.append((float(x), float(y)))
        self.data.shots = new_shots
        self.update_metrics_and_visualization()

    def add_shot(self):
        add_window = tk.Toplevel(self.master)
        add_window.title("Add New Shot")
        add_window.geometry("300x170")
        add_window.grab_set()

        tk.Label(add_window, text="X Coordinate (cm):").pack(pady=10)
        x_entry = tk.Entry(add_window)
        x_entry.pack()

        tk.Label(add_window, text="Y Coordinate (cm):").pack(pady=10)
        y_entry = tk.Entry(add_window)
        y_entry.pack()

        def submit():
            x_str = x_entry.get().strip()
            y_str = y_entry.get().strip()
            if x_str == "" or y_str == "":
                messagebox.showerror("Input Error", "Both X and Y coordinates must be filled.")
                return
            try:
                x_val = float(x_str)
                y_val = float(y_str)
                self.data.add_shot(x_val, y_val)
                order = len(self.data.shots)
                self.tree.insert('', 'end', values=(order, x_val, y_val))
                add_window.destroy()
                self.update_metrics_and_visualization()
            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid numeric values for coordinates.")

        tk.Button(add_window, text="Add Shot", command=submit).pack(pady=10)

    def remove_shot(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Selection Error", "Please select a shot to remove.")
            return
        for item in selected_item:
            index = self.tree.index(item)
            self.tree.delete(item)
            self.data.remove_shot(index)
        self.update_metrics_and_visualization()

    def clear_data(self):
        if not self.data.shots:
            return
        if not messagebox.askyesno("Clear Data", "Are you sure you want to clear all data?"):
            return
        self.data.shots.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.update_metrics_and_visualization()

    def import_from_excel(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            self.data.import_from_excel(file_path)
            # Clear tree
            for item in self.tree.get_children():
                self.tree.delete(item)
            # Repopulate
            for i, (x, y) in enumerate(self.data.shots, 1):
                self.tree.insert('', 'end', values=(i, x, y))
            self.update_metrics_and_visualization()
            messagebox.showinfo("Import Successful", f"Data imported successfully from {file_path}")
        except Exception as e:
            messagebox.showerror("Import Error", f"An error occurred while importing: {e}")

    def export_to_excel(self):
        if not self.data.shots:
            messagebox.showerror("No Data", "No data to export.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension='.xlsx', 
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if not file_path:
            return
        try:
            self.data.export_to_excel(file_path)
            messagebox.showinfo("Export Successful", f"Data exported successfully to {file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while exporting: {e}")

    def update_metrics_and_visualization(self):
        self.data.calculate_metrics()
        self.data.calculate_probabilities()
        self.metric_pvalues = self.data.compute_metric_pvalues()
        self.update_metric_labels()
        self.update_visualization()

    def _fmt_se(self, key):
        """Format a bootstrap SE value from self.metric_pvalues, or return ''."""
        pv = getattr(self, 'metric_pvalues', {})
        se = pv.get(f'{key}_se')
        if se is not None:
            return f"\u00b1 {se:.2f} cm"
        return ""

    def _fmt_p(self, key):
        """Format a p-value from self.metric_pvalues, or return ''."""
        pv = getattr(self, 'metric_pvalues', {})
        p = pv.get(f'{key}_p')
        if p is not None:
            return f"p = {p:.4f}"
        return ""

    def _clear_p_labels(self):
        """Clear all second-column p-value/SE labels."""
        _s = self._set_label_text
        for lbl in [
            self.avg_p_label, self.stdX_p_label, self.stdY_p_label,
            self.rms_x_p_label, self.rms_y_p_label, self.rms_radial_p_label,
            self.range_x_50_p_label, self.range_y_50_p_label,
            self.cep_50_p_label, self.avg_abs_x_p_label, self.avg_abs_y_p_label,
            self.cep_95_p_label, self.cep_100_p_label,
            self.range_x_100_p_label, self.range_y_100_p_label,
            self.gauss_x_p_label, self.gauss_y_p_label,
        ]:
            _s(lbl, "")

    def update_metric_labels(self):
        _s = self._set_label_text
        if not self.data.shots:
            # Clear everything
            _s(self.avg_label, "Average Coordinates: N/A")
            _s(self.stdX_label, "Standard Deviation X: N/A")
            _s(self.stdY_label, "Standard Deviation Y: N/A")
            _s(self.rms_x_label, "RMS X: N/A")
            _s(self.rms_y_label, "RMS Y: N/A")
            _s(self.rms_radial_label, "RMS Radial: N/A")
            _s(self.range_x_50_label, "50% X Range: N/A")
            _s(self.range_y_50_label, "50% Y Range: N/A")
            _s(self.cep_50_label, "CEP50 (radius): N/A")
            _s(self.cumulative_x_error_label, "Average absolute X: N/A")
            _s(self.cumulative_y_error_label, "Average absolute Y: N/A")
            _s(self.cep_95_label, "CEP95 (radius): N/A")
            _s(self.cep_100_label, "CEP100 (radius): N/A")
            _s(self.range_x_100_label, "100% X Range: N/A")
            _s(self.range_y_100_label, "100% Y Range: N/A")
            _s(self.gauss_x_label, "Gaussianity X: N/A")
            _s(self.gauss_y_label, "Gaussianity Y: N/A")
            self._clear_p_labels()
            return

        # Basic
        avg_x, avg_y = self.data.avg_coords
        _s(self.avg_label, f"Average Coordinates: ({avg_x:.2f}, {avg_y:.2f}) cm")
        # p-value for mean: t-test H0: mean_x=0, mean_y=0 (bias test)
        mean_x_p = self._fmt_p('mean_x')
        mean_y_p = self._fmt_p('mean_y')
        if mean_x_p and mean_y_p:
            _s(self.avg_p_label, f"bias: X {mean_x_p}, Y {mean_y_p}")
        else:
            _s(self.avg_p_label, "")

        if self.data.stdX_dev is not None:
            _s(self.stdX_label, f"Standard Deviation X: {self.data.stdX_dev:.2f} cm")
            _s(self.stdY_label, f"Standard Deviation Y: {self.data.stdY_dev:.2f} cm")
            _s(self.stdX_p_label, self._fmt_se('stdX'))
            _s(self.stdY_p_label, self._fmt_se('stdY'))
        else:
            _s(self.stdX_label, "Standard Deviation X: N/A")
            _s(self.stdY_label, "Standard Deviation Y: N/A")
            _s(self.stdX_p_label, "")
            _s(self.stdY_p_label, "")

        # If 2+ shots, we can do extra stats
        distances = self.data.distances
        if len(distances) > 1:
            x_vals = [s[0] for s in self.data.shots]
            y_vals = [s[1] for s in self.data.shots]

            # RMS
            rms_x = np.sqrt(np.mean((np.array(x_vals) - avg_x)**2))
            rms_y = np.sqrt(np.mean((np.array(y_vals) - avg_y)**2))
            rms_radial = np.sqrt(np.mean(np.array(distances)**2))
            _s(self.rms_x_label, f"RMS X: {rms_x:.2f} cm")
            _s(self.rms_y_label, f"RMS Y: {rms_y:.2f} cm")
            _s(self.rms_radial_label, f"RMS Radial: {rms_radial:.2f} cm")
            _s(self.rms_x_p_label, self._fmt_se('rms_x'))
            _s(self.rms_y_p_label, self._fmt_se('rms_y'))
            _s(self.rms_radial_p_label, self._fmt_se('rms_radial'))

            x_50_range = 2 * np.percentile(np.abs(np.array(x_vals) - avg_x), 50)
            y_50_range = 2 * np.percentile(np.abs(np.array(y_vals) - avg_y), 50)
            _s(self.range_x_50_label, f"50% X Range: {x_50_range:.2f} cm")
            _s(self.range_y_50_label, f"50% Y Range: {y_50_range:.2f} cm")
            _s(self.range_x_50_p_label, self._fmt_se('range_x_50'))
            _s(self.range_y_50_p_label, self._fmt_se('range_y_50'))

            cep_50 = np.percentile(distances, 50)
            _s(self.cep_50_label, f"CEP50 (radius): {cep_50:.2f} cm")
            _s(self.cep_50_p_label, self._fmt_se('cep_50'))

            cum_x_error = float(np.sum(np.abs(np.array(x_vals) - avg_x)))
            cum_y_error = float(np.sum(np.abs(np.array(y_vals) - avg_y)))
            n = len(distances)
            _s(self.cumulative_x_error_label, f"Average absolute X: {cum_x_error/n:.2f} cm")
            _s(self.cumulative_y_error_label, f"Average absolute Y: {cum_y_error/n:.2f} cm")
            _s(self.avg_abs_x_p_label, self._fmt_se('avg_abs_x'))
            _s(self.avg_abs_y_p_label, self._fmt_se('avg_abs_y'))

            cep_95 = np.percentile(distances, 95)
            _s(self.cep_95_label, f"CEP95 (radius): {cep_95:.2f} cm")
            _s(self.cep_95_p_label, self._fmt_se('cep_95'))

            cep_100 = max(distances)
            range_x_100 = max(x_vals) - min(x_vals)
            range_y_100 = max(y_vals) - min(y_vals)
            _s(self.cep_100_label, f"CEP100 (radius): {cep_100:.2f} cm")
            _s(self.range_x_100_label, f"100% X Range: {range_x_100:.2f} cm")
            _s(self.range_y_100_label, f"100% Y Range: {range_y_100:.2f} cm")
            _s(self.cep_100_p_label, self._fmt_se('cep_100'))
            _s(self.range_x_100_p_label, self._fmt_se('range_x_100'))
            _s(self.range_y_100_p_label, self._fmt_se('range_y_100'))

            # Gaussianity (Shapiro-Wilk W statistic; 1.0 = perfect Gaussian)
            if self.data.gaussianity_x_w is not None:
                _s(self.gauss_x_label, f"Gaussianity X: W={self.data.gaussianity_x_w:.4f}")
                _s(self.gauss_y_label, f"Gaussianity Y: W={self.data.gaussianity_y_w:.4f}")
                _s(self.gauss_x_p_label, f"p = {self.data.gaussianity_x_p:.4f}")
                _s(self.gauss_y_p_label, f"p = {self.data.gaussianity_y_p:.4f}")
            else:
                _s(self.gauss_x_label, "Gaussianity X: N/A (need 3+ shots)")
                _s(self.gauss_y_label, "Gaussianity Y: N/A (need 3+ shots)")
                _s(self.gauss_x_p_label, "")
                _s(self.gauss_y_p_label, "")
        else:
            # If only 1 shot
            _s(self.rms_x_label, "RMS X: N/A")
            _s(self.rms_y_label, "RMS Y: N/A")
            _s(self.rms_radial_label, "RMS Radial: N/A")
            _s(self.range_x_50_label, "50% X Range: N/A")
            _s(self.range_y_50_label, "50% Y Range: N/A")
            _s(self.cep_50_label, "CEP50 (radius): N/A")
            _s(self.cumulative_x_error_label, "Average absolute X: N/A")
            _s(self.cumulative_y_error_label, "Average absolute Y: N/A")
            _s(self.cep_95_label, "CEP95 (radius): N/A")
            _s(self.cep_100_label, "CEP100 (radius): N/A")
            _s(self.range_x_100_label, "100% X Range: N/A")
            _s(self.range_y_100_label, "100% Y Range: N/A")
            _s(self.gauss_x_label, "Gaussianity X: N/A")
            _s(self.gauss_y_label, "Gaussianity Y: N/A")
            self._clear_p_labels()

        # Probability
        if self.data.prob_hit_one_shot is not None:
            _s(self.prob_xy_label, f"Probability of one shot hitting: {self.data.prob_hit_one_shot * 100:.2f}%")
        else:
            _s(self.prob_xy_label, "Probability of one shot hitting: N/A")

        if self.data.prob_binomial is not None:
            _s(self.prob_binomial_label, f"Probability of reaching desired result: {self.data.prob_binomial * 100:.2f}%")
            _s(self.prob_lower_label, f"Lower Probability (95% confidence): {self.data.prob_binomial_lower_95 * 100:.2f}%")
            _s(self.prob_higher_label, f"Higher Probability (95% confidence): {self.data.prob_binomial_higher_95 * 100:.2f}%")
            _s(self.prob_lower_50_label, f"Lower Probability (50% confidence): {self.data.prob_binomial_lower_50 * 100:.2f}%")
            _s(self.prob_higher_50_label, f"Higher Probability (50% confidence): {self.data.prob_binomial_higher_50 * 100:.2f}%")
        else:
            _s(self.prob_binomial_label, "Probability of reaching desired result: N/A")
            _s(self.prob_lower_label, "Lower Probability (95% confidence): N/A")
            _s(self.prob_higher_label, "Higher Probability (95% confidence): N/A")
            _s(self.prob_lower_50_label, "Lower Probability (50% confidence): N/A")
            _s(self.prob_higher_50_label, "Higher Probability (50% confidence): N/A")

    def on_prob_input_change(self, *args):
        try:
            t = int(self.trials_var.get().strip())
        except:
            t = None
        try:
            h = int(self.hits_var.get().strip())
        except:
            h = None
        
        if t is not None:
            try:
                self.data.set_trials(t)
            except ValueError:
                pass
        if h is not None:
            try:
                self.data.set_hits(h)
            except ValueError:
                pass
        
        self.data.calculate_probabilities()
        self.update_metric_labels()

    def setup_visualization_plot(self):
        self.visualization_frame = tk.Frame(self.plots_frame)
        self.visualization_frame.pack(side='left', fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.density_frame = tk.Frame(self.plots_frame)
        self.density_frame.pack(side='right', fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig_vis, self.ax_vis = plt.subplots(figsize=(6,6))
        self.canvas_vis = FigureCanvasTkAgg(self.fig_vis, master=self.visualization_frame)
        self.canvas_vis.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_vis.get_tk_widget().config(height=500)
        self.ax_vis.set_title('Shot Distribution')
        self.ax_vis.set_xlabel('X (cm)')
        self.ax_vis.set_ylabel('Y (cm)')
        self.ax_vis.grid(True)
        self.fig_vis.tight_layout()
        self.canvas_vis.draw()

        self.fig_density, (self.ax_density_x, self.ax_density_y) = plt.subplots(2, 1, figsize=(6,6))
        self.canvas_density = FigureCanvasTkAgg(self.fig_density, master=self.density_frame)
        self.canvas_density.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_density.get_tk_widget().config(height=500)
        self.ax_density_x.set_title('X Coordinate Density')
        self.ax_density_x.set_xlabel('X (cm)')
        self.ax_density_x.set_ylabel('Density')
        self.ax_density_x.grid(True)

        self.ax_density_y.set_title('Y Coordinate Density')
        self.ax_density_y.set_xlabel('Y (cm)')
        self.ax_density_y.set_ylabel('Density')
        self.ax_density_y.grid(True)

        self.fig_density.tight_layout()
        self.canvas_density.draw()

    def update_visualization(self):
        # Distribution plot
        self.ax_vis.clear()
        self.ax_vis.set_title('Shot Distribution')
        self.ax_vis.set_xlabel('X (cm)')
        self.ax_vis.set_ylabel('Y (cm)')
        self.ax_vis.grid(True)

        shots = self.data.shots
        if shots:
            x_vals = [s[0] for s in shots]
            y_vals = [s[1] for s in shots]
            self.ax_vis.scatter(x_vals, y_vals, c='blue', label='Shots')

            if self.data.avg_coords:
                avg_x, avg_y = self.data.avg_coords
                self.ax_vis.scatter(avg_x, avg_y, c='green', marker='x', s=100, label='Center')

            self.ax_vis.legend()
            self.ax_vis.set_aspect('equal', adjustable='datalim')

        self.fig_vis.tight_layout()
        self.canvas_vis.draw()

        # Density plots (X & Y)
        self.ax_density_x.clear()
        self.ax_density_y.clear()

        if len(shots) > 2:
            from scipy.stats import gaussian_kde
            x_vals = [s[0] for s in shots]
            y_vals = [s[1] for s in shots]

            # Plot X density (skip KDE if all values identical)
            if len(set(x_vals)) > 1:
                kde_x = gaussian_kde(x_vals, bw_method='scott')
                x_range = np.linspace(min(x_vals) - 10, max(x_vals) + 10, 1000)
                y_kde_x = kde_x(x_range)
                self.ax_density_x.plot(x_range, y_kde_x, color='blue', lw=2)
                self.ax_density_x.fill_between(x_range, y_kde_x, color='skyblue', alpha=0.5)
            else:
                self.ax_density_x.axvline(x_vals[0], color='blue', lw=2, label=f'x = {x_vals[0]:.2f}')
                self.ax_density_x.legend()

            # Plot Y density (skip KDE if all values identical)
            if len(set(y_vals)) > 1:
                kde_y = gaussian_kde(y_vals, bw_method='scott')
                y_range = np.linspace(min(y_vals) - 10, max(y_vals) + 10, 1000)
                y_kde_y = kde_y(y_range)
                self.ax_density_y.plot(y_range, y_kde_y, color='blue', lw=2)
                self.ax_density_y.fill_between(y_range, y_kde_y, color='skyblue', alpha=0.5)
            else:
                self.ax_density_y.axvline(y_vals[0], color='blue', lw=2, label=f'y = {y_vals[0]:.2f}')
                self.ax_density_y.legend()

        self.ax_density_x.set_title('X Coordinate Density')
        self.ax_density_x.set_xlabel('X (cm)')
        self.ax_density_x.set_ylabel('Density')
        self.ax_density_x.grid(True)

        self.ax_density_y.set_title('Y Coordinate Density')
        self.ax_density_y.set_xlabel('Y (cm)')
        self.ax_density_y.set_ylabel('Density')
        self.ax_density_y.grid(True)

        self.fig_density.tight_layout()
        self.canvas_density.draw()

    # ----------------------------------------------------------------
    # ROLLING DISPARITY TAB
    # ----------------------------------------------------------------

    def build_disparity_ui(self, parent):
        # Controls row
        ctrl = tk.Frame(parent)
        ctrl.pack(pady=10, padx=10, fill=tk.X)

        tk.Label(ctrl, text="Window size (n):").pack(side='left', padx=5)
        self.disparity_n_var = tk.StringVar(value="10")
        tk.Entry(ctrl, textvariable=self.disparity_n_var, width=8).pack(side='left', padx=5)
        tk.Button(ctrl, text="Compute", command=self.on_compute_disparity).pack(side='left', padx=10)

        self.disparity_summary_var = tk.StringVar(value="")
        summary = tk.Entry(ctrl, textvariable=self.disparity_summary_var, state='readonly',
                           readonlybackground=self._label_bg, relief='flat',
                           borderwidth=0, highlightthickness=0, width=60)
        summary.pack(side='left', padx=10)
        summary._var = self.disparity_summary_var
        self.disparity_summary_label = summary

        # Results treeview
        results_frame = tk.Frame(parent)
        results_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        cols = ('rank', 'start', 'end', 'disparity', 'dist_to_mean')
        self.disp_tree = ttk.Treeview(results_frame, columns=cols, show='headings', height=20)
        self.disp_tree.heading('rank', text='Rank',
                               command=lambda: self._sort_disp_tree('rank'))
        self.disp_tree.heading('start', text='From shot #',
                               command=lambda: self._sort_disp_tree('start'))
        self.disp_tree.heading('end', text='To shot #',
                               command=lambda: self._sort_disp_tree('end'))
        self.disp_tree.heading('disparity', text='Disparity (cm)',
                               command=lambda: self._sort_disp_tree('disparity'))
        self.disp_tree.heading('dist_to_mean', text='Dist to mean (cm)',
                               command=lambda: self._sort_disp_tree('dist_to_mean'))
        self.disp_tree.column('rank', width=60, anchor='center')
        self.disp_tree.column('start', width=100, anchor='center')
        self.disp_tree.column('end', width=100, anchor='center')
        self.disp_tree.column('disparity', width=140, anchor='center')
        self.disp_tree.column('dist_to_mean', width=140, anchor='center')
        self._disp_sort_col = 'rank'
        self._disp_sort_rev = False
        self.disp_tree.pack(side='left', fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.disp_tree.yview)
        self.disp_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

        # Detail area: shows shots in the selected window
        detail_frame = tk.LabelFrame(parent, text="Window detail")
        detail_frame.pack(pady=10, padx=10, fill=tk.X)

        self.disp_detail_var = tk.StringVar(value="Select a row above to see the shots in that window.")
        detail_label = tk.Label(detail_frame, textvariable=self.disp_detail_var,
                                justify='left', anchor='w', wraplength=800)
        detail_label.pack(padx=10, pady=5, fill=tk.X)

        self.disp_tree.bind('<<TreeviewSelect>>', self.on_disparity_select)
        self._disparity_results = []

    def on_compute_disparity(self):
        try:
            n = int(self.disparity_n_var.get())
        except ValueError:
            messagebox.showerror("Input Error", "Window size must be an integer.")
            return
        if n < 2:
            messagebox.showerror("Input Error", "Window size must be at least 2.")
            return
        if len(self.data.shots) < n:
            messagebox.showwarning("Not enough data",
                                   f"Need at least {n} shots, have {len(self.data.shots)}.")
            return

        results = self.data.rolling_disparity(n)
        self._disparity_results = results

        # Clear tree
        for item in self.disp_tree.get_children():
            self.disp_tree.delete(item)

        # Populate
        for rank, r in enumerate(results, 1):
            self.disp_tree.insert('', 'end', values=(
                rank,
                r['start'] + 1,   # 1-based for display
                r['end'] + 1,
                f"{r['disparity']:.2f}",
                f"{r['dist_to_mean']:.2f}",
            ))

        worst = results[0]['disparity'] if results else 0
        best = results[-1]['disparity'] if results else 0
        self.disparity_summary_label._var.set(
            f"Windows: {len(results)}  |  "
            f"Worst: {worst:.2f} cm  |  Best: {best:.2f} cm"
        )
        self.disp_detail_var.set("Select a row above to see the shots in that window.")

    def on_disparity_select(self, event):
        sel = self.disp_tree.selection()
        if not sel:
            return
        item = sel[0]
        rank = int(self.disp_tree.set(item, 'rank'))
        r = self._disparity_results[rank - 1]
        lines = [f"Window shots #{r['start']+1}\u2013#{r['end']+1}  "
                 f"(disparity {r['disparity']:.2f} cm):"]
        for i, (x, y) in enumerate(r['shots']):
            shot_num = r['start'] + i + 1
            lines.append(f"  #{shot_num}: X={x:.2f}, Y={y:.2f}")
        self.disp_detail_var.set("\n".join(lines))

    def _sort_disp_tree(self, col):
        if self._disp_sort_col == col:
            self._disp_sort_rev = not self._disp_sort_rev
        else:
            self._disp_sort_col = col
            self._disp_sort_rev = False
        items = [(self.disp_tree.set(k, col), k) for k in self.disp_tree.get_children()]
        items.sort(key=lambda t: float(t[0]), reverse=self._disp_sort_rev)
        for i, (_, k) in enumerate(items):
            self.disp_tree.move(k, '', i)
        labels = {
            'rank': 'Rank', 'start': 'From shot #', 'end': 'To shot #',
            'disparity': 'Disparity (cm)', 'dist_to_mean': 'Dist to mean (cm)',
        }
        for c, base in labels.items():
            arrow = ''
            if c == col:
                arrow = ' \u25bc' if self._disp_sort_rev else ' \u25b2'
            self.disp_tree.heading(c, text=base + arrow)

    # ----------------------------------------------------------------
    # SECOND TAB: Compare Two Datasets
    # ----------------------------------------------------------------

    def build_compare_ui(self, parent):
        """Sets up the tab that allows comparing two data sets, each with their own radius."""
        # Radius entries for A and B
        tk.Label(parent, text="Radius for A:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.radiusA_var = tk.StringVar(value="15.0")
        radiusA_entry = tk.Entry(parent, textvariable=self.radiusA_var, width=10)
        radiusA_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        tk.Label(parent, text="Dataset A file:").grid(row=0, column=2, padx=5, pady=5, sticky='e')
        self.labelA = tk.Label(parent, text="(no file loaded)")
        self.labelA.grid(row=0, column=3, sticky='w', padx=5, pady=5)
        tk.Button(parent, text="Load A", command=self.on_load_A).grid(row=0, column=4, padx=5, pady=5)

        tk.Label(parent, text="Radius for B:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.radiusB_var = tk.StringVar(value="15.0")
        radiusB_entry = tk.Entry(parent, textvariable=self.radiusB_var, width=10)
        radiusB_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        tk.Label(parent, text="Dataset B file:").grid(row=1, column=2, padx=5, pady=5, sticky='e')
        self.labelB = tk.Label(parent, text="(no file loaded)")
        self.labelB.grid(row=1, column=3, sticky='w', padx=5, pady=5)
        tk.Button(parent, text="Load B", command=self.on_load_B).grid(row=1, column=4, padx=5, pady=5)

        tk.Button(parent, text="Compare", command=self.on_compare).grid(row=2, column=0, columnspan=5, pady=10)

        self.compare_result_label = self._selectable_label(parent, "Probability A > B: N/A",
                                                              row=3, column=0, columnspan=5, padx=10, pady=10)

    def on_load_A(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.compare_dataA.import_from_excel(path)
            self.compare_dataA.calculate_metrics()
            # require at least 2 shots
            if self.compare_dataA.stdX_dev is None or self.compare_dataA.stdY_dev is None:
                raise ValueError("Not enough shots to compute stdev in dataset A.")
            self.labelA.config(text=path)
        except Exception as e:
            messagebox.showerror("Error loading A", str(e))

    def on_load_B(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.compare_dataB.import_from_excel(path)
            self.compare_dataB.calculate_metrics()
            if self.compare_dataB.stdX_dev is None or self.compare_dataB.stdY_dev is None:
                raise ValueError("Not enough shots to compute stdev in dataset B.")
            self.labelB.config(text=path)
        except Exception as e:
            messagebox.showerror("Error loading B", str(e))

    def on_compare(self):
        # Set each dataset's radius from the user entry
        try:
            rA = float(self.radiusA_var.get())
            rB = float(self.radiusB_var.get())
            self.compare_dataA.set_radius(rA)
            self.compare_dataB.set_radius(rB)
        except ValueError:
            messagebox.showerror("Radius Error", "Radius must be numeric and > 0.")
            return

        if not self.compare_dataA.shots or not self.compare_dataB.shots:
            messagebox.showwarning("Missing data", "Please load both dataset A and B first.")
            return

        try:
            frac = compare_datasets(self.compare_dataA, self.compare_dataB,
                                    n_boot=10000, mc_size=10000)
            pct = frac * 100
            self._set_label_text(self.compare_result_label, f"Probability A > B: {pct:.2f}%")
        except Exception as e:
            messagebox.showerror("Comparison Error", str(e))


###############################################################################
# 3) CLI mode: text-based commands to do the same add/list/remove/import/export
###############################################################################

def cli_mode():
    print("Entering CLI mode. Type 'help' for commands, 'exit' to quit.")
    data = ShotsData(radius=15.0)

    def do_help():
        print("Available commands:")
        print("  help                       Show this help")
        print("  list                       Show all shots")
        print("  add <x> <y>               Add a shot")
        print("  remove <index>            Remove a shot by its index (from 'list')")
        print("  import <file.xlsx>        Import from Excel")
        print("  export <file.xlsx>        Export to Excel")
        print("  set radius <value>        Set radius (cm)")
        print("  set trials <N>            Set number of trials")
        print("  set hits <N>              Set number of hits")
        print("  calc                      Calculate all metrics & probabilities")
        print("  metrics                   Print out the latest metrics")
        print("  compare <A.xlsx> <B.xlsx> [rA] [rB]   Compare two datasets with optional radii")
        print("  exit                      Quit the program")

    def do_list():
        shots = data.list_shots()
        if not shots:
            print("No shots.")
        else:
            for i, x, y in shots:
                print(f"{i}: X={x}, Y={y}")
    
    def do_add(args):
        if len(args) < 2:
            print("Usage: add <x> <y>")
            return
        try:
            x = float(args[0])
            y = float(args[1])
            data.add_shot(x, y)
            print(f"Shot added: (X={x}, Y={y})")
        except ValueError:
            print("Invalid numeric values.")
    
    def do_remove(args):
        if not args:
            print("Usage: remove <index>")
            return
        try:
            index = int(args[0])
            data.remove_shot(index)
            print(f"Removed shot at index {index}")
        except ValueError:
            print("Index must be an integer.")
    
    def do_import(args):
        if not args:
            print("Usage: import <file.xlsx>")
            return
        path = " ".join(args)
        try:
            data.import_from_excel(path)
            print(f"Imported shots from {path}")
        except Exception as e:
            print(f"Import error: {e}")
    
    def do_export(args):
        if not args:
            print("Usage: export <file.xlsx>")
            return
        path = " ".join(args)
        try:
            data.export_to_excel(path)
            print(f"Exported shots to {path}")
        except Exception as e:
            print(f"Export error: {e}")
    
    def do_set(args):
        if len(args) < 2:
            print("Usage: set radius/trials/hits <value>")
            return
        field = args[0]
        value = args[1]
        try:
            val = float(value)
        except:
            print("Value must be numeric.")
            return
        if field == "radius":
            try:
                data.set_radius(val)
                print(f"Radius set to {val}")
            except ValueError as e:
                print(e)
        elif field == "trials":
            try:
                ival = int(value)
                data.set_trials(ival)
                print(f"Trials set to {ival}")
            except ValueError as e:
                print(e)
        elif field == "hits":
            try:
                ival = int(value)
                data.set_hits(ival)
                print(f"Hits set to {ival}")
            except ValueError as e:
                print(e)
        else:
            print(f"Unknown field '{field}'.")
    
    def do_calc():
        data.calculate_metrics()
        data.calculate_probabilities()
        print("Metrics & probabilities recalculated. Use 'metrics' to view.")
    
    def do_metrics():
        data.calculate_metrics()
        data.calculate_probabilities()
        
        if not data.shots:
            print("No shots to show metrics for.")
            return
        
        avg_coords = data.avg_coords
        print(f"Average coords: {avg_coords}")
        print(f"Std dev X={data.stdX_dev}, Y={data.stdY_dev}")
        
        if data.prob_hit_one_shot is not None:
            print(f"Probability of 1 shot hitting: {data.prob_hit_one_shot*100:.2f}%")
        if data.prob_binomial is not None:
            print(f"Probability of reaching desired result (binomial): {data.prob_binomial*100:.2f}%")
            print(f"95% CI: [{data.prob_binomial_lower_95*100:.2f}%, {data.prob_binomial_higher_95*100:.2f}%]")
            print(f"50% CI: [{data.prob_binomial_lower_50*100:.2f}%, {data.prob_binomial_higher_50*100:.2f}%]")

    def do_compare(args):
        """
        Usage: compare <fileA.xlsx> <fileB.xlsx> [rA] [rB]

        If rA/rB not provided, defaults to 15.
        """
        if len(args) < 2:
            print("Usage: compare <A.xlsx> <B.xlsx> [rA] [rB]")
            return

        fileA = args[0]
        fileB = args[1]
        radiusA = 15.0
        radiusB = 15.0
        if len(args) >= 3:
            try:
                radiusA = float(args[2])
            except ValueError:
                print("Invalid radiusA.")
                return
        if len(args) >= 4:
            try:
                radiusB = float(args[3])
            except ValueError:
                print("Invalid radiusB.")
                return
        
        dsA = ShotsData()
        dsB = ShotsData()
        try:
            dsA.import_from_excel(fileA)
            dsA.set_radius(radiusA)
            dsA.calculate_metrics()

            dsB.import_from_excel(fileB)
            dsB.set_radius(radiusB)
            dsB.calculate_metrics()
        except Exception as e:
            print(f"Error loading data: {e}")
            return
        
        # now compare
        try:
            frac = compare_datasets(dsA, dsB, n_boot=10000, mc_size=10000)
            print(f"Probability dataset A (radius={radiusA}) is better than B (radius={radiusB}): {frac*100:.2f}%")
        except Exception as e:
            print(f"Comparison error: {e}")

    do_help()
    while True:
        try:
            line = input("cli> ").strip()
        except EOFError:
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]
        
        if cmd == 'exit':
            print("Goodbye!")
            break
        elif cmd == 'help':
            do_help()
        elif cmd == 'list':
            do_list()
        elif cmd == 'add':
            do_add(args)
        elif cmd == 'remove':
            do_remove(args)
        elif cmd == 'import':
            do_import(args)
        elif cmd == 'export':
            do_export(args)
        elif cmd == 'set':
            do_set(args)
        elif cmd == 'calc':
            do_calc()
        elif cmd == 'metrics':
            do_metrics()
        elif cmd == 'compare':
            do_compare(args)
        else:
            print(f"Unknown command: {cmd}. Type 'help' for a list.")


###############################################################################
# 4) main(): Decide whether to run the GUI or CLI based on arguments
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of GUI")
    args = parser.parse_args()

    if args.cli:
        cli_mode()
    else:
        root = tk.Tk()
        root.state('zoomed')  # Maximize window on Windows
        app = ShotAccuracyApp(root)
        root.mainloop()

if __name__ == "__main__":
    main()
