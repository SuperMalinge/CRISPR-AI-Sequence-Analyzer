import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from Bio import SeqIO
from Bio.Seq import Seq

class CRISPRPredictor(tf.keras.Model):
    def __init__(self):
        super(CRISPRPredictor, self).__init__()
        self.sequence_encoder = tf.keras.layers.LSTM(128, return_sequences=True)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=16)
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(4)

    def call(self, inputs):
        x = self.sequence_encoder(inputs)
        attention_output = self.attention(x, x)
        x = tf.concat([x, attention_output], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

class CRISPRGui:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CRISPR-AI Advanced Sequence Analyzer")
        self.root.geometry("1400x800")
        self.model = CRISPRPredictor()
        self.setup_gui()

    def setup_gui(self):
        # Main container with scrolling
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Input Tab
        self.create_input_tab()
        
        # Analysis Tab
        self.create_analysis_tab()
        
        # Results Tab
        self.create_results_tab()

    def create_input_tab(self):
        input_frame = ttk.Frame(self.notebook)
        self.notebook.add(input_frame, text="Sequence Input")

        # Sequence input section
        input_section = ttk.LabelFrame(input_frame, text="DNA Sequence Input")
        input_section.pack(fill=tk.X, padx=10, pady=5)

        self.sequence_input = tk.Text(input_section, height=6, width=70)
        self.sequence_input.pack(padx=5, pady=5)

        # Control buttons
        btn_frame = ttk.Frame(input_section)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(btn_frame, text="Analyze Sequence", 
                  command=self.analyze_sequence).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear", 
                  command=lambda: self.sequence_input.delete(1.0, tk.END)).pack(side=tk.LEFT)

        # Parameters section
        param_section = ttk.LabelFrame(input_frame, text="Analysis Parameters")
        param_section.pack(fill=tk.X, padx=10, pady=5)

        # Add parameter controls
        self.create_parameter_controls(param_section)

    def create_analysis_tab(self):
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="Analysis")

        # Create visualization panels with descriptive headers
        left_panel = ttk.LabelFrame(analysis_frame, text="Cutting Efficiency Analysis")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        right_panel = ttk.LabelFrame(analysis_frame, text="Off-Target Risk Assessment")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add description labels
        ttk.Label(left_panel, text="This graph shows the predicted cutting efficiency along the DNA sequence.\nHigher values indicate better target sites.", 
                wraplength=300).pack(pady=5)

        ttk.Label(right_panel, text="This heatmap displays potential off-target binding risks.\nDarker colors indicate higher risk regions.", 
                wraplength=300).pack(pady=5)

        # Efficiency plot with enhanced labels
        self.fig1, self.ax1 = plt.subplots(figsize=(6, 4))
        self.ax1.set_title('CRISPR Cutting Efficiency Profile', fontsize=12)
        self.ax1.set_xlabel('Position in Target Sequence (bp)', fontsize=10)
        self.ax1.set_ylabel('Efficiency Score (0-1)', fontsize=10)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, left_panel)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Off-target plot with enhanced labels
        self.fig2, self.ax2 = plt.subplots(figsize=(6, 4))
        self.ax2.set_title('Off-Target Risk Analysis', fontsize=12)
        self.ax2.set_xlabel('Sequence Position (bp)', fontsize=10)
        self.ax2.set_ylabel('Risk Level', fontsize=10)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, right_panel)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


    def create_results_tab(self):
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")

        # Results text area
        self.results_text = tk.Text(results_frame, height=20, width=70)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def create_parameter_controls(self, parent):
        param_frame = ttk.Frame(parent)
        param_frame.pack(fill=tk.X, padx=5, pady=5)

        # Efficiency threshold control
        eff_frame = ttk.LabelFrame(param_frame, text="Efficiency Threshold (0.0 - 1.0)")
        eff_frame.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        
        self.eff_value = tk.StringVar(value="0.7")
        self.efficiency_threshold = ttk.Scale(eff_frame, from_=0, to=1, orient=tk.HORIZONTAL,
                                            command=lambda x: self.eff_value.set(f"{float(x):.2f}"))
        self.efficiency_threshold.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(eff_frame, textvariable=self.eff_value).pack(side=tk.LEFT, padx=5)
        ttk.Label(eff_frame, text="Controls minimum acceptable cutting efficiency\nHigher values = stricter selection",
                wraplength=200).pack(pady=5)

        # Off-target tolerance control
        off_frame = ttk.LabelFrame(param_frame, text="Off-target Tolerance (0.0 - 1.0)")
        off_frame.grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        
        self.off_value = tk.StringVar(value="0.3")
        self.offtarget_tolerance = ttk.Scale(off_frame, from_=0, to=1, orient=tk.HORIZONTAL,
                                        command=lambda x: self.off_value.set(f"{float(x):.2f}"))
        self.offtarget_tolerance.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(off_frame, textvariable=self.off_value).pack(side=tk.LEFT, padx=5)
        ttk.Label(off_frame, text="Sets maximum allowed off-target effects\nLower values = higher specificity",
                wraplength=200).pack(pady=5)


    def analyze_sequence(self):
        sequence = self.sequence_input.get("1.0", tk.END).strip()
        if not sequence:
            messagebox.showwarning("Input Required", "Please enter a DNA sequence.")
            return

        # Process sequence
        seq_numeric = self.encode_sequence(sequence)
        predictions = self.model(tf.expand_dims(seq_numeric, 0))
        
        # Update visualizations
        self.update_visualizations(sequence, predictions)
        self.update_results(predictions)
        
        # Switch to analysis tab
        self.notebook.select(1)

    def encode_sequence(self, sequence):
        encoding = {'A': [1,0,0,0], 'T': [0,1,0,0], 
                   'G': [0,0,1,0], 'C': [0,0,0,1]}
        return np.array([encoding.get(base.upper(), [0,0,0,0]) for base in sequence])

    def update_visualizations(self, sequence, predictions):
        # Update efficiency plot
        self.ax1.clear()
        efficiency_scores = predictions[0, :, 0].numpy()
        self.ax1.plot(efficiency_scores, 'b-', linewidth=2)
        self.ax1.set_title('CRISPR Cutting Efficiency Profile', fontsize=12, pad=10)
        self.ax1.set_xlabel('Position in Target Sequence (bp)', fontsize=10)
        self.ax1.set_ylabel('Predicted Efficiency Score (0-1)', fontsize=10)
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add explanation text box
        self.ax1.text(0.05, 0.95, 
            'Higher values indicate better cutting efficiency\nOptimal targets shown in peak regions', 
            transform=self.ax1.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))
        self.canvas1.draw()

        # Update off-target plot
        self.ax2.clear()
        off_target_matrix = predictions[0, :, 2].numpy().reshape(-1, 1)
        sns.heatmap(off_target_matrix, ax=self.ax2, cmap='YlOrRd',
                    xticklabels=False, cbar_kws={'label': 'Risk Level (0-1)'})
        self.ax2.set_title('Off-target Risk Analysis', fontsize=12, pad=10)
        self.ax2.set_xlabel('Target Position (bp)', fontsize=10)
        self.ax2.set_ylabel('Risk Score', fontsize=10)
        
        # Add explanation text box
        self.ax2.text(1.15, 0.5, 
            'Darker colors indicate\nhigher off-target risk\nAvoid regions with\nintense red coloring', 
            transform=self.ax2.transAxes,
            bbox=dict(facecolor='white', alpha=0.8))
        self.canvas2.draw()


    def update_results(self, predictions):
        mean_efficiency = tf.reduce_mean(predictions[0, :, 0])
        mean_specificity = tf.reduce_mean(predictions[0, :, 1])
        max_off_target = tf.reduce_max(predictions[0, :, 2])
        mod_success = tf.reduce_mean(predictions[0, :, 3])

        results = f"""
        CRISPR-Cas9 Analysis Results
        ===========================

        Efficiency Metrics:
        • Mean Cutting Efficiency: {mean_efficiency:.3f}
        • Target Specificity: {mean_specificity:.3f}
        • Maximum Off-target Risk: {max_off_target:.3f}
        • Modification Success Probability: {mod_success:.3f}

        Recommendations:
        • {'High efficiency target' if mean_efficiency > 0.7 else 'Consider alternative target site'}
        • {'Low off-target risk' if max_off_target < 0.3 else 'High off-target risk - verify specificity'}
        • {'Good modification potential' if mod_success > 0.6 else 'Limited modification potential'}

        Additional Notes:
        • Sequence length analyzed: {predictions.shape[1]} bp
        • Analysis completed with current parameter settings
        """

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = CRISPRGui()
    app.run()
