from preprocessor.preprocessor import Preprocessor
from ml_analyzers.ml_text_analyzer import MLTextAnalyzer  
from content_analyzers.audio_analyzer import AudioAnalyzer
from content_analyzers.text_analyzer import TextAnalyzer
from content_analyzers.image_analyzer import ImageAnalyzer
from content_analyzers.video_analyzer import VideoAnalyzer
from aggregator.aggregator import Aggregator
from engines.decision_engine import DecisionEngine
from blockchain.blockchain import Blockchain
from utils.content_type import ContentType

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk 

# Import ttkbootstrap for modern styling
import ttkbootstrap as tb
from ttkbootstrap.constants import *

# For logging
import logging

# Configure logging
logging.basicConfig(
    filename='ai_content_detector.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Initialize global variables for predictions, true labels, and probabilities
# (Note: Consider encapsulating these within classes to avoid using globals)
predictions = []
true_labels = []
probabilities = []

def process_and_analyze_content(content_path_or_url, is_ai_generated=False, ml_text_analyzer=None):
    """
    Processes the content and analyzes it using the appropriate analyzers based on content type.
    
    Parameters:
        content_path_or_url (str): Path or URL to the content.
        is_ai_generated (bool): Indicates if the content is AI-generated (for true labels).
        ml_text_analyzer (MLTextAnalyzer): Instance of MLTextAnalyzer.
    
    Returns:
        dict: Aggregated results, blockchain verification, and decision.
    """
    global predictions, true_labels, probabilities
    
    try:
        # Preprocessing
        preprocessor = Preprocessor()
        content = preprocessor.process(content_path_or_url)  # Returns dict with 'data' and 'type'
        if content is None:
            logging.warning(f"Failed to preprocess content: {content_path_or_url}")
            return None

        content_type = content['type']
        reports = []

        # Analyzers
        if content_type == ContentType.IMAGE:
            image_analyzer = ImageAnalyzer()
            analyzer_report = image_analyzer.analyze(content['data'])
            reports.append(analyzer_report)
        elif content_type == ContentType.VIDEO:
            video_analyzer = VideoAnalyzer()
            analyzer_report = video_analyzer.analyze(content['data'])
            reports.append(analyzer_report)
        elif content_type == ContentType.AUDIO:
            audio_analyzer = AudioAnalyzer()
            analyzer_report = audio_analyzer.analyze(content['data'])
            reports.append(analyzer_report)
        elif content_type == ContentType.TEXT:
            # Text Analysis
            text_analyzer = TextAnalyzer(corpus=content['data'])
            analyzer_report = text_analyzer.analyze(content['data'])
            reports.append(analyzer_report)

            # Machine Learning Analysis
            if ml_text_analyzer is None:
                logging.error("MLTextAnalyzer instance is not provided.")
                return None
            ml_text_report = ml_text_analyzer.analyze_text(content['data'], analyzer_report)
            reports.append(ml_text_report)
            
            # Store prediction and true label
            prediction_label = ml_text_report.get('ml_text_ai_generated_label', 'Human-Written')
            prediction = 1 if prediction_label == 'AI-Generated' else 0
            true_label = 1 if is_ai_generated else 0
            
            # Store predicted probability
            probability = ml_text_report.get('ml_text_ai_generated_probability', 0.0)
            
            # Pass to MLTextAnalyzer for metric calculation
            ml_text_analyzer.store_prediction(true_label, prediction, probability)
                
        else:
            logging.warning(f"Unsupported content type: {content_type}")
            return None

        # Aggregation
        aggregator = Aggregator()
        aggregated_results = aggregator.aggregate(reports)

        # Blockchain Verification
        blockchain = Blockchain()
        blockchain_result = blockchain.blockchain_verification(content)
        
        # Decision Engine
        decision_engine = DecisionEngine()
        decision = decision_engine.make_decision(aggregated_results)

        # Compile Final Results
        final_results = {
            'aggregated_results': aggregated_results,
            'blockchain_result': blockchain_result,
            'decision': decision,
        }

        logging.info(f"Successfully processed content: {content_path_or_url}")
        return final_results

    except Exception as e:
        logging.error(f"Error processing content {content_path_or_url}: {e}")
        return None

class DragDropGUI(tb.Window):
    def __init__(self):
        super().__init__(themename="darkly")  # Set the ttkbootstrap theme to 'darkly'
        self.title("AI Content Detector")
        self.geometry("1000x700")  # Adjusted size for better layout

        # Set window transparency (0.0 to 1.0)
        self.attributes('-alpha', 0.90)  # More transparent

        # Initialize Notebook (Tabbed Interface)
        self.notebook = tb.Notebook(self, bootstyle="dark")
        self.notebook.pack(fill='both', expand=True)

        # Define Tabs
        self.upload_tab = tb.Frame(self.notebook, padding=20)
        self.results_tab = tb.Frame(self.notebook, padding=20)

        self.notebook.add(self.upload_tab, text="Upload")
        self.notebook.add(self.results_tab, text="Results")

        # Initialize MLTextAnalyzer once
        try:
            self.ml_text_analyzer = MLTextAnalyzer()
        except Exception as e:
            logging.error(f"Failed to initialize MLTextAnalyzer: {e}")
            self.show_error_message("Failed to initialize MLTextAnalyzer.")
            self.ml_text_analyzer = None

        # Setup Tabs
        self.setup_upload_tab()
        self.setup_results_tab()

    def setup_upload_tab(self):
        """
        Sets up the Upload tab with a label and a button to select files.
        """
        try:
            # Upload Label
            upload_label = tb.Label(
                self.upload_tab,
                text="Analyze file with the button below",
                font=("Helvetica", 16),
                bootstyle="inverse",
                padding=10,
                background="#222222",
                foreground="white"
            )
            upload_label.pack(pady=50)

            # Select File Button
            select_button = tb.Button(
                self.upload_tab,
                text="Analyze File",
                bootstyle="success-outline",
                command=self.select_file
            )
            select_button.pack(pady=20)
        except Exception as e:
            logging.error(f"Error setting up Upload tab: {e}")
            self.show_error_message("Failed to set up Upload tab.")

    def setup_results_tab(self):
        """
        Sets up the Results tab to display analysis results in tables.
        """
        try:
            # Create a canvas and scrollbar
            self.results_canvas = tk.Canvas(self.results_tab, bg='#2e2e2e', highlightthickness=0)
            self.results_scrollbar = ttk.Scrollbar(self.results_tab, orient="vertical", command=self.results_canvas.yview)
            self.results_scrollable_frame = ttk.Frame(self.results_canvas)

            # Bind the scrollable frame to the canvas
            self.results_scrollable_frame.bind(
                "<Configure>",
                lambda e: self.results_canvas.configure(
                    scrollregion=self.results_canvas.bbox("all")
                )
            )

            self.results_canvas.create_window((0, 0), window=self.results_scrollable_frame, anchor="nw")
            self.results_canvas.configure(yscrollcommand=self.results_scrollbar.set)

            # Pack canvas and scrollbar
            self.results_canvas.pack(side="left", fill="both", expand=True)
            self.results_scrollbar.pack(side="right", fill="y")

            # Default message
            self.results_default_label = tb.Label(
                self.results_scrollable_frame,
                text="No data available yet. Please analyze some files first.",
                font=("Helvetica", 12),
                bootstyle="secondary",
                anchor='center',
                justify='center'
            )
            self.results_default_label.pack(pady=200)
        except Exception as e:
            logging.error(f"Error setting up Results tab: {e}")
            self.show_error_message("Failed to set up Results tab.")

    def select_file(self):
        """
        Opens a file dialog to select a file and processes it.
        """
        try:
            file_path = filedialog.askopenfilename()
            if file_path:
                if self.ml_text_analyzer is None:
                    self.show_error_message("MLTextAnalyzer is not initialized.")
                    return

                # Reset metrics before new analysis
                self.ml_text_analyzer.reset_metrics()

                # Process and analyze content
                results = process_and_analyze_content(
                    file_path, 
                    is_ai_generated=False,  # Adjust based on your logic
                    ml_text_analyzer=self.ml_text_analyzer
                )
                if results is None:
                    logging.warning(f"Analysis could not be completed for file: {file_path}")
                    self.show_error_message("Analysis could not be completed.")
                    return
                # Update Results Tab
                self.update_results_tab(results)
                # Optionally, switch to Results tab
                self.notebook.select(self.results_tab)
                logging.info(f"Successfully analyzed file: {file_path}")
        except Exception as e:
            logging.error(f"Error during file selection and analysis: {e}")
            self.show_error_message("An unexpected error occurred during analysis.")

    def update_results_tab(self, results):
        """
        Updates the Results tab with the latest analysis results in table format.
        """
        try:
            # Clear previous content
            for widget in self.results_scrollable_frame.winfo_children():
                widget.destroy()

            # Check if no content has been analyzed
            if not results:
                self.results_default_label = tb.Label(
                    self.results_scrollable_frame,
                    text="No data available yet. Please analyze some files first.",
                    font=("Helvetica", 12),
                    bootstyle="secondary",
                    anchor='center',
                    justify='center'
                )
                self.results_default_label.pack(pady=200)
                return

            # Display Decision
            decision = results.get('decision', 'N/A')
            ai_probability = results['aggregated_results']['ml_text_ai_generated_probability']
            
            decision_label = tb.Label(
                self.results_scrollable_frame,
                text=f"Decision: {decision} ({ai_probability:.4f}%)",
                font=("Helvetica", 16, 'bold'),
                bootstyle="warning",
                anchor='w'
            )
            decision_label.pack(anchor='w', padx=10, pady=10)

            # Separator
            separator = ttk.Separator(self.results_scrollable_frame, orient='horizontal')
            separator.pack(fill='x', padx=10, pady=10)

            # Display Aggregated Results in a Table
            agg_results = results.get('aggregated_results', {})
            agg_label = tb.Label(
                self.results_scrollable_frame,
                text="Aggregated Results:",
                font=("Helvetica", 14, 'bold'),
                bootstyle="info",
                anchor='w'
            )
            agg_label.pack(anchor='w', padx=10, pady=5)

            # **Create a Frame for Aggregated Results Table and Scrollbar**
            agg_frame = tb.Frame(self.results_scrollable_frame)
            agg_frame.pack(padx=20, pady=10, fill='x')

            # Create Treeview for Aggregated Results
            agg_tree = ttk.Treeview(agg_frame, columns=("Metric", "Value"), show='headings', height=10)
            agg_tree.heading("Metric", text="Metric")
            agg_tree.heading("Value", text="Value")

            # **Set column widths and disable stretching**
            agg_tree.column("Metric", width=200)
            agg_tree.column("Value", width=500) 

            # Insert data into Treeview
            for key, value in agg_results.items():
                agg_tree.insert("", "end", values=(key, value))

            # Style the Treeview
            style = ttk.Style(self)
            style.configure("Treeview",
                            background="#2e2e2e",
                            foreground="white",
                            fieldbackground="#2e2e2e",
                            rowheight=25)
            style.map("Treeview",
                    background=[("selected", "#4d4d4d")],
                    foreground=[("selected", "white")])

            # **Add Horizontal Scrollbar to Aggregated Results**
            agg_h_scrollbar = ttk.Scrollbar(agg_frame, orient="horizontal", command=agg_tree.xview)
            agg_tree.configure(xscrollcommand=agg_h_scrollbar.set)
            agg_h_scrollbar.pack(side='bottom', fill='x')

            # Pack the Aggregated Results Treeview
            agg_tree.pack(side='top', fill='x')

            # Separator
            separator2 = ttk.Separator(self.results_scrollable_frame, orient='horizontal')
            separator2.pack(fill='x', padx=10, pady=10)

            # Display Blockchain Verification in a Table
            blockchain_result = results.get('blockchain_result', 'N/A')
            blockchain_label = tb.Label(
                self.results_scrollable_frame,
                text="Blockchain Verification:",
                font=("Helvetica", 14, 'bold'),
                bootstyle="info",
                anchor='w'
            )
            blockchain_label.pack(anchor='w', padx=10, pady=5)

            # **Create a Frame for Blockchain Verification Table and Scrollbar**
            blockchain_frame = tb.Frame(self.results_scrollable_frame)
            blockchain_frame.pack(padx=20, pady=10, fill='x')

            # Create Treeview for Blockchain Verification
            blockchain_tree = ttk.Treeview(blockchain_frame, columns=("Verification", "Status"), show='headings', height=5)
            blockchain_tree.heading("Verification", text="Verification")
            blockchain_tree.heading("Status", text="Status")

            # **Set column widths and disable stretching**
            blockchain_tree.column("Verification", width=300)
            blockchain_tree.column("Status", width=500)

            # Insert data into Treeview
            if isinstance(blockchain_result, dict):
                for key, value in blockchain_result.items():
                    blockchain_tree.insert("", "end", values=(key, value))
            else:
                blockchain_tree.insert("", "end", values=("Verification", blockchain_result))

            # Style the Treeview
            blockchain_tree.configure(style="Treeview")

            # **Add Horizontal Scrollbar to Blockchain Verification**
            blockchain_h_scrollbar = ttk.Scrollbar(blockchain_frame, orient="horizontal", command=blockchain_tree.xview)
            blockchain_tree.configure(xscrollcommand=blockchain_h_scrollbar.set)
            blockchain_h_scrollbar.pack(side='bottom', fill='x')

            # Pack the Blockchain Verification Treeview
            blockchain_tree.pack(side='top', fill='x')

        except Exception as e:
            logging.error(f"Error updating Results tab: {e}")
            self.show_error_message("Failed to update Results tab.")

    def show_error_message(self, message):
        """
        Displays an error message in a popup window.
        """
        try:
            error_window = tb.Toplevel(self)
            error_window.title("Error")
            error_window.geometry("400x200")
            error_window.resizable(False, False)
            error_window.configure(bg='#2e2e2e')  # Match dark theme

            # Style for error message
            error_label = tb.Label(
                error_window, 
                text=message, 
                font=("Helvetica", 12),
                bootstyle="danger",
                wraplength=350,
                anchor='center',
                justify='center'
            )
            error_label.pack(padx=20, pady=20)

            ok_button = tb.Button(
                error_window, 
                text="OK", 
                bootstyle="primary-outline",
                command=error_window.destroy
            )
            ok_button.pack(pady=10)
        except Exception as e:
            logging.error(f"Error displaying error message: {e}")
            print(f"Error displaying error message: {e}")

def main():
    try:
        app = DragDropGUI()
        app.mainloop()
    except Exception as e:
        logging.error(f"Error running the application: {e}")
        print(f"Error running the application: {e}")

if __name__ == "__main__":
    main()
