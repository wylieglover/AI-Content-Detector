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

def process_and_analyze_content(content_path_or_url):
    """
    Processes the content and analyzes it using the appropriate analyzers based on content type.
    """
    # Preprocessing
    preprocessor = Preprocessor()
    content = preprocessor.process(content_path_or_url) # content['data'], content['type']
    if content is None:
        print("Failed to preprocess content.")
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
        text_analyzer = TextAnalyzer()
        analyzer_report = text_analyzer.analyze(content['data'])
        reports.append(analyzer_report)

        # Machine Learning Analysis
        ml_text_analyzer = MLTextAnalyzer()
        ml_text_report = ml_text_analyzer.analyze_text(content['data'], analyzer_report)
        reports.append(ml_text_report)
    else:
        print(f"Unsupported content type: {content_type}")
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

    return final_results

class DragDropGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Content Detector")
        self.geometry("800x600")

        self.label = tk.Label(
            self,
            text="Click the button below to select a file",
            width=40,
            height=10,
            borderwidth=2,
            relief="groove"
        )
        self.label.pack(expand=True)

        self.button = tk.Button(self, text="Select File", command=self.select_file)
        self.button.pack(pady=10)

    def select_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            results = process_and_analyze_content(file_path)
            if results is None:
                print("Analysis could not be completed.")
                return
            self.display_results(results)
            
    def display_results(self, results):
        # Extract results
        aggregated_results = results['aggregated_results']
        blockchain_result = results['blockchain_result']
        decision = results['decision']

        # Create a new window to display results
        result_window = tk.Toplevel(self)
        result_window.title("Analysis Results")
        result_window.geometry("800x600")  # Adjust the size as needed

        # Create a canvas and scrollbar
        canvas = tk.Canvas(result_window)
        scrollbar = ttk.Scrollbar(result_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        # Bind the scrollable frame to the canvas
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Apply styles
        style = ttk.Style()
        style.configure("TLabel", font=('Arial', 10))
        style.configure("Header.TLabel", font=('Arial', 12, 'bold'))
        style.configure("Title.TLabel", font=('Arial', 14, 'bold'))

        # Display Decision at the top with sufficient padding
        decision_label = ttk.Label(
            scrollable_frame,
            text=f"Decision: {decision}",
            style="Header.TLabel",
            justify='left',
            anchor='w'
        )
        decision_label.pack(anchor='w', padx=10, pady=10)  # Increase top padding

        separator = ttk.Separator(scrollable_frame, orient='horizontal')
        separator.pack(fill='x', padx=10, pady=10)

        # Display Aggregated Results
        agg_label = ttk.Label(
            scrollable_frame,
            text="Aggregated Results:",
            style="Header.TLabel",
            anchor='w'
        )
        agg_label.pack(anchor='w', padx=10, pady=10)

        for key, value in aggregated_results.items():
            result = ttk.Label(scrollable_frame, text=f"{key}: {value}", anchor='w')
            result.pack(anchor='w', padx=20)

        # Add another separator
        separator2 = ttk.Separator(scrollable_frame, orient='horizontal')
        separator2.pack(fill='x', padx=10, pady=10)

        # Display Blockchain Verification
        blockchain_label = ttk.Label(
            scrollable_frame,
            text="Blockchain Verification:",
            style="Header.TLabel",
            anchor='w'
        )
        blockchain_label.pack(anchor='w', padx=10, pady=10)

        blockchain_result_label = ttk.Label(
            scrollable_frame,
            text=blockchain_result,
            wraplength=450,
            anchor='w',
            justify='left'
        )
        blockchain_result_label.pack(anchor='w', padx=20)

def main():
    app = DragDropGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
