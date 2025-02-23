import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
import logging
from flask import Flask, request, jsonify, send_file, make_response
import base64
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time
import sys
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import textwrap
import seaborn as sns
import json
from io import BytesIO
from reportlab.lib.image import ImageReader

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes and origins

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Google API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Google API configured successfully")
else:
    logger.warning("No Google API key found")

# Create download directory if it doesn't exist
if not os.path.exists('download'):
    os.makedirs('download')

@app.after_request
def after_request(response):
    """Add CORS headers to all responses."""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def generate_research_paper(topic):
    """Generate research paper content using Google's Gemini API."""
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""Generate a comprehensive research paper on the topic: {topic}
        Include the following sections:
        1. Abstract
        2. Introduction
        3. Literature Review
        4. Methodology
        5. Results and Discussion
        6. Conclusion
        7. References

        Format the paper in markdown with appropriate headings (#, ##) for each section.
        Make it detailed and academically rigorous."""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating research paper: {str(e)}")
        return None

def generate_graph(topic):
    """Generate a graph based on the topic."""
    try:
        # Create sample data for visualization
        years = pd.date_range(start='2020', end='2024', freq='Y')
        data = np.random.normal(10, 2, size=len(years))
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        plt.plot(years, data, marker='o')
        
        # Customize the plot
        plt.title(f'Trend Analysis: {topic}')
        plt.xlabel('Year')
        plt.ylabel('Impact Score')
        
        # Save plot to BytesIO
        img_stream = BytesIO()
        plt.savefig(img_stream, format='png', bbox_inches='tight')
        img_stream.seek(0)
        graph_data = base64.b64encode(img_stream.read()).decode('utf-8')
        
        plt.close()
        return graph_data
    except Exception as e:
        logger.error(f"Error generating graph: {str(e)}")
        return None

def create_pdf(topic, content):
    """Create a PDF file with the research paper content and graph."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"download/research_paper_{timestamp}.pdf"
        
        # Create PDF
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        
        # Title
        c.setFont("Helvetica-Bold", 24)
        c.drawString(72, height - 72, f"Research Paper: {topic}")
        
        # Date
        c.setFont("Helvetica", 12)
        c.drawString(72, height - 100, f"Generated on: {datetime.now().strftime('%B %d, %Y')}")
        
        # Content
        y = height - 150
        c.setFont("Helvetica", 12)
        for line in content.split('\n'):
            if y < 72:  # Start new page if near bottom
                c.showPage()
                y = height - 72
            
            # Handle section titles
            if line.startswith('#'):
                c.setFont("Helvetica-Bold", 14)
                line = line.lstrip('#').strip()
            else:
                c.setFont("Helvetica", 12)
            
            # Wrap text to fit page width
            wrapped_text = textwrap.wrap(line, width=80)
            for text in wrapped_text:
                c.drawString(72, y, text)
                y -= 20

        # Add graph on a new page
        c.showPage()
        
        # Generate graph
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        years = pd.date_range(start='2020', end='2024', freq='Y')
        data = np.random.normal(10, 2, size=len(years))
        plt.plot(years, data, marker='o')
        plt.title(f'Trend Analysis: {topic}')
        plt.xlabel('Year')
        plt.ylabel('Impact Score')
        
        # Save graph to temporary file
        temp_graph = BytesIO()
        plt.savefig(temp_graph, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        temp_graph.seek(0)
        
        # Add graph title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height - 72, "Research Analysis Graph")
        
        # Draw graph on PDF
        c.drawImage(ImageReader(temp_graph), 72, height - 500, width=450, height=350)
        
        c.save()
        return filename
    except Exception as e:
        logger.error(f"Error creating PDF: {str(e)}")
        return None

@app.route('/generate', methods=['POST'])
def generate():
    """Generate endpoint with improved error handling and response formatting."""
    try:
        # Get topic
        data = request.get_json()
        if not data or not data.get("topic"):
            return jsonify({"error": "Topic is required", "success": False}), 400
        
        topic = data["topic"].strip()
        if not topic:
            return jsonify({"error": "Empty topic provided", "success": False}), 400
            
        logger.info(f"Starting generation for topic: {topic}")
        
        # Generate research paper
        content = generate_research_paper(topic)
        if not content:
            return jsonify({
                "error": "Failed to generate content",
                "success": False
            }), 500
        
        # Generate graph
        graph_data = generate_graph(topic)
        
        # Create PDF
        pdf_file = create_pdf(topic, content)
        if not pdf_file:
            return jsonify({
                "error": "Failed to create PDF file",
                "success": False,
                "paper_content": content
            }), 500
            
        # Get PDF data
        try:
            with open(pdf_file, 'rb') as f:
                pdf_data = base64.b64encode(f.read()).decode('utf-8')
                
            response_data = {
                "success": True,
                "paper_content": content,
                "pdf_file": os.path.basename(pdf_file),
                "pdf_data": pdf_data,
                "graph_data": graph_data
            }
            
            logger.info("Successfully generated research paper and PDF")
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error reading PDF file: {str(e)}")
            return jsonify({
                "error": "Error processing PDF file",
                "success": False,
                "paper_content": content
            }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in generate endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download endpoint for PDF files."""
    try:
        return send_file(
            f"download/{filename}",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    logger.info("Starting server on all interfaces (0.0.0.0:5000)")
    app.run(host='0.0.0.0', port=5000, debug=True) 