#!/usr/bin/env python3
"""
Comprehensive Intent Analysis for CEO Messages in Airport Sustainability PDFs

This script extracts and analyzes CEO messages from airport sustainability reports
to identify key intents and sustainability commitments.
"""

import os
import re
import json
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# PDF Processing
import PyPDF2
import pdfplumber

# NLP and ML
import nltk
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect

# Data Processing and Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

class AirportCEOIntentAnalyzer:
    """
    Comprehensive analyzer for CEO messages in airport sustainability PDFs
    """
    
    def __init__(self):
        """Initialize the analyzer with NLP models and configurations"""
        self.setup_logging()
        self.load_models()
        self.setup_intent_categories()
        self.setup_sustainability_keywords()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ceo_intent_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_models(self):
        """Load required NLP models"""
        self.logger.info("Loading NLP models...")
        
        try:
            # Load spaCy model for named entity recognition
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize transformers pipeline for zero-shot classification
        try:
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                return_all_scores=True
            )
        except Exception as e:
            self.logger.warning(f"Could not load zero-shot classifier: {e}")
            self.zero_shot_classifier = None
            
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
    def setup_intent_categories(self):
        """Define intent categories for CEO messages"""
        self.intent_categories = {
            "commitment": [
                "commitment to sustainability",
                "environmental pledge",
                "promise to reduce emissions",
                "dedication to green practices",
                "vow to improve environment"
            ],
            "vision": [
                "future vision",
                "strategic outlook",
                "long-term goals",
                "environmental future",
                "sustainable development vision"
            ],
            "achievement": [
                "sustainability accomplishments",
                "environmental achievements",
                "carbon reduction success",
                "green milestones",
                "environmental progress"
            ],
            "investment": [
                "sustainability investment",
                "green technology funding",
                "environmental spending",
                "renewable energy investment",
                "infrastructure modernization"
            ],
            "collaboration": [
                "partnership for sustainability",
                "stakeholder engagement",
                "community collaboration",
                "industry cooperation",
                "environmental partnerships"
            ],
            "regulation_compliance": [
                "regulatory compliance",
                "environmental standards",
                "policy adherence",
                "legal requirements",
                "industry regulations"
            ],
            "innovation": [
                "sustainable innovation",
                "green technology development",
                "environmental solutions",
                "clean technology adoption",
                "innovative practices"
            ],
            "responsibility": [
                "corporate responsibility",
                "environmental stewardship",
                "social responsibility",
                "ethical practices",
                "sustainable operations"
            ]
        }
        
    def setup_sustainability_keywords(self):
        """Define sustainability-related keywords and phrases"""
        self.sustainability_keywords = {
            "emissions": [
                "carbon emissions", "greenhouse gas", "co2", "carbon footprint",
                "emissions reduction", "net zero", "carbon neutral", "decarbonization"
            ],
            "energy": [
                "renewable energy", "solar power", "wind power", "energy efficiency",
                "clean energy", "sustainable energy", "green energy", "biofuel"
            ],
            "waste": [
                "waste reduction", "recycling", "circular economy", "waste management",
                "zero waste", "landfill diversion", "waste minimization"
            ],
            "transportation": [
                "sustainable transport", "electric vehicles", "hybrid aircraft",
                "sustainable aviation fuel", "SAF", "biofuel", "electric ground support"
            ],
            "infrastructure": [
                "green building", "LEED certification", "sustainable infrastructure",
                "energy efficient buildings", "smart technology", "green construction"
            ],
            "biodiversity": [
                "biodiversity", "ecosystem", "wildlife protection", "habitat conservation",
                "environmental impact", "nature conservation"
            ],
            "water": [
                "water conservation", "water management", "stormwater", "water efficiency",
                "water recycling", "water treatment"
            ]
        }
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using multiple methods for robustness
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        self.logger.info(f"Extracting text from: {pdf_path}")
        text = ""
        
        # Method 1: Using pdfplumber (better for structured PDFs)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            self.logger.warning(f"pdfplumber extraction failed: {e}")
            
        # Method 2: Using PyPDF2 as fallback
        if not text.strip():
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                self.logger.error(f"PyPDF2 extraction failed: {e}")
                
        if not text.strip():
            raise ValueError(f"Could not extract text from {pdf_path}")
            
        return text
        
    def identify_ceo_messages(self, text: str) -> List[str]:
        """
        Identify and extract CEO messages from the document
        
        Args:
            text: Full document text
            
        Returns:
            List of CEO message segments
        """
        self.logger.info("Identifying CEO messages...")
        
        ceo_messages = []
        
        # Common patterns for CEO messages
        ceo_patterns = [
            r"(?i)(message from|letter from|foreword by|statement by).*?(?:ceo|chief executive|president)",
            r"(?i)(?:ceo|chief executive|president).*?(?:message|letter|foreword|statement)",
            r"(?i)dear (?:stakeholders|shareholders|passengers|customers|community)",
            r"(?i)(?:signed|sincerely|regards),?\s*(?:\w+\s+)*(?:ceo|chief executive|president)"
        ]
        
        # Split text into sections
        sections = re.split(r'\n\s*\n', text)
        
        for section in sections:
            # Check if section contains CEO-related patterns
            for pattern in ceo_patterns:
                if re.search(pattern, section):
                    # Extract the message (assuming it's within a reasonable length)
                    if 100 <= len(section) <= 5000:  # Reasonable message length
                        ceo_messages.append(section.strip())
                        break
                        
        # If no specific CEO messages found, look for introduction/foreword sections
        if not ceo_messages:
            intro_patterns = [
                r"(?i)(?:introduction|foreword|preface|executive summary)",
                r"(?i)(?:welcome|greetings|dear)"
            ]
            
            for section in sections:
                for pattern in intro_patterns:
                    if re.search(pattern, section) and 200 <= len(section) <= 3000:
                        ceo_messages.append(section.strip())
                        break
                        
        self.logger.info(f"Found {len(ceo_messages)} CEO message segments")
        return ceo_messages
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\b\d{1,3}\b', '', text)  # Remove standalone numbers
        text = re.sub(r'page \d+', '', text, flags=re.IGNORECASE)
        
        # Clean up common PDF artifacts
        text = re.sub(r'[^\w\s.,!?;:()"-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
        
    def extract_sustainability_themes(self, text: str) -> Dict[str, List[str]]:
        """
        Extract sustainability themes and related sentences
        
        Args:
            text: Message text
            
        Returns:
            Dictionary of themes and related sentences
        """
        themes = {}
        sentences = nltk.sent_tokenize(text)
        
        for theme, keywords in self.sustainability_keywords.items():
            theme_sentences = []
            for sentence in sentences:
                for keyword in keywords:
                    if keyword.lower() in sentence.lower():
                        theme_sentences.append(sentence.strip())
                        break
            themes[theme] = list(set(theme_sentences))  # Remove duplicates
            
        return themes
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of the text
        
        Args:
            text: Input text
            
        Returns:
            Sentiment scores
        """
        # VADER sentiment analysis
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        return {
            "vader_positive": vader_scores['pos'],
            "vader_negative": vader_scores['neg'],
            "vader_neutral": vader_scores['neu'],
            "vader_compound": vader_scores['compound'],
            "textblob_polarity": textblob_polarity,
            "textblob_subjectivity": textblob_subjectivity
        }
        
    def classify_intents(self, text: str) -> Dict[str, float]:
        """
        Classify intents in the CEO message
        
        Args:
            text: CEO message text
            
        Returns:
            Intent classification scores
        """
        intent_scores = {}
        
        if self.zero_shot_classifier:
            # Use transformer-based zero-shot classification
            candidate_labels = list(self.intent_categories.keys())
            
            try:
                result = self.zero_shot_classifier(text, candidate_labels)
                for label, score in zip(result['labels'], result['scores']):
                    intent_scores[label] = score
            except Exception as e:
                self.logger.warning(f"Zero-shot classification failed: {e}")
                
        # Fallback: Keyword-based classification
        if not intent_scores:
            intent_scores = self._classify_intents_keywords(text)
            
        return intent_scores
        
    def _classify_intents_keywords(self, text: str) -> Dict[str, float]:
        """
        Fallback keyword-based intent classification
        
        Args:
            text: Input text
            
        Returns:
            Intent scores based on keyword matching
        """
        text_lower = text.lower()
        intent_scores = {}
        
        for intent, keywords in self.intent_categories.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1
            # Normalize by number of keywords and text length
            intent_scores[intent] = score / (len(keywords) * (len(text) / 1000))
            
        return intent_scores
        
    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and entities
        """
        entities = {}
        
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                entities[ent.label_].append(ent.text)
                
        # Remove duplicates
        for ent_type in entities:
            entities[ent_type] = list(set(entities[ent_type]))
            
        return entities
        
    def analyze_commitment_strength(self, text: str) -> Dict[str, float]:
        """
        Analyze the strength of sustainability commitments
        
        Args:
            text: CEO message text
            
        Returns:
            Commitment strength indicators
        """
        # Strong commitment indicators
        strong_indicators = [
            r"commit", r"pledge", r"promise", r"will", r"shall", r"must",
            r"target", r"goal", r"achieve", r"deliver", r"ensure"
        ]
        
        # Weak commitment indicators
        weak_indicators = [
            r"hope", r"try", r"attempt", r"consider", r"explore",
            r"may", r"might", r"could", r"should", r"would like"
        ]
        
        # Time-bound indicators
        time_indicators = [
            r"\d{4}", r"by \d{4}", r"within \d+", r"next \d+",
            r"short-term", r"long-term", r"immediate"
        ]
        
        text_lower = text.lower()
        
        strong_count = sum(len(re.findall(pattern, text_lower)) for pattern in strong_indicators)
        weak_count = sum(len(re.findall(pattern, text_lower)) for pattern in weak_indicators)
        time_count = sum(len(re.findall(pattern, text_lower)) for pattern in time_indicators)
        
        total_words = len(text.split())
        
        return {
            "strong_commitment_density": strong_count / total_words if total_words > 0 else 0,
            "weak_commitment_density": weak_count / total_words if total_words > 0 else 0,
            "time_bound_density": time_count / total_words if total_words > 0 else 0,
            "commitment_confidence": (strong_count - weak_count) / max(strong_count + weak_count, 1)
        }
        
    def generate_word_cloud(self, text: str, output_path: str = "ceo_wordcloud.png"):
        """
        Generate and save word cloud from text
        
        Args:
            text: Input text
            output_path: Path to save the word cloud image
        """
        try:
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('CEO Message Word Cloud')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Word cloud saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Could not generate word cloud: {e}")
            
    def analyze_single_pdf(self, pdf_path: str) -> Dict:
        """
        Perform comprehensive analysis on a single PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Analysis results dictionary
        """
        self.logger.info(f"Analyzing PDF: {pdf_path}")
        
        try:
            # Extract text
            full_text = self.extract_text_from_pdf(pdf_path)
            
            # Identify CEO messages
            ceo_messages = self.identify_ceo_messages(full_text)
            
            if not ceo_messages:
                self.logger.warning(f"No CEO messages found in {pdf_path}")
                return {"error": "No CEO messages found"}
                
            # Combine all CEO messages
            combined_message = " ".join(ceo_messages)
            processed_text = self.preprocess_text(combined_message)
            
            # Perform various analyses
            analysis_results = {
                "pdf_path": pdf_path,
                "message_count": len(ceo_messages),
                "total_length": len(processed_text),
                "word_count": len(processed_text.split()),
                "language": self._detect_language(processed_text),
                "sustainability_themes": self.extract_sustainability_themes(processed_text),
                "sentiment_analysis": self.analyze_sentiment(processed_text),
                "intent_classification": self.classify_intents(processed_text),
                "named_entities": self.extract_named_entities(processed_text),
                "commitment_strength": self.analyze_commitment_strength(processed_text),
                "raw_messages": ceo_messages,
                "processed_text": processed_text
            }
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing {pdf_path}: {e}")
            return {"error": str(e)}
            
    def _detect_language(self, text: str) -> str:
        """Detect language of the text"""
        try:
            return detect(text)
        except:
            return "unknown"
            
    def analyze_multiple_pdfs(self, pdf_directory: str) -> List[Dict]:
        """
        Analyze multiple PDFs in a directory
        
        Args:
            pdf_directory: Directory containing PDF files
            
        Returns:
            List of analysis results
        """
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDF files to analyze")
        
        results = []
        for pdf_file in pdf_files:
            result = self.analyze_single_pdf(str(pdf_file))
            results.append(result)
            
        return results
        
    def generate_comparative_report(self, results: List[Dict], output_path: str = "ceo_intent_analysis_report.html"):
        """
        Generate a comprehensive comparative report
        
        Args:
            results: List of analysis results
            output_path: Path to save the HTML report
        """
        # Filter out error results
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            self.logger.error("No valid results to generate report")
            return
            
        # Create visualizations and analysis
        self._create_visualizations(valid_results)
        
        # Generate HTML report
        html_content = self._generate_html_report(valid_results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        self.logger.info(f"Comprehensive report saved to: {output_path}")
        
    def _create_visualizations(self, results: List[Dict]):
        """Create various visualizations for the analysis"""
        # Sentiment comparison
        self._plot_sentiment_comparison(results)
        
        # Intent distribution
        self._plot_intent_distribution(results)
        
        # Commitment strength comparison
        self._plot_commitment_strength(results)
        
        # Sustainability themes heatmap
        self._plot_sustainability_themes(results)
        
    def _plot_sentiment_comparison(self, results: List[Dict]):
        """Plot sentiment comparison across PDFs"""
        plt.figure(figsize=(12, 6))
        
        pdf_names = [os.path.basename(r['pdf_path']) for r in results]
        compound_scores = [r['sentiment_analysis']['vader_compound'] for r in results]
        
        bars = plt.bar(range(len(pdf_names)), compound_scores, 
                       color=['green' if score > 0 else 'red' for score in compound_scores])
        
        plt.xlabel('Airport PDF')
        plt.ylabel('Sentiment Score (VADER Compound)')
        plt.title('CEO Message Sentiment Comparison Across Airports')
        plt.xticks(range(len(pdf_names)), pdf_names, rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('sentiment_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_intent_distribution(self, results: List[Dict]):
        """Plot intent distribution heatmap"""
        if not results:
            return
            
        # Collect intent data
        intent_data = []
        pdf_names = []
        
        for result in results:
            pdf_names.append(os.path.basename(result['pdf_path']))
            intent_scores = result['intent_classification']
            intent_data.append([intent_scores.get(intent, 0) for intent in self.intent_categories.keys()])
            
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            intent_data,
            xticklabels=list(self.intent_categories.keys()),
            yticklabels=pdf_names,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Intent Score'}
        )
        plt.title('Intent Classification Heatmap - CEO Messages')
        plt.xlabel('Intent Categories')
        plt.ylabel('Airport PDFs')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('intent_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_commitment_strength(self, results: List[Dict]):
        """Plot commitment strength comparison"""
        plt.figure(figsize=(12, 6))
        
        pdf_names = [os.path.basename(r['pdf_path']) for r in results]
        commitment_scores = [r['commitment_strength']['commitment_confidence'] for r in results]
        
        bars = plt.bar(range(len(pdf_names)), commitment_scores,
                       color=['darkgreen' if score > 0.5 else 'orange' if score > 0 else 'red' 
                             for score in commitment_scores])
        
        plt.xlabel('Airport PDF')
        plt.ylabel('Commitment Confidence Score')
        plt.title('Sustainability Commitment Strength - CEO Messages')
        plt.xticks(range(len(pdf_names)), pdf_names, rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(y=0.5, color='blue', linestyle='--', alpha=0.5, label='High Confidence Threshold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('commitment_strength.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_sustainability_themes(self, results: List[Dict]):
        """Plot sustainability themes frequency"""
        theme_counts = {theme: 0 for theme in self.sustainability_keywords.keys()}
        
        for result in results:
            for theme, sentences in result['sustainability_themes'].items():
                theme_counts[theme] += len(sentences)
                
        plt.figure(figsize=(10, 6))
        themes = list(theme_counts.keys())
        counts = list(theme_counts.values())
        
        bars = plt.bar(themes, counts, color='steelblue')
        plt.xlabel('Sustainability Themes')
        plt.ylabel('Frequency Across All PDFs')
        plt.title('Sustainability Theme Distribution in CEO Messages')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('sustainability_themes.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _generate_html_report(self, results: List[Dict]) -> str:
        """Generate comprehensive HTML report"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>CEO Intent Analysis Report - Airport Sustainability</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #2E8B57; }
                .summary { background-color: #f0f8f0; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .pdf-analysis { border: 1px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 5px; }
                .intent-scores { display: flex; flex-wrap: wrap; gap: 10px; }
                .intent-score { background-color: #e8f5e8; padding: 5px 10px; border-radius: 3px; }
                .theme-list { columns: 2; }
                img { max-width: 100%; height: auto; margin: 10px 0; }
                table { border-collapse: collapse; width: 100%; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>CEO Intent Analysis Report - Airport Sustainability PDFs</h1>
        """
        
        # Executive Summary
        html += self._generate_executive_summary(results)
        
        # Visualizations
        html += """
            <h2>Comparative Analysis Visualizations</h2>
            <h3>Sentiment Analysis</h3>
            <img src="sentiment_comparison.png" alt="Sentiment Comparison">
            
            <h3>Intent Distribution</h3>
            <img src="intent_distribution.png" alt="Intent Distribution">
            
            <h3>Commitment Strength</h3>
            <img src="commitment_strength.png" alt="Commitment Strength">
            
            <h3>Sustainability Themes</h3>
            <img src="sustainability_themes.png" alt="Sustainability Themes">
        """
        
        # Individual PDF Analysis
        html += "<h2>Individual PDF Analysis</h2>"
        
        for result in results:
            html += self._generate_pdf_section(result)
            
        html += """
        </body>
        </html>
        """
        
        return html
        
    def _generate_executive_summary(self, results: List[Dict]) -> str:
        """Generate executive summary section"""
        total_pdfs = len(results)
        avg_sentiment = np.mean([r['sentiment_analysis']['vader_compound'] for r in results])
        avg_commitment = np.mean([r['commitment_strength']['commitment_confidence'] for r in results])
        
        # Most common intents
        intent_totals = {}
        for result in results:
            for intent, score in result['intent_classification'].items():
                intent_totals[intent] = intent_totals.get(intent, 0) + score
                
        top_intents = sorted(intent_totals.items(), key=lambda x: x[1], reverse=True)[:3]
        
        summary = f"""
        <div class="summary">
            <h2>Executive Summary</h2>
            <ul>
                <li><strong>Total PDFs Analyzed:</strong> {total_pdfs}</li>
                <li><strong>Average Sentiment Score:</strong> {avg_sentiment:.3f} 
                    ({'Positive' if avg_sentiment > 0.1 else 'Negative' if avg_sentiment < -0.1 else 'Neutral'})</li>
                <li><strong>Average Commitment Confidence:</strong> {avg_commitment:.3f}
                    ({'Strong' if avg_commitment > 0.5 else 'Moderate' if avg_commitment > 0 else 'Weak'})</li>
                <li><strong>Top Intent Categories:</strong> 
                    {', '.join([f"{intent} ({score:.2f})" for intent, score in top_intents])}</li>
            </ul>
        </div>
        """
        
        return summary
        
    def _generate_pdf_section(self, result: Dict) -> str:
        """Generate individual PDF analysis section"""
        pdf_name = os.path.basename(result['pdf_path'])
        
        # Top intents
        top_intents = sorted(result['intent_classification'].items(), 
                           key=lambda x: x[1], reverse=True)[:5]
        
        # Named entities summary
        entities_summary = []
        for ent_type, entities in result['named_entities'].items():
            if entities:
                entities_summary.append(f"<strong>{ent_type}:</strong> {', '.join(entities[:5])}")
        
        section = f"""
        <div class="pdf-analysis">
            <h3>{pdf_name}</h3>
            <p><strong>Word Count:</strong> {result['word_count']}</p>
            <p><strong>Language:</strong> {result['language']}</p>
            <p><strong>Sentiment:</strong> {result['sentiment_analysis']['vader_compound']:.3f}</p>
            <p><strong>Commitment Confidence:</strong> {result['commitment_strength']['commitment_confidence']:.3f}</p>
            
            <h4>Top Intent Classifications:</h4>
            <div class="intent-scores">
        """
        
        for intent, score in top_intents:
            section += f'<div class="intent-score">{intent}: {score:.3f}</div>'
            
        section += """
            </div>
            
            <h4>Sustainability Themes:</h4>
            <div class="theme-list">
        """
        
        for theme, sentences in result['sustainability_themes'].items():
            if sentences:
                section += f"<p><strong>{theme.title()}:</strong> {len(sentences)} mentions</p>"
                
        section += f"""
            </div>
            
            <h4>Key Entities:</h4>
            <p>{'; '.join(entities_summary) if entities_summary else 'No entities extracted'}</p>
        </div>
        """
        
        return section
        
    def save_results_to_json(self, results: List[Dict], output_path: str = "ceo_analysis_results.json"):
        """Save analysis results to JSON file"""
        # Clean results for JSON serialization
        clean_results = []
        for result in results:
            if "error" not in result:
                clean_result = result.copy()
                # Remove non-serializable elements if any
                clean_results.append(clean_result)
                
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Results saved to: {output_path}")


def main():
    """
    Main function to demonstrate usage
    """
    # Initialize analyzer
    analyzer = AirportCEOIntentAnalyzer()
    
    # Example usage for single PDF
    # result = analyzer.analyze_single_pdf("sample_airport_sustainability_report.pdf")
    
    # Example usage for multiple PDFs
    # results = analyzer.analyze_multiple_pdfs("pdf_directory")
    
    # Generate word cloud for a sample text
    sample_text = """
    As CEO of this airport, I am committed to achieving net-zero carbon emissions by 2030. 
    We will invest significantly in renewable energy, sustainable aviation fuels, and green infrastructure. 
    Our partnership with stakeholders ensures environmental stewardship and innovation in clean technology.
    """
    
    analyzer.generate_word_cloud(sample_text, "sample_wordcloud.png")
    
    # Example analysis
    sample_analysis = analyzer.analyze_sentiment(sample_text)
    print("Sample Sentiment Analysis:", sample_analysis)
    
    sample_intents = analyzer.classify_intents(sample_text)
    print("Sample Intent Classification:", sample_intents)
    
    sample_commitment = analyzer.analyze_commitment_strength(sample_text)
    print("Sample Commitment Strength:", sample_commitment)


if __name__ == "__main__":
    main()