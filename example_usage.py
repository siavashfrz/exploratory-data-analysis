#!/usr/bin/env python3
"""
Example Usage Script for Airport CEO Intent Analyzer

This script demonstrates how to use the AirportCEOIntentAnalyzer class
for analyzing CEO messages in airport sustainability PDFs.
"""

import os
from airport_ceo_intent_analyzer import AirportCEOIntentAnalyzer

def create_sample_data():
    """Create sample CEO messages for demonstration"""
    sample_messages = {
        "heathrow_sample.txt": """
        Message from the CEO - Heathrow Airport
        
        Dear Stakeholders,
        
        As CEO of Heathrow Airport, I am proud to announce our ambitious commitment to achieve 
        net-zero carbon emissions by 2030. This represents one of the most aggressive 
        sustainability targets in the aviation industry.
        
        We will invest £2 billion over the next decade in renewable energy infrastructure, 
        including the largest airport solar farm in Europe. Our partnership with leading 
        technology companies will drive innovation in sustainable aviation fuels and 
        electric ground support equipment.
        
        We pledge to reduce our carbon footprint by 50% within the next five years through 
        energy-efficient building upgrades, waste reduction programs, and enhanced 
        public transportation connectivity.
        
        Together with our airline partners and the broader aviation community, we are 
        committed to leading the industry's environmental transformation.
        
        Sincerely,
        John Holland-Kaye, CEO
        """,
        
        "changi_sample.txt": """
        CEO Statement - Singapore Changi Airport
        
        Welcome to our sustainability journey at Changi Airport. As we look towards the 
        future, environmental stewardship remains at the core of our operations.
        
        I am pleased to share our achievements in 2023: we successfully reduced energy 
        consumption by 15% through smart building technologies and achieved LEED Gold 
        certification for our new terminal.
        
        Our vision for 2025 includes the implementation of a comprehensive circular 
        economy model, where 80% of our waste will be recycled or repurposed. We hope 
        to explore innovative solutions in biodiversity conservation and water management.
        
        While we face challenges, our commitment to sustainable aviation and green 
        technology adoption remains unwavering. We will continue to collaborate with 
        stakeholders to ensure Changi remains a model for environmental excellence.
        
        Best regards,
        Lee Seow Hiang, CEO
        """,
        
        "schiphol_sample.txt": """
        Letter from the CEO - Amsterdam Airport Schiphol
        
        Dear Passengers and Partners,
        
        At Schiphol, we believe that aviation must become more sustainable, and we are 
        taking decisive action to make this happen. Our goal is clear: become the most 
        sustainable airport in the world by 2030.
        
        We have made significant investments in renewable energy, with our airport now 
        powered by 100% wind energy. Our waste-to-energy facility processes 95% of 
        airport waste, contributing to our circular economy objectives.
        
        Innovation drives our sustainability efforts. We are testing hydrogen-powered 
        ground vehicles and implementing AI-driven energy management systems. Our 
        collaboration with KLM on sustainable aviation fuel trials shows promising results.
        
        We must acknowledge that achieving net-zero emissions requires industry-wide 
        transformation. However, I am confident that through continued investment and 
        partnership, we will deliver on our environmental promises.
        
        Dick Benschop, CEO
        Amsterdam Airport Schiphol
        """
    }
    
    # Create sample directory and files
    os.makedirs("sample_ceo_messages", exist_ok=True)
    
    for filename, content in sample_messages.items():
        with open(f"sample_ceo_messages/{filename}", 'w', encoding='utf-8') as f:
            f.write(content)
    
    print("Sample CEO messages created in 'sample_ceo_messages' directory")

def analyze_single_message_example():
    """Example of analyzing a single CEO message"""
    print("\n" + "="*60)
    print("SINGLE MESSAGE ANALYSIS EXAMPLE")
    print("="*60)
    
    analyzer = AirportCEOIntentAnalyzer()
    
    # Sample CEO message
    sample_message = """
    As CEO of this major international airport, I am committed to achieving carbon neutrality 
    by 2025. We will invest $500 million in renewable energy infrastructure, including solar 
    panels and wind turbines. Our partnership with local communities ensures sustainable 
    development that benefits everyone. We must take immediate action to reduce our 
    environmental impact and lead the aviation industry towards a greener future.
    """
    
    print("Analyzing sample CEO message...")
    
    # Sentiment Analysis
    sentiment = analyzer.analyze_sentiment(sample_message)
    print(f"\nSentiment Analysis:")
    print(f"  Compound Score: {sentiment['vader_compound']:.3f}")
    print(f"  Positive: {sentiment['vader_positive']:.3f}")
    print(f"  Negative: {sentiment['vader_negative']:.3f}")
    print(f"  Neutral: {sentiment['vader_neutral']:.3f}")
    
    # Intent Classification
    intents = analyzer.classify_intents(sample_message)
    print(f"\nIntent Classification (Top 5):")
    sorted_intents = sorted(intents.items(), key=lambda x: x[1], reverse=True)[:5]
    for intent, score in sorted_intents:
        print(f"  {intent}: {score:.3f}")
    
    # Commitment Strength
    commitment = analyzer.analyze_commitment_strength(sample_message)
    print(f"\nCommitment Strength:")
    print(f"  Confidence Score: {commitment['commitment_confidence']:.3f}")
    print(f"  Strong Commitment Density: {commitment['strong_commitment_density']:.4f}")
    print(f"  Time-bound Density: {commitment['time_bound_density']:.4f}")
    
    # Sustainability Themes
    themes = analyzer.extract_sustainability_themes(sample_message)
    print(f"\nSustainability Themes:")
    for theme, sentences in themes.items():
        if sentences:
            print(f"  {theme}: {len(sentences)} mentions")
    
    # Generate word cloud
    analyzer.generate_word_cloud(sample_message, "single_message_wordcloud.png")
    print(f"\nWord cloud saved as 'single_message_wordcloud.png'")

def analyze_multiple_messages_example():
    """Example of analyzing multiple CEO messages"""
    print("\n" + "="*60)
    print("MULTIPLE MESSAGES ANALYSIS EXAMPLE")
    print("="*60)
    
    # Create sample data first
    create_sample_data()
    
    analyzer = AirportCEOIntentAnalyzer()
    
    # Analyze sample text files (simulating PDF extraction)
    results = []
    sample_dir = "sample_ceo_messages"
    
    for filename in os.listdir(sample_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(sample_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simulate the analysis process
            processed_text = analyzer.preprocess_text(content)
            
            analysis_result = {
                "pdf_path": filepath,
                "message_count": 1,
                "total_length": len(processed_text),
                "word_count": len(processed_text.split()),
                "language": analyzer._detect_language(processed_text),
                "sustainability_themes": analyzer.extract_sustainability_themes(processed_text),
                "sentiment_analysis": analyzer.analyze_sentiment(processed_text),
                "intent_classification": analyzer.classify_intents(processed_text),
                "named_entities": analyzer.extract_named_entities(processed_text),
                "commitment_strength": analyzer.analyze_commitment_strength(processed_text),
                "raw_messages": [content],
                "processed_text": processed_text
            }
            
            results.append(analysis_result)
            print(f"Analyzed: {filename}")
    
    # Generate comparative visualizations
    if results:
        analyzer._create_visualizations(results)
        print(f"\nGenerated visualizations:")
        print("  - sentiment_comparison.png")
        print("  - intent_distribution.png") 
        print("  - commitment_strength.png")
        print("  - sustainability_themes.png")
        
        # Generate comprehensive report
        analyzer.generate_comparative_report(results, "comprehensive_analysis_report.html")
        
        # Save results to JSON
        analyzer.save_results_to_json(results, "analysis_results.json")
        
        print(f"\nReport saved as 'comprehensive_analysis_report.html'")
        print(f"JSON results saved as 'analysis_results.json'")
        
        # Print summary statistics
        print(f"\n" + "-"*40)
        print("SUMMARY STATISTICS")
        print("-"*40)
        
        avg_sentiment = sum(r['sentiment_analysis']['vader_compound'] for r in results) / len(results)
        avg_commitment = sum(r['commitment_strength']['commitment_confidence'] for r in results) / len(results)
        
        print(f"Total airports analyzed: {len(results)}")
        print(f"Average sentiment score: {avg_sentiment:.3f}")
        print(f"Average commitment confidence: {avg_commitment:.3f}")
        
        # Most common intents across all messages
        intent_totals = {}
        for result in results:
            for intent, score in result['intent_classification'].items():
                intent_totals[intent] = intent_totals.get(intent, 0) + score
        
        top_intents = sorted(intent_totals.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"Top intents across all messages:")
        for intent, total_score in top_intents:
            print(f"  {intent}: {total_score:.2f}")

def demonstrate_pdf_analysis():
    """Demonstrate PDF analysis capabilities"""
    print("\n" + "="*60)
    print("PDF ANALYSIS DEMONSTRATION")
    print("="*60)
    
    analyzer = AirportCEOIntentAnalyzer()
    
    # This would be used with actual PDF files
    print("To analyze actual PDF files:")
    print("1. Place your airport sustainability PDF reports in a directory")
    print("2. Use the following code:")
    print()
    print("# Analyze single PDF")
    print('result = analyzer.analyze_single_pdf("path/to/sustainability_report.pdf")')
    print()
    print("# Analyze multiple PDFs")
    print('results = analyzer.analyze_multiple_pdfs("path/to/pdf_directory")')
    print()
    print("# Generate comprehensive report")
    print('analyzer.generate_comparative_report(results)')
    print()
    print("Note: Ensure PDFs contain CEO messages, forewords, or executive summaries")
    print("The analyzer will automatically identify and extract relevant sections")

def advanced_analysis_features():
    """Demonstrate advanced analysis features"""
    print("\n" + "="*60)
    print("ADVANCED ANALYSIS FEATURES")
    print("="*60)
    
    analyzer = AirportCEOIntentAnalyzer()
    
    # Sample message with rich content
    complex_message = """
    Dear Stakeholders and Community Partners,
    
    As we navigate the challenges of 2024, Zurich Airport remains steadfast in our 
    commitment to environmental excellence. Our board has approved a CHF 800 million 
    investment program focused on three key areas: carbon reduction, biodiversity 
    protection, and circular economy implementation.
    
    I am proud to announce that we have achieved a 35% reduction in Scope 1 and 2 
    emissions since 2019, ahead of our 2025 target. Our new geothermal energy system, 
    operational since January, provides heating for 60% of our terminal buildings.
    
    Looking ahead, we will launch pilot programs for sustainable aviation fuel (SAF) 
    blending by Q3 2024, with the goal of offering 10% SAF blend to all carriers by 2026. 
    Our partnership with ETH Zurich on hydrogen aircraft refueling infrastructure 
    represents a significant step towards zero-emission aviation.
    
    However, we must acknowledge that individual airport efforts alone cannot solve 
    aviation's environmental challenges. Industry-wide collaboration, regulatory support, 
    and technological breakthroughs are essential. We hope to see accelerated development 
    of electric aircraft and improved air traffic management systems.
    
    Our commitment extends beyond operations to our supply chain and stakeholder 
    community. By 2025, all major suppliers must demonstrate measurable sustainability 
    improvements, and we will continue investing in local environmental projects.
    
    Thank you for your continued support as we build a more sustainable future for aviation.
    
    Dr. André Dosé, CEO
    Flughafen Zürich AG
    """
    
    print("Analyzing complex CEO message with advanced features...")
    
    # Named Entity Recognition
    entities = analyzer.extract_named_entities(complex_message)
    print(f"\nNamed Entities Extracted:")
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"  {entity_type}: {', '.join(entity_list[:5])}")
    
    # Detailed sentiment analysis
    sentiment = analyzer.analyze_sentiment(complex_message)
    print(f"\nDetailed Sentiment Analysis:")
    print(f"  VADER Compound: {sentiment['vader_compound']:.3f}")
    print(f"  TextBlob Polarity: {sentiment['textblob_polarity']:.3f}")
    print(f"  TextBlob Subjectivity: {sentiment['textblob_subjectivity']:.3f}")
    
    # Comprehensive commitment analysis
    commitment = analyzer.analyze_commitment_strength(complex_message)
    print(f"\nCommitment Strength Analysis:")
    for metric, value in commitment.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Sustainability themes with details
    themes = analyzer.extract_sustainability_themes(complex_message)
    print(f"\nSustainability Themes (with sentence count):")
    for theme, sentences in themes.items():
        if sentences:
            print(f"  {theme.title()}: {len(sentences)} mentions")
            # Show first sentence as example
            if sentences:
                print(f"    Example: {sentences[0][:100]}...")
    
    # Language detection
    language = analyzer._detect_language(complex_message)
    print(f"\nDetected Language: {language}")
    
    print(f"\nMessage Statistics:")
    print(f"  Total characters: {len(complex_message)}")
    print(f"  Word count: {len(complex_message.split())}")
    print(f"  Sentence count: {len(complex_message.split('.'))}")

def main():
    """Main function to run all examples"""
    print("Airport CEO Intent Analyzer - Example Usage")
    print("=" * 60)
    
    # Run all example functions
    analyze_single_message_example()
    analyze_multiple_messages_example()
    demonstrate_pdf_analysis()
    advanced_analysis_features()
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETE")
    print("="*60)
    print("Check the generated files:")
    print("- comprehensive_analysis_report.html (main report)")
    print("- analysis_results.json (raw data)")
    print("- Various .png visualization files")
    print("- sample_ceo_messages/ directory with sample data")

if __name__ == "__main__":
    main()