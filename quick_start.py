#!/usr/bin/env python3
"""
Quick Start Script for Airport CEO Intent Analyzer

This script provides a minimal example to get started quickly with the analyzer.
It demonstrates core functionality without requiring PDF files.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

def check_dependencies():
    """Check if essential dependencies are installed"""
    required_packages = [
        'nltk', 'textblob', 'vaderSentiment', 'numpy', 
        'matplotlib', 'wordcloud', 'langdetect'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All essential packages are installed!")
    return True

def quick_demo():
    """Run a quick demonstration of the analyzer"""
    print("\n" + "="*60)
    print("AIRPORT CEO INTENT ANALYZER - QUICK DEMO")
    print("="*60)
    
    if not check_dependencies():
        return
    
    try:
        from airport_ceo_intent_analyzer import AirportCEOIntentAnalyzer
        
        # Initialize analyzer
        print("\n🔧 Initializing analyzer...")
        analyzer = AirportCEOIntentAnalyzer()
        
        # Sample CEO message for analysis
        sample_ceo_message = """
        Dear Stakeholders,
        
        As CEO of Metropolitan Airport, I am proud to announce our commitment to achieving 
        carbon neutrality by 2030. This ambitious goal reflects our dedication to 
        environmental stewardship and sustainable aviation.
        
        Over the next five years, we will invest $2 billion in renewable energy infrastructure, 
        including solar panels, wind turbines, and energy-efficient terminal buildings. 
        Our partnership with leading technology companies will accelerate the adoption of 
        sustainable aviation fuels and electric ground support equipment.
        
        We have already achieved significant milestones: reducing energy consumption by 25%, 
        implementing a comprehensive waste recycling program, and obtaining LEED Gold 
        certification for our new terminal. These accomplishments demonstrate our unwavering 
        commitment to environmental excellence.
        
        Looking ahead, we will continue to innovate and collaborate with airlines, 
        government agencies, and local communities to ensure that aviation becomes more 
        sustainable. Together, we can build a greener future for air travel.
        
        Thank you for your continued support in this vital mission.
        
        Sincerely,
        Sarah Johnson, CEO
        Metropolitan Airport Authority
        """
        
        print("📝 Analyzing sample CEO message...")
        
        # Perform various analyses
        print("\n1️⃣ Sentiment Analysis:")
        sentiment = analyzer.analyze_sentiment(sample_ceo_message)
        print(f"   Overall Sentiment: {sentiment['vader_compound']:.3f} "
              f"({'Positive' if sentiment['vader_compound'] > 0.1 else 'Neutral' if sentiment['vader_compound'] > -0.1 else 'Negative'})")
        print(f"   Positive: {sentiment['vader_positive']:.3f}")
        print(f"   Negative: {sentiment['vader_negative']:.3f}")
        print(f"   Neutral: {sentiment['vader_neutral']:.3f}")
        
        print("\n2️⃣ Intent Classification:")
        intents = analyzer.classify_intents(sample_ceo_message)
        sorted_intents = sorted(intents.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (intent, score) in enumerate(sorted_intents, 1):
            print(f"   {i}. {intent.replace('_', ' ').title()}: {score:.3f}")
        
        print("\n3️⃣ Commitment Strength:")
        commitment = analyzer.analyze_commitment_strength(sample_ceo_message)
        confidence = commitment['commitment_confidence']
        print(f"   Commitment Confidence: {confidence:.3f} "
              f"({'Strong' if confidence > 0.5 else 'Moderate' if confidence > 0 else 'Weak'})")
        print(f"   Strong Language Density: {commitment['strong_commitment_density']:.4f}")
        print(f"   Time-bound References: {commitment['time_bound_density']:.4f}")
        
        print("\n4️⃣ Sustainability Themes:")
        themes = analyzer.extract_sustainability_themes(sample_ceo_message)
        theme_count = 0
        for theme, sentences in themes.items():
            if sentences:
                theme_count += 1
                print(f"   {theme.replace('_', ' ').title()}: {len(sentences)} mentions")
        
        if theme_count == 0:
            print("   No specific sustainability themes detected")
        
        print("\n5️⃣ Named Entities:")
        entities = analyzer.extract_named_entities(sample_ceo_message)
        if entities:
            for entity_type, entity_list in entities.items():
                if entity_list:
                    print(f"   {entity_type}: {', '.join(entity_list[:3])}")
        else:
            print("   No named entities extracted (spaCy model may not be installed)")
        
        print("\n6️⃣ Generating Word Cloud...")
        try:
            analyzer.generate_word_cloud(sample_ceo_message, "quick_demo_wordcloud.png")
            print("   ✅ Word cloud saved as 'quick_demo_wordcloud.png'")
        except Exception as e:
            print(f"   ⚠️ Could not generate word cloud: {e}")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
        # Summary
        top_intent = max(intents.items(), key=lambda x: x[1])
        print(f"\n📊 SUMMARY:")
        print(f"   Primary Intent: {top_intent[0].replace('_', ' ').title()} ({top_intent[1]:.3f})")
        print(f"   Sentiment: {'Positive' if sentiment['vader_compound'] > 0.1 else 'Neutral' if sentiment['vader_compound'] > -0.1 else 'Negative'} ({sentiment['vader_compound']:.3f})")
        print(f"   Commitment Level: {'Strong' if confidence > 0.5 else 'Moderate' if confidence > 0 else 'Weak'} ({confidence:.3f})")
        print(f"   Sustainability Themes: {theme_count} detected")
        
        # Next steps
        print(f"\n🚀 NEXT STEPS:")
        print("   1. Install additional dependencies for full functionality:")
        print("      pip install transformers torch spacy")
        print("      python -m spacy download en_core_web_sm")
        print("   2. Try analyzing your own PDFs:")
        print("      result = analyzer.analyze_single_pdf('your_file.pdf')")
        print("   3. Run the comprehensive example:")
        print("      python example_usage.py")
        print("   4. Check the README.md for detailed documentation")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Check the error message and ensure all dependencies are installed correctly")

def installation_guide():
    """Provide installation guidance"""
    print("\n" + "="*60)
    print("INSTALLATION GUIDE")
    print("="*60)
    
    print("\n📋 REQUIREMENTS:")
    print("   - Python 3.8 or higher")
    print("   - 2GB+ RAM (4GB+ recommended for transformer models)")
    print("   - Internet connection for initial setup")
    
    print("\n⬇️ INSTALLATION STEPS:")
    print("   1. Install basic dependencies:")
    print("      pip install -r requirements.txt")
    print()
    print("   2. Install spaCy English model (optional, for named entity recognition):")
    print("      python -m spacy download en_core_web_sm")
    print()
    print("   3. Test installation:")
    print("      python quick_start.py")
    
    print("\n🔧 TROUBLESHOOTING:")
    print("   - If PDF extraction fails, try: pip install pymupdf")
    print("   - For memory issues, disable transformer models in the code")
    print("   - Check README.md for detailed troubleshooting")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--install-guide":
        installation_guide()
    else:
        quick_demo()

if __name__ == "__main__":
    main()