#!/usr/bin/env python3
"""
Basic Quick Start Script for Airport CEO Intent Analyzer

This script demonstrates core functionality without requiring heavy transformer libraries.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

def check_basic_dependencies():
    """Check if basic dependencies are installed"""
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
    
    print("✅ All basic packages are installed!")
    return True

def basic_demo():
    """Run a basic demonstration without transformer dependencies"""
    print("\n" + "="*60)
    print("AIRPORT CEO INTENT ANALYZER - BASIC DEMO")
    print("="*60)
    
    if not check_basic_dependencies():
        return
    
    # Import only the necessary components
    import nltk
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from langdetect import detect
    import re
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
    
    # Initialize basic components
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
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
    
    # 1. Basic text statistics
    print("\n1️⃣ Text Statistics:")
    word_count = len(sample_ceo_message.split())
    char_count = len(sample_ceo_message)
    sentences = nltk.sent_tokenize(sample_ceo_message)
    sentence_count = len(sentences)
    
    print(f"   Words: {word_count}")
    print(f"   Characters: {char_count}")
    print(f"   Sentences: {sentence_count}")
    
    # 2. Language detection
    try:
        language = detect(sample_ceo_message)
        print(f"   Detected Language: {language}")
    except:
        print(f"   Detected Language: unknown")
    
    # 3. Sentiment Analysis
    print("\n2️⃣ Sentiment Analysis:")
    sentiment = sentiment_analyzer.polarity_scores(sample_ceo_message)
    print(f"   Overall Sentiment: {sentiment['compound']:.3f} "
          f"({'Positive' if sentiment['compound'] > 0.1 else 'Neutral' if sentiment['compound'] > -0.1 else 'Negative'})")
    print(f"   Positive: {sentiment['pos']:.3f}")
    print(f"   Negative: {sentiment['neg']:.3f}")
    print(f"   Neutral: {sentiment['neu']:.3f}")
    
    # TextBlob sentiment
    blob = TextBlob(sample_ceo_message)
    print(f"   TextBlob Polarity: {blob.sentiment.polarity:.3f}")
    print(f"   TextBlob Subjectivity: {blob.sentiment.subjectivity:.3f}")
    
    # 4. Basic Intent Classification (keyword-based)
    print("\n3️⃣ Intent Classification (Keyword-based):")
    
    intent_keywords = {
        "commitment": ["commit", "pledge", "promise", "will", "shall", "dedicated", "commitment"],
        "achievement": ["achieved", "accomplished", "success", "milestones", "obtained", "certification"],
        "investment": ["invest", "investment", "funding", "billion", "million", "spending"],
        "collaboration": ["partnership", "collaborate", "together", "community", "stakeholders"],
        "innovation": ["innovate", "innovation", "technology", "adoption", "development"],
        "vision": ["future", "ahead", "vision", "goal", "mission", "build"],
        "responsibility": ["stewardship", "responsibility", "environmental", "sustainable"]
    }
    
    text_lower = sample_ceo_message.lower()
    intent_scores = {}
    
    for intent, keywords in intent_keywords.items():
        score = 0
        for keyword in keywords:
            score += text_lower.count(keyword)
        intent_scores[intent] = score
    
    # Show top intents
    sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (intent, score) in enumerate(sorted_intents, 1):
        print(f"   {i}. {intent.replace('_', ' ').title()}: {score} mentions")
    
    # 5. Commitment Strength Analysis
    print("\n4️⃣ Commitment Strength Analysis:")
    
    strong_indicators = ["commit", "pledge", "promise", "will", "shall", "must", "target", "goal"]
    weak_indicators = ["hope", "try", "attempt", "consider", "may", "might", "could"]
    time_indicators = ["2030", "2025", "five years", "next", "by", "within"]
    
    strong_count = sum(text_lower.count(word) for word in strong_indicators)
    weak_count = sum(text_lower.count(word) for word in weak_indicators)
    time_count = sum(text_lower.count(phrase) for phrase in time_indicators)
    
    commitment_confidence = (strong_count - weak_count) / max(strong_count + weak_count, 1)
    
    print(f"   Strong Language Count: {strong_count}")
    print(f"   Weak Language Count: {weak_count}")
    print(f"   Time-bound References: {time_count}")
    print(f"   Commitment Confidence: {commitment_confidence:.3f}")
    print(f"   Assessment: {'Strong' if commitment_confidence > 0.5 else 'Moderate' if commitment_confidence > 0 else 'Weak'}")
    
    # 6. Sustainability Themes
    print("\n5️⃣ Sustainability Themes:")
    
    sustainability_themes = {
        "emissions": ["carbon", "emissions", "co2", "neutral", "footprint"],
        "energy": ["energy", "renewable", "solar", "wind", "efficiency"],
        "waste": ["waste", "recycling", "circular"],
        "certification": ["leed", "certification", "standards"],
        "technology": ["technology", "innovation", "fuel", "equipment"],
        "collaboration": ["partnership", "collaboration", "community"]
    }
    
    theme_count = 0
    for theme, keywords in sustainability_themes.items():
        mentions = sum(text_lower.count(keyword) for keyword in keywords)
        if mentions > 0:
            theme_count += 1
            print(f"   {theme.title()}: {mentions} mentions")
    
    if theme_count == 0:
        print("   No specific sustainability themes detected")
    
    # 7. Generate Word Cloud
    print("\n6️⃣ Generating Word Cloud...")
    try:
        # Remove common words and create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=50,
            colormap='viridis',
            stopwords={'the', 'and', 'to', 'of', 'in', 'we', 'our', 'will', 'a', 'an', 'as', 'for', 'with', 'by', 'on'}
        ).generate(sample_ceo_message)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('CEO Message Word Cloud')
        plt.tight_layout()
        plt.savefig('basic_demo_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Word cloud saved as 'basic_demo_wordcloud.png'")
    except Exception as e:
        print(f"   ⚠️ Could not generate word cloud: {e}")
    
    print("\n" + "="*60)
    print("BASIC ANALYSIS COMPLETE!")
    print("="*60)
    
    # Summary
    top_intent = max(intent_scores.items(), key=lambda x: x[1])
    print(f"\n📊 SUMMARY:")
    print(f"   Primary Intent: {top_intent[0].replace('_', ' ').title()} ({top_intent[1]} mentions)")
    print(f"   Sentiment: {'Positive' if sentiment['compound'] > 0.1 else 'Neutral' if sentiment['compound'] > -0.1 else 'Negative'} ({sentiment['compound']:.3f})")
    print(f"   Commitment Level: {'Strong' if commitment_confidence > 0.5 else 'Moderate' if commitment_confidence > 0 else 'Weak'} ({commitment_confidence:.3f})")
    print(f"   Sustainability Themes: {theme_count} detected")
    
    # Next steps
    print(f"\n🚀 NEXT STEPS:")
    print("   1. For advanced features, install transformer libraries:")
    print("      pip install transformers torch")
    print("   2. For spaCy named entity recognition:")
    print("      python -m spacy download en_core_web_sm")
    print("   3. Try analyzing your own PDFs with the full script:")
    print("      python airport_ceo_intent_analyzer.py")
    print("   4. Check the README.md for detailed documentation")

def main():
    """Main function"""
    basic_demo()

if __name__ == "__main__":
    main()