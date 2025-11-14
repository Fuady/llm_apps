"""
Social Media Marketing AI Assistant with Ollama

A comprehensive Streamlit app for social media marketers to:
- Create campaign strategies
- Generate content (articles, captions, carousels)
- Analyze metrics and performance data
- All powered by local AI models via Ollama

Installation:
1. pip install streamlit requests pandas plotly python-dotenv
2. Install Ollama: https://ollama.ai
3. Pull models: ollama pull llama2, ollama pull mistral, ollama pull llama3
4. Run: streamlit run social_media_ai_app.py
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Page config
st.set_page_config(
    page_title="Social Media Marketing AI Assistant",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions

def get_ollama_models(base_url):
    """Fetch available models from Ollama."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if "models" in data:
            return [model["name"] for model in data["models"]]
        return []
    except Exception as e:
        return []

def generate_with_ollama(prompt: str, model: str, max_tokens: int = 1000, temperature: float = 0.7):
    """Call Ollama API to generate content."""
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature
        }
    }
    try:
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "No response generated")
    except requests.exceptions.ConnectionError:
        return "[Error] Cannot connect to Ollama. Make sure Ollama is running (try: ollama serve)"
    except requests.exceptions.Timeout:
        return "[Error] Request timed out. Try a smaller model or reduce max_tokens."
    except Exception as e:
        return f"[Error] {str(e)}"

def build_campaign_prompt(title, audience, goal, context, brand_voice, constraints, tone, optimize_for):
    """Build prompt for campaign creation."""
    prompt = f"""You are an expert social media marketing strategist. Create a comprehensive campaign strategy.

Campaign Details:
- Title/Name: {title}
- Target Audience: {audience}
- Goal: {goal}
- Brand Voice: {brand_voice}
- Tone: {tone}
- Optimization Focus: {optimize_for}
- Additional Context: {context}
- Constraints: {constraints}

Please provide a detailed campaign strategy including:
1. Campaign Overview & Objectives
2. Platform Strategy (which platforms to use and why)
3. Content Pillars (3-5 main themes)
4. Posting Schedule & Frequency
5. Key Messages & Hashtags
6. Success Metrics & KPIs
7. Timeline & Milestones
8. Budget Allocation Suggestions (if applicable)

Format the response clearly with headers and bullet points."""
    return prompt

def build_content_prompt(content_type, title, audience, goal, context, brand_voice, constraints, tone, length, optimize_for):
    """Build prompt for content creation."""
    length_map = {
        "Short (50-150 words)": "50-150 words",
        "Medium (150-350 words)": "150-350 words",
        "Long (350-800 words)": "350-800 words"
    }
    
    prompts = {
        "Article": f"""You are a professional content writer. Write a {length_map[length]} article.

Details:
- Title: {title}
- Target Audience: {audience}
- Goal: {goal}
- Brand Voice: {brand_voice}
- Tone: {tone}
- Optimization: {optimize_for}
- Context: {context}
- Constraints: {constraints}

Write an engaging article with:
- Compelling headline
- Strong introduction
- Well-structured body with subheadings
- Clear conclusion with call-to-action
- SEO-friendly content (if optimizing for SEO)""",

        "Instagram Caption": f"""You are a social media copywriter. Create an engaging Instagram caption.

Details:
- Campaign: {title}
- Audience: {audience}
- Goal: {goal}
- Brand Voice: {brand_voice}
- Tone: {tone}
- Optimization: {optimize_for}
- Context: {context}
- Constraints: {constraints}

Create a caption that includes:
- Hook in the first line
- Engaging body text ({length_map[length]})
- Clear call-to-action
- 5-10 relevant hashtags
- Emoji usage (if appropriate for brand)""",

        "Twitter/X Thread": f"""You are a social media expert. Create a Twitter/X thread.

Details:
- Topic: {title}
- Audience: {audience}
- Goal: {goal}
- Brand Voice: {brand_voice}
- Tone: {tone}
- Optimization: {optimize_for}
- Context: {context}
- Constraints: {constraints}

Create a thread with:
- 5-8 tweets (each under 280 characters)
- Strong hook in first tweet
- Clear narrative flow
- Relevant hashtags
- Call-to-action in final tweet""",

        "LinkedIn Post": f"""You are a professional B2B content creator. Write a LinkedIn post.

Details:
- Topic: {title}
- Audience: {audience}
- Goal: {goal}
- Brand Voice: {brand_voice}
- Tone: {tone}
- Context: {context}
- Constraints: {constraints}

Create a post with:
- Professional yet engaging opening
- Value-driven content ({length_map[length]})
- Industry insights or data
- Clear takeaways
- Professional call-to-action""",

        "Carousel Content": f"""You are a visual content strategist. Create content for a carousel post.

Details:
- Topic: {title}
- Audience: {audience}
- Goal: {goal}
- Brand Voice: {brand_voice}
- Tone: {tone}
- Context: {context}

Create content for 7-10 carousel slides:
- Slide 1: Eye-catching title/hook
- Slides 2-8: Key points (one per slide, concise)
- Final slide: Summary + CTA

Format each slide clearly with:
SLIDE [number]: [Title]
[Brief content - 20-30 words max per slide]"""
    }
    
    return prompts.get(content_type, prompts["Instagram Caption"])

def build_analysis_prompt(metrics_data, title, audience, goal, context):
    """Build prompt for metrics analysis."""
    prompt = f"""You are a data-driven social media analyst. Analyze the following campaign performance.

Campaign Details:
- Campaign: {title}
- Audience: {audience}
- Goal: {goal}
- Context: {context}

Metrics Data:
{metrics_data}

Please provide a comprehensive analysis including:
1. Performance Overview (what the numbers tell us)
2. Key Wins & Successes
3. Areas for Improvement
4. Audience Insights
5. Content Performance Analysis
6. Actionable Recommendations (at least 5 specific actions)
7. Projected Improvements (if recommendations are implemented)

Be specific, data-driven, and actionable in your analysis."""
    return prompt

# Initialize session state
if 'generated_content' not in st.session_state:
    st.session_state.generated_content = []

# Header
st.markdown('<div class="main-header">📱 Social Media Marketing AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Create campaigns, generate content, and analyze metrics with AI - All on your local machine</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🤖 AI Model Settings")
    
    # Check Ollama connection
    available_models = get_ollama_models(OLLAMA_URL)
    
    if available_models:
        st.success(f"✓ Connected ({len(available_models)} models)")
        
        # Filter for common models or show all
        default_models = ["llama2", "llama3", "mistral", "gemma", "phi"]
        suggested_models = [m for m in available_models if any(d in m.lower() for d in default_models)]
        
        if len(suggested_models) >= 3:
            model_options = suggested_models[:3]
        else:
            model_options = available_models[:3] if len(available_models) >= 3 else available_models
        
        selected_model = st.selectbox(
            "Select Ollama Model",
            options=model_options,
            help="Choose the AI model for content generation"
        )
    else:
        st.error("⚠️ Cannot connect to Ollama")
        st.info("Start Ollama: `ollama serve`\nInstall models: `ollama pull llama2`")
        selected_model = st.text_input("Enter model name", value="llama2")
    
    if st.button("🔄 Refresh Models"):
        st.rerun()
    
    st.markdown("---")
    
    # Generation Settings
    st.header("⚙️ Generation Settings")
    
    tone = st.selectbox(
        "Tone",
        ["Professional", "Casual", "Playful", "Authoritative", "Inspirational", "Conversational"],
        help="The overall tone of the generated content"
    )
    
    length = st.selectbox(
        "Length",
        ["Short (50-150 words)", "Medium (150-350 words)", "Long (350-800 words)"],
        index=1,
        help="Approximate length of generated content"
    )
    
    st.markdown("---")
    
    # Optimization Purpose
    st.header("🎯 Optimization Purpose")
    
    optimize_for = st.selectbox(
        "Optimize For",
        ["Engagement (Likes, Comments, Shares)", "SEO (Search Engine Optimization)", "Readability (Easy to understand)", "Conversion (Sales, Sign-ups)"],
        help="Primary optimization objective"
    )
    
    st.markdown("---")
    
    # Advanced Settings
    with st.expander("🔧 Advanced Settings"):
        temperature = st.slider(
            "Creativity (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=256,
            max_value=2048,
            value=1000,
            step=128,
            help="Maximum length of generated response"
        )

# Main Content Area
st.markdown("---")

# Task Selection
task_type = st.radio(
    "**Select Task Type:**",
    ["📋 Create Campaign Strategy", "✍️ Create Content", "📊 Analyze Metrics"],
    horizontal=True
)

st.markdown("---")

# Input Form
with st.form("main_form"):
    st.subheader("📝 Input Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        title = st.text_input(
            "Campaign/Content Title *",
            placeholder="e.g., Summer Product Launch 2024",
            help="Name of your campaign or content piece"
        )
        
        audience = st.text_input(
            "Target Audience *",
            placeholder="e.g., Young professionals aged 25-35, tech-savvy",
            help="Who are you trying to reach?"
        )
        
        goal = st.text_input(
            "Primary Goal *",
            placeholder="e.g., Increase brand awareness by 30%",
            help="What do you want to achieve?"
        )
    
    with col2:
        brand_voice = st.text_input(
            "Brand Voice",
            placeholder="e.g., Friendly, innovative, customer-focused",
            help="How should your brand sound?"
        )
        
        constraints = st.text_input(
            "Constraints",
            placeholder="e.g., Budget: $5000, Timeline: 2 months",
            help="Any limitations or requirements"
        )
        
        # Content type selection (only for Create Content task)
        if task_type == "✍️ Create Content":
            content_type = st.selectbox(
                "Content Type *",
                ["Instagram Caption", "Twitter/X Thread", "LinkedIn Post", "Article", "Carousel Content"]
            )
    
    context = st.text_area(
        "Additional Context",
        placeholder="Any additional information that would help create better content...",
        height=100,
        help="Provide any relevant background information"
    )
    
    # Metrics input (only for Analyze Metrics task)
    if task_type == "📊 Analyze Metrics":
        st.subheader("📈 Campaign Metrics")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            impressions = st.number_input("Impressions", min_value=0, value=10000)
            reach = st.number_input("Reach", min_value=0, value=8000)
            engagement = st.number_input("Engagement", min_value=0, value=500)
        
        with col_m2:
            likes = st.number_input("Likes", min_value=0, value=300)
            comments = st.number_input("Comments", min_value=0, value=50)
            shares = st.number_input("Shares", min_value=0, value=40)
        
        with col_m3:
            clicks = st.number_input("Clicks", min_value=0, value=200)
            conversions = st.number_input("Conversions", min_value=0, value=25)
            spend = st.number_input("Ad Spend ($)", min_value=0.0, value=500.0)
    
    st.markdown("---")
    
    col_submit1, col_submit2, col_submit3 = st.columns([2, 1, 2])
    with col_submit2:
        submitted = st.form_submit_button("🚀 Generate", use_container_width=True)

# Process Form Submission
if submitted:
    # Validation
    if not title or not audience or not goal:
        st.error("⚠️ Please fill in all required fields (marked with *)")
    else:
        st.markdown("---")
        
        with st.spinner("🤖 AI is working on your request... This may take 30-60 seconds."):
            
            # Build appropriate prompt based on task type
            if task_type == "📋 Create Campaign Strategy":
                prompt = build_campaign_prompt(
                    title, audience, goal, context, brand_voice, 
                    constraints, tone, optimize_for
                )
            
            elif task_type == "✍️ Create Content":
                prompt = build_content_prompt(
                    content_type, title, audience, goal, context, 
                    brand_voice, constraints, tone, length, optimize_for
                )
            
            else:  # Analyze Metrics
                metrics_data = f"""
Impressions: {impressions:,}
Reach: {reach:,}
Engagement: {engagement:,}
Likes: {likes:,}
Comments: {comments:,}
Shares: {shares:,}
Clicks: {clicks:,}
Conversions: {conversions:,}
Ad Spend: ${spend:,.2f}

Calculated Metrics:
- Engagement Rate: {(engagement/reach*100):.2f}%
- Click-Through Rate: {(clicks/impressions*100):.2f}%
- Conversion Rate: {(conversions/clicks*100):.2f}% (of clicks)
- Cost Per Conversion: ${(spend/conversions):.2f} (if conversions > 0)
- Cost Per Click: ${(spend/clicks):.2f} (if clicks > 0)
"""
                prompt = build_analysis_prompt(metrics_data, title, audience, goal, context)
            
            # Generate content
            result = generate_with_ollama(prompt, selected_model, max_tokens, temperature)
            
            # Display results
            if result.startswith("[Error]"):
                st.error(result)
            else:
                st.success("✅ Generation Complete!")
                
                # Display result in a nice box
                st.markdown("### 📄 Generated Output")
                st.markdown(result)
                
                # Add to history
                st.session_state.generated_content.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "task_type": task_type,
                    "title": title,
                    "content": result
                })
                
                # Download button
                col_dl1, col_dl2, col_dl3 = st.columns([2, 1, 2])
                with col_dl2:
                    st.download_button(
                        label="📥 Download Result",
                        data=result,
                        file_name=f"{title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                # Show metrics visualization for analysis task
                if task_type == "📊 Analyze Metrics":
                    st.markdown("---")
                    st.markdown("### 📊 Metrics Visualization")
                    
                    col_v1, col_v2 = st.columns(2)
                    
                    with col_v1:
                        # Engagement metrics
                        fig1 = go.Figure(data=[
                            go.Bar(name='Likes', x=['Engagement'], y=[likes], marker_color='#1f77b4'),
                            go.Bar(name='Comments', x=['Engagement'], y=[comments], marker_color='#ff7f0e'),
                            go.Bar(name='Shares', x=['Engagement'], y=[shares], marker_color='#2ca02c')
                        ])
                        fig1.update_layout(title="Engagement Breakdown", barmode='group', height=300)
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col_v2:
                        # Funnel metrics
                        funnel_data = pd.DataFrame({
                            'Stage': ['Impressions', 'Reach', 'Engagement', 'Clicks', 'Conversions'],
                            'Count': [impressions, reach, engagement, clicks, conversions]
                        })
                        fig2 = px.funnel(funnel_data, x='Count', y='Stage', title="Conversion Funnel")
                        fig2.update_layout(height=300)
                        st.plotly_chart(fig2, use_container_width=True)

# History Section
if st.session_state.generated_content:
    st.markdown("---")
    with st.expander("📚 Generation History", expanded=False):
        for idx, item in enumerate(reversed(st.session_state.generated_content[-5:])):
            st.markdown(f"**{item['timestamp']}** - {item['task_type']} - *{item['title']}*")
            with st.expander(f"View content #{len(st.session_state.generated_content) - idx}"):
                st.markdown(item['content'])
                st.download_button(
                    label="Download",
                    data=item['content'],
                    file_name=f"history_{idx}.txt",
                    key=f"history_download_{idx}"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Social Media Marketing AI Assistant</strong></p>
    <p>Powered by Ollama • All processing happens locally on your machine</p>
    <p style='font-size: 0.8rem;'>Need help? Make sure Ollama is running: <code>ollama serve</code></p>
</div>
""", unsafe_allow_html=True)
