# 📱 Social Media Marketing AI Assistant

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Ollama](https://img.shields.io/badge/AI-Ollama-orange)

A powerful, privacy-focused AI assistant for social media marketers that runs entirely on your local machine. Create campaigns, generate engaging content, and analyze metrics using state-of-the-art language models through Ollama.

## ✨ Features

### 🎯 Three Core Capabilities

1. **📋 Campaign Strategy Creation**
   - Complete campaign planning with platform-specific strategies
   - Content pillars and posting schedules
   - Success metrics and KPIs definition
   - Timeline and milestone planning
   - Budget allocation suggestions

2. **✍️ Content Generation**
   - **Instagram Captions** - Engaging captions with hashtags and emojis
   - **Twitter/X Threads** - Multi-tweet narratives (5-8 tweets)
   - **LinkedIn Posts** - Professional B2B content
   - **Blog Articles** - Long-form content with SEO optimization
   - **Carousel Content** - Slide-by-slide content for visual posts

3. **📊 Metrics Analysis**
   - Performance data interpretation
   - Key insights and patterns identification
   - Actionable recommendations
   - Visual analytics with interactive charts
   - ROI and conversion tracking

### 🎨 Advanced Features

- **Multiple AI Models** - Support for Llama 2, Llama 3, Mistral, and more
- **Customizable Tone** - Professional, Casual, Playful, Authoritative, Inspirational, Conversational
- **Length Control** - Short (50-150 words), Medium (150-350 words), Long (350-800 words)
- **Optimization Focus** - Engagement, SEO, Readability, or Conversion
- **Generation History** - Track and download your last 5 generations
- **Privacy First** - All processing happens locally, no data sent to external servers

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Ollama installed on your system

### Installation

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd social-media-ai-assistant
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama**
   
   Visit [https://ollama.ai](https://ollama.ai) and follow installation instructions for your operating system.

   **macOS / Linux:**
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

   **Windows:**
   Download the installer from [ollama.ai/download](https://ollama.ai/download)

4. **Pull AI models**
   ```bash
   ollama pull llama2
   ollama pull mistral
   ollama pull llama3
   ```

   You can pull any model you prefer. See available models at [ollama.ai/library](https://ollama.ai/library)

5. **Start Ollama server**
   ```bash
   ollama serve
   ```

6. **Run the application**
   ```bash
   streamlit run social_media_ai_app.py
   ```

The app will automatically open in your default browser at `http://localhost:8501`

## 📋 Requirements

Create a `requirements.txt` file with the following dependencies:

```
streamlit>=1.28.0
requests>=2.31.0
pandas>=2.0.0
plotly>=5.17.0
python-dotenv>=1.0.0
```

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## 🎮 Usage Guide

### 1. Select Your AI Model

In the sidebar, choose from available Ollama models. The app will automatically detect models installed on your system.

### 2. Configure Generation Settings

- **Tone**: Choose how your content should sound (Professional, Casual, etc.)
- **Length**: Select desired content length
- **Optimization**: Pick your primary objective (Engagement, SEO, Readability, Conversion)

### 3. Choose Your Task

#### Creating a Campaign Strategy

1. Select "📋 Create Campaign Strategy"
2. Fill in:
   - Campaign Title
   - Target Audience
   - Primary Goal
   - Brand Voice
   - Constraints (budget, timeline, etc.)
   - Additional Context
3. Click "🚀 Generate"

**Example Output:**
- Campaign overview and objectives
- Platform-specific strategies
- Content pillars and themes
- Posting schedule
- Success metrics
- Timeline and milestones

#### Generating Content

1. Select "✍️ Create Content"
2. Choose content type (Instagram, Twitter, LinkedIn, Article, Carousel)
3. Fill in campaign details
4. Click "🚀 Generate"

**Example: Instagram Caption**
```
🌟 Launching something incredible! 

We've spent months perfecting this just for you. 
Our new AI-powered tool helps marketers create 
stunning campaigns in minutes, not hours. ✨

Ready to transform your workflow? 
Link in bio 👆

#MarketingAI #SocialMediaMarketing #ContentCreation 
#DigitalMarketing #MarketingTools #AITools
```

#### Analyzing Metrics

1. Select "📊 Analyze Metrics"
2. Input your campaign metrics:
   - Impressions, Reach, Engagement
   - Likes, Comments, Shares
   - Clicks, Conversions, Ad Spend
3. Click "🚀 Generate"

**The AI will provide:**
- Performance overview
- Key wins and successes
- Areas for improvement
- Actionable recommendations
- Visual analytics

### 4. Download and Use

- Click "📥 Download Result" to save your generated content
- Access previous generations in the "📚 Generation History" section

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root (optional):

```env
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

### Advanced Settings

In the sidebar, expand "🔧 Advanced Settings" to adjust:

- **Creativity (Temperature)**: 0.0 (focused) to 1.0 (creative)
  - Recommended: 0.7 for balanced output
- **Max Tokens**: 256 to 2048
  - Recommended: 1000 for most tasks

## 📊 Supported Content Types

| Content Type | Platform | Best For | Typical Length |
|-------------|----------|----------|----------------|
| Instagram Caption | Instagram | Visual posts | 50-200 words |
| Twitter/X Thread | Twitter/X | Stories, tutorials | 5-8 tweets |
| LinkedIn Post | LinkedIn | B2B, thought leadership | 150-300 words |
| Article | Blog, website | In-depth content | 350-800 words |
| Carousel | Instagram, LinkedIn | Educational content | 7-10 slides |

## 🔧 Troubleshooting

### Ollama Connection Issues

**Problem:** "Cannot connect to Ollama"

**Solutions:**
1. Ensure Ollama is running: `ollama serve`
2. Check if Ollama is on the correct port (default: 11434)
3. Verify firewall settings

### No Models Available

**Problem:** "No models found"

**Solutions:**
1. Pull at least one model: `ollama pull llama2`
2. Verify installation: `ollama list`
3. Click "🔄 Refresh Models" in the app

### Slow Generation

**Problem:** Content takes too long to generate

**Solutions:**
1. Use a smaller model (e.g., `llama2:7b` instead of `llama2:13b`)
2. Reduce max_tokens in Advanced Settings
3. Lower the temperature setting
4. Check your system resources (RAM, CPU)

### Model Not Loading

**Problem:** "[Error] Model not found"

**Solutions:**
1. Pull the specific model: `ollama pull <model-name>`
2. Check model name spelling
3. List available models: `ollama list`

## 💡 Best Practices

### For Campaign Creation
- Be specific about your target audience demographics
- Include clear, measurable goals
- Mention budget and timeline constraints
- Describe your brand voice in detail

### For Content Generation
- Use relevant keywords in your context
- Specify any hashtags or mentions to include
- Mention character limits if applicable
- Include examples of successful past content

### For Metrics Analysis
- Input accurate, complete metrics data
- Provide context about campaign goals
- Mention any external factors that affected performance
- Include previous campaign data for comparison

## 🔐 Privacy & Security

- ✅ **100% Local Processing** - All AI computations happen on your machine
- ✅ **No Data Transmission** - Your content never leaves your computer
- ✅ **No API Keys Required** - No third-party services involved
- ✅ **Offline Capable** - Works without internet connection (after models are downloaded)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Roadmap

- [ ] Add support for Facebook and TikTok content
- [ ] Implement content calendar view
- [ ] Add competitor analysis features
- [ ] Include A/B testing recommendations
- [ ] Export to CSV/PDF functionality
- [ ] Multi-language support
- [ ] Integration with social media APIs
- [ ] Content scheduling capability

## 🆘 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the [Ollama documentation](https://github.com/ollama/ollama)
- Review [Streamlit documentation](https://docs.streamlit.io)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) for making local LLMs accessible
- [Streamlit](https://streamlit.io) for the amazing web framework
- [Plotly](https://plotly.com) for interactive visualizations
- The open-source AI community

## 📞 Contact

For professional inquiries or collaborations, please reach out through GitHub.

---

**Made with ❤️ for Social Media Marketers**

*Star ⭐ this repository if you find it helpful!*
