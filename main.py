import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from few_shot import FewShotPosts
from post_generator import generate_post, analyze_post, generate_image_prompt, generate_reply_suggestions, generate_hook_variations

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


# Options for length, language, platform and tone
length_options = ["Short", "Medium", "Long"]
language_options = ["English", "Hinglish", "Spanish", "French", "German", "Mandarin"]
tone_options = ["Professional", "Casual", "Storytelling", "Motivational", "Humorous", "Educational"]
platform_options = ["LinkedIn", "Twitter", "Instagram"]

def scrape_url_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text[:5000] # Limit to avoid context overflow
    except Exception as e:
        return f"Error scraping URL: {str(e)}"

def text_to_speech(text):
    if not TTS_AVAILABLE:
        return None
    # Initialize engine
    engine = pyttsx3.init()
    # Path for temporary audio file
    audio_path = "temp_audio.mp3"
    if os.path.exists(audio_path):
        os.remove(audio_path)
    engine.save_to_file(text, audio_path)
    engine.runAndWait()
    return audio_path


# Main app layout
def main():
    st.set_page_config(page_title="AI Social Media Manager", page_icon="🚀", layout="wide")
    st.title("AI Social Media Manager")
    
    # Initialize session state for history
    if "post_history" not in st.session_state:
        st.session_state.post_history = []

    # --- SIDEBAR: Brand Persona & History ---
    st.sidebar.title("Your Brand Persona")
    st.sidebar.info("Tell the AI who you are so it can write authentically in your voice.")
    author_background = st.sidebar.text_area(
        "Who are you?", 
        placeholder="e.g. I am a Junior Data Analyst learning Python and GenAI. I love sharing my coding journey...",
        height=150
    )
    st.sidebar.divider()
    
    st.sidebar.title("Session History")
    if not st.session_state.post_history:
        st.sidebar.write("No posts generated yet.")
    else:
        for i, past_post in enumerate(reversed(st.session_state.post_history)):
            with st.sidebar.expander(f"Post {len(st.session_state.post_history)-i}: {past_post['topic'][:15]}..."):
                st.write(past_post['post'])
                if st.button("Copy", key=f"copy_{i}"):
                    st.write("*(Highlight and copy the text above)*")

    st.sidebar.divider()
    
    # Platform selector
    selected_platform = st.sidebar.selectbox("Select Target Platform", options=platform_options)

    # Create columns for the dropdowns
    col1, col2, col3, col4 = st.columns(4)

    fs = FewShotPosts()
    tags = fs.get_tags()
    
    with col1:
        # Dropdown for Topic (Tags)
        selected_tag = st.selectbox("Topic", options=tags)

    with col2:
        # Dropdown for Length
        selected_length = st.selectbox("Length", options=length_options)

    with col3:
        # Dropdown for Language
        selected_language = st.selectbox("Language", options=language_options)
        
    with col4:
        # Dropdown for Tone
        selected_tone = st.selectbox("Tone", options=tone_options)

    # Content Repurposer / Source Material
    st.markdown("###Content Repurposing (Optional)")
    source_url = st.text_input("Paste an Article/Blog URL to repurpose:")
    
    source_material = st.text_area(
        "Or paste raw notes/transcript here:",
        height=100, 
        placeholder="Paste your source text here..."
    )

    # Custom Topic Input
    st.markdown("###Core Topic")
    custom_topic = st.text_input("Or enter your own custom topic (leaves dropdown selected tag otherwise):")
    final_topic = custom_topic if custom_topic else selected_tag

    # Generate Button
    if st.button("Generate Post", type="primary", use_container_width=True):
        with st.spinner("Generating post and hashtags..."):
            
            # Handle URL scraping if provided
            final_source_material = source_material
            if source_url:
                with st.spinner("Scraping URL content..."):
                    scraped_text = scrape_url_text(source_url)
                    final_source_material = scraped_text + "\n\n" + source_material
            
            post, hashtags = generate_post(selected_length, selected_language, final_topic, selected_tone, selected_platform, author_background, final_source_material)
            
            # Save to session history
            st.session_state.post_history.append({"topic": final_topic, "post": post})
            st.success("Post generated successfully!")
            
        # Editable Output
        st.subheader("Your Generated Post")
        edited_post = st.text_area("Edit your post below:", value=post, height=250)
        
        # Action Buttons Row
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("Regenerate Post"):
                st.rerun()
        with action_col2:
            # Download to CSV feature
            df_export = pd.DataFrame({"Platform": [selected_platform], "Topic": [final_topic], "Content": [edited_post], "Hashtags": [hashtags]})
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button("Download to CSV (For Hootsuite/Buffer)", data=csv, file_name="social_post.csv", mime="text/csv")

        # Audio Player (Text-to-Speech)
        if TTS_AVAILABLE:
            st.write("🔊 **Listen to Post:**")
            audio_file = text_to_speech(edited_post)
            if audio_file and os.path.exists(audio_file):
                st.audio(audio_file, format="audio/mp3")

        st.subheader("Suggested Hashtags")
        st.write(hashtags)
        
        st.divider()
        st.subheader("Advanced AI Features")

        # Hook A/B Testing Generator
        with st.expander("Generate Hook Variations (A/B Testing)"):
            with st.spinner("Writing alternative hooks..."):
                hooks = generate_hook_variations(edited_post)
                st.info("The first 2 lines determine your reach. Try swapping your current intro with one of these variations:")
                st.markdown(hooks)

        # Performance Prediction Feature
        with st.expander("Analyze Post Performance (AI Predictor)"):
            with st.spinner("Analyzing post metrics..."):
                analysis = analyze_post(edited_post)
                st.markdown(analysis)
                
        # Image Prompt Generator
        with st.expander("Generate GenAI Image Prompt (Midjourney/DALL-E)"):
            with st.spinner("Crafting visual prompt..."):
                image_prompt = generate_image_prompt(edited_post)
                st.info("Copy this prompt into Midjourney, DALL-E, or Stable Diffusion to create a matching graphic!")
                st.code(image_prompt, language="text")

        # Comment & Auto-Reply Strategy
        with st.expander("Generate Engagement Strategy (Comments & Replies)"):
            with st.spinner("Brainstorming community engagement..."):
                replies = generate_reply_suggestions(edited_post)
                st.markdown(replies)
        
        with st.expander("See similar reference posts used (Few-Shot / RAG examples)"):
            # Uses the newly added RAG approach behind the scenes
            examples = fs.get_similar_posts_by_topic(final_topic)
            if examples:
                for i, ex in enumerate(examples):
                    st.markdown(f"**Example {i+1}:**")
                    st.write(ex['text'])
                    st.divider()
                    if i == 1: # Showing max 2 items to match the generator logic
                        break
            else:
                st.write("No direct examples found for these exact criteria.")



# Run the app
if __name__ == "__main__":
    main()
