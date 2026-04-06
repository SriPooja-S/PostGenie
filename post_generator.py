from llm_helper import llm
from few_shot import FewShotPosts

few_shot = FewShotPosts()


def get_length_str(length):
    if length == "Short":
        return "1 to 5 lines"
    if length == "Medium":
        return "6 to 10 lines"
    if length == "Long":
        return "11 to 15 lines"


def generate_post(length, language, tag, tone, platform="LinkedIn", author_background="", source_material=""):
    prompt = get_prompt(length, language, tag, tone, platform, author_background, source_material)
    response = llm.invoke(prompt)
    post_content = response.content
    
    # Generate Hashtags
    hashtags_prompt = f"Generate 10 relevant {platform} hashtags for a post about: {tag}. Only return space-separated hashtags. No intro."
    hashtags_response = llm.invoke(hashtags_prompt)
    hashtags = hashtags_response.content
    
    return post_content, hashtags

def generate_hook_variations(post):
    prompt = f"""
    The hook (first 2 lines) is the most important part of a social media post. 
    Based on the following post, generate 3 ALTERNATIVE, highly-engaging hooks (A/B Testing variations) that the user could swap out.
    Make one Data-Driven, one Story-Driven, and one Controversial/Bold.
    Just output the 3 hooks separated by newlines, clearly labeled.
    
    Post: {post}
    """
    response = llm.invoke(prompt)
    return response.content

def analyze_post(post):
    analysis_prompt = f"""
    Analyze this social media post and provide a short rating:
    - Engagement Potential (1-10)
    - Clarity (1-10)
    - Virality Potential (1-10)
    
    Provide the response in a brief Markdown format.
    Post: {post}
    """
    response = llm.invoke(analysis_prompt)
    return response.content

def generate_image_prompt(post):
    prompt = f"""
    Based on the following social media post, write a highly detailed, imaginative image generation prompt that could be used in Midjourney or DALL-E to create an eye-catching, accompanying graphic.
    Keep the prompt concise (under 40 words), focusing on visual elements, art style, lighting, and mood. No intro or preamble.
    
    Post: {post}
    """
    response = llm.invoke(prompt)
    return response.content

def generate_reply_suggestions(post):
    prompt = f"""
    Based on this social media post, pretend you are managing the community engagement for it. 
    Generate 3 realistic engaging comments that the audience might leave, AND generate the author's perfect reply to each.
    Format nicely with emojis.
    
    Post: {post}
    """
    response = llm.invoke(prompt)
    return response.content

def get_prompt(length, language, tag, tone, platform, author_background, source_material):
    length_str = get_length_str(length)

    prompt = f'''
    Generate a {platform} post using the below information. No preamble.

    1) Topic: {tag}
    2) Length: {length_str}
    3) Language: {language}
    4) Tone/Style: {tone}
    5) Author Background/Context: {author_background if author_background else "General professional"}
    '''
    
    if source_material:
        prompt += f"\n6) Source Material to Repurpose: {source_material}\nMake sure to extract the key insights from this material and transform it into the post.\n"
        
    prompt += '''
    Structure the post carefully:
    - Start with a strong, attention-grabbing Hook (first 1-2 lines)
    - Main Body Content with value (incorporate the Author Background naturally if provided)
    - End with a clear Call to Action (CTA)
    
    If Language is Hinglish then it means it is a mix of Hindi and English. 
    The script for the generated post should always be English.
    '''
    # prompt = prompt.format(post_topic=tag, post_length=length_str, post_language=language)

    # Use RAG to get most context-relevant examples based on topic
    examples = few_shot.get_similar_posts_by_topic(tag)

    if len(examples) > 0:
        prompt += "\n4) Use the writing style and format as per the following examples:\n"

    for i, post in enumerate(examples):
        post_text = post['text']
        prompt += f'\n\n Example {i+1}: \n\n {post_text}'

        if i == 1: # Use max two samples
            break

    return prompt


if __name__ == "__main__":
    post, hashtags = generate_post("Medium", "English", "Mental Health", "Professional")
    print(post)
    print("\nHashtags:\n", hashtags)