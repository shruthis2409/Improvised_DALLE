#pip install streamlit openai pyshorteners requests translate opencv-python numpy language-tool-python
import subprocess
req=
subprocess.check_call(['pip','install','-r',req])
import streamlit as st
import openai
import pyshorteners
from PIL import Image
import io
import requests
import sqlite3
from translate import Translator
import cv2
import numpy as np
from language_tool_python import LanguageTool

# Set your OpenAI API key
openai.api_key = "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"

# Create or connect to the SQLite database
conn = sqlite3.connect('image_variations.db')
cursor = conn.cursor()

# Create a table to store image variations
cursor.execute('''
    CREATE TABLE IF NOT EXISTS image_variations (
        id INTEGER PRIMARY KEY,
        original_image_url TEXT,
        variation_url TEXT
    )
''')
conn.commit()

# Streamlit app title
st.title("Multilingual Improvised DALL-E Image Generation with Streamlit and OpenAI GPT-3")

# Add a mode selection radio button
mode = st.radio("Select Mode", ["Generate Image"])

# To generate shortened URL
def generate_short_url(image_url):
    return pyshorteners.Shortener().tinyurl.short(image_url)

# To generate AI image
def generate_image():
    text_prompt = st.text_input("Enter a text prompt:")
    target_language = st.selectbox("Select Target Language", ["en", "fr", "es", "de"], index=0)  # Set English as default language

    if st.button("Generate Image") and text_prompt:
        try:
            # Translate the text prompt to English if the selected language is not English
            if target_language != "en":
                translator = Translator(to_lang=target_language)
                text_prompt = translator.translate(text_prompt)

            # Spell check the text prompt only if the selected language is English
            if target_language == "en":
                tool = LanguageTool('en-US')
                matches = tool.check(text_prompt)

                # Display spelling suggestions, if any
                if matches:
                    st.warning("Spelling Errors in the Text Prompt:")
                    return None  # Return None to prevent image generation when there are spelling errors

            # Generate image using translated prompt
            response_gpt3 = openai.Image.create(
                prompt=f"{text_prompt} ",
                n=1
            )
            image_url = response_gpt3['data'][0]['url']
            return image_url
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")

# To generate image variation and save to the database
def create_variation(image_url):
    try:
        # Download the image
        image_content = Image.open(io.BytesIO(requests.get(image_url).content)).convert("RGB")

        # Resize the image using OpenCV to the fixed resolution
        user_selected_resolution = 1024  # Change this value if needed
        image_cv2 = cv2.cvtColor(np.array(image_content), cv2.COLOR_RGB2BGR)
        resized_image_cv2 = cv2.resize(image_cv2, (user_selected_resolution, user_selected_resolution))
        resized_image_pil = Image.fromarray(cv2.cvtColor(resized_image_cv2, cv2.COLOR_BGR2RGB))

        # Convert the image to PNG format
        png_buffer = io.BytesIO()
        resized_image_pil.save(png_buffer, format="PNG")
        png_image = png_buffer.getvalue()

        # Check if the image is less than 4 MB
        if len(png_image) < 4 * 1024 * 1024:
            response_gpt3 = openai.Image.create_variation(
                image=png_image,
                n=1
            )
            variation_url = response_gpt3['data'][0]['url']

            # Save to the database
            cursor.execute('INSERT INTO image_variations (original_image_url, variation_url) VALUES (?, ?)',
                           (image_url, variation_url))
            conn.commit()

            st.image(variation_url, caption='Generated Image', use_column_width=True)
        else:
            st.error("Error: Converted PNG image size exceeds 4 MB.")
    except Exception as e:
        st.error(f"Error creating variation: {str(e)}")

# Generate image and create variation
if mode == "Generate Image":
    generated_image_url = generate_image()
    if generated_image_url:
        create_variation(generated_image_url)

# Close the database connection
conn.close()
