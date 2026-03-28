import streamlit as st
from openai import OpenAI
from PIL import Image
import io

# --- MOBILE UI SETUP ---
st.set_page_config(page_title="Persona AI", page_icon="🎨", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .stButton>button { 
        width: 100%; border-radius: 50px; height: 3.5rem; 
        background: linear-gradient(90deg, #10a37f 0%, #00775a 100%);
        color: white; border: none; font-weight: bold; font-size: 1.2rem;
    }
    .stSelectbox label { color: #888; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- OPENAI AUTH (SECURE) ---
def get_openai_client():
    try:
        # This will look for OPENAI_API_KEY in your Streamlit Secrets
        api_key = st.secrets["OPENAI_API_KEY"]
        return OpenAI(api_key=api_key)
    except Exception:
        st.error("🔑 API Key Missing: Please add 'OPENAI_API_KEY' to your Streamlit Secrets.")
        return None

# --- HELPER: FORMAT IMAGE FOR DALL-E 2 ---
def prepare_image_for_variation(uploaded_file):
    """DALL-E 2 variations require a square PNG < 4MB."""
    img = Image.open(uploaded_file)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Square Crop
    width, height = img.size
    side = min(width, height)
    left = (width - side) / 2
    top = (height - side) / 2
    img = img.crop((left, top, left + side, top + side))
    
    # Resize and Compress
    img = img.resize((1024, 1024))
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG')
    return byte_arr.getvalue()

# --- STYLE PROMPTS ---
STYLES = {
    "Cyberpunk": "A high-detail cyberpunk portrait, neon violet lighting, futuristic city background, cinematic.",
    "3D Pixar": "A cute 3D Pixar-style character, big expressive eyes, smooth textures, vibrant colors.",
    "Oil Painting": "A classic royal oil painting, rich brushstrokes, museum quality, golden lighting.",
    "Viking Warrior": "A cinematic warrior portrait, fur armor, snowy mountain background, epic lighting."
}

# --- MAIN APP ---
st.title("✨ Persona AI")
st.write("Transform your portrait using OpenAI.")

client = get_openai_client()

if client:
    # 1. Camera Input
    user_photo = st.camera_input("Take a selfie", label_visibility="collapsed")

    if user_photo:
        st.divider()
        
        # 2. Configuration Grid
        col1, col2 = st.columns(2)
        with col1:
            selected_model = st.selectbox("AI MODEL:", ["dall-e-3", "dall-e-2"])
        with col2:
            selected_style = st.selectbox("ART STYLE:", list(STYLES.keys()))

        # 3. Execution
        if st.button(f"GENERATE {selected_style.upper()} ✨"):
            with st.spinner(f"Using {selected_model} to reimagining you..."):
                try:
                    if selected_model == "dall-e-2":
                        # Image-to-Image Variation
                        img_bytes = prepare_image_for_variation(user_photo)
                        response = client.images.create_variation(
                            image=img_bytes,
                            n=1,
                            size="1024x1024"
                        )
                    else:
                        # DALL-E 3 (Prompt based stylization)
                        # Note: DALL-E 3 API is currently text-only. It generates based on prompt.
                        prompt = f"A professional portrait in {STYLES[selected_style]}. Ensure it looks like a high-quality headshot."
                        response = client.images.generate(
                            model="dall-e-3",
                            prompt=prompt,
                            n=1,
                            size="1024x1024"
                        )

                    # Display Result
                    image_url = response.data[0].url
                    st.success("Transformation Complete!")
                    st.image(image_url, use_container_width=True)
                    
                    st.info("💡 Pro Tip: DALL-E 2 keeps your pose/face shape. DALL-E 3 creates high-quality art based on your style choice.")
                    
                except Exception as e:
                    st.error(f"OpenAI Error: {e}")
