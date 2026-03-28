import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image
import json
from google.oauth2 import service_account

# --- MOBILE UI & STYLING ---
st.set_page_config(page_title="Persona AI", page_icon="🎨", layout="centered")

st.markdown("""
    <style>
    /* Dark Theme */
    .stApp { background-color: #0e1117; color: white; }
    
    /* Action Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 50px;
        height: 3.5rem;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4);
        margin-top: 10px;
    }
    
    /* Clean up the selectbox label */
    .stSelectbox label { color: #a0aec0; font-weight: bold; }
    
    /* Hide default streamlit menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- GOOGLE AUTH ---
def init_google_ai():
    try:
        if "gcp_service_account" in st.secrets:
            creds_info = json.loads(st.secrets["gcp_service_account"])
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
            creds = service_account.Credentials.from_service_account_info(creds_info)
            vertexai.init(project=creds_info["project_id"], location="us-central1", credentials=creds)
            return True
        return False
    except Exception as e:
        st.error(f"Auth Error: {e}")
        return False

# --- STYLE PROMPTS ---
STYLES = {
    "Cyberpunk": "A high-detail cyberpunk portrait of the person in [1], neon violet lighting, futuristic city background, 8k cinematic.",
    "3D Pixar": "A cute 3D Pixar-style animated character portrait of the person in [1], big expressive eyes, smooth textures, vibrant lighting.",
    "Oil Painting": "A classic royal oil painting of the person in [1], museum quality, visible brushstrokes, rich classical lighting.",
    "Viking": "A cinematic warrior portrait of the person in [1] as a Viking, wearing furs and leather, snowy mountain background, fierce expression."
}

# --- APP MAIN ---
st.title("✨ Persona AI")
st.write("Snap a selfie and choose your transformation style.")

if init_google_ai():
    # 1. Camera Input
    user_photo = st.camera_input("Take a selfie", label_visibility="collapsed")
    
    if user_photo:
        st.divider()
        
        # 2. Style Selection (Dropdown Menu)
        selected_style = st.selectbox(
            "CHOOSE YOUR STYLE:", 
            options=list(STYLES.keys()),
            index=0
        )
        
        st.write(f"Selected: **{selected_style}**")

        # 3. Generation Button
        if st.button(f"GENERATE {selected_style.upper()} ✨"):
            with st.spinner("AI is reimagining your portrait..."):
                try:
                    # Initialize Imagen 3 Capability Model
                    model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")
                    ref_img = Image(image_bytes=user_photo.getvalue())
                    
                    # Generate Image using Subject Reference
                    response = model.generate_images(
                        prompt=STYLES[selected_style],
                        reference_images=[ref_img],
                        number_of_images=1
                    )

                    if response and response.images:
                        st.success("Your transformation is ready!")
                        st.image(response.images[0]._image_bytes, use_container_width=True)
                        
                        # Save/Download Button
                        st.download_button(
                            label="📥 Save to Photos",
                            data=response.images[0]._image_bytes,
                            file_name=f"persona_{selected_style.lower()}.png",
                            mime="image/png"
                        )
                    else:
                        st.warning("Could not generate image. The photo might have triggered a safety filter.")
                except Exception as e:
                    st.error(f"API Error: {e}")
