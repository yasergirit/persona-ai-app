import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image
import json
from google.oauth2 import service_account

# --- MOBILE UI ---
st.set_page_config(page_title="Persona AI", page_icon="🎨")

# --- GOOGLE AUTH FIX ---
def init_google_ai():
    try:
        if "gcp_service_account" in st.secrets:
            # Load secrets as a dictionary
            # We convert to a standard dict to ensure compatibility
            creds_info = dict(st.secrets["gcp_service_account"])
            
            # THE FIX: Replace double-escaped newlines with real newlines
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
            
            creds = service_account.Credentials.from_service_account_info(creds_info)
            vertexai.init(project=creds_info["project_id"], location="us-central1", credentials=creds)
            return True
        else:
            st.error("Setup Required: Paste your JSON key into Streamlit Secrets.")
            return False
    except Exception as e:
        st.error(f"Authentication Error: {e}")
        return False

# --- APP LAYOUT ---
st.title("✨ Persona AI")

if init_google_ai():
    img_file = st.camera_input("Take a selfie")
    if not img_file:
        img_file = st.file_uploader("Or upload from gallery", type=["jpg", "png"])

    if img_file:
        st.subheader("Select Style")
        styles = {
            "Cyberpunk": "A high-detail cyberpunk portrait of [1], neon lighting, futuristic.",
            "3D Pixar": "A 3D animated character version of [1], Pixar style, vibrant.",
            "Oil Painting": "A classic oil painting of [1], rich brushstrokes, gold lighting.",
            "Viking": "A cinematic warrior portrait of [1], fur clothing, snowy mountains."
        }
        
        choice = st.selectbox("Style:", list(styles.keys()))
        
        if st.button("CREATE MAGIC ✨"):
            with st.spinner("AI is reimagining you..."):
                try:
                    model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")
                    user_img = Image(image_bytes=img_file.getvalue())
                    
                    response = model.edit_image(
                        prompt=styles[choice],
                        reference_images=[user_img],
                        number_of_images=1,
                        guidance_scale=60 
                    )
                    
                    st.image(response.images[0]._image_bytes, use_container_width=True)
                    st.download_button("Save Photo", response.images[0]._image_bytes, "art.png")
                except Exception as e:
                    st.error(f"Generation failed: {e}")
