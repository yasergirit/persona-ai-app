import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image
import json
from google.oauth2 import service_account

# --- MOBILE UI SETUP ---
st.set_page_config(page_title="Persona AI", page_icon="🎨", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .stButton>button { 
        width: 100%; border-radius: 30px; height: 3.5rem; 
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white; border: none; font-weight: bold; font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.3);
    }
    .stCameraInput > label { display: none; }
    </style>
    """, unsafe_allow_html=True)

# --- AUTHENTICATION FIX ---
def init_google_ai():
    try:
        # Check if the secret exists in Streamlit Cloud
        if "gcp_service_account" in st.secrets:
            # We load the whole block as a string and parse it as JSON
            # This avoids TOML formatting errors entirely
            creds_json = st.secrets["gcp_service_account"]
            creds_info = json.loads(creds_json)
            
            # Fix newline issue
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
            
            creds = service_account.Credentials.from_service_account_info(creds_info)
            vertexai.init(project=creds_info["project_id"], location="us-central1", credentials=creds)
            return True
        else:
            st.error("🔑 Setup Required: Go to Settings -> Secrets and paste your JSON.")
            return False
    except Exception as e:
        st.error(f"⚠️ Auth Error: {str(e)}")
        return False

# --- MAIN APP ---
st.title("✨ Persona AI")
st.write("Turn your selfie into a masterpiece.")

if init_google_ai():
    # 1. Camera / Upload
    input_file = st.camera_input("Take a selfie")
    if not input_file:
        input_file = st.file_uploader("Or choose from library", type=["jpg", "png", "jpeg"])

    if input_file:
        st.divider()
        st.subheader("Choose Your Style")
        
        # Optimized Styles for Imagen 3 [Subject Reference]
        styles = {
            "Cyberpunk": "A futuristic cyberpunk portrait of [1], neon lighting, purple and blue tones, 8k cinematic.",
            "Disney 3D": "A cute 3D Pixar-style animated character version of [1], soft lighting, vibrant colors.",
            "Oil Painting": "A regal oil painting on canvas of [1], thick brushstrokes, classical lighting, masterpiece.",
            "Warrior": "A cinematic portrait of [1] as a legendary Viking warrior, fur armor, snowy mountains, 8k."
        }
        
        # Grid layout for mobile buttons
        style_list = list(styles.keys())
        choice = st.selectbox("Select Style", style_list)
        
        if st.button("CREATE MAGIC ✨"):
            with st.spinner("Reimagining you..."):
                try:
                    model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")
                    user_img = Image(image_bytes=input_file.getvalue())
                    
                    # Call the API
                    response = model.edit_image(
                        prompt=styles[choice],
                        reference_images=[user_img],
                        number_of_images=1,
                        guidance_scale=60 # Balanced likeness vs style
                    )
                    
                    # 2. Check if image was generated (Safety Filter Check)
                    if response.images:
                        result_img = response.images[0]
                        st.success("Your transformation is complete!")
                        st.image(result_img._image_bytes, use_container_width=True)
                        
                        st.download_button(
                            label="📥 Save to Photos",
                            data=result_img._image_bytes,
                            file_name="persona_result.png",
                            mime="image/png"
                        )
                    else:
                        st.warning("⚠️ AI Safety Filter Triggered. This happens if the AI thinks the photo is too realistic or high-risk. Try a different photo or style.")
                        
                except Exception as e:
                    st.error(f"API Error: {str(e)}")
