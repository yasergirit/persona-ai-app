import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image
import json
from google.oauth2 import service_account

# --- MOBILE UI SETUP ---
st.set_page_config(page_title="Persona AI", page_icon="🎨")

# CSS to make it look like a real app on iPhone
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .stButton>button { 
        width: 100%; border-radius: 25px; height: 3.5rem; 
        background: linear-gradient(90deg, #4F46E5, #7C3AED);
        color: white; border: none; font-weight: bold; font-size: 1.1rem;
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    </style>
    """, unsafe_allow_html=True)

# --- GOOGLE AUTH FROM STREAMLIT SECRETS ---
def init_google_ai():
    try:
        # Looks for 'gcp_service_account' in Streamlit Cloud Secrets
        if "gcp_service_account" in st.secrets:
            key_dict = json.loads(st.secrets["gcp_service_account"])
            creds = service_account.Credentials.from_service_account_info(key_dict)
            vertexai.init(project=key_dict["project_id"], location="us-central1", credentials=creds)
            return True
        else:
            st.error("Setup Required: Paste your Google JSON key into Streamlit Secrets.")
            return False
    except Exception as e:
        st.error(f"Auth Error: {e}")
        return False

# --- MAIN APP ---
st.title("✨ Persona AI")
st.write("Take a selfie and transform your style.")

if init_google_ai():
    # iPhone Camera Input
    img_file = st.camera_input("Take a selfie")
    
    if not img_file:
        img_file = st.file_uploader("Or upload from gallery", type=["jpg", "png"])

    if img_file:
        st.divider()
        st.subheader("Select Your Vibe")
        
        # Style Dictionary (Using the [1] reference tag for Imagen 3)
        styles = {
            "Cyberpunk": "A high-detail cyberpunk portrait of [1], neon lighting, futuristic.",
            "3D Pixar": "A 3D animated character version of [1], Pixar style, vibrant.",
            "Oil Painting": "A classic oil painting of [1], rich brushstrokes, gold lighting.",
            "Viking": "A cinematic warrior portrait of [1] with fur clothing and snowy mountains."
        }
        
        # Style Selection
        choice = st.selectbox("Choose Style:", list(styles.keys()))
        
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
                    
                    # Display Result
                    st.success("Your Masterpiece is Ready!")
                    result_bytes = response.images[0]._image_bytes
                    st.image(result_bytes, use_container_width=True)
                    
                    # Native iPhone Save button
                    st.download_button(
                        label="📥 Save to Photos",
                        data=result_bytes,
                        file_name="persona_art.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Generation failed: {e}")
