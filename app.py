import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image
import json
from google.oauth2 import service_account

# --- MOBILE-FIRST UI ---
st.set_page_config(page_title="Persona AI", page_icon="📸")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    .stButton>button { 
        width: 100%; border-radius: 30px; height: 3.5rem; 
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white; border: none; font-weight: bold; font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.3);
    }
    .stCameraInput { border-radius: 20px; overflow: hidden; }
    </style>
    """, unsafe_allow_html=True)

# --- GOOGLE AUTHENTICATION ---
def init_google():
    try:
        if "gcp_service_account" in st.secrets:
            # This pulls the JSON from your Streamlit Cloud Secrets dashboard
            info = json.loads(st.secrets["gcp_service_account"])
            creds = service_account.Credentials.from_service_account_info(info)
            vertexai.init(project=info["project_id"], location="us-central1", credentials=creds)
            return True
        else:
            st.warning("⚠️ Setup Required: Please add your GCP JSON to Streamlit Secrets.")
            return False
    except Exception as e:
        st.error(f"Authentication Error: {e}")
        return False

# --- MAIN APP ---
st.title("✨ Persona AI")
st.write("Turn your selfie into a stylized masterpiece.")

if init_google():
    # iPhone-ready camera input
    user_file = st.camera_input("Take a selfie")
    
    if not user_file:
        user_file = st.file_uploader("Or upload a portrait", type=["jpg", "jpeg", "png"])

    if user_file:
        st.divider()
        st.subheader("Choose Your Style")
        
        # 2x2 Grid for styles
        styles = {
            "Cyberpunk": "A high-quality cyberpunk portrait of [1], neon lighting, futuristic fashion, night city background.",
            "Disney Pixar": "A cute 3D animated character version of [1], smooth textures, expressive eyes, Pixar style.",
            "Oil Painting": "A classic textured oil painting of [1] on canvas, rich brushstrokes, dramatic lighting.",
            "Pencil Sketch": "A professional graphite pencil sketch of [1], charcoal shading, artistic hand-drawn look."
        }
        
        selected_style = st.selectbox("Style Selection", list(styles.keys()), label_visibility="collapsed")
        
        if st.button("CREATE ART ✨"):
            with st.spinner("AI is reimagining you... (takes ~15 seconds)"):
                try:
                    # Initialize Imagen 3
                    model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")
                    
                    # Convert uploaded file to Vertex Image format
                    source_img = Image(image_bytes=user_file.getvalue())
                    
                    # Generate the Image-to-Image result
                    response = model.edit_image(
                        prompt=styles[selected_style],
                        reference_images=[source_img],
                        number_of_images=1,
                        guidance_scale=60 # Balanced between likeness and style
                    )
                    
                    # Output Result
                    st.success("Success!")
                    result_img = response.images[0]._image_bytes
                    st.image(result_img, use_container_width=True)
                    
                    # Download for iPhone
                    st.download_button(
                        label="📥 Save to My Photos",
                        data=result_img,
                        file_name=f"persona_{selected_style.lower()}.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
