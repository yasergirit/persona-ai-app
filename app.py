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
    
    /* Grid Container */
    .style-card {
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid #2d3748;
        transition: all 0.3s ease;
        margin-bottom: 10px;
        text-align: center;
        background: #1a202c;
    }
    
    /* Highlight for Selected Style */
    .selected-card {
        border: 4px solid #805ad5 !important;
        box-shadow: 0 0 20px rgba(128, 90, 213, 0.6);
        transform: scale(1.02);
    }

    .style-label {
        font-size: 0.9rem;
        font-weight: bold;
        padding: 5px 0;
        color: #e2e8f0;
    }

    /* Action Button */
    .stButton>button {
        width: 100%;
        border-radius: 50px;
        height: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.4);
    }
    
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

# --- STYLE DEFINITIONS WITH PORTRAIT THUMBNAILS ---
STYLES = {
    "Cyberpunk": {
        "prompt": "A futuristic cyberpunk portrait of the person in [1], neon violet lighting, high-tech background, 8k cinematic.",
        "thumb": "https://images.unsplash.com/photo-1614728263952-84ea256f9679?w=400&h=400&fit=crop"
    },
    "3D Pixar": {
        "prompt": "A cute 3D Pixar-style animated character portrait of the person in [1], big expressive eyes, smooth textures, vibrant lighting.",
        "thumb": "https://images.unsplash.com/photo-1590602847861-f357a9332bbc?w=400&h=400&fit=crop"
    },
    "Oil Painting": {
        "prompt": "A classic royal oil painting of the person in [1], museum quality, visible brushstrokes, rich classical lighting.",
        "thumb": "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=400&h=400&fit=crop"
    },
    "Viking": {
        "prompt": "A cinematic warrior portrait of the person in [1] as a Viking, wearing furs and leather, snowy mountain background, fierce expression.",
        "thumb": "https://images.unsplash.com/photo-1533619239233-6280475a633a?w=400&h=400&fit=crop"
    }
}

# --- APP MAIN ---
st.title("✨ Persona AI")
st.write("Snap a selfie and pick your new look.")

if init_google_ai():
    # 1. Camera Input
    user_photo = st.camera_input("Take a selfie", label_visibility="collapsed")
    
    if user_photo:
        st.write("### 1. Choose Your Style")
        
        # Selection Logic
        if 'choice' not in st.session_state:
            st.session_state.choice = "Cyberpunk"

        # 2x2 Grid Layout
        col1, col2 = st.columns(2)
        style_list = list(STYLES.keys())

        # Style 1 & 2
        with col1:
            for i in [0, 2]:
                name = style_list[i]
                selected_class = "selected-card" if st.session_state.choice == name else ""
                st.markdown(f'<div class="style-card {selected_class}">', unsafe_allow_html=True)
                st.image(STYLES[name]["thumb"], use_container_width=True)
                if st.button(name, key=name):
                    st.session_state.choice = name
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        # Style 3 & 4
        with col2:
            for i in [1, 3]:
                name = style_list[i]
                selected_class = "selected-card" if st.session_state.choice == name else ""
                st.markdown(f'<div class="style-card {selected_class}">', unsafe_allow_html=True)
                st.image(STYLES[name]["thumb"], use_container_width=True)
                if st.button(name, key=name):
                    st.session_state.choice = name
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # 3. Generation
        if st.button(f"TRANSFORM TO {st.session_state.choice.upper()} ✨"):
            with st.spinner("AI is reimagining your portrait..."):
                try:
                    model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")
                    ref_img = Image(image_bytes=user_photo.getvalue())
                    
                    response = model.generate_images(
                        prompt=STYLES[st.session_state.choice]["prompt"],
                        reference_images=[ref_img],
                        number_of_images=1
                    )

                    if response and response.images:
                        st.success("Your transformation is ready!")
                        st.image(response.images[0]._image_bytes, use_container_width=True)
                        
                        st.download_button(
                            label="📥 Save to Photos",
                            data=response.images[0]._image_bytes,
                            file_name=f"persona_{st.session_state.choice}.png",
                            mime="image/png"
                        )
                    else:
                        st.warning("Could not generate image. Please try another pose.")
                except Exception as e:
                    st.error(f"API Error: {e}")
