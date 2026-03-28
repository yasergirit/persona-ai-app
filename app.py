import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image
import json
from google.oauth2 import service_account

# --- MOBILE UI SETUP ---
st.set_page_config(page_title="Persona AI", page_icon="🎨", layout="centered")

# Custom CSS for the Image Grid and Buttons
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    
    /* Style Thumbnail Styling */
    .style-container {
        border-radius: 15px;
        overflow: hidden;
        border: 2px solid #374151;
        transition: 0.3s;
        cursor: pointer;
    }
    .style-container:hover { border-color: #6366f1; transform: scale(1.02); }
    .selected-style { border: 4px solid #a855f7 !important; box-shadow: 0 0 15px rgba(168, 85, 247, 0.5); }
    
    /* Button Styling */
    .stButton>button { 
        width: 100%; border-radius: 30px; height: 3.5rem; 
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white; border: none; font-weight: bold; font-size: 1.1rem;
        margin-top: 20px;
    }
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

# --- STYLE DEFINITIONS ---
# You can replace these URLs with your own 1x1 images
STYLES = {
    "Cyberpunk": {
        "prompt": "A high-detail cyberpunk style portrait of the person in [1], neon lighting, futuristic city background.",
        "thumb": "https://images.unsplash.com/photo-1605810230434-7631ac76ec81?w=400&h=400&fit=crop"
    },
    "3D Pixar": {
        "prompt": "A cute 3D Pixar character of the person in [1], Disney style animation, vibrant colors.",
        "thumb": "https://images.unsplash.com/photo-1599408162162-cdbaf14a8571?w=400&h=400&fit=crop"
    },
    "Oil Painting": {
        "prompt": "A classic oil painting of the person in [1], rich brushstrokes, museum masterpiece texture.",
        "thumb": "https://images.unsplash.com/photo-1579783902614-a3fb3927b6a5?w=400&h=400&fit=crop"
    },
    "Viking": {
        "prompt": "A cinematic portrait of the person in [1] as a Viking warrior, fur armor, snowy mountains.",
        "thumb": "https://images.unsplash.com/photo-1552168324-d612d77725e3?w=400&h=400&fit=crop"
    }
}

# --- MAIN APP ---
st.title("✨ Persona AI")

if init_google_ai():
    # 1. Image Capture
    img_file = st.camera_input("Snap a selfie", label_visibility="collapsed")
    
    if img_file:
        st.write("### 1. Select Your Style")
        
        # Initialize session state for selection
        if 'selected_style' not in st.session_state:
            st.session_state.selected_style = "Cyberpunk"

        # Create 2x2 Grid
        col1, col2 = st.columns(2)
        
        style_names = list(STYLES.keys())
        
        # Grid Display Logic
        for i, name in enumerate(style_names):
            target_col = col1 if i % 2 == 0 else col2
            with target_col:
                # Highlight if selected
                is_selected = "selected-style" if st.session_state.selected_style == name else ""
                
                st.markdown(f'<div class="style-container {is_selected}">', unsafe_allow_html=True)
                st.image(STYLES[name]["thumb"], use_container_width=True)
                if st.button(name, key=f"btn_{name}"):
                    st.session_state.selected_style = name
                    st.rerun() # Refresh to show the border highlight
                st.markdown('</div>', unsafe_allow_html=True)

        st.divider()
        
        # 2. Generate Button
        if st.button(f"CREATE {st.session_state.selected_style.upper()} MAGIC ✨"):
            with st.spinner("AI is reimagining you..."):
                try:
                    model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")
                    user_img = Image(image_bytes=img_file.getvalue())
                    
                    selected_prompt = STYLES[st.session_state.selected_style]["prompt"]
                    
                    response = model.generate_images(
                        prompt=selected_prompt,
                        reference_images=[user_img],
                        number_of_images=1
                    )

                    if response and response.images:
                        st.success("Transformation Complete!")
                        st.image(response.images[0]._image_bytes, use_container_width=True)
                        st.download_button("📥 Save Photo", response.images[0]._image_bytes, "art.png")
                    else:
                        st.warning("Safety filter triggered. Try a different photo.")
                        
                except Exception as e:
                    st.error(f"API Error: {str(e)}")
