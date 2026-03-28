import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image
import json
from google.oauth2 import service_account

# --- MOBILE UI SETUP ---
st.set_page_config(page_title="Persona AI", page_icon="🎨")

# CSS: iPhone için daha şık görünüm
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .stButton>button { 
        width: 100%; border-radius: 30px; height: 3.5rem; 
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white; border: none; font-weight: bold; font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- GOOGLE AUTH ---
def init_google_ai():
    try:
        if "gcp_service_account" in st.secrets:
            creds_info = json.loads(st.secrets["gcp_service_account"])
            # Newline temizliği
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
            
            creds = service_account.Credentials.from_service_account_info(creds_info)
            vertexai.init(project=creds_info["project_id"], location="us-central1", credentials=creds)
            return True
        else:
            st.error("🔑 Kurulum Gerekli: Streamlit Secrets kısmına JSON anahtarını yapıştırın.")
            return False
    except Exception as e:
        st.error(f"Kimlik Doğrulama Hatası: {e}")
        return False

# --- MAIN APP ---
st.title("✨ Persona AI")
st.write("Kendini farklı dünyalarda keşfet.")

if init_google_ai():
    # iPhone Kamerası
    img_file = st.camera_input("Bir selfie çek")
    
    if img_file:
        st.subheader("Bir Stil Seç")
        # [1] etiketi Imagen 3'e "bu referans resimdeki kişiyi kullan" der
        styles = {
            "Cyberpunk": "A high-detail cyberpunk portrait of the person in [1], neon lights, futuristic city background, 8k cinematic.",
            "3D Pixar": "A cute 3D Pixar-style animated character of the person in [1], soft lighting, vibrant colors, Disney style.",
            "Yağlı Boya": "A classic royal oil painting of the person in [1], rich brushstrokes, golden lighting, museum masterpiece.",
            "Viking": "A cinematic portrait of the person in [1] as a Viking warrior, fur clothing, snowy mountain background."
        }
        
        choice = st.selectbox("Stil:", list(styles.keys()))
        
        if st.button("SİHRİ BAŞLAT ✨"):
            with st.spinner("Yapay zeka seni yeniden tasarlıyor..."):
                try:
                    # Imagen 3 Modeli (Subject Reference destekleyen sürüm)
                    model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")
                    
                    # Resmi Vertex AI formatına çevir
                    user_img = Image(image_bytes=img_file.getvalue())
                    
                    # Hata veren edit_image yerine generate_images kullanıyoruz
                    response = model.generate_images(
                        prompt=styles[choice],
                        reference_images=[user_img], # Referans resim listesi
                        number_of_images=1,
                        # guidance_scale 60-70 arası benzerliği iyi korur
                    )
                    
                    if response.images:
                        st.success("Dönüşüm Tamamlandı!")
                        result_bytes = response.images[0]._image_bytes
                        st.image(result_bytes, use_container_width=True)
                        
                        st.download_button(
                            label="📥 Fotoğrafı Kaydet",
                            data=result_bytes,
                            file_name="persona_art.png",
                            mime="image/png"
                        )
                    else:
                        st.warning("Görsel oluşturulamadı (Güvenlik filtresine takılmış olabilir).")
                        
                except Exception as e:
                    st.error(f"API Hatası: {e}")
