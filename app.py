import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image
import json
from google.oauth2 import service_account

# --- UI AYARLARI ---
st.set_page_config(page_title="Persona AI", page_icon="🎨")

# --- AUTH ---
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
        st.error(f"Kimlik Doğrulama Hatası: {e}")
        return False

# --- ANA EKRAN ---
st.title("✨ Persona AI")

if init_google_ai():
    img_file = st.camera_input("Kamerayı Aç")
    
    if img_file:
        styles = {
            "Cyberpunk": "A cyberpunk portrait of the person in [1], neon lights, futuristic.",
            "3D Pixar": "A cute 3D Pixar character of the person in [1], vibrant colors.",
            "Yağlı Boya": "An oil painting of the person in [1], artistic brushstrokes.",
            "Viking": "A Viking warrior portrait of the person in [1], fur armor, snow."
        }
        choice = st.selectbox("Stil Seç:", list(styles.keys()))
        
        if st.button("OLUŞTUR ✨"):
            with st.spinner("Yapay zeka işleniyor... (Bu işlem 20-30 sn sürebilir)"):
                try:
                    # NOT: Imagen 3 Subject Reference için model ismi sabittir
                    model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")
                    
                    # Resmi Vertex AI formatına çevir
                    user_img = Image(image_bytes=img_file.getvalue())
                    
                    # GÜNCEL KULLANIM: 
                    # Eğer requirements.txt 1.72.0 ise bu parametre çalışacaktır.
                    response = model.generate_images(
                        prompt=styles[choice],
                        reference_images=[user_img], # Subject Reference
                        number_of_images=1,
                    )
                    
                    if response.images:
                        st.image(response.images[0]._image_bytes, use_container_width=True)
                        st.download_button("Kaydet", response.images[0]._image_bytes, "sonuc.png")
                    else:
                        st.warning("Görsel oluşturulamadı. Lütfen farklı bir açıdan selfie çekin.")
                        
                except Exception as e:
                    # Eğer hala hata alırsak sürüm bilgisini ekrana yazdıralım (Hata ayıklama için)
                    import google.cloud.aiplatform as aiplatform
                    st.error(f"API Hatası: {e}")
                    st.info(f"Yüklü SDK Sürümü: {aiplatform.__version__}")
