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
st.write("Selfie'ni çek ve yapay zekanın seni yeniden tasarlamasını izle.")

if init_google_ai():
    # iPhone Kamerası
    img_file = st.camera_input("Kamerayı Aç")
    
    if img_file:
        st.subheader("Bir Stil Seç")
        
        # NOT: [1] etiketi, yapay zekaya 'yüklediğin resimdeki kişiyi buraya koy' der.
        styles = {
            "Cyberpunk": "A high-detail cyberpunk style portrait of the person in [1], neon lighting, futuristic city background, cinematic colors.",
            "3D Pixar": "A cute 3D Pixar character of the person in [1], Disney style animation, vibrant colors, soft lighting.",
            "Oil Painting": "A classic oil painting of the person in [1], rich brushstrokes, golden lighting, artistic canvas texture.",
            "Viking": "A cinematic portrait of the person in [1] as a Viking warrior, wearing furs, snowy mountain background, 8k resolution."
        }
        choice = st.selectbox("Stil Seçin:", list(styles.keys()))
        
        if st.button("OLUŞTUR ✨"):
            with st.spinner("Yapay zeka portreni hazırlıyor..."):
                try:
                    # Model: Capability-001 (Özelleştirme yeteneği olan model)
                    model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")
                    
                    # Resmi Vertex AI formatına çevir
                    user_img = Image(image_bytes=img_file.getvalue())
                    
                    # --- GÜNCEL SUBJECT REFERENCE ÇAĞRISI ---
                    # Hata almamak için parametreyi 'reference_images' olarak veriyoruz.
                    # Imagen 3'te maskesiz dönüşüm için en doğru yol budur.
                    
                    response = model.generate_images(
                        prompt=styles[choice],
                        reference_images=[user_img], # Selfie'yi referans olarak gönderiyoruz
                        number_of_images=1,
                        # guidance_scale=60 # Varsayılan olarak benzerliği korur
                    )

                    if response and response.images:
                        st.success("Dönüşüm Tamamlandı!")
                        st.image(response.images[0]._image_bytes, use_container_width=True)
                        st.download_button("📥 Fotoğrafı Kaydet", response.images[0]._image_bytes, "persona_art.png")
                    else:
                        st.warning("Görsel oluşturulamadı. Güvenlik filtresi (Safety Filter) nedeniyle olabilir.")
                        
                except Exception as e:
                    st.error(f"Dönüştürme Hatası: {str(e)}")
                    st.info("Eğer 'unexpected keyword' hatası alırsanız, Streamlit'ten uygulamayı 'Reboot' edin.")
