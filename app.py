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
st.write("Selfie'ni çek ve stilini seç!")

if init_google_ai():
    img_file = st.camera_input("Kamerayı Aç")
    
    if img_file:
        styles = {
            "Cyberpunk": "A high-detail cyberpunk portrait of the person, neon lights, futuristic city background, 8k cinematic.",
            "3D Pixar": "A cute 3D Pixar-style animated character of the person, vibrant colors, Disney style.",
            "Yağlı Boya": "A classic royal oil painting of the person, rich brushstrokes, golden lighting, masterpiece.",
            "Viking": "A cinematic portrait of the person as a Viking warrior, fur clothing, snowy mountain background."
        }
        choice = st.selectbox("Stil Seçin:", list(styles.keys()))
        
        if st.button("STİLİ UYGULA ✨"):
            with st.spinner("Yapay zeka seni dönüştürüyor..."):
                try:
                    # Model: Capability-001 özelleştirilmiş işlemler için en iyisidir
                    model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")
                    
                    # Resmi Vertex AI formatına çevir
                    user_img = Image(image_bytes=img_file.getvalue())
                    
                    # --- HATA ÇÖZÜCÜ PARAMETRE YAPISI ---
                    # Hata mesajında istenen 'context_images' parametresini kullanıyoruz.
                    # 'edit_image' fonksiyonu maskesiz (mask-free) düzenleme için bu bağlamı ister.
                    
                    response = model.edit_image(
                        prompt=styles[choice],
                        base_image=user_img,         # Düzenlenecek temel resim
                        context_images=[user_img],   # Hata mesajının istediği kritik parametre!
                        number_of_images=1,
                    )

                    if response.images:
                        st.success("İşte Yeni Tarzın!")
                        st.image(response.images[0]._image_bytes, use_container_width=True)
                        st.download_button("📥 Fotoğrafı İndir", response.images[0]._image_bytes, "persona_art.png")
                    else:
                        st.warning("Görsel oluşturulamadı. Lütfen farklı bir ışıkta tekrar deneyin.")
                        
                except Exception as e:
                    # Eğer hala parametre hatası verirse, 'reference_images' olarak dene (Fallback)
                    try:
                        response = model.generate_images(
                            prompt=styles[choice],
                            reference_images=[user_img],
                            number_of_images=1
                        )
                        st.image(response.images[0]._image_bytes, use_container_width=True)
                    except Exception as fallback_error:
                        st.error(f"Dönüştürme Hatası: {str(e)}")
                        st.info("İpucu: Fotoğrafın çok yakın veya çok uzak olmamasına dikkat edin.")
