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
    img_file = st.camera_input("Bir selfie çekin")
    
    if img_file:
        styles = {
            "Cyberpunk": "A high-detail cyberpunk portrait of [1], neon lights, futuristic city.",
            "3D Pixar": "A cute 3D Pixar character of [1], vibrant colors, Disney style.",
            "Yağlı Boya": "An oil painting of [1], artistic brushstrokes, masterpiece.",
            "Viking": "A Viking warrior portrait of [1], fur armor, cinematic background."
        }
        choice = st.selectbox("Stil Seç:", list(styles.keys()))
        
        if st.button("OLUŞTUR ✨"):
            with st.spinner("Yapay zeka seni yeniden tasarlıyor..."):
                try:
                    # Model: Mutlaka capability-001 olmalı
                    model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")
                    user_img = Image(image_bytes=img_file.getvalue())
                    
                    # --- HATA ÖNLEYİCİ ÇAĞRI (DYNAMIC CALL) ---
                    # SDK sürümleri arasındaki parametre farkını çözmek için:
                    try:
                        # 1. Yöntem: En güncel yöntem (reference_images)
                        response = model.generate_images(
                            prompt=styles[choice],
                            reference_images=[user_img],
                            number_of_images=1,
                        )
                    except TypeError:
                        # 2. Yöntem: Eğer reference_images hatası verirse, 'image' parametresini dene
                        # Bazı SDK sürümlerinde Subject Reference bu isimle çağrılır
                        try:
                            response = model.generate_images(
                                prompt=styles[choice],
                                image=user_img,
                                number_of_images=1,
                            )
                        except TypeError:
                            # 3. Yöntem: Edit Image fallback (Garantili yöntem)
                            # Eğer capability modelleri çalışmazsa edit modunda dene
                            response = model.edit_image(
                                prompt=styles[choice],
                                base_image=user_img,
                                number_of_images=1,
                            )

                    if response.images:
                        st.success("Dönüşüm Tamamlandı!")
                        st.image(response.images[0]._image_bytes, use_container_width=True)
                        st.download_button("📥 Kaydet", response.images[0]._image_bytes, "persona.png")
                    else:
                        st.warning("Görsel oluşturulamadı. Güvenlik filtresi (Safety Filter) nedeniyle olabilir.")
                        
                except Exception as e:
                    st.error(f"API Hatası: {str(e)}")
