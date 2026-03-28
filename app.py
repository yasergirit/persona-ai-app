import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image
import json
from google.oauth2 import service_account
from openai import OpenAI

# --- UI AYARLARI ---
st.set_page_config(page_title="Persona AI", page_icon="🎨")

# --- AUTH (GOOGLE) ---
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
        st.error(f"Google Auth Hatası: {e}")
        return False

# --- ANA EKRAN ---
st.title("✨ Persona AI")
st.write("Selfie'ni çek ve favori yapay zeka modelini seç!")

if init_google_ai():
    # iPhone Kamerası
    img_file = st.camera_input("Kamerayı Aç")
    
    if img_file:
        st.divider()
        
        # 1. Sağlayıcı Seçimi
        ai_provider = st.selectbox("Yapay Zeka Modeli Seçin:", ["Google Imagen 3 (Yüzü Korur)", "OpenAI DALL-E 3 (Yüksek Kalite Art)"])
        
        # 2. Stil Seçimi
        styles = {
            "Cyberpunk": "A high-detail cyberpunk style portrait, neon lighting, futuristic city background, cinematic colors.",
            "3D Pixar": "A cute 3D Pixar character, Disney style animation, vibrant colors, soft lighting.",
            "Oil Painting": "A classic royal oil painting, rich brushstrokes, golden lighting, artistic canvas texture.",
            "Viking": "A cinematic warrior portrait as a Viking warrior, wearing furs, snowy mountain background, 8k resolution."
        }
        choice = st.selectbox("Stil Seçin:", list(styles.keys()))
        
        # --- OLUŞTURMA BUTONU ---
        if st.button("SİHRİ BAŞLAT ✨"):
            with st.spinner(f"{ai_provider} kullanılarak portreniz hazırlanıyor..."):
                try:
                    result_image = None
                    
                    if ai_provider == "OpenAI DALL-E 3 (Yüksek Kalite Art)":
                        # OpenAI API Çağrısı
                        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                        # DALL-E 3 için prompt düzenleme (Resmi birebir görmediği için betimleyici ekliyoruz)
                        full_prompt = f"Professional portrait of a person in {styles[choice]} style. Realistic human features, high quality art."
                        
                        response = client.images.generate(
                            model="dall-e-3",
                            prompt=full_prompt,
                            n=1,
                            size="1024x1024"
                        )
                        # DALL-E 3 URL döner
                        result_image = response.data[0].url
                        
                    else:
                        # Google Imagen 3 Çağrısı (Subject Reference)
                        model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")
                        user_img = Image(image_bytes=img_file.getvalue())
                        
                        # [1] etiketi ekleyerek yüz korumayı aktifleştiriyoruz
                        google_prompt = styles[choice].replace("portrait", "portrait of the person in [1]")
                        
                        response = model.generate_images(
                            prompt=google_prompt,
                            reference_images=[user_img],
                            number_of_images=1
                        )
                        # Google bytes döner
                        if response.images:
                            result_image = response.images[0]._image_bytes

                    # --- SONUCU GÖSTER ---
                    if result_image:
                        st.success("Dönüşüm Tamamlandı!")
                        st.image(result_image, use_container_width=True)
                        
                        # Kaydetme butonu (Eğer OpenAI ise URL'den indirmek için link verir)
                        if isinstance(result_image, str): # OpenAI URL durumu
                            st.markdown(f'[📥 Fotoğrafı İndir]({result_image})')
                        else: # Google Bytes durumu
                            st.download_button("📥 Fotoğrafı Kaydet", result_image, "persona_art.png", "image/png")
                    else:
                        st.warning("Görsel oluşturulamadı. Lütfen tekrar deneyin.")
                        
                except Exception as e:
                    st.error(f"Hata Oluştu: {str(e)}")
