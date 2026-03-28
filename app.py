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
            "Cyberpunk": "A high-detail cyberpunk portrait of the person in [1], neon lights, futuristic city background, 8k cinematic.",
            "3D Pixar": "A cute 3D Pixar-style animated character of the person in [1], Disney style, vibrant colors.",
            "Yağlı Boya": "A classic royal oil painting of the person in [1], rich brushstrokes, museum masterpiece.",
            "Viking": "A cinematic portrait of the person in [1] as a Viking warrior, fur clothing, snowy mountain."
        }
        choice = st.selectbox("Stil Seçin:", list(styles.keys()))
        
        if st.button("OLUŞTUR ✨"):
            with st.spinner("Yapay zeka en uygun yöntemi deniyor..."):
                try:
                    # Model: Capability-001 kişi referansı için en iyisidir
                    model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")
                    user_img = Image(image_bytes=img_file.getvalue())
                    
                    response = None
                    
                    # --- YÖNTEM 1: SUBJECT REFERENCE (Modern Yöntem) ---
                    try:
                        response = model.generate_images(
                            prompt=styles[choice],
                            reference_images=[user_img], # En yeni parametre ismi
                            number_of_images=1
                        )
                    except Exception:
                        # --- YÖNTEM 2: CONTEXT IMAGES (Alternatif Yöntem) ---
                        try:
                            response = model.generate_images(
                                prompt=styles[choice],
                                context_images=[user_img], # Bazı sürümlerde bu isimdedir
                                number_of_images=1
                            )
                        except Exception:
                            # --- YÖNTEM 3: EDIT IMAGE (Fallback / Geri Dönüş) ---
                            # Eğer yukarıdakiler "unexpected keyword" derse, en stabil edit moduna döneriz.
                            # Hata mesajındaki 'Must provide at least one context_image' uyarısını burada çözeriz.
                            try:
                                # Bu yöntem, maskesiz düzenleme (stylization) için en garantisidir.
                                response = model.edit_image(
                                    prompt=styles[choice],
                                    base_image=user_img,
                                    number_of_images=1
                                )
                            except Exception as e_final:
                                st.error(f"Tüm yöntemler denendi ancak başarılı olunamadı: {str(e_final)}")

                    if response and response.images:
                        st.success("Dönüşüm Tamamlandı!")
                        st.image(response.images[0]._image_bytes, use_container_width=True)
                        st.download_button("📥 Kaydet", response.images[0]._image_bytes, "sonuc.png")
                    elif response:
                        st.warning("Güvenlik filtresi nedeniyle görsel oluşturulamadı.")
                        
                except Exception as e:
                    st.error(f"Kritik Hata: {str(e)}")
