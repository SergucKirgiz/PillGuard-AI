import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.models import load_model
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Akıllı İlaç Rehberi", page_icon="💊")


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


hastane_url = "https://lottie.host/d0e26947-724d-4c08-bd9d-eac4072dab6a/W7fFJBog7D.json"
hap_url = "https://lottie.host/44802932-64ec-43bd-be41-ae3fd4014972/6TmNOgs95F.json"

lottie_hospital = load_lottieurl(hastane_url)
lottie_pill_json = load_lottieurl(hap_url)

col1, col2 = st.columns([1, 4])
with col1:
    if lottie_hospital:
        st_lottie(lottie_hospital, height=100, key="hospital")
with col2:
    st.title("PillGuard AI")
    st.caption("Yapay Zeka Destekli İlaç Tanıma Sistemi")

st.divider()

ilac_detaylari = {
    'advantan': {
        'ad': "ADVANTAN %0.1 Krem",
        'ozet': "Egzama ve şiddetli kaşıntı/kızarıklık durumları içindir.",
        'etken': "Metilprednisolon aseponat",
        'endikasyon': "İltihabi ve kaşıntılı deri bozukluklarının tedavisi.",
        'kullanim': "Günde bir kez sorunlu bölgeye ince bir tabaka sürülür.",
        'uyari': "Yüz bölgesinde ve açık yaralarda doktor kontrolü dışında kullanmayınız."
    },
    'ferrozinc': {
        'ad': "FERROZINC Şurup/Kapsül",
        'ozet': "Vücuttaki demir ve çinko eksikliğini gidermek (kansızlık vb.) içindir.",
        'etken': "Demir (II) Sülfat + Çinko Sülfat + Vitamin C",
        'endikasyon': "Demir ve çinko eksikliği tedavisi ve korunması.",
        'kullanim': "Yemeklerden 1 saat önce veya 2 saat sonra (aç karnına) alınır.",
        'uyari': "Süt ürünleri ve çay ile aynı anda almayınız, emilimi azaltır."
    },
    'magnezic': {
        'ad': "MAGNEZIC 365 mg Tablet",
        'ozet': "Kas krampları, yorgunluk ve magnezyum eksikliği içindir.",
        'etken': "Magnezyum",
        'endikasyon': "Magnezyum eksikliğine bağlı kas ve sinir sistemi bozuklukları.",
        'kullanim': "Günde 1 veya 2 tablet bol su ile yutulur.",
        'uyari': "İshal yapabilir, doz aşımından kaçınınız."
    },
    'parol': {
        'ad': "PAROL 500 mg Tablet",
        'ozet': "Hafif ve orta şiddetli ağrılar (baş, diş) ve ateş düşürmek içindir.",
        'etken': "Parasetamol",
        'endikasyon': "Ağrı kesici ve ateş düşürücü.",
        'kullanim': "Yetişkinlerde 4-6 saatte bir 1-2 tablet.",
        'uyari': "Alkolle birlikte almayınız, karaciğere zarar verebilir."
    },
    'parolplus': {
        'ad': "PAROL PLUS Tablet",
        'ozet': "Şiddetli baş ağrısı, migren ve daha güçlü ağrı kesici etkisi içindir.",
        'etken': "Parasetamol + Propifenazon + Kafein",
        'endikasyon': "Şiddetli ağrıların ve ateşli hastalıkların tedavisi.",
        'kullanim': "Günde 1-3 kez 1 tablet, tok karnına.",
        'uyari': "Kafein içerdiği için uykusuzluk ve çarpıntı yapabilir."
    },
    'randutil': {
        'ad': "RANDUTIL 500 mg Film Tablet",
        'ozet': "Romatizma, eklem ağrıları ve şiddetli kas ağrıları içindir.",
        'etken': "Naproksen Sodyum",
        'endikasyon': "İltihaplı romatizma ve akut kas-iskelet sistemi ağrıları.",
        'kullanim': "Genellikle 12 saatte bir 1 tablet.",
        'uyari': "Mide rahatsızlığına yol açabilir, mutlaka tok karnına içiniz."
    },
    'tribeksol': {
        'ad': "TRIBEKSOL Film Tablet",
        'ozet': "B vitamini eksikliği, sinir hasarı ve halsizlik durumları içindir.",
        'etken': "Vitamin B1 + B6 + B12",
        'endikasyon': "Nevrit, polinevrit ve genel B vitamini eksikliği.",
        'kullanim': "Günde 1 tablet, çiğnenmeden yutulur.",
        'uyari': "İdrar rengini parlak sarıya çevirebilir, bu normaldir."
    }
}

class_names = sorted(['advantan', 'ferrozinc', 'magnezic', 'parol', 'parolplus', 'randutil', 'tribeksol'])


@st.cache_resource
def model_yukle():
    return load_model("model/model.keras")


model = model_yukle()

st.markdown("""
    ### İlacınızı Saniyeler İçinde Tanıyın
    **AI-Med Lens**, derin öğrenme teknolojisini kullanarak ilaç kutularını görsel olarak analiz eder ve size en doğru kullanım bilgilerini sunar.

    *Not: Bu uygulama bilgilendirme amaçlıdır. İlaçları kullanmadan önce mutlaka doktorunuza danışınız.*
""")

st.divider()

kamera_aktif = st.toggle("📸 Kamerayı Başlat")

if kamera_aktif:
    foto = st.camera_input("İlaç kutusunu net bir şekilde çekin")
    if foto is not None:
        img = Image.open(foto)
        st.image(img, caption="Çekilen Fotoğraf", use_container_width=True)

        loading_col1, loading_col2 = st.columns([1, 4])
        with loading_col1:
            if lottie_pill_json:
                st_lottie(lottie_pill_json, height=100, key="loading_animation", speed=1.5)
            else:
                st.write("🔄")
        with loading_col2:
            st.info("### Görüntü analiz ediliyor...")
            st.write("Yapay zeka modeli ilaç kutusunu inceliyor, lütfen bekleyin.")

        img_hazir = img.resize((224, 224))
        img_array = image.img_to_array(img_hazir)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        tahminler = model.predict(img_array)
        indeks = np.argmax(tahminler)
        eminlik = np.max(tahminler) * 100
        sonuc_adi = class_names[indeks]

        if eminlik > 65:
            detay = ilac_detaylari.get(sonuc_adi, {"ad": sonuc_adi.upper(), "ozet": "Bilgi yok."})
            st.success(f"✅ Tahmin: {detay['ad']} (%{eminlik:.1f} doğruluk)")
            st.markdown(f"### 🎯 Ne İçin Kullanılır?\n{detay.get('ozet', '')}")
            with st.expander("📄 Detaylı Reçete Bilgileri"):
                st.write(f"**Etken Madde:** {detay.get('etken', 'Bilinmiyor')}")
                st.write(f"**Kullanım Amacı:** {detay.get('endikasyon', 'Bilinmiyor')}")
                st.write(f"**Nasıl Kullanılır:** {detay.get('kullanim', 'Bilinmiyor')}")
                st.error(f"⚠️ Uyarı: {detay.get('uyari', 'Bilinmiyor')}")
        else:
            st.warning("⚠️ İlaç tam anlaşılamadı. Lütfen ışığı ayarlayıp tekrar deneyin.")
else:
    st.write("Sistemi kullanmak için yukarıdaki anahtarı açın.")