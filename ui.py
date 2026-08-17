import streamlit as st
import torch
import pickle
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# اضافه کردن BiLSTMClassifier به ایمپورت‌ها (مطمئن شو در models.py وجود دارد)
from models import RNNClassifier, GRUClassifier, LSTMClassifier, TextCNNClassifier, BiLSTMClassifier 
from preprocess import PreprocessManager

# ------------------ UI Config ------------------
st.set_page_config(page_title="SnappFood Advanced Analysis", page_icon="📊", layout="wide")

st.markdown("""
<style>
    @import url('https://v1.fontapi.ir/css/Vazir');
    html, body, [class*="css"] { font-family: 'Vazir', sans-serif; direction: rtl; text-align: right; }
    .stApp { background-color: #ffffff; }
    .stButton>button { background-color: #FF00A4; color: white; border-radius: 12px; height: 3em; width: 100%; border:none; font-weight:bold;}
    .model-card {
        background-color: white; padding: 1rem; border-radius: 20px;
        border: 1px solid #eee; text-align: center; margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .ensemble-card {
        background: linear-gradient(135deg, #FF00A4 0%, #ff52b8 100%);
        color: white !important; padding: 1.5rem; border-radius: 25px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ Logic & Helper ------------------
def get_dynamic_label(score, threshold):
    diff = score - threshold
    if diff >= 0.3: return "مثبت", "😊", "#28a745"
    elif diff >= 0.1: return "نسبتاً مثبت", "🙂", "#77cc33"
    elif diff > -0.1: return "خنثی", "😐", "#ffcc00"
    elif diff > -0.3: return "نسبتاً منفی", "🙁", "#ff9900"
    else: return "منفی", "😢", "#dc3545"

def create_gauge(score, threshold, color):
    # مطمئن شویم color حتما یک رشته (string) حاوی کد رنگ است
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'font': {'size': 24, 'color': color}}, 
        gauge = {
            'axis': {'range': [0, 1], 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.3}, # اینجا باید رشته رنگ باشد
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#eeeeee",
            'steps': [
                {'range': [0, threshold], 'color': '#fff5f5'}, # قرمز بسیار ملایم
                {'range': [threshold, 1], 'color': '#f6ffed'}  # سبز بسیار ملایم
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    
    fig.update_layout(
        height=160, 
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': "Vazir"}
    )
    return fig

# ------------------ Resource Loading ------------------
MODEL_DIR = Path("models")
pm = PreprocessManager()

@st.cache_resource
@st.cache_resource
def load_all():
    # 1. مدل‌های کلاسیک (همان قبلی)
    with open(MODEL_DIR / "Random_Forest.pkl", "rb") as f: rf = pickle.load(f)
    with open(MODEL_DIR / "XGBoost.pkl", "rb") as f: xgb = pickle.load(f)
    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "rb") as f: tfidf = pickle.load(f)
    with open(MODEL_DIR / "Logistic_Regression.pkl", "rb") as f:
        lr = pickle.load(f)
        if not hasattr(lr, 'multi_class'):
            lr.multi_class = 'ovr' # یا 'auto'

    # 2. تنظیمات اولیه
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(MODEL_DIR / "vocab.pkl", "rb") as f: vocab = pickle.load(f)

    # 3. لود کردن مدل‌های عصبی (بسیار ساده‌تر شد)
    n_models = {}
    neural_files = {
        "RNN": "RNN_full_model.pth",
        "LSTM": "LSTM_full_model.pth",
        "GRU": "GRU_full_model.pth",
        "BiLSTM": "BiLSTM_full_model.pth",
        "TextCNN": "TextCNN_full_model.pth"
    }

    for name, filename in neural_files.items():
        # اضافه کردن weights_only=False برای اجازه دادن به لود کل کلاس مدل
        try:
            m = torch.load(MODEL_DIR / filename, map_location=device, weights_only=False)
            n_models[name] = m.to(device).eval()
        except Exception as e:
            st.error(f"خطا در لود مدل {name}: {e}")

    # 4. BERT (معمولاً طبق استاندارد HuggingFace لود می‌شود)
    b_tk = AutoTokenizer.from_pretrained(MODEL_DIR / "ParsBERT_model")
    b_md = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / "ParsBERT_model")
    
    return {"RF": rf, "XGB": xgb, "Logistic_Regression": lr}, tfidf, vocab, n_models, b_md, b_tk, device

c_dict, tfidf, vocab, n_dict, b_model, b_tk, dev = load_all()

def get_score(text, m_name):
    if m_name in c_dict:
        # مدل‌های کلاسیک (SVM, Naive Bayes, ...)
        p = pm.preprocess(text, "classic")
        return c_dict[m_name].predict_proba(tfidf.transform([p]))[0][1]
        
    elif m_name in n_dict:
        # مدل‌های عصبی (BiLSTM, CNN, ...)
        p = pm.preprocess(text, "neural")
        tokens = [vocab.get(t, 1) for t in p.split()]
        
        # پدینگ استاندارد برای طول 128 (مطابق آموزش)
        if len(tokens) < 128:
            tokens += [0] * (128 - len(tokens))
        else:
            tokens = tokens[:128]
            
        input_ts = torch.tensor(tokens).unsqueeze(0).to(dev)
        
        with torch.no_grad():
            raw_output = n_dict[m_name](input_ts)
            # اگر خروجی مدل شما تک‌مقدار است از sigmoid استفاده کنید
            probability = torch.sigmoid(raw_output).item()
            return probability
            
    else: # ParsBERT
        # برای BERT بهتر است از همان متنی که به مدل دادی (p) استفاده کنی
        p = pm.preprocess(text, "bert")
        inputs = b_tk(p, return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
        
        with torch.no_grad(): 
            outputs = b_model(**inputs)
            # تبدیل Logits به احتمال با Softmax
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            return probs[0][1].item()

# ------------------ Sidebar ------------------
with st.sidebar:
    st.image("https://snappfood.ir/static/images/snappfood-logo.svg", width=120)
    st.title("پنل تنظیمات")
    
    selected = st.multiselect("مدل‌ها:", 
                             ["Logistic_Regression", "RF", "XGB", "RNN", "LSTM", "GRU", "BiLSTM", "TextCNN", "ParsBERT"],
                             default=["Logistic_Regression", "ParsBERT", "BiLSTM"])
    
    threshold = st.slider("آستانه اطمینان", 0.0, 1.0, 0.5, 0.05)

    # --- بخش جدید: آمار مدل‌ها در سایدبار ---
    st.markdown("---")
    st.subheader("📊 دقت مدل‌ها (Test)")
    
    # مقادیر را بر اساس نتایج واقعی‌ات تنظیم کن
    m_stats = {"ParsBERT": 0.87, "BiLSTM": 0.85, "LSTM":0.85 ,"Logistic": 0.87,"GRU":0.85 ,"RNN":0.84 ,"TextCNN":0.83 , "XGB": 0.86 , "Random_Forest":0.86}
    
    for m, acc in m_stats.items():
        # اصلاح فرمت نمایش درصد (استفاده از .1f برای یک رقم, اعشار)
        st.write(f"**{m}**: {acc*100:.1f}%")
        st.progress(acc)
st.markdown("<h1 style='text-align: center; color: #FF00A4;'>تحلیل پیشرفته نظرات کاربران 🍽️</h1>", unsafe_allow_html=True)
# --- مثال‌های پیشنهادی برای تست سریع ---
st.markdown("##### 💡 تست سریع با جملات نمونه:")
cols_ex = st.columns(4)
examples = [
    ("مثبت صریح", "غذا عالی و گرم بود، ممنون."),
    ("منفی صریح", "اصلا کیفیت نداشت، خیلی دیر رسید."),
    ("کنایه (Sarcasm)", "واقعا خسته نباشید با این سرویس‌دهی ضعیفتون!"),
    ("دوپهلو", "مرغش سرخ نشده بود ولی سیب زمینی هاش طلایی شده بود")
]

# مدیریت کلیک روی دکمه‌ها
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

for i, (label, exp_text) in enumerate(examples):
    if cols_ex[i].button(label):
        st.session_state.input_text = exp_text

txt = st.text_area("متن نظر را وارد کنید:", value=st.session_state.input_text, height=100)
# ------------------ Main UI ------------------
if st.button("تحلیل و رسم نمودارها") and txt:
    # 1. محاسبه نتایج
    results = {m: get_score(txt, m) for m in selected}
    
    # ------------------ بخش اول: نمایش مدل‌های تکی ------------------
    st.markdown("### 🥧 تفکیک مدل‌ها و وضعیت اطمینان")
    grid_cols = st.columns(len(selected))
    
    for i, (name, score) in enumerate(results.items()):
        label, emoji, color = get_dynamic_label(score, threshold)
        with grid_cols[i]:
            st.markdown(f"""<div class="model-card"><small>{name}</small>""", unsafe_allow_html=True)
            
            # نمودار عقربه‌ای
            st.plotly_chart(create_gauge(score, threshold, color), use_container_width=True)
            
            # نمودار میله‌ای توزیع برچسب برای هر مدل
            # این نمودار نشان می‌دهد امتیاز در کل طیف کجا قرار دارد
            fig_dist = px.bar(
                x=[score], y=[" "], 
                orientation='h', 
                range_x=[0, 1],
                color_discrete_sequence=[color],
                height=50
            )
            fig_dist.update_layout(margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})

            st.markdown(f"""
                <b style="color:{color}; font-size:1.2em;">{score:.2f}</b><br>
                <span style="color:{color};">{emoji} {label}</span>
            </div>
            """, unsafe_allow_html=True)

    # ------------------ بخش دوم: نمودار میله‌ای کلی مقایسه‌ای ------------------
    st.markdown("---")
    st.markdown("### 📊 مقایسه قدرت تشخیص مدل‌ها")
    fig_compare = px.bar(
        x=list(results.keys()), 
        y=list(results.values()), 
        color=list(results.values()),
        color_continuous_scale='RdPu', 
        labels={'x':'مدل', 'y':'امتیاز مثبت بودن'}
    )
    fig_compare.update_layout(height=300)
    st.plotly_chart(fig_compare, use_container_width=True)

    

    # ------------------ بخش سوم: مدل ترکیبی هوشمند ------------------
    if len(selected) > 1:
        st.markdown("---")
        st.markdown("### 🏆 خروجی نهایی مدل ترکیبی هوشمند (Weighted Ensemble)")
        
        
        weights = {"ParsBERT": 0.5, "BiLSTM": 0.3, "XGB": 0.2}
        
        # پیدا کردن مدل‌های انتخاب شده که در لیست وزنی ما هستند
        available_weighted = [m for m in weights.keys() if m in selected]
        

        if available_weighted:
            # محاسبه میانگین وزنی
            total_w = sum(weights[m] for m in available_weighted)
            ens_score = sum(results[m] * weights[m] for m in available_weighted) / total_w
            info_msg = f"این نتیجه با تمرکز بر دقت مدل‌های برتر ({', '.join(available_weighted)}) محاسبه شده است."
        else:
            # اگر هیچکدام از مدل‌های برتر انتخاب نشده بودند، میانگین ساده بگیر
            ens_score = sum(results.values()) / len(results)
            info_msg = "این نتیجه از میانگین ساده مدل‌های انتخاب شده به دست آمده است."
            
        ens_label, ens_emoji, ens_color = get_dynamic_label(ens_score, threshold)

        c_ens1, c_ens2 = st.columns([1, 2])
        with c_ens1:
            st.markdown(f"""
            <div class="ensemble-card" style="background: linear-gradient(135deg, {ens_color} 0%, #333 150%);">
                <p>نتیجه نهایی سیستم</p>
                <h2 style="color:white;">{ens_emoji} {ens_label}</h2>
                <h1 style="color:white; font-size: 3.5em; margin:0;">{ens_score:.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
        with c_ens2:
            st.success(info_msg)
            st.write("📊 **تحلیل لایه‌ای:**")
            st.caption("مدل‌های Transformer (ParsBERT) وزن ۵۰٪ را در تصمیم نهایی دارند به دلیل درک بهتر کنایه.")

    # ------------------ بخش چهارم: نمودار روند احساسات در طول متن ------------------
    st.markdown("---")
    st.markdown("### 📈 تحلیل روند احساسات (جمله به جمله)")
    
    # جدا کردن متن بر اساس نقطه
    # جدا کردن متن به جملات
    raw_sentences = txt.split('.')
    valid_sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 3]

    if len(valid_sentences) > 1:
        avg_scores = []
        
        # برای هر جمله...
        for s in valid_sentences:
            # امتیاز تمام مدل‌هایی که کاربر تیک زده را می‌گیریم
            # متغیر selected همان لیست مدل‌های تیک خورده در سایدبار است
            current_scores = [get_score(s, m) for m in selected]
            
            # میانگین امتیاز مدل‌های انتخاب شده برای این جمله
            sentence_avg = sum(current_scores) / len(current_scores)
            avg_scores.append(sentence_avg)
        
        # رسم نمودار بر اساس میانگین مدل‌های انتخاب شده
        st.line_chart(avg_scores)
        st.info(f"نمودار بر اساس میانگین {len(selected)} مدل انتخاب شده رسم شده است.")

    
    
