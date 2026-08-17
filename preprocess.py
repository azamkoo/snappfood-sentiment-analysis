
import re
from hazm import Normalizer, Lemmatizer

use_hazm = False
try:
    from hazm import Normalizer, Lemmatizer
    normalizer = Normalizer()
    lemmatizer = Lemmatizer()
    use_hazm = True
except ImportError:
    pass


persian_stopwords = set([
    "و", "در", "به", "از", "که", "این", "با", "را", "برای",
    "اما", "یا", "اگر", "نه", "هم", "تا", "یک", "من", "تو",
    "ایشان","بود","داشت","خواه","کند","همین"
])


def normalize_persian(text):
    text = text.replace("ي", "ی").replace("ك", "ک")
    return text

def remove_noise(text):
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    return text

def remove_english_and_digits(text):
    text = re.sub(r"[A-Za-z]", " ", text)
    text = re.sub(r"[0-9٠-٩۰-۹]", " ", text)
    return text

def remove_punctuation_keep_persian(text):
    return re.sub(r"[^\u0600-\u06FF\s‌]", " ", text)

def remove_stretch(text):
    return re.sub(r'(.)\1{2,}', r'\1', text)

def fix_spaces_and_halfspace(text):
    text = text.replace("\u200c", "‌")
    return re.sub(r"\s+", " ", text).strip()

def remove_stopwords_func(text):
    words = [w for w in text.split() if w and w not in persian_stopwords]
    return " ".join(words)

def simple_lemmatize(word):
    orig = word
    for suf in ["تر", "ترین", "ها", "های", "ام", "ی", "هایم", "هایش", "شان", "ترين"]:
        if word.endswith(suf) and len(word) - len(suf) > 2:
            return word[:-len(suf)]
    for pref in ["می", "نمی", "بی"]:
        if word.startswith(pref) and len(word) - len(pref) > 2:
            return word[len(pref):]
    return orig

def lemmatize_text(text):
    tokens = text.split()
    if use_hazm:
        try:
            return " ".join([lemmatizer.lemmatize(t) for t in tokens])
        except Exception:
            return " ".join([simple_lemmatize(t) for t in tokens])
    else:
        return " ".join([simple_lemmatize(t) for t in tokens])

def remove_spam_and_short(text):
    if not isinstance(text, str): return ""
    if re.search(r"(www|http|@)", text): return ""
    if re.search(r"\d{6,}", text): return ""
    if len(text.split()) <= 1: return ""
    return text

# -Preprocess Manager
class PreprocessManager:
    def __init__(self):
        pass

    #  Classic models: heavy cleaning
    def classic_pipeline(self, text):
        t = text
        if use_hazm:
            try: t = normalizer.normalize(t)
            except: pass
        t = t.lower()
        t = normalize_persian(t)
        t = remove_noise(t)
        t = remove_english_and_digits(t)
        t = remove_punctuation_keep_persian(t)
        t = remove_stretch(t)
        t = fix_spaces_and_halfspace(t)
        t = remove_stopwords_func(t)
       # t = lemmatize_text(t)
        t = fix_spaces_and_halfspace(t)
        t = remove_spam_and_short(t)
        return t

    #  Neural models: medium cleaning
    def neural_pipeline(self, text):
        t = text
        if use_hazm:
            try: t = normalizer.normalize(t)
            except: pass
        t = t.lower()
        t = normalize_persian(t)
        t = remove_noise(t)
        t = remove_punctuation_keep_persian(t)
        t = fix_spaces_and_halfspace(t)

        return t

    #  BERT: minimal cleaning
    def bert_pipeline(self, text):
        if not isinstance(text, str):
            return ""
        if use_hazm:
            try: text = normalizer.normalize(text)
            except: pass
        text = normalizer.normalize(text)
        text = re.sub(r'<[^>]+>', ' ', text)  # HTML
        text = re.sub(r'http\S+', ' ', text)  # URLs
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        return text.strip()

    #  Unified preprocess function
    def preprocess(self, text, model_type="classic"):
        """
        model_type: "classic", "neural", "bert"
        """
        if model_type == "classic":
            return self.classic_pipeline(text)
        elif model_type == "neural":
            return self.neural_pipeline(text)
        elif model_type == "bert":
            return self.bert_pipeline(text)
        else:
            # fallback
            return self.neural_pipeline(text)
        

