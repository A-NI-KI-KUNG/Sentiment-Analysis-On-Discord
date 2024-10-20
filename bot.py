import discord
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from dotenv import load_dotenv
import os
import random
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # เพิ่มสำหรับ BoW และ Word Vectorizer

# โหลด .env file
load_dotenv()

# โหลดโมเดลและ tokenizer
model_path = 'sentiment-model'  # โฟลเดอร์ที่คุณบันทึกโมเดลหลังจากฝึกเสร็จ
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# ตั้งค่า device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

# ฟังก์ชันทำความสะอาดข้อความ
def clean_text(text):
    excluded_words = ['ผม', 'ฉัน', 'เธอ', 'เรา', 'คุณ']
    pattern = r'\s*(?:' + '|'.join(excluded_words) + r')\s*'
    cleaned_text = re.sub(pattern, ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# ฟังก์ชันสำหรับ Bag-of-Words (BoW)
def bow_vectorizer(texts):
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(texts)
    return bow_matrix.toarray(), vectorizer.get_feature_names_out()

# ฟังก์ชันสำหรับ TF-IDF Vectorizer
def tfidf_vectorizer(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix.toarray(), vectorizer.get_feature_names_out()

# ฟังก์ชันสำหรับทำนายความรู้สึก (ยังใช้ BERT)
def predict_sentiment(text):
    print(f"ข้อความที่ได้รับจาก Discord: '{text}'")
    cleaned_text = clean_text(text)
    print(f"ประโยคที่เอาไปวิเคราะห์: '{cleaned_text}'")

    # Bag-of-Words ตัวอย่างการใช้งาน
    bow_matrix, bow_features = bow_vectorizer([cleaned_text])
    print("Bag-of-Words Matrix:", bow_matrix)
    print("Bag-of-Words Features:", bow_features)

    # TF-IDF ตัวอย่างการใช้งาน
    tfidf_matrix, tfidf_features = tfidf_vectorizer([cleaned_text])
    print("TF-IDF Matrix:", tfidf_matrix)
    print("TF-IDF Features:", tfidf_features)

    # ใช้ BERT ในการพยากรณ์
    inputs = tokenizer(cleaned_text, return_tensors='pt', padding=True, truncation=True, max_length=64)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class

# ฟังก์ชันสุ่มลิงก์
def get_random_link(sentiment):
    good_links = [
        "https://youtu.be/sm7ULd-vULY?si=jDW2WjHkWQhdmBG3",
        "https://youtu.be/nKSpbpSQvr4?si=J0VhKcMtjwf51cXl",
        "https://youtu.be/ALuAqImqCOY?si=y0H_nu-KaouVLTSC"
    ]

    bad_links = [
        "https://youtu.be/RiZ2N3A5siI?si=Rrt7ovXEDLCNcGDd",
        "https://youtu.be/ojvL5nLJKas?si=Qi7MOjoBVBUGtMip",
        "https://youtu.be/zhcn1-R-2d0?si=RaSDURcSK_te2Q9p",
        "https://youtu.be/7KIQkaPgFhM?si=2uRkSpzhvZDI9XKt"
    ]

    if sentiment == 1:  # ถ้าความรู้สึกดี
        return random.choice(good_links)
    else:  # ถ้าความรู้สึกไม่ดี
        return random.choice(bad_links)

# ตั้งค่า Discord bot
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    for guild in client.guilds:
        for channel in guild.text_channels:
            await channel.send(f'สวัสดีเจ้านาย! {client.user.name} พร้อมที่จะรับฟังทุกเรื่องเลยเจ้านาย!\nกรุณาใช้คำสั่ง "T!" ตามด้วยข้อความเพื่อเรียกใช้บอท.')
            return

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('T!'):
        text_to_analyze = message.content[len('T!'):].strip()
        if len(text_to_analyze) < 3:
            await message.channel.send('กรุณาพิมพ์ข้อความอย่างน้อย 3 ตัวอักษร.')
            return

        sentiment = predict_sentiment(text_to_analyze)
        await message.channel.send(f'เจ้านายพูดแต่เรื่อง: {"ดี -3- " if sentiment == 1 else "ไม่ดี T_T"}')

        link = get_random_link(sentiment)
        await message.channel.send(f'งั้นผมขอแนะนำลิงก์: {link}')
    else:
        await message.channel.send('กรุณาใช้คำสั่ง "T!" ตามด้วยข้อความเพื่อเรียกใช้บอท.')

DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
client.run(DISCORD_TOKEN)
