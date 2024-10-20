import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim import AdamW
import matplotlib.pyplot as plt
import numpy as np

# สร้างคลาสสำหรับ Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience  # จำนวน epoch ที่จะรอหากไม่พบการปรับปรุง
        self.verbose = verbose  # แสดงข้อความหรือไม่
        self.counter = 0  # ตัวนับ epoch ที่ไม่มีการปรับปรุง
        self.best_loss = None  # ค่าความสูญเสียที่ดีที่สุด

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss  # กำหนดค่าแรกเริ่ม
        elif val_loss < self.best_loss:
            self.best_loss = val_loss  # อัปเดตค่าความสูญเสียที่ดีที่สุด
            self.counter = 0  # รีเซ็ตตัวนับ
        else:
            self.counter += 1  # เพิ่มตัวนับหากไม่มีการปรับปรุง
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered!")  # แสดงข้อความเมื่อมีการหยุด
                return True  # ส่งคืน True เพื่อหยุดการฝึก
        return False  # ส่งคืน False หากไม่หยุด

# โหลดชุดข้อมูลจากไฟล์ Excel
file_path = 'test file.xlsx'  # เปลี่ยนเป็นที่อยู่ไฟล์ของคุณ
data = pd.read_excel(file_path)

# แปลงค่าป้ายกำกับเป็นรูปแบบตัวเลข
data['ระดับความรู้สึก'] = data['ระดับความรู้สึก'].map({'ดี': 1, 'ไม่ดี': 0})

# ตรวจสอบค่า NaN
if data['ระดับความรู้สึก'].isnull().any():
    data = data.dropna(subset=['ระดับความรู้สึก'])  # ลบแถวที่มีค่า NaN

# แบ่งข้อมูลออกเป็นชุดฝึกและชุดทดสอบ
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['ประโยค'], data['ระดับความรู้สึก'], test_size=0.3, random_state=42
)

# โหลด BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# ทำการ Tokenize ข้อความ
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=64)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=64)

# สร้างคลาส Dataset ที่กำหนดเอง
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  # เก็บข้อมูลที่ถูก Tokenize
        self.labels = labels  # เก็บป้ายกำกับ

    def __len__(self):
        return len(self.labels)  # คืนค่าจำนวนข้อมูลใน Dataset

    def __getitem__(self, idx):
        # คืนค่าข้อมูลและป้ายกำกับในรูปแบบ Tensor
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# สร้างชุดข้อมูลสำหรับฝึกและทดสอบ
train_dataset = SentimentDataset(train_encodings, train_labels.tolist())
val_dataset = SentimentDataset(val_encodings, val_labels.tolist())

# โหลดโมเดล BERT ที่ผ่านการฝึกมาแล้ว
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)

# ตั้งค่า Optimizer
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.1)

# สร้าง DataLoader สำหรับการประมวลผลเป็นชุด
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# ย้ายโมเดลไปยังอุปกรณ์ที่เหมาะสม (GPU หรือ CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# กำหนดลิสต์สำหรับเก็บค่าความสูญเสียและความถูกต้อง
train_losses = []
val_losses = []
val_accuracies = []

# ฟังก์ชันการฝึก
def train(model, train_loader):
    model.train()  # ตั้งค่าโมเดลเป็นโหมดฝึก
    total_loss = 0  # ตัวแปรเก็บค่าความสูญเสียทั้งหมด
    for batch in tqdm(train_loader):  # สำหรับแต่ละชุดข้อมูลใน train_loader
        optimizer.zero_grad()  # รีเซ็ตค่าการกราด
        # ย้ายข้อมูลไปยังอุปกรณ์ที่เหมาะสม
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)  # เรียกใช้โมเดล
        loss = outputs.loss  # ดึงค่าความสูญเสีย
        total_loss += loss.item()  # เพิ่มค่าความสูญเสียลงในตัวแปรรวม
        loss.backward()  # ทำการย้อนกลับเพื่อคำนวณกราด
        optimizer.step()  # อัปเดตน้ำหนักของโมเดล
    return total_loss / len(train_loader)  # คืนค่าความสูญเสียเฉลี่ย

# ฟังก์ชันการประเมิน
def evaluate(model, val_loader):
    model.eval()  # ตั้งค่าโมเดลเป็นโหมดทดสอบ
    total_loss = 0  # ตัวแปรเก็บค่าความสูญเสียทั้งหมด
    correct = 0  # ตัวแปรเก็บจำนวนการทำนายที่ถูกต้อง
    entropy_values = []  # ลิสต์สำหรับเก็บค่า entropy
    with torch.no_grad():  # ปิดการคำนวณกราด
        for batch in val_loader:  # สำหรับแต่ละชุดข้อมูลใน val_loader
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)  # เรียกใช้โมเดล
            loss = outputs.loss  # ดึงค่าความสูญเสีย
            total_loss += loss.item()  # เพิ่มค่าความสูญเสียลงในตัวแปรรวม
            preds = torch.argmax(outputs.logits, dim=1)  # ทำนายผลจาก logits
            correct += (preds == labels).sum().item()  # นับจำนวนการทำนายที่ถูกต้อง

            # คำนวณค่า entropy
            probabilities = torch.softmax(outputs.logits, dim=1)  # แปลง logits เป็นความน่าจะเป็น
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)  # คำนวณค่า entropy
            entropy_values.extend(entropy.cpu().numpy())  # เพิ่มค่า entropy ลงในลิสต์

    accuracy = correct / len(val_loader.dataset)  # คำนวณความถูกต้อง
    return total_loss / len(val_loader), accuracy, entropy_values  # คืนค่าความสูญเสีย, ความถูกต้อง, และค่า entropy

# เริ่มต้น Early Stopping
early_stopping = EarlyStopping(patience=2, verbose=True)

# กระบวนการฝึกพร้อม Early Stopping
epochs = 5
entropy_per_epoch = []  # ลิสต์สำหรับเก็บค่า entropy เฉลี่ยของแต่ละ epoch   
for epoch in range(epochs):  # สำหรับแต่ละ epoch
    print(f'Epoch {epoch + 1}/{epochs}')
    train_loss = train(model, train_loader)  # ฝึกโมเดล
    val_loss, val_accuracy, val_entropy = evaluate(model, val_loader)  # ประเมินโมเดล
    avg_entropy = np.mean(val_entropy)  # คำนวณค่า entropy เฉลี่ยของ epoch นี้
    entropy_per_epoch.append(avg_entropy)  # เก็บค่า entropy เฉลี่ย
    
    # เก็บค่าความสูญเสียและความถูกต้องในลิสต์
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f} | Average Entropy: {avg_entropy:.4f}')
    
    # ตรวจสอบ Early Stopping
    if early_stopping(val_loss):
        print("Stopping training early")  # แสดงข้อความเมื่อมีการหยุด
        break

# บันทึกโมเดลที่ฝึกเสร็จแล้ว
model.save_pretrained('sentiment-model')
tokenizer.save_pretrained('sentiment-model')

# การวาดกราฟค่าความสูญเสียและความถูกต้อง
plt.figure(figsize=(12, 5))

# กราฟสำหรับ Loss
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Training Loss', marker='o')  # เส้นค่าความสูญเสียในการฝึก
plt.plot(val_losses, label='Validation Loss', marker='o')  # เส้นค่าความสูญเสียในการทดสอบ
plt.title('Loss Over Epochs')  # ชื่อกราฟ
plt.xlabel('Epoch')  # ป้ายแกน X
plt.ylabel('Loss')  # ป้ายแกน Y
plt.legend()  # แสดงตำนาน

# กราฟสำหรับ Accuracy
plt.subplot(1, 3, 2)
plt.plot(val_accuracies, label='Validation Accuracy', marker='o')  # เส้นค่าความถูกต้องในการทดสอบ
plt.title('Validation Accuracy Over Epochs')  # ชื่อกราฟ
plt.xlabel('Epoch')  # ป้ายแกน X
plt.ylabel('Accuracy')  # ป้ายแกน Y
plt.legend()  # แสดงตำนาน

# กราฟสำหรับ Average Entropy
plt.subplot(1, 3, 3)
plt.plot(entropy_per_epoch, marker='o')  # เส้นค่า entropy เฉลี่ย
plt.title('Average Entropy per Epoch')  # ชื่อกราฟ
plt.xlabel('Epoch')  # ป้ายแกน X
plt.ylabel('Average Entropy')  # ป้ายแกน Y
plt.grid()  # แสดงกริดในกราฟ

# แสดงกราฟ
plt.tight_layout()
plt.show()
