from urllib import request
import numpy as np
import pandas as pd
import re
import torch
import random
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt
from flask import Flask,render_template
from flask_sqlalchemy import SQLAlchemy
from flask import url_for
from sklearn.metrics import classification_report
#GPU'u belirtmek
df = pd.read_excel('cb_dataset.xlsx')
data = pd.read_json('cb_data.json')

#Etiketler sayısal kodlamalara dönüştürüldü.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
#Sınıfların dağılımı kontrol edildi. Toplamda 5 adet tagımız mevcut.
df['label'].value_counts(normalize = True)

#tüm verileri eğitim için kullandık.
train_text, train_labels = df['text'], df['label']
text = df['text']
label = df['label']

#Bert algoritmasının tokenizer kısmı yüklendi.
from transformers import AutoModel, BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert = AutoModel.from_pretrained('bert-base-uncased')

#Datasetteki text bölümündeki tüm verileri eğitim için alındı ve histograma göre max uzunluğu 8 yapıldı
seq_len = [len(i.split()) for i in train_text]
pd.Series(seq_len).hist(bins = 10)
max_seq_len = 8

tokens_train = tokenizer(
    train_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

#Eğitim seti için tanımladığımız torchtensor kısmı.
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#burada batch size boyutu veriliyor yani ağa verilen alt örnek sayısı
batch_size = 16
# burada train_data kısmını oluşturuyoruz
train_data = TensorDataset(train_seq, train_mask, train_y)
# eğitim kısmında verileri örnekleme kısmı
train_sampler = RandomSampler(train_data)
# Burada eğitim kümesi oluturuluyor
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)


# Bert mimarisi sınıfı kuruluyor
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert

        # dropout katmanı
        self.dropout = nn.Dropout(0.2)

        # relu etkinleştime işlemi
        self.relu = nn.ReLU()
        # dense katmanı
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 5)
        # softmax etkinleştime işlemi
        self.softmax = nn.LogSoftmax(dim=1)
        # forward geçişimi fonksiyon olarak tanımlama

    def forward(self, sent_id, mask):
        # girdileri modele geçirme
        cls_hs = self.bert(sent_id, attention_mask=mask)[0][:, 0]

        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        # output katmanı
        x = self.fc3(x)

        # softmax aktivasyonunu uygulama
        x = self.softmax(x)
        return x


# parametreleri dondruma kısmı  bu model ağırlıklarının güncellenmesini önleyecektir.
for param in bert.parameters():
      param.requires_grad = False
# modeli Gpu da çalıştırma
model = BERT_Arch(bert)
model = model.to()
from torchinfo import summary
summary(model)

from transformers import AdamW
# optimizer kısmını tanımlama
optimizer = AdamW(model.parameters(), lr = 1e-4)


from sklearn.utils.class_weight import compute_class_weight
#Sınıfların ağırlıklarını hesaplama kısm
class_wts = compute_class_weight(class_weight = "balanced",
                                 classes = np.unique(train_labels),
                                 y = train_labels)
print(class_wts)

# sınıf ağırlıklarını tensöre dönüştürme
weights= torch.tensor(class_wts,dtype=torch.float)
weights = weights.to()
# loss fonksiyonu kullanıldı yani kayıp var mı yok mu ona bakıldı
cross_entropy = nn.NLLLoss(weight=weights)


# doğrulama kaybını depolamak için boş listeler
train_losses=[]
# eğitim dönemi sayısı
epochs = 1
# Daha iyi sonuçlar elde etmek için öğrenme oranı
lr_sch = torch.optim.lr_scheduler.StepLR(optimizer,
                                        step_size=3,
                                        gamma=0.1)


# egitim fonksiyonu oluşturuldu
def egitim():
    model.train()
    total_loss = 0

    # model tahminlerini kaydetmek için boş liste
    total_preds = []

    # gruplar üzerinde tekrarlama
    for step, batch in enumerate(train_dataloader):

        # Her 50 işlemden  sonra ilerleme güncellemesi.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        # işi gpu da yapma
        batch = [r.to() for r in batch]
        sent_id, mask, labels = batch
        # mevcut kısımdan  model tahminleri alma
        preds = model(sent_id, mask)
        # gerçek ve tahmin edilen değerler arasındaki kaybı hesaplama
        loss = cross_entropy(preds, labels)
        # toplam kayba ekleme
        total_loss = total_loss + loss.item()
        # gradyanları hesaplamak için geriye doğru geçiş
        loss.backward()
        # gradyanları 1.0'a kırpma. Patlayan gradyan problemini önlemeye yardımcı olur
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # parametreleri güncelleme
        optimizer.step()
        # hesaplanmış gradyanları temizleme
        optimizer.zero_grad()
        # model tahminleri gpu da şuan onları cpu ya alma
        preds = preds.detach().cpu().numpy()
        # model tahminlerini ekleme
        total_preds.append(preds)
        # eğitim kaybını hesaplama
    avg_loss = total_loss / len(train_dataloader)

    # toplam tahmin kısmı
    total_preds = np.concatenate(total_preds, axis=0)
    # kayıp ve tahminleri döndürür
    return avg_loss, total_preds


for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

    # modeli eğitme
    train_loss, _ = egitim()

    # eğitim ve doğrulama kaybı ekleme
    train_losses.append(train_loss)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f'\nTraining Loss: {train_loss:.3f}')


# girilen inputun kategorisini tahmin etme
def tahmin_Et(str):
    str = re.sub(r'[^a-zA-Z ]+', '', str)
    test_text = [str]
    model.eval()
    # Tokenizer kısmı
    tokens_test_data = tokenizer(
        test_text,
        max_length=max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )
    # burada verileri anlama kısmı
    test_seq = torch.tensor(tokens_test_data['input_ids'])
    test_mask = torch.tensor(tokens_test_data['attention_mask'])

    preds = None
    with torch.no_grad():
        preds = model(test_seq.to(), test_mask.to())
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)
    print('Konu Tahmin : ', le.inverse_transform(preds)[0])
    return le.inverse_transform(preds)[0]


def cevap_Ver(message):
    intent = tahmin_Et(message)
    for i in data['intents']:
        if i["tag"] == intent:
            result = random.choice(i["responses"])
            break
    return  result


ilkproje = Flask(__name__,template_folder='template',static_folder='static')
ilkproje.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:12345@localhost/ChatBot'
ilkproje.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
ilkproje.secret_key = 'secret string'
db = SQLAlchemy(ilkproje)
class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    soru = db.Column(db.String(500), nullable=False)
    konu = db.Column(db.String(500), nullable=False)

    def __init__(self, soru, konu):
        self.soru = soru
        self.konu = konu
@ilkproje.route('/')
def index():
   imag = url_for('static', filename='logo-icon.svg')
   imag1 = url_for('static', filename='page-bg.jpg')
   return render_template('index.html', imag=imag,imag1=imag1)
@ilkproje.route("/Api/<string:mesaj>")
def anasayfa(mesaj:str):
    konu=tahmin_Et(mesaj)
    entry = Chat(mesaj, konu)
    db.session.add(entry)
    db.session.commit()
    mesaj1 = cevap_Ver(mesaj)
    return mesaj1
if __name__ == '__main__':
    db.create_all()
    ilkproje.run(debug=False)