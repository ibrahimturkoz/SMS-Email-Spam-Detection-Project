import pandas as pd  # Veri analizi ve dosya okuma işlemleri için pandas kütüphanesini içeri aktarır.
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test setlerine bölmek için kullanılan fonksiyonu çağırır.
from sklearn.feature_extraction.text import CountVectorizer  # Metinleri kelime sayım matrisine dönüştürmek için kullanılan aracı ekler.
from sklearn.naive_bayes import MultinomialNB  # Metin sınıflandırmada yüksek performans gösteren Naive Bayes algoritmasını dahil eder.
from sklearn.metrics import accuracy_score, classification_report  # Modelin başarı oranını ve detaylı raporunu ölçmek için araçları yükler.

# 1. VERİ YÜKLEME VE ÖN İŞLEME
df = pd.read_csv('spam.csv', encoding='latin-1')  # Kaggle'dan indirilen CSV dosyasını uygun karakter kodlamasıyla okur.
df = df[['v1', 'v2']]  # Veri setindeki gereksiz boş sütunları atıp sadece etiket (v1) ve mesaj (v2) sütunlarını alır.
df.columns = ['label', 'message']  # Sütun isimlerini daha anlaşılır olması için 'label' ve 'message' olarak değiştirir.
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})  # 'ham' etiketini 0, 'spam' etiketini ise makinenin anlaması için 1 sayısına çevirir.

# 2. VERİYİ BÖLME
X = df['message']  # Mesaj içeriklerini (bağımsız değişken) X değişkenine atar.
y = df['label_num']  # Mesajın spam olup olmadığı bilgisini (hedef değişken) y değişkenine atar.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Verinin %80'ini eğitim, %20'sini test için ayırır.

# 3. ÖZELLİK ÇIKARIMI (METNİ SAYIYA DÖNÜŞTÜRME)
cv = CountVectorizer()  # Kelimeleri sayısallaştırmak için CountVectorizer nesnesi oluşturur.
X_train_count = cv.fit_transform(X_train.values)  # Eğitim verisindeki kelimeleri öğrenir ve mesajları sayısal bir tabloya dönüştürür.

# 4. MODEL EĞİTİMİ
model = MultinomialNB()  # Multinomial Naive Bayes sınıflandırıcı modelini tanımlar.
model.fit(X_train_count, y_train)  # Hazırlanan sayısal eğitim verileriyle modeli eğitmeye başlar.

# 5. TEST VE DEĞERLENDİRME
X_test_count = cv.transform(X_test)  # Test verisindeki mesajları, eğitimde oluşturulan kelime listesine göre sayıya çevirir.
y_pred = model.predict(X_test_count)  # Modelin test verileri üzerinde tahmin yapmasını sağlar.

print(f"Model Başarı Yüzdesi: {accuracy_score(y_test, y_pred) * 100:.2f}")  # Tahminlerin gerçek sonuçlarla ne kadar eşleştiğini yüzde olarak yazdırır.
print("\n--- Detaylı Analiz Raporu ---\n")  # Çıktı ekranında görsel bir başlık oluşturur.
print(classification_report(y_test, y_pred))  # Hassasiyet ve geri çağırma gibi detaylı başarı metriklerini ekrana basar.

# 6. CANLI TAHMİN ÖRNEĞİ
yeni_mesaj = ["URGENT! Your mobile number has been awarded a £2000 prize! Call 0905xxx to claim."]  # Test etmek için örnek bir spam mesajı tanımlar.
yeni_mesaj_sayisal = cv.transform(yeni_mesaj)  # Bu yeni mesajı modelin anlayabileceği sayısal formata sokar.
tahmin = model.predict(yeni_mesaj_sayisal)  # Eğitilen modelin bu mesajın ne olduğunu tahmin etmesini sağlar.

if tahmin[0] == 1:  # Eğer modelin tahmin sonucu 1 (yani spam) ise alt satıra geçer.
    print("Sonuç: Bu bir SPAM mesajdır!")  # Ekrana mesajın spam olduğunu yazdırır.
else:  # Eğer sonuç 1 değilse (yani 0 ise) alt satıra geçer.
    print("Sonuç: Bu güvenli (HAM) bir mesajdır.")  # Ekrana mesajın güvenli olduğunu yazdırır.