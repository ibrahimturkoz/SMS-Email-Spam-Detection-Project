# 📧 SMS & Email Spam Detection Project

Bu proje, makine öğrenmesi (Machine Learning) tekniklerini kullanarak gelen mesajların **SPAM** (istenmeyen) veya
**HAM** (güvenli) olduğunu yüksek doğrulukla tespit etmek için geliştirilmiştir.

---

## 🚀 Başarı Oranı: %98.39
Modelimiz, test verileri üzerinde yapılan değerlendirmeler sonucunda **%98.39** doğruluk (accuracy) oranına ulaşmıştır.

---


## 🛠️ Kullanılan Teknolojiler
* **Python 3.x**
* **Pandas:** Veri manipülasyonu.
* **Scikit-Learn:** Makine öğrenmesi modeli ve metin işleme.
* **Multinomial Naive Bayes:** Sınıflandırma algoritması.
* **CountVectorizer:** Metin verilerini sayısal vektörlere dönüştürme.

---

## 📊 Veri Seti
Projelerimizde Kaggle'ın popüler [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
veri seti kullanılmıştır. 
* **Toplam Veri:** 5,574 Mesaj

---

### 📈 Model Performans Tablosu

| Metrik Parametresi | Skor / Değer | Açıklama |
| :--- | :--- | :--- |
| **Model Doğruluğu** | `%98.39` | Test setindeki genel başarı oranı. |
| **Algoritma** | `MultinomialNB` | Naive Bayes tabanlı sınıflandırıcı. |
| **Eğitim Verisi** | `4,459` | Modelin eğitildiği mesaj sayısı. |
| **Test Verisi** | `1,115` | Başarının ölçüldüğü mesaj sayısı. |
| **Vektörleştirme** | `CountVectorize` | Metinden sayıya dönüştürme yöntemi. |
| **Random State** | `42` | Sonuçların tutarlılığı için kullanılan sabit. |

---

## 📝 Algoritma Hakkında

Projede Multinomial Naive Bayes algoritması tercih edilmiştir. Bu algoritma, kelimelerin bir belgede bulunma olasılıklarını 
hesaplayarak metin sınıflandırma (text classification) işlemlerinde çok yüksek hız ve doğruluk sağlar.

Metinler, bilgisayarın anlayabileceği sayısal değerlere CountVectorizer yöntemiyle dönüştürülmüştür.

---

## 💻 Örnek Kullanım (Kod Bloğu)

Projeyi kendi yerelinizde test etmek için aşağıdaki Python kod yapısını kullanabilirsiniz:

```python
from spam_detector import spam_kontrol

# Test etmek istediğiniz mesajı buraya yazın
mesaj_1 = "Tebrikler! 1000 TL değerinde hediye çeki kazandınız. Hemen tıklayın!"
mesaj_2 = "Merhaba İbrahim, bugün saat 15:00'da toplantımız var, unutma."

# Tahmin sonuçlarını yazdır
print(f"Mesaj 1 Analizi: {spam_kontrol(mesaj_1)}") # Çıktı: SPAM
print(f"Mesaj 2 Analizi: {spam_kontrol(mesaj_2)}") # Çıktı: GÜVENLİ (HAM)

```

## ✅ Özellikler

1-%98+ üzerinde yüksek doğruluk.

2-Karmaşıklıktan uzak, optimize edilmiş kod yapısı.

3-Gerçek zamanlı mesaj testi yapabilme özelliği.

---

## Geliştirici: İbrahim Türköz

