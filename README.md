# 🚦 STOP Tabelası Tespit Projesi

Bu proje, STOP (dur) tabelalarının görüntülerde tespit edilmesini sağlayan basit bir Python projesidir.

## Klasör Yapısı

- `dataset/` → STOP tabelalarının bulunduğu test görselleri  
- `results/` → Tespit edilen STOP tabelalı görsellerin çıktıları  
- `main.py` → STOP tabelası tespit algoritmasının bulunduğu ana Python dosyası  
- `.gitignore` → Versiyon kontrolüne dahil edilmeyecek dosya ve klasörler  
- `README.md` → Bu açıklama dosyası  

## Kullanılan Teknolojiler

- Python 3.x  
- OpenCV  
- NumPy  

## ⚙️ Kurulum ve Çalıştırma

1. Gerekli kütüphaneleri kur:

```bash
pip install opencv-python numpy
```

2. Projeyi çalıştır:

```bash
python main.py
```

Proje çalıştırıldığında, `dataset/` klasörü içindeki görseller üzerinde STOP tabelası tespiti yapılır ve sonuç görselleri `results/` klasörüne kaydedilir.

## Örnek Kullanım

```bash
python main.py
``` 

## 🔗 GitHub

Bu proje GitHub üzerinde barındırılmaktadır:  
👉 [stop-sign-detector-python](https://github.com/UmutUsenmez/stop-sign-detector-python)
