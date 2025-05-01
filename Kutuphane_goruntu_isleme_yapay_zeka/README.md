# Kütüphane Doluluk Takip Sistemi

Bu proje, kütüphanelerdeki oturan insan sayısını ve doluluk oranını otomatik olarak tespit eden bir web uygulamasıdır. YOLO (You Only Look Once) nesne tespit modeli kullanılarak insanları ve sandalyeleri tespit eder.

## Özellikler

- Görüntü yükleme ve analiz etme
- Oturan insan sayısını tespit etme
- Toplam insan ve sandalye sayısını tespit etme
- Geçmiş kayıtları görüntüleme
- Modern ve kullanıcı dostu arayüz

## Kurulum

1. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

2. YOLO modelini indirin:
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo12m.pt
```

3. Uygulamayı başlatın:
```bash
python app.py
```

4. Tarayıcınızda `http://localhost:5000` adresine gidin.

## Kullanım

1. Ana sayfada "Görüntü Yükle" butonuna tıklayın
2. Analiz etmek istediğiniz görüntüyü seçin
3. "Analiz Et" butonuna tıklayın
4. Sonuçları görüntüleyin
5. Geçmiş kayıtları görüntülemek için "Geçmiş Kayıtlar" butonuna tıklayın

## Teknik Detaylar

- Flask web framework'ü kullanılmıştır
- SQLite veritabanı ile veri depolama
- YOLO v8 nesne tespit modeli
- OpenCV ile görüntü işleme
- Bootstrap ile modern UI tasarımı

## Geliştirici Notları

- `app.py`: Ana uygulama dosyası
- `models.py`: Veritabanı modelleri
- `templates/`: HTML şablonları
- `static/`: Statik dosyalar (CSS, JS, görüntüler) 