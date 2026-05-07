🃏 Klasifikácia hracích kariet pomocou CNN
Záverečná práca v rámci študijného programu MSc Umelá Inteligencia a Strojové Učenie (AI a ML)
Škola: VITA ACADEMY

📌 O projekte
Táto aplikácia slúži ako edukačný systém na rozpoznávanie hracích kariet pomocou konvolučných neurónových sietí (CNN). Cieľom je ukázať celý životný cyklus ML projektu – od prípravy datasetu, cez tréning modelu, až po klasifikáciu nových obrázkov – v jednom prehľadnom GUI rozhraní.
Systém bol navrhnutý tak, aby bežal aj na bežnom počítači bez dedikovanej GPU a aby bol plne reprodukovateľný.

📁 Štruktúra projektu
projekt/
│
├── cards_classifier.py       # Hlavný súbor aplikácie
├── icon.png                  # Ikona okna (voliteľné)
│
├── dataset/                  # Dataset hracích kariet
│   ├── train/                # Trénovacie dáta (7 624 obrázkov, 53 tried)
│   ├── valid/                # Validačné dáta (265 obrázkov)
│   └── test/                 # Testovacie dáta (265 obrázkov)
│
└── models/                   # Generuje sa automaticky po tréningu
    ├── model_best.keras       # Uložený model
    ├── class_names.json       # Názvy tried
    ├── accuracy_plot.png      # Graf presnosti
    ├── loss_plot.png          # Graf straty
    ├── confusion_matrix.png   # Confusion matrix
    └── metrics_plot.png       # Graf metrík

⚙️ Inštalácia
1. Klonovanie repozitára

git clone https://github.com/<tvoj-github-nick>/cards-classifier.git
cd cards-classifier

2. Inštalácia závislostí

pip install tensorflow pillow matplotlib scikit-learn

3. Stiahnutie datasetu
Dataset hracích kariet je dostupný na platforme Kaggle:

🔗 Cards Image Dataset – Playing Card Images
https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification

Po stiahnutí rozbaľ dataset do priečinka dataset/ tak, aby mal štruktúru dataset/train/, dataset/valid/, dataset/test/.

🚀 Spustenie
python cards_classifier.py
Po spustení sa otvorí grafické GUI rozhranie aplikácie.

🖥️ Popis GUI aplikácie
Aplikácia obsahuje niekoľko záložiek a ovládacích prvkov:
Nastavenia tréningu
Pred spustením tréningu môžeš nastaviť tieto parametre priamo v GUI:
ParameterPredvolená hodnotaPopisEpochy10Počet epoch tréninguBatch size32Veľkosť dávky pri tréninguDropout0.2Miera regularizácie (0.1 / 0.2 / 0.3 / 0.4)
Priebeh tréningu

Reálne logovanie po každej epoche (accuracy, loss, val_accuracy, val_loss)
Tlačidlo STOP na prerušenie tréningu kedykoľvek
Automatické uloženie najlepšieho modelu

Grafy a metriky
Po tréningu aplikácia automaticky vygeneruje:

Graf presnosti (accuracy) počas tréningu
Graf straty (loss) počas tréningu
Confusion matrix
Prehľad metrík (precision, recall, F1-score)

Klasifikácia obrázkov

Načítanie celého testovacieho priečinka (dataset/test/)
Prechádzanie obrázkov pomocou tlačidiel alebo klávesových šípok
Upload vlastného obrázka z počítača
Zobrazenie Top-3 predikovaných tried s percentuálnou pravdepodobnosťou
Porovnanie predikcie so skutočnou triedou (podľa názvu priečinka)


🧠 Architektúra CNN modelu

Vstupný rozmer obrázka: 224 × 224 px (RGB)
Počet tried: 53 (52 kariet + žolík)
Celkový počet parametrov modelu: ~12 945 269
Regularizácia: Dropout (nastaviteľný v GUI)
Callback: automatické ukladanie najlepšieho modelu podľa validačnej presnosti
