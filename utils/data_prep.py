# ::::::::::::::::::::::::::::::::::::::::::::::::::::
# Datenvorbereitung Model 1a
# ::::::::::::::::::::::::::::::::::::::::::::::::::::

# ::::::::::::::::::::::::::::::::::::::::::::::::::::
# Lade Pakete
# ::::::::::::::::::::::::::::::::::::::::::::::::::::

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ::::::::::::::::::::::::::::::::::::::::::::::::::::
# Daten einlesen
# ::::::::::::::::::::::::::::::::::::::::::::::::::::

df = pd.read_csv("data/informatikkurse_export_DE.csv", sep=';')
df = df[df['filter_model_1a'] == 1]

print(df.head())
len(df)

# ::::::::::::::::::::::::::::::::::::::::::::::::::::
# Daten Splitten in Trainings- und Testdaten
# ::::::::::::::::::::::::::::::::::::::::::::::::::::

# Split-Ratio (hier 80% Training, 20% Test)
test_size = 0.01  

# Split der Daten
df_train, df_test = train_test_split(
    df, 
    test_size=test_size, 
    random_state=42 # ⬅️ Wichtig für Reproduzierbarkeit
)

# Ausgabe der Ergebnisse
print(f"Anzahl der Trainingsdaten: {len(df_train)}")    
print(f"Anzahl der Testdaten: {len(df_test)}")

# Speichern der aufgeteilten Datensätze
output_dir = Path("data/processed") 
output_dir.mkdir(parents=True, exist_ok=True)  # Erstelle das Verzeichnis, falls es nicht existiert
df_train.to_csv(output_dir / "train_data.csv", index=False)
df_test.to_csv(output_dir / "test_data.csv", index=False)