import math
import os
def calculate_sample_size(N=None, p=0.5, z=1.96, e=0.03):
    """
    Berechnet die benötigte Stichprobengröße basierend auf der Cochran-Formel.
    
    Parameter:
    N: Gesamtpopulation (Optional). Falls None, wird n_0 berechnet.
    p: Erwarteter Anteil (Default 0.5 für maximale Stichprobe).
    z: Z-Wert (Default 1.96 für 95% Konfidenz).
    e: Fehlermarge (Default 0.03 für 3%).
    """
    
    # 1. Basis-Stichprobengröße (n0) berechnen
    n_0 = (z**2 * p * (1 - p)) / (e**2)
    
    if N is None:
        return math.ceil(n_0)
    
    # 2. Korrekturterm für kleine Populationen berechnen
    # Korrekturterm = 1 + (z^2 * p * (1-p)) / (e^2 * N)
    correction_term = 1 + (n_0 / N)
    
    # Finale Stichprobengröße = n0 / Korrekturterm
    sample_size = n_0 / correction_term
    
    return math.ceil(sample_size)

import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
import time
from pathlib import Path

# Initialisierung
# 1. Variablen aus .env laden
load_dotenv()

# 2. Key aus der Umgebung ziehen
api_key = os.getenv("OPENAI_API_KEY")

# 3. Den Client initialisieren
client = OpenAI(api_key=api_key)

import pandas as pd
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def label_courses_with_gpt(df, topic_list_str, output_path, max_new_topics=20, model_name="gpt-4o-mini"):
    """
    Labels Kurse dynamisch: Nutzt die bestehende Liste und ergänzt sie um bis zu 20 neue Topics,
    die für nachfolgende Zeilen sofort verfügbar sind.
    """
    # Bestehende Topics in eine Liste laden
    current_topics = [t.strip() for t in topic_list_str.strip().split('\n') if t.strip()]
    new_topics_count = 0
    
    if 'Labels_ChatGPT' not in df.columns:
        df['Labels_ChatGPT'] = None
    if 'Begründung_KI' not in df.columns:
        df['Begründung_KI'] = None

    total = len(df)
    print(f"Starte dynamisches Labeling. Basis-Topics: {len(current_topics)}")

    for i in range(total):
        # Überspringe bereits bearbeitete Zeilen
        if pd.notna(df.iloc[i]['Labels_ChatGPT']) and df.iloc[i]['Labels_ChatGPT'] != "":
            continue

        titel = df.iloc[i]['veranstaltung_titel']
        beschreibung = df.iloc[i]['kursbeschreibung'] if pd.notna(df.iloc[i]['kursbeschreibung']) else "Keine Beschreibung vorhanden."
        
        # Die Liste wird bei jedem Durchgang aktuell zusammengebaut (inkl. der "gelernten" Topics)
        dynamic_topic_list = "\n".join(current_topics)
        
        prompt = f"""
        Du bist ein Experte für Informatik-Lehre. Ein Kurs kann mehrere Schwerpunkte haben.
        
        AKTUELL ZUGELASSENE TOPICS:
        {dynamic_topic_list}

        KURS-DATEN:
        Titel: {titel}
        Beschreibung: {beschreibung}

        ANWEISUNG:
        1. Wähle bis zu 4 passende Topics aus der Liste. Nutze EXAKT die Namen.
        2. Falls ein wichtiges Thema fehlt UND wir noch nicht {max_new_topics} neue Topics erfunden haben, darfst du ein NEUES Topic erstellen.
        3. Format für neue Topics: 'NEW_[Themenname]' (z.B. 'NEW_Web-Development').
        4. Wichtigstes zuerst, Trennung durch Komma.
        5. Wenn absolut nichts passt: '99_Sonstiges'.

        ANTWORTE NUR IM FORMAT:
        Topics: [Deine Auswahl]
        Grund: [Kurze Begründung, max. 15 Wörter]
        """

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein präziser Klassifikator, der eine Themenliste dynamisch erweitern kann."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                seed=42
            )
            
            answer = response.choices[0].message.content
            topics_part = answer.split("Topics:")[1].split("Grund:")[0].strip().replace("[", "").replace("]", "")
            reason_part = answer.split("Grund:")[1].strip()

            # Neue Topics in unsere laufende Liste aufnehmen
            found_topics = [t.strip() for t in topics_part.split(",")]
            for ft in found_topics:
                if ft.startswith("NEW_") and ft not in current_topics:
                    if new_topics_count < max_new_topics:
                        current_topics.append(ft)
                        new_topics_count += 1
                        print(f"(!) Neues Topic gelernt: {ft} ({new_topics_count}/{max_new_topics})")
                    else:
                        # Falls Limit erreicht, das NEW_ Tag ignorieren oder auf Sonstiges umleiten
                        print(f"Limit erreicht. Ignoriere neues Topic: {ft}")

            df.at[df.index[i], 'Labels_ChatGPT'] = topics_part
            df.at[df.index[i], 'Begründung_KI'] = reason_part
            
        except Exception as e:
            print(f"Fehler bei Zeile {i} (Index {df.index[i]}): {e}")

        # Zwischenspeichern
        if (i + 1) % 10 == 0 or (i + 1) == total:
            df.to_csv(output_path, index=False, encoding="utf-8")
            print(f"Fortschritt: {i+1}/{total} gespeichert.")

        time.sleep(0.05)

    print(f"Fertig! Neue Topics insgesamt erstellt: {new_topics_count}")
    return df, current_topics