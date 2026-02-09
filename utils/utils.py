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

import pandas as pd
import time

import pandas as pd
import time

def label_courses_with_gpt(df, topic_list_str, output_path, max_new_topics=20, model_name="gpt-4o-mini"):
    """
    Labels Kurse dynamisch und lernt nur REIN INFORMATISCHE Topics dazu.
    Trennt mehrere ausgewählte Topics im Output durch " ; ".
    """
    # Basis-Liste erstellen
    current_topics = [t.strip() for t in topic_list_str.strip().split('\n') if t.strip()]
    newly_learned = [] # Liste für die separat zurückgegebenen neuen Topics
    new_topics_count = 0
    
    # Farbcodes für das Terminal
    RED = "\033[91m"
    RESET = "\033[0m"
    
    # Spalten initialisieren
    if 'Label_ChatGPT' not in df.columns:
        df['Label_ChatGPT'] = None
    if 'Begründung_KI' not in df.columns:
        df['Begründung_KI'] = None

    total = len(df)
    print(f"Starte dynamisches Informatik-Labeling. Basis-Topics: {len(current_topics)}")

    for i in range(total):
        # Überspringen, wenn bereits gelabelt
        if pd.notna(df.iloc[i]['Label_ChatGPT']) and df.iloc[i]['Label_ChatGPT'] != "":
            continue

        titel = df.iloc[i]['veranstaltung_titel']
        beschreibung = df.iloc[i]['kursbeschreibung'] if pd.notna(df.iloc[i]['kursbeschreibung']) else "Keine Beschreibung vorhanden."

        print(f"Verarbeite Kurs {i+1}/{total}: {titel}")
        
        dynamic_topic_list = "\n".join(current_topics)
        
        prompt = f"""
        Du bist ein Experte für universitäre Informatik-Lehre. Deine Aufgabe ist eine SEHR KONSERVATIVE Klassifizierung.

        ZUGELASSENE TOPICS:
        {dynamic_topic_list}

        KURS-DATEN:
        Titel: {titel}
        Beschreibung: {beschreibung}

        STRENGSTE ANWEISUNGEN:
        1. HAUPTFOKUS: Wähle NUR EIN EINZIGES Topic aus der Liste, das den Kern des Kurses am besten trifft.
        2. KEINE REDUNDANZ: Wähle NICHT 'Software Engineering', wenn bereits ein spezielleres Thema wie 'Webprogrammierung' passt.
        3. MULTI-LABEL: Nur wenn der Kurs explizit zwei völlig unterschiedliche Fachgebiete gleichwertig behandelt, sind 2 Labels erlaubt. 3 oder 4 Labels sind STRENGSTENS VERBOTEN.
        4. NEUE TOPICS: Erstelle ein NEW_ Topic nur, wenn es eine fundamentale Informatik-Disziplin ist, die absolut nicht in der Liste steht.
        5. 'Sonstiges / Keine Zuordnung möglich' ist die Standardwahl für alles, was interdisziplinär oder fachfremd ist.

        ANTWORTE NUR IM FORMAT:
        Topics: [Topic A, Topic B]
        Grund: [Kurzer Satz zur fachlichen Einordnung]
        """

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein strenger Informatik-Professor. Falls du mehrere Topics wählst, trenne sie innerhalb der Klammern mit Kommata."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                seed=42
            )
            
            answer = response.choices[0].message.content
            
            # Parsing: Inhalt zwischen "Topics:" und "Grund:" extrahieren
            topics_part_raw = answer.split("Topics:")[1].split("Grund:")[0].strip().replace("[", "").replace("]", "")
            reason_part = answer.split("Grund:")[1].strip()

            # In Liste umwandeln, trimmen und leere Einträge entfernen
            found_topics = [t.strip() for t in topics_part_raw.split(",") if t.strip()]
            
            # Neue Topics verarbeiten
            for ft in found_topics:
                if ft.startswith("NEW_") and ft not in current_topics:
                    if new_topics_count < max_new_topics:
                        current_topics.append(ft)
                        newly_learned.append(ft)
                        new_topics_count += 1
                        print(f"{RED}(!) Neues Informatik-Topic gelernt: {ft} ({new_topics_count}/{max_new_topics}){RESET}")

            # --- Speichern im DF mit Semikolon-Trennung ---
            df.at[df.index[i], 'Label_ChatGPT'] = " ; ".join(found_topics)
            df.at[df.index[i], 'Begründung_KI'] = reason_part
            
        except Exception as e:
            print(f"Fehler bei Zeile {i}: {e}")

        # Zwischenspeichern alle 10 Zeilen
        if (i + 1) % 10 == 0 or (i + 1) == total:
            df.to_csv(output_path, index=False, encoding="utf-8")

        time.sleep(0.05)

    return df, current_topics, newly_learned