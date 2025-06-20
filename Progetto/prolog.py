from pyswip import Prolog
import csv

def validazione_righe_csv(row):
  """
  Valida una riga CSV.
     Args:
        row (list): Una riga del file CSV.
     Returns:
         tuple: I valori validati o None se non validi.
   """
  try:
       nazione = row[0]
       acqua_annuale = int(row[1].replace('.',''))
       acqua_pro_capite = int(row[2].replace('.',''))
       popolazione = int(row[3].replace('.',''))
       return nazione, acqua_annuale, acqua_pro_capite, popolazione
  except (IndexError): 
       print(f"Riga non valida: {row}")
       return None


def carica_risorse_idriche(file_csv, file_pl):
   """
   Carica le risorse idriche da un file CSV e genera i fatti.
      Args:
        file_csv (str): Il percorso del file CSV.
        file_pl (str): Il percorso del file Prolog
   """
   prolog = Prolog()

   try:
     prolog.consult(file_pl)
   except Exception as e:
     print(f"Errore durante la consultazione del file Prolog: {e}")

   with open(file_csv, mode='r') as file:
     reader = csv.reader(file, delimiter = ';')
     next(reader)  # Salta l'intestazione
     for row in reader:
        righe_valide = validazione_righe_csv(row)
        if righe_valide:
           nazione, acqua_annuale, acqua_pro_capite, popolazione = righe_valide
           prolog.assertz(f"acqua_annuale('{nazione}', {acqua_annuale})")
           prolog.assertz(f"acqua_pro_capite('{nazione}', {acqua_pro_capite})")
           prolog.assertz(f"popolazione('{nazione}', {popolazione})")
   return prolog
