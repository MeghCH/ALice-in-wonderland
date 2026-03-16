import requests
import os

def clean_text(text):
    start = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    if start in text:
        text = text.split(start)[1]
    if end in text:
        text = text.split(end)[0]
    
    return text.strip()

# Téléchargement des livres + nettoyage du header & footer
def downclean(book_id):
    file_name = f"{book_id}.txt"
    # Si le fichier existe -> lire depuis le disque 
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            return f.read()
        lien = requests.get (f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt")
        if lien.status_code == 200:
            texte = clean_text(lien.text)
        with open(file_name, "w") as f:
            f.write(texte)
        return texte
    # sinon -> télécharger -> nettoyer -> sauvegarder -> retourner
    else:
        lien_bis = requests.get (f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt")
        if lien_bis.status_code == 200:
            texte = clean_text(lien_bis.text)
            with open(file_name, "w") as f:
                f.write(texte)
            return texte
        else:
            return None 