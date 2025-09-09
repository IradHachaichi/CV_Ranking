import re
import spacy
from symspellpy import SymSpell, Verbosity
import os
import pandas as pd
import string

class GeneralOCRCorrector:
    def __init__(self, language="fr"):
        self.language = language
        try:
            self.nlp = spacy.load(f"{language}_core_news_md")
        except OSError:
            print(f"Warning: {language}_core_news_md not found. Install with: python -m spacy download {language}_core_news_md")
            self.nlp = None
        
        # Initialize spell checker
        self.spell_checker = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
        self.load_custom_dictionary()
        
        # Enhanced common words for French
        self.common_words = {
            'le', 'de', 'et', 'à', 'un', 'il', 'être', 'en', 'avoir', 'que', 'pour',
            'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus',
            'par', 'grand', 'venir', 'temps', 'très', 'savoir', 'voir', 'sans', 'petit',
            'si', 'après', 'bien', 'où', 'même', 'prendre', 'nom', 'partir', 'rien',
            'vouloir', 'dire', 'manière', 'tenir', 'devenir', 'quel', 'déjà', 'nouveau',
            'an', 'donner', 'homme', 'année', 'propre', 'mon', 'tant', 'moins', 'selon',
            'cela', 'entre', 'pouvoir', 'faire', 'aller', 'mettre', 'jour', 'ordre',
            'groupe', 'vers', 'chose', 'cas', 'gouvernement', 'lieu', 'vie', 'fin',
            'sorte', 'fois', 'moment', 'exemple', 'personne', 'main', 'partie', 'nombre',
            'point', 'monde', 'cours', 'place', 'contre', 'forme', 'question', 'droit',
            'eau', 'développement', 'économie', 'gestion', 'projet', 'recherche',
            'système', 'données', 'analyse', 'résultat', 'information', 'travail',
            'entreprise', 'marché', 'produit', 'service', 'client', 'équipe',
            'expérience', 'compétence', 'formation', 'diplôme', 'université', 'école',
            'marketing', 'assistant', 'compétences', 'professionnel', 'personnel',
            'communication', 'réseautage', 'esprit', 'équipe', 'développer',
            'capacité', 'travailler', 'pression', 'exécuter', 'évoluer', 'suivre',
            'instructions', 'obtenir', 'résultats', 'confiant', 'proposer', 'idées',
            'intéressantes', 'publicitaires', 'campagnes', 'coordonnées', 'juin',
            'janvier', 'maintenir', 'organiser', 'nombreux', 'dossiers', 'bureau',
            'constamment', 'listes', 'contacts', 'diffusion', 'présence', 'ligne',
            'mise', 'jour', 'site', 'web', 'société', 'médias', 'sociaux', 'suivi',
            'cours', 'potentiels', 'préparation', 'présentations', 'clients',
            'licence', 'juillet', 'paris', 'france'
        }
        
        # General OCR character confusion patterns
        self.char_confusion_patterns = [
            (r'\b(\w*)[0O](\w*)\b', self._number_to_letter_callback),
            (r'\b(\w*)[1Il|](\w*)\b', self._vertical_char_callback),
            (r'\b(\w*)[5S](\w*)\b', self._s_confusion_callback),
            (r'\b(\w*)[8B](\w*)\b', self._b_confusion_callback),
            (r'\brn\b', 'm'),
            (r'vv', 'w'),
            (r'cl', 'cl'),
            (r'fi', 'fi'),
        ]
        
        # Pattern-based corrections
        self.correction_patterns = [
            (r'(\w+)IES\b', r'\1LES'),
            (r'\bIA\b', 'LA'),
            (r'([A-Z])\1{2,}', r'\1\1'),
            (r'\bEXTERIENCE\b', 'EXPÉRIENCE'),
            (r'\bCOMPETENCES?\b', 'COMPÉTENCES'),
            (r'\bDEVELOPPER\b', 'DÉVELOPPER'),
            (r'\bORGANISER\b', 'ORGANISER'),
            (r'\bMAINTENIR\b', 'MAINTENIR')
        ]

    def load_custom_dictionary(self):
        """Load custom dictionary if available"""
        dictionary_path = "frequency_dictionary_fr_custom.txt"
        try:
            if os.path.exists(dictionary_path):
                df = pd.read_csv(dictionary_path, sep=" ", header=None, names=["word", "freq"])
                df = df[df["freq"].astype(str).str.match(r'^\d+$', na=False)]
                df.to_csv("temp_dict.txt", sep=" ", index=False, header=False)
                self.spell_checker.load_dictionary("temp_dict.txt", term_index=0, count_index=1)
                os.remove("temp_dict.txt")
        except Exception as e:
            print(f"Dictionary loading failed: {e}")

    def _number_to_letter_callback(self, match):
        """Callback for number to letter conversion in words"""
        prefix, suffix = match.groups()
        full_word = match.group(0)
        if len(prefix + suffix) > 0 and (prefix.isalpha() or suffix.isalpha()):
            return full_word.replace('0', 'O')
        return full_word
    
    def _vertical_char_callback(self, match):
        """Handle vertical character confusions (1, I, l, |)"""
        prefix, suffix = match.groups()
        full_word = match.group(0)
        if len(prefix + suffix) > 0 and (prefix.isalpha() or suffix.isalpha()):
            return re.sub(r'[1Il|]', 'I', full_word)
        return full_word
    
    def _s_confusion_callback(self, match):
        """Handle S/5 confusion"""
        prefix, suffix = match.groups()
        full_word = match.group(0)
        if len(prefix + suffix) > 0 and (prefix.isalpha() or suffix.isalpha()):
            return full_word.replace('5', 'S')
        return full_word
    
    def _b_confusion_callback(self, match):
        """Handle B/8 confusion"""
        prefix, suffix = match.groups()
        full_word = match.group(0)
        if len(prefix + suffix) > 0 and (prefix.isalpha() or suffix.isalpha()):
            return full_word.replace('8', 'B')
        return full_word

    def clean_artifacts(self, text):
        """Clean OCR artifacts with enhanced pipeline"""
        text = re.sub(r'[\.]{3,}', ' ', text)
        text = re.sub(r'[\s]{2,}', ' ', text)
        text = re.sub(r'\b[B-HJ-Z]\b(?![a-zA-Z0-9])', '', text)
        for pattern, replacement in self.char_confusion_patterns:
            if callable(replacement):
                text = re.sub(pattern, replacement, text)
            else:
                text = re.sub(pattern, replacement, text)
        text = re.sub(r'[~@#&*+]{2,}', ' ', text)
        text = re.sub(r'\s+[&@#~*+]+\s+', ' ', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([A-Z]{2,})([a-z])', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text.strip())
        return text

    def spell_check_word(self, word):
        """Adaptive spell checking for individual words"""
        if len(word) < 2 or word.isdigit():
            return word
        if '@' in word or '.' in word and len(word.split('.')) > 1:
            return word
        if word.lower() in self.common_words:
            return word
        word_length = len(word)
        max_distance = 1 if word_length < 5 else 2 if word_length < 10 else 3
        if hasattr(self.spell_checker, '_words') and self.spell_checker._words:
            suggestions = self.spell_checker.lookup(
                word.lower(), 
                Verbosity.CLOSEST, 
                max_edit_distance=max_distance
            )
            if suggestions:
                best_suggestion = suggestions[0]
                confidence_threshold = 0.7
                if (best_suggestion.distance / len(word)) < confidence_threshold:
                    corrected = best_suggestion.term
                    if word.isupper():
                        return corrected.upper()
                    elif word.istitle():
                        return corrected.title()
                    else:
                        return corrected
        return word

    def context_aware_correction(self, text):
        """Apply context-aware corrections using spaCy if available"""
        if self.nlp is None:
            words = text.split()
            corrected_words = []
            for word in words:
                clean_word = ''.join(c for c in word if c.isalpha())
                punctuation = ''.join(c for c in word if not c.isalpha())
                if clean_word:
                    corrected_word = self.spell_check_word(clean_word)
                    corrected_words.append(corrected_word + punctuation)
                elif punctuation:
                    corrected_words.append(punctuation)
            return ' '.join(corrected_words)
        
        doc = self.nlp(text)
        corrected_tokens = []
        for token in doc:
            if token.pos_ in ["PUNCT", "NUM", "SPACE"] or re.match(r'[\w\.-]+@[\w\.-]+', token.text):
                corrected_tokens.append(token.text)
            elif token.is_alpha:
                corrected_tokens.append(self.spell_check_word(token.text))
            else:
                corrected_tokens.append(token.text)
        return ' '.join(corrected_tokens)

    def reconstruct_text_structure(self, text):
        """Intelligently reconstruct text structure"""
        sentences = re.split(r'([.!?]+)', text)
        reconstructed = []
        i = 0
        while i < len(sentences):
            sentence = sentences[i].strip()
            if len(sentence) < 3:
                i += 1
                continue
            punctuation = ''
            if i + 1 < len(sentences) and sentences[i + 1].strip():
                punctuation = sentences[i + 1].strip()
                i += 1
            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:].lower()
                if not punctuation and not sentence[-1] in '.!?':
                    sentence += '.'
                elif punctuation:
                    sentence += punctuation
                reconstructed.append(sentence)
            i += 1
        return ' '.join(reconstructed)

    def clean_text(self, text):
        """Main cleaning function with enhanced pipeline"""
        # Apply pattern-based corrections
        for pattern, replacement in self.correction_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Clean artifacts
        text = self.clean_artifacts(text)
        
        # Context-aware correction
        text = self.context_aware_correction(text)
        
        # Format for professional documents
        text = re.sub(r'\b(JUIN|JUILLET|JANVIER|FÉVRIER|MARS|AVRIL|MAI|SEPTEMBRE|OCTOBRE|NOVEMBRE|DÉCEMBRE)\b', 
                     lambda m: m.group(1).title(), text)
        text = re.sub(r'\b(\d{4})\b', r'\1', text)
        text = re.sub(r'\b(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\b', 
                     r'\1 \2 \3 \4 \5', text)
        text = re.sub(r'([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+)\.([a-zA-Z]{2,})', 
                     r'\1@\2.\3', text)
        
        # Reconstruct text structure
        text = self.reconstruct_text_structure(text)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text

def clean_ocr_text(raw_text, language="fr"):
    """Helper function to clean OCR paragraph text"""
    corrector = GeneralOCRCorrector(language=language)
    return corrector.clean_text(raw_text)
