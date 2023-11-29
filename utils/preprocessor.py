import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd


class Preprocessor:
    def __init__(
        self,
        remove_url=True,
        remove_punct=True,
        remove_stopwords=True,
        tokenize_words=True,
        lemmatize_words=True,
    ) -> None:
        self.methods = []
        if remove_url:
            self.methods.append(self._remove_URL)
        if remove_punct:
            self.methods.append(self._remove_punct)
        if remove_stopwords:
            nltk.download("stopwords")
            self.methods.append(self._remove_stopwords)
        if tokenize_words:
            nltk.download("punkt")
            self.methods.append(self._tokenize_words)
        if lemmatize_words:
            nltk.download("wordnet")
            self.methods.append(self._lemmatize_words)

    def apply(self, data: pd.Series | str | list) -> str | list[str]:
        return self(data)

    def __call__(self, data: str | list) -> str | list[str]:
        """Apply cleaning methods on the data and return the cleaned data"""
        result = data
        for method in self.methods:
            if isinstance(data, pd.Series):
                result = result.map(method)
            else:
                result = method(result)
        return result

    def _remove_URL(self, text: str) -> str:
        url = re.compile(r"https?://\S+|www\.\S+")
        return url.sub(r"", text)

    def _remove_punct(self, text: str) -> str:
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)

    def _remove_stopwords(self, text: str) -> str:
        stop = set(stopwords.words("english"))

        filtered_words = [
            word.lower() for word in text.split() if word.lower() not in stop
        ]
        return " ".join(filtered_words)

    def _tokenize_words(self, text: str) -> list[str]:
        return nltk.tokenize.word_tokenize(text)

    def _lemmatize_words(self, text: list[str]) -> list[str]:
        lemmatizer = WordNetLemmatizer()
        result_sentence = []
        for token in text:
            result_sentence.append(lemmatizer.lemmatize(token))
        return result_sentence
