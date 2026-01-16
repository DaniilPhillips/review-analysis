"""
Модуль предобработки текстовых данных

Этапы предобработки:
1. Очистка текста — удаление лишних символов, приведение к нижнему регистру
2. Токенизация — разбиение на слова/токены
3. Лемматизация — приведение к начальной форме (опционально)
4. Кодирование — преобразование текста в числовые последовательности
5. Паддинг — выравнивание последовательностей до одной длины
"""

import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PREPROCESSING_CONFIG, CATEGORIES


class TextPreprocessor:
    """Препроцессор текстовых данных для обучения нейросети"""
    
    # Стоп-слова русского языка (частичный список)
    STOP_WORDS = {
        "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
        "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к", 
        "у", "же", "вы", "за", "бы", "по", "только", "её", "мне", "было",
        "вот", "от", "меня", "ещё", "нет", "о", "из", "ему", "теперь",
        "когда", "уже", "для", "вам", "ведь", "там", "мой", "если", "этот"
    }
    
    def __init__(self, config: Dict = None):
        """
        Инициализация препроцессора
        
        Args:
            config: Словарь с параметрами предобработки
        """
        self.config = config or PREPROCESSING_CONFIG
        self.max_words = self.config["max_words"]
        self.max_sequence_length = self.config["max_sequence_length"]
        
        # Словари для кодирования
        self.word_to_index: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.index_to_word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self.word_counts: Dict[str, int] = {}
        
        # Кодировщики для категориальных признаков
        self.category_to_index: Dict[str, int] = {}
        self.subject_to_index: Dict[str, int] = {}
        
        self.is_fitted = False
        
    def clean_text(self, text: str) -> str:
        """
        Очистка текста от шума
        
        Args:
            text: Исходный текст
            
        Returns:
            Очищенный текст
        """
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление HTML-тегов
        text = re.sub(r'<[^>]+>', '', text)
        
        # Удаление URL
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Удаление email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Оставляем только буквы и пробелы
        text = re.sub(r'[^а-яёa-z\s]', ' ', text)
        
        # Удаление множественных пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str, remove_stopwords: bool = False) -> List[str]:
        """
        Токенизация текста
        
        Args:
            text: Очищенный текст
            remove_stopwords: Удалять ли стоп-слова
            
        Returns:
            Список токенов
        """
        tokens = text.split()
        
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.STOP_WORDS]
        
        # Фильтрация коротких токенов
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    def fit(self, texts: List[str], categories: List[str] = None, 
            subjects: List[str] = None):
        """
        Обучение препроцессора на текстах
        
        Строит словарь на основе частотности слов
        
        Args:
            texts: Список текстов для обучения
            categories: Список категорий
            subjects: Список субъектов
        """
        # Подсчёт частотности слов
        for text in texts:
            cleaned = self.clean_text(text)
            tokens = self.tokenize(cleaned)
            for token in tokens:
                self.word_counts[token] = self.word_counts.get(token, 0) + 1
        
        # Сортировка по частотности и построение словаря
        sorted_words = sorted(
            self.word_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.max_words - 2]  # -2 для PAD и UNK
        
        for word, _ in sorted_words:
            idx = len(self.word_to_index)
            self.word_to_index[word] = idx
            self.index_to_word[idx] = word
        
        # Кодирование категорий
        if categories:
            unique_categories = sorted(set(categories))
            self.category_to_index = {cat: i for i, cat in enumerate(unique_categories)}
        
        # Кодирование субъектов
        if subjects:
            unique_subjects = sorted(set(subjects))
            self.subject_to_index = {subj: i for i, subj in enumerate(unique_subjects)}
        
        self.is_fitted = True
        print(f"✅ Препроцессор обучен:")
        print(f"   Размер словаря: {len(self.word_to_index)}")
        print(f"   Категорий: {len(self.category_to_index)}")
        print(f"   Субъектов: {len(self.subject_to_index)}")
    
    def text_to_sequence(self, text: str) -> List[int]:
        """
        Преобразование текста в последовательность индексов
        
        Args:
            text: Исходный текст
            
        Returns:
            Список индексов слов
        """
        if not self.is_fitted:
            raise ValueError("Препроцессор не обучен. Вызовите fit() сначала.")
        
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        
        sequence = []
        for token in tokens:
            idx = self.word_to_index.get(token, 1)  # 1 = UNK
            sequence.append(idx)
        
        return sequence
    
    def pad_sequence(self, sequence: List[int]) -> np.ndarray:
        """
        Паддинг последовательности до фиксированной длины
        
        Args:
            sequence: Последовательность индексов
            
        Returns:
            Выровненная последовательность
        """
        if len(sequence) > self.max_sequence_length:
            # Обрезаем с конца (сохраняем начало)
            return np.array(sequence[:self.max_sequence_length])
        else:
            # Дополняем нулями (pre-padding)
            padding = [0] * (self.max_sequence_length - len(sequence))
            return np.array(padding + sequence)
    
    def encode_category(self, category: str) -> int:
        """Кодирование категории в индекс"""
        return self.category_to_index.get(category, 0)
    
    def encode_subject(self, subject: str) -> int:
        """Кодирование субъекта в индекс"""
        return self.subject_to_index.get(subject, 0)
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Полное преобразование датасета
        
        Args:
            df: DataFrame с колонками review_text, rating, category, subject_id,
                sentiment_class, reputation_index
                
        Returns:
            Кортеж из:
            - X_text: последовательности токенов [n_samples, max_seq_len]
            - X_rating: оценки [n_samples, 1]
            - X_category: закодированные категории [n_samples, 1]
            - y_class: классы тональности [n_samples]
            - y_index: индексы репутации [n_samples]
        """
        if not self.is_fitted:
            raise ValueError("Препроцессор не обучен. Вызовите fit() сначала.")
        
        n_samples = len(df)
        
        # Преобразование текстов
        X_text = np.zeros((n_samples, self.max_sequence_length), dtype=np.int32)
        for i, text in enumerate(df["review_text"]):
            sequence = self.text_to_sequence(text)
            X_text[i] = self.pad_sequence(sequence)
        
        # Оценки (нормализация к 0-1)
        X_rating = (df["rating"].values - 1) / 4  # [1,5] -> [0,1]
        X_rating = X_rating.reshape(-1, 1).astype(np.float32)
        
        # Категории (one-hot encoding)
        n_categories = len(self.category_to_index)
        X_category = np.zeros((n_samples, n_categories), dtype=np.float32)
        for i, cat in enumerate(df["category"]):
            cat_idx = self.encode_category(cat)
            X_category[i, cat_idx] = 1.0
        
        # Целевые переменные
        y_class = df["sentiment_class"].values.astype(np.int32)
        y_index = df["reputation_index"].values.astype(np.float32)
        
        return X_text, X_rating, X_category, y_class, y_index
    
    def get_vocab_size(self) -> int:
        """Возвращает размер словаря"""
        return len(self.word_to_index)
    
    def get_num_categories(self) -> int:
        """Возвращает количество категорий"""
        return len(self.category_to_index)
    
    def decode_sequence(self, sequence: np.ndarray) -> str:
        """
        Декодирование последовательности обратно в текст
        
        Args:
            sequence: Последовательность индексов
            
        Returns:
            Восстановленный текст
        """
        words = []
        for idx in sequence:
            if idx == 0:  # PAD
                continue
            word = self.index_to_word.get(idx, "<UNK>")
            words.append(word)
        return " ".join(words)


def main():
    """Демонстрация работы препроцессора"""
    from data_generator import ReviewDataGenerator
    
    # Генерируем тестовые данные
    generator = ReviewDataGenerator({"num_samples": 100, "num_subjects": 10, "random_seed": 42})
    df = generator.generate_dataset()
    
    # Создаём и обучаем препроцессор
    preprocessor = TextPreprocessor()
    preprocessor.fit(
        texts=df["review_text"].tolist(),
        categories=df["category"].tolist(),
        subjects=df["subject_id"].tolist()
    )
    
    # Преобразуем данные
    X_text, X_rating, X_category, y_class, y_index = preprocessor.transform(df)
    
    print("\n📊 Результаты предобработки:")
    print(f"   X_text shape: {X_text.shape}")
    print(f"   X_rating shape: {X_rating.shape}")
    print(f"   X_category shape: {X_category.shape}")
    print(f"   y_class shape: {y_class.shape}")
    print(f"   y_index shape: {y_index.shape}")
    
    # Пример декодирования
    print("\n🔄 Пример преобразования:")
    sample_text = df["review_text"].iloc[0]
    print(f"   Оригинал: {sample_text}")
    sequence = preprocessor.text_to_sequence(sample_text)
    print(f"   Последовательность: {sequence[:10]}...")
    decoded = preprocessor.decode_sequence(X_text[0])
    print(f"   Декодировано: {decoded}")


if __name__ == "__main__":
    main()
