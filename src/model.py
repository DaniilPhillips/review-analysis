"""
Архитектура нейросети для анализа репутации

Модель: Bidirectional LSTM с мультизадачным обучением

Обоснование выбора:
1. LSTM — эффективно работает с последовательностями, запоминает контекст
2. Bidirectional — учитывает контекст слева и справа
3. Мультизадачность — одновременно решает классификацию и регрессию

Архитектура:
┌─────────────────┐   ┌─────────────┐   ┌──────────────┐
│ Текст (tokens)  │   │   Оценка    │   │   Категория  │
└────────┬────────┘   └──────┬──────┘   └──────┬───────┘
         │                   │                  │
    ┌────▼────┐              │                  │
    │Embedding│              │                  │
    └────┬────┘              │                  │
         │                   │                  │
    ┌────▼─────────┐         │                  │
    │ Bi-LSTM (64) │         │                  │
    └────┬─────────┘         │                  │
         │                   │                  │
    ┌────▼────┐              │                  │
    │ Dropout │              │                  │
    └────┬────┘              │                  │
         │                   │                  │
         └─────────┬─────────┴──────────────────┘
                   │ Concatenate
              ┌────▼────┐
              │Dense(64)│
              └────┬────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    ┌────▼────┐        ┌─────▼─────┐
    │Dense(32)│        │ Dense(32) │
    └────┬────┘        └─────┬─────┘
         │                   │
    ┌────▼─────┐       ┌─────▼──────┐
    │Softmax(3)│       │  Linear(1) │
    │(класс)   │       │  (индекс)  │
    └──────────┘       └────────────┘
"""

import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PREPROCESSING_CONFIG, MODEL_CONFIG, MODELS_DIR

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class ReputationModel:
    """
    Модель анализа влияния отзывов на репутацию
    
    Мультизадачная архитектура:
    - Задача 1: Классификация тональности (3 класса)
    - Задача 2: Регрессия индекса репутации (-1 до 1)
    """
    
    def __init__(self, vocab_size: int, num_categories: int,
                 config: Dict = None, preprocessing_config: Dict = None):
        """
        Инициализация модели
        
        Args:
            vocab_size: Размер словаря
            num_categories: Количество категорий
            config: Параметры модели
            preprocessing_config: Параметры предобработки
        """
        self.vocab_size = vocab_size
        self.num_categories = num_categories
        self.config = config or MODEL_CONFIG
        self.preprocessing_config = preprocessing_config or PREPROCESSING_CONFIG
        
        self.model: Optional[Model] = None
        self._build_model()
        
    def _build_model(self):
        """Построение архитектуры нейросети"""
        
        # Параметры
        max_seq_len = self.preprocessing_config["max_sequence_length"]
        embedding_dim = self.preprocessing_config["embedding_dim"]
        lstm_units = self.config["lstm_units"]
        dense_units = self.config["dense_units"]
        dropout_rate = self.config["dropout_rate"]
        
        # ===== ВХОДЫ =====
        # Вход для текста
        text_input = Input(
            shape=(max_seq_len,), 
            dtype="int32", 
            name="text_input"
        )
        
        # Вход для оценки
        rating_input = Input(
            shape=(1,), 
            dtype="float32", 
            name="rating_input"
        )
        
        # Вход для категории (one-hot)
        category_input = Input(
            shape=(self.num_categories,), 
            dtype="float32", 
            name="category_input"
        )
        
        # ===== ТЕКСТОВАЯ ВЕТКА =====
        # Эмбеддинг слой
        x = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,  # Маскирование PAD токенов
            name="embedding"
        )(text_input)
        
        # Пространственный дропаут
        x = layers.SpatialDropout1D(0.2, name="spatial_dropout")(x)
        
        # Bidirectional LSTM
        x = layers.Bidirectional(
            layers.LSTM(
                lstm_units, 
                return_sequences=False,
                kernel_regularizer=keras.regularizers.l2(0.01)
            ),
            name="bilstm"
        )(x)
        
        # Dropout после LSTM
        text_features = layers.Dropout(dropout_rate, name="lstm_dropout")(x)
        
        # ===== ОБЪЕДИНЕНИЕ ПРИЗНАКОВ =====
        # Конкатенация всех входов
        combined = layers.Concatenate(name="concatenate")([
            text_features,
            rating_input,
            category_input
        ])
        
        # Общий Dense слой
        shared = layers.Dense(
            dense_units * 2, 
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(0.01),
            name="shared_dense"
        )(combined)
        shared = layers.BatchNormalization(name="batch_norm")(shared)
        shared = layers.Dropout(dropout_rate, name="shared_dropout")(shared)
        
        # ===== ВЫХОДЫ =====
        # Ветка классификации тональности
        class_branch = layers.Dense(
            dense_units, 
            activation="relu", 
            name="class_dense"
        )(shared)
        class_output = layers.Dense(
            3, 
            activation="softmax", 
            name="sentiment_output"
        )(class_branch)
        
        # Ветка регрессии индекса репутации
        index_branch = layers.Dense(
            dense_units, 
            activation="relu", 
            name="index_dense"
        )(shared)
        index_output = layers.Dense(
            1, 
            activation="tanh",  # Ограничение выхода [-1, 1]
            name="reputation_output"
        )(index_branch)
        
        # ===== СБОРКА МОДЕЛИ =====
        self.model = Model(
            inputs=[text_input, rating_input, category_input],
            outputs=[class_output, index_output],
            name="reputation_model"
        )
        
        # Компиляция модели
        self.model.compile(
            optimizer=Adam(learning_rate=self.config["learning_rate"]),
            loss={
                "sentiment_output": "sparse_categorical_crossentropy",
                "reputation_output": "mse"
            },
            loss_weights={
                "sentiment_output": 1.0,  # Вес задачи классификации
                "reputation_output": 0.5  # Вес задачи регрессии
            },
            metrics={
                "sentiment_output": ["accuracy"],
                "reputation_output": ["mae"]
            }
        )
        
        print("✅ Модель построена:")
        self.model.summary()
    
    def get_callbacks(self, model_path: Path = None) -> list:
        """
        Создание callbacks для обучения
        
        Args:
            model_path: Путь для сохранения лучшей модели
            
        Returns:
            Список callbacks
        """
        callbacks = []
        
        # Early Stopping — остановка при ухудшении валидации
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # Уменьшение learning rate при плато
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Сохранение лучшей модели
        if model_path:
            checkpoint = ModelCheckpoint(
                filepath=str(model_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        return callbacks
    
    def fit(self, X_text: np.ndarray, X_rating: np.ndarray, 
            X_category: np.ndarray, y_class: np.ndarray, 
            y_index: np.ndarray, **kwargs) -> keras.callbacks.History:
        """
        Обучение модели
        
        Args:
            X_text: Последовательности токенов
            X_rating: Оценки
            X_category: One-hot категории
            y_class: Классы тональности
            y_index: Индексы репутации
            **kwargs: Дополнительные параметры для fit()
            
        Returns:
            История обучения
        """
        model_path = MODELS_DIR / "best_model.keras"
        callbacks = self.get_callbacks(model_path)
        
        history = self.model.fit(
            x={
                "text_input": X_text,
                "rating_input": X_rating,
                "category_input": X_category
            },
            y={
                "sentiment_output": y_class,
                "reputation_output": y_index
            },
            callbacks=callbacks,
            **kwargs
        )
        
        return history
    
    def predict(self, X_text: np.ndarray, X_rating: np.ndarray,
                X_category: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказание модели
        
        Args:
            X_text: Последовательности токенов
            X_rating: Оценки
            X_category: One-hot категории
            
        Returns:
            (вероятности классов, индексы репутации)
        """
        predictions = self.model.predict({
            "text_input": X_text,
            "rating_input": X_rating,
            "category_input": X_category
        })
        
        return predictions[0], predictions[1]
    
    def evaluate(self, X_text: np.ndarray, X_rating: np.ndarray,
                 X_category: np.ndarray, y_class: np.ndarray,
                 y_index: np.ndarray) -> Dict[str, float]:
        """
        Оценка модели на тестовых данных
        
        Returns:
            Словарь с метриками
        """
        results = self.model.evaluate(
            x={
                "text_input": X_text,
                "rating_input": X_rating,
                "category_input": X_category
            },
            y={
                "sentiment_output": y_class,
                "reputation_output": y_index
            },
            return_dict=True
        )
        
        return results
    
    def save(self, path: Path = None):
        """Сохранение модели"""
        path = path or MODELS_DIR / "reputation_model.keras"
        self.model.save(str(path))
        print(f"✅ Модель сохранена: {path}")
    
    def load(self, path: Path = None):
        """Загрузка модели"""
        path = path or MODELS_DIR / "reputation_model.keras"
        self.model = keras.models.load_model(str(path))
        print(f"✅ Модель загружена: {path}")


def main():
    """Демонстрация архитектуры модели"""
    # Создаём модель с тестовыми параметрами
    model = ReputationModel(
        vocab_size=5000,
        num_categories=10
    )
    
    # Тестовые данные
    n_samples = 32
    X_text = np.random.randint(0, 5000, (n_samples, 100))
    X_rating = np.random.rand(n_samples, 1).astype(np.float32)
    X_category = np.eye(10)[np.random.randint(0, 10, n_samples)].astype(np.float32)
    
    # Тестовое предсказание
    class_probs, rep_index = model.predict(X_text, X_rating, X_category)
    
    print("\n📊 Тестовое предсказание:")
    print(f"   Размер вероятностей классов: {class_probs.shape}")
    print(f"   Размер индексов репутации: {rep_index.shape}")
    print(f"   Пример вероятностей: {class_probs[0]}")
    print(f"   Пример индекса: {rep_index[0][0]:.4f}")


if __name__ == "__main__":
    main()
