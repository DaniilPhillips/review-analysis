"""
Конфигурация информационно-аналитической системы анализа отзывов
"""

from pathlib import Path

# Пути
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Создание директорий
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Параметры датасета
DATASET_CONFIG = {
    "num_samples": 10000,           # Количество отзывов
    "num_subjects": 50,             # Количество субъектов (компаний)
    "random_seed": 42,              # Для воспроизводимости
}

# Категории товаров/услуг
CATEGORIES = [
    "Электроника",
    "Одежда",
    "Продукты питания",
    "Бытовая техника",
    "Косметика",
    "Спорт и отдых",
    "Книги",
    "Доставка",
    "Ресторан",
    "Отель"
]

# Классы репутации
REPUTATION_CLASSES = {
    0: "негативная",
    1: "нейтральная", 
    2: "позитивная"
}

# Параметры предобработки текста
PREPROCESSING_CONFIG = {
    "max_words": 10000,             # Размер словаря
    "max_sequence_length": 100,     # Максимальная длина последовательности
    "embedding_dim": 128,           # Размерность эмбеддингов
}

# Параметры модели
MODEL_CONFIG = {
    "lstm_units": 128,               # Количество LSTM нейронов
    "dense_units": 64,              # Нейроны в Dense слоях
    "dropout_rate": 0.3,            # Dropout для регуляризации
    "learning_rate": 0.001,         # Скорость обучения
}

# Параметры обучения
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 5,
    "validation_split": 0.2,
    "test_split": 0.1,
    "early_stopping_patience": 3,
}
