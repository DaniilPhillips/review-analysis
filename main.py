"""
Главный модуль информационно-аналитической системы
анализа влияния отзывов на репутацию

Запуск:
    python main.py --generate   # Генерация датасета
    python main.py --train      # Обучение модели
    python main.py --analyze    # Аналитика
    python main.py --full       # Полный пайплайн
    python main.py --demo       # Демонстрация предсказаний
"""

import argparse
import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import DATA_DIR, MODELS_DIR, LOGS_DIR


def generate_dataset():
    """Генерация синтетического датасета"""
    print("\n" + "="*60)
    print("📝 ГЕНЕРАЦИЯ СИНТЕТИЧЕСКОГО ДАТАСЕТА")
    print("="*60)
    
    from src.data_generator import ReviewDataGenerator
    
    generator = ReviewDataGenerator()
    df = generator.generate_dataset()
    generator.save_dataset(df)
    
    print("\n📊 Примеры сгенерированных отзывов:")
    for i, row in df.head(3).iterrows():
        print(f"\n   [{row['subject_name']}] Оценка: {row['rating']}/5")
        print(f"   Категория: {row['category']}")
        print(f"   Текст: {row['review_text'][:80]}...")
        labels = ["негативная", "нейтральная", "позитивная"]
        print(f"   Тональность: {labels[row['sentiment_class']]}, Индекс: {row['reputation_index']:.2f}")


def train_model():
    """Обучение модели"""
    print("\n" + "="*60)
    print("🚀 ОБУЧЕНИЕ МОДЕЛИ")
    print("="*60)
    
    from src.training import ModelTrainer
    
    trainer = ModelTrainer()
    df = trainer.load_data()
    train_df, val_df, test_df = trainer.split_data(df)
    data = trainer.prepare_data(train_df, val_df, test_df)
    trainer.train(data)
    metrics = trainer.evaluate(data)
    trainer.plot_training_history()
    trainer.plot_confusion_matrix(data)
    trainer.save_artifacts()
    
    return metrics


def run_analytics():
    """Запуск аналитики"""
    print("\n" + "="*60)
    print("📊 АНАЛИТИКА РЕПУТАЦИИ")
    print("="*60)
    
    import pandas as pd
    from src.analytics import ReputationAnalytics
    
    analytics = ReputationAnalytics()
    df = pd.read_csv(DATA_DIR / "synthetic_dataset.csv")
    
    report = analytics.generate_report(df)
    print(report)
    
    analytics.plot_reputation_distribution(df)
    analytics.plot_subject_comparison(df)
    
    with open(LOGS_DIR / "analytics_report.txt", "w", encoding="utf-8") as f:
        f.write(report)


def demo_predictions():
    """Демонстрация предсказаний на новых отзывах"""
    print("\n" + "="*60)
    print("🔮 ДЕМОНСТРАЦИЯ ПРЕДСКАЗАНИЙ")
    print("="*60)
    
    from src.analytics import ReputationAnalytics
    import numpy as np
    
    analytics = ReputationAnalytics()
    analytics.load_model()
    
    # Тестовые отзывы
    test_reviews = [
        ("Отличный товар! Качество превосходное, доставка быстрая.", 5, "Электроника"),
        ("Не рекомендую. Сломался через неделю, деньги не вернули.", 1, "Бытовая техника"),
        ("Нормально, ничего особенного. Соответствует цене.", 3, "Одежда"),
    ]
    
    texts = [r[0] for r in test_reviews]
    ratings = [r[1] for r in test_reviews]
    categories = [r[2] for r in test_reviews]
    
    class_probs, rep_indices = analytics.predict_reputation(texts, ratings, categories)
    
    labels = ["негативная", "нейтральная", "позитивная"]
    print("\n📋 Результаты анализа:")
    for i, (text, rating, cat) in enumerate(test_reviews):
        pred_class = np.argmax(class_probs[i])
        confidence = class_probs[i][pred_class] * 100
        
        print(f"\n{'─'*50}")
        print(f"Текст: {text[:60]}...")
        print(f"Оценка: {rating}/5, Категория: {cat}")
        print(f"Предсказанная тональность: {labels[pred_class]} ({confidence:.1f}%)")
        print(f"Индекс репутационного влияния: {rep_indices[i]:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Информационно-аналитическая система анализа репутации"
    )
    parser.add_argument("--generate", action="store_true", help="Генерация датасета")
    parser.add_argument("--train", action="store_true", help="Обучение модели")
    parser.add_argument("--analyze", action="store_true", help="Аналитика")
    parser.add_argument("--demo", action="store_true", help="Демо предсказаний")
    parser.add_argument("--full", action="store_true", help="Полный пайплайн")
    
    args = parser.parse_args()
    
    if args.full or (not any([args.generate, args.train, args.analyze, args.demo])):
        generate_dataset()
        train_model()
        run_analytics()
        demo_predictions()
    else:
        if args.generate:
            generate_dataset()
        if args.train:
            train_model()
        if args.analyze:
            run_analytics()
        if args.demo:
            demo_predictions()
    
    print("\n✅ Выполнение завершено!")


if __name__ == "__main__":
    main()
