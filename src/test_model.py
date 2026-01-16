"""
Скрипт тестирования обученной модели на случайных примерах

Использование:
    python test_model.py              # 5 случайных примеров
    python test_model.py -n 10        # 10 примеров
    python test_model.py --errors     # Только ошибочные предсказания
    python test_model.py --class 0    # Только негативные отзывы
"""

import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import DATA_DIR, MODELS_DIR, TRAINING_CONFIG
from src.preprocessing import TextPreprocessor
from src.model import ReputationModel


def load_model_and_preprocessor():
    """Загрузка обученной модели и препроцессора"""
    print("📦 Загрузка модели...")
    
    # Загрузка препроцессора
    with open(MODELS_DIR / "preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    
    # Создание и загрузка модели
    model = ReputationModel(
        vocab_size=preprocessor.get_vocab_size(),
        num_categories=preprocessor.get_num_categories()
    )
    model.load()
    
    return model, preprocessor


def get_test_data(preprocessor):
    """Получение тестовой выборки"""
    df = pd.read_csv(DATA_DIR / "synthetic_dataset.csv")
    
    # Воспроизводим то же разделение, что при обучении
    test_size = TRAINING_CONFIG["test_split"]
    _, test_df = train_test_split(
        df, 
        test_size=test_size,
        random_state=42,
        stratify=df["sentiment_class"]
    )
    
    return test_df


def predict_single(model, preprocessor, text, rating, category):
    """Предсказание для одного примера"""
    df = pd.DataFrame({
        "review_text": [text],
        "rating": [rating],
        "category": [category],
        "subject_id": ["TEST"],
        "sentiment_class": [0],
        "reputation_index": [0]
    })
    
    X_text, X_rating, X_category, _, _ = preprocessor.transform(df)
    class_probs, rep_index = model.predict(X_text, X_rating, X_category)
    
    return class_probs[0], rep_index[0][0]


def display_prediction(row, pred_class, pred_probs, pred_index):
    """Красивый вывод результата"""
    labels = ["негативная", "нейтральная", "позитивная"]
    true_label = labels[row["sentiment_class"]]
    pred_label = labels[pred_class]
    
    is_correct = row["sentiment_class"] == pred_class
    status = "✅" if is_correct else "❌"
    
    print("\n" + "═" * 70)
    print(f"📝 Текст: {row['review_text']}")
    print(f"⭐ Оценка: {row['rating']}/5 | 📦 Категория: {row['category']}")
    print(f"🏢 Субъект: {row['subject_name']}")
    print("─" * 70)
    print(f"   Истинная тональность:     {true_label}")
    print(f"   Предсказанная тональность: {pred_label} {status}")
    print(f"   Уверенность: негатив={pred_probs[0]*100:.1f}%, "
          f"нейтраль={pred_probs[1]*100:.1f}%, позитив={pred_probs[2]*100:.1f}%")
    print("─" * 70)
    print(f"   Истинный индекс репутации:     {row['reputation_index']:+.3f}")
    print(f"   Предсказанный индекс:          {pred_index:+.3f}")
    print(f"   Ошибка (|разница|):            {abs(row['reputation_index'] - pred_index):.3f}")
    
    return is_correct


def main():
    parser = argparse.ArgumentParser(description="Тестирование модели на случайных примерах")
    parser.add_argument("-n", "--num", type=int, default=5, help="Количество примеров")
    parser.add_argument("--errors", action="store_true", help="Показать только ошибки")
    parser.add_argument("--class", dest="target_class", type=int, choices=[0, 1, 2],
                        help="Фильтр по классу (0=негатив, 1=нейтраль, 2=позитив)")
    parser.add_argument("--seed", type=int, default=None, help="Seed для воспроизводимости")
    args = parser.parse_args()
    
    # Загрузка
    model, preprocessor = load_model_and_preprocessor()
    test_df = get_test_data(preprocessor)
    
    print(f"\n📊 Размер тестовой выборки: {len(test_df)}")
    
    # Фильтрация по классу
    if args.target_class is not None:
        test_df = test_df[test_df["sentiment_class"] == args.target_class]
        labels = ["негативных", "нейтральных", "позитивных"]
        print(f"   Фильтр: только {labels[args.target_class]} ({len(test_df)} шт.)")
    
    # Установка seed
    if args.seed:
        np.random.seed(args.seed)
    
    # Выбор примеров
    if args.errors:
        # Режим поиска ошибок — проходим по всем и показываем только ошибки
        print("\n🔍 Поиск ошибочных предсказаний...")
        errors_found = 0
        
        for idx, row in test_df.iterrows():
            pred_probs, pred_index = predict_single(
                model, preprocessor,
                row["review_text"], row["rating"], row["category"]
            )
            pred_class = np.argmax(pred_probs)
            
            if pred_class != row["sentiment_class"]:
                display_prediction(row, pred_class, pred_probs, pred_index)
                errors_found += 1
                
                if errors_found >= args.num:
                    break
        
        if errors_found == 0:
            print("✅ Ошибок не найдено!")
        else:
            print(f"\n📈 Найдено ошибок: {errors_found}")
    else:
        # Обычный режим — случайные примеры
        samples = test_df.sample(n=min(args.num, len(test_df)))
        
        correct = 0
        total_mae = 0
        
        print(f"\n🎲 Случайные примеры из тестовой выборки:")
        
        for idx, row in samples.iterrows():
            pred_probs, pred_index = predict_single(
                model, preprocessor,
                row["review_text"], row["rating"], row["category"]
            )
            pred_class = np.argmax(pred_probs)
            
            is_correct = display_prediction(row, pred_class, pred_probs, pred_index)
            if is_correct:
                correct += 1
            total_mae += abs(row["reputation_index"] - pred_index)
        
        # Итоговая статистика
        print("\n" + "═" * 70)
        print(f"📊 ИТОГО по {len(samples)} примерам:")
        print(f"   Точность классификации: {correct}/{len(samples)} ({correct/len(samples)*100:.1f}%)")
        print(f"   Средняя ошибка индекса (MAE): {total_mae/len(samples):.3f}")
        print("═" * 70)


if __name__ == "__main__":
    main()
