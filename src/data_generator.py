"""
Генератор синтетического датасета отзывов (v2 - реалистичный)

Улучшения для избежания overfitting:
1. Смешанные отзывы (позитив+негатив в одном тексте)
2. Label noise — случайные ошибки в разметке
3. Больше вариативности в формулировках
4. Неоднозначные/сложные случаи
5. Разная длина отзывов
"""

import random
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASET_CONFIG, CATEGORIES, DATA_DIR


class ReviewDataGenerator:
    """Генератор реалистичных синтетических отзывов"""
    
    # Позитивные фразы (разбиты на части для комбинирования)
    POSITIVE_STARTERS = [
        "Отличный товар", "Прекрасная покупка", "Очень доволен", "Замечательно",
        "Рекомендую", "Супер", "Великолепно", "Лучшее что покупал", "Топ",
        "Качественный продукт", "Хороший выбор", "Понравилось", "Всё отлично",
    ]
    
    POSITIVE_DETAILS = [
        "качество на высоте", "работает идеально", "полностью доволен",
        "превзошло ожидания", "стоит своих денег", "буду заказывать ещё",
        "всё как в описании", "быстрая доставка", "отличный сервис",
        "приятно удивлён", "радует каждый день", "нет нареканий",
    ]
    
    # Негативные фразы
    NEGATIVE_STARTERS = [
        "Ужасное качество", "Не рекомендую", "Разочарован", "Плохой товар",
        "Деньги на ветер", "Брак", "Отвратительно", "Худшая покупка",
        "Не советую", "Полное разочарование", "Жалею что купил", "Кошмар",
    ]
    
    NEGATIVE_DETAILS = [
        "сломалось через неделю", "не работает", "не соответствует описанию",
        "пришло поврежденное", "деньги не вернули", "хамское обслуживание",
        "долгая доставка", "потерянная посылка", "низкое качество",
        "быстро износилось", "не стоит таких денег", "обман покупателей",
    ]
    
    # Нейтральные фразы
    NEUTRAL_STARTERS = [
        "Нормально", "Средне", "Обычный товар", "Ничего особенного",
        "Так себе", "Приемлемо", "На троечку", "Могло быть лучше",
        "Неплохо но", "Есть недостатки", "Соответствует цене", "Пойдёт",
    ]
    
    NEUTRAL_DETAILS = [
        "за такие деньги норм", "есть и плюсы и минусы", "ожидал большего",
        "не впечатлило", "обычное качество", "ничего выдающегося",
        "можно взять если нужно", "не хуже других", "средний уровень",
        "бывает и лучше", "не идеально но сойдёт", "без восторга",
    ]
    
    # Слова-усилители (добавляют шум)
    INTENSIFIERS = ["очень", "крайне", "весьма", "довольно", "слегка", "немного", ""]
    
    # Связки для смешанных отзывов
    CONJUNCTIONS = [
        ", но ", ", однако ", ", хотя ", ". Но ", ". Однако ", 
        ", при этом ", ", правда ", ". С другой стороны, ",
    ]
    
    # Аспекты для категорий
    ASPECTS = {
        "Электроника": ["экран", "батарея", "камера", "звук", "дизайн", "скорость"],
        "Одежда": ["материал", "размер", "цвет", "качество пошива", "посадка"],
        "Продукты питания": ["вкус", "свежесть", "упаковка", "срок годности"],
        "Бытовая техника": ["мощность", "шум", "надёжность", "функционал"],
        "Косметика": ["запах", "эффект", "состав", "текстура", "упаковка"],
        "Спорт и отдых": ["удобство", "прочность", "материал", "дизайн"],
        "Книги": ["содержание", "перевод", "печать", "переплёт"],
        "Доставка": ["скорость", "упаковка", "курьер", "цена доставки"],
        "Ресторан": ["еда", "обслуживание", "атмосфера", "цены", "чистота"],
        "Отель": ["номер", "персонал", "расположение", "завтрак", "чистота"],
    }
    
    COMPANY_PREFIXES = ["ТехноМир", "МегаМаркет", "СуперСервис", "Глобал", "Премиум", 
                        "Экспресс", "Максимум", "Оптима", "Люкс", "Стандарт"]
    COMPANY_SUFFIXES = ["Плюс", "Про", "Маркет", "Лайн", "Групп", 
                        "Трейд", "Хаб", "Центр", "Сток", "Шоп"]
    
    def __init__(self, config: Dict = None):
        self.config = config or DATASET_CONFIG
        random.seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])
        self.subjects = self._generate_subjects()
        
    def _generate_subjects(self) -> Dict[str, Dict]:
        subjects = {}
        for i in range(self.config["num_subjects"]):
            subject_id = f"SUBJ_{i:04d}"
            base_quality = np.random.uniform(0.2, 0.85)
            prefix = random.choice(self.COMPANY_PREFIXES)
            suffix = random.choice(self.COMPANY_SUFFIXES)
            subjects[subject_id] = {
                "name": f"{prefix}{suffix}",
                "base_quality": base_quality,
                "primary_category": random.choice(CATEGORIES)
            }
        return subjects
    
    def _generate_pure_positive(self, aspect: str) -> str:
        """Чисто позитивный отзыв"""
        intensifier = random.choice(self.INTENSIFIERS)
        starter = random.choice(self.POSITIVE_STARTERS)
        detail = random.choice(self.POSITIVE_DETAILS)
        
        templates = [
            f"{starter}! {intensifier} {detail}.",
            f"{starter}, {aspect} {detail}.",
            f"{intensifier.capitalize()} {detail}. {starter}!",
            f"{aspect.capitalize()} {detail}. {starter}.",
        ]
        return random.choice(templates)
    
    def _generate_pure_negative(self, aspect: str) -> str:
        """Чисто негативный отзыв"""
        intensifier = random.choice(self.INTENSIFIERS)
        starter = random.choice(self.NEGATIVE_STARTERS)
        detail = random.choice(self.NEGATIVE_DETAILS)
        
        templates = [
            f"{starter}! {intensifier} {detail}.",
            f"{starter}, {aspect} {detail}.",
            f"{intensifier.capitalize()} {detail}. {starter}!",
            f"{aspect.capitalize()} {detail}. {starter}.",
        ]
        return random.choice(templates)
    
    def _generate_pure_neutral(self, aspect: str) -> str:
        """Чисто нейтральный отзыв"""
        starter = random.choice(self.NEUTRAL_STARTERS)
        detail = random.choice(self.NEUTRAL_DETAILS)
        
        templates = [
            f"{starter}. {detail.capitalize()}.",
            f"{starter}, {aspect} {detail}.",
            f"{detail.capitalize()}. {starter}.",
        ]
        return random.choice(templates)
    
    def _generate_mixed_review(self, aspect: str, primary_sentiment: int) -> Tuple[str, int]:
        """
        Смешанный отзыв (например: позитив + негатив)
        Возвращает текст и РЕАЛЬНЫЙ класс (может отличаться от шаблона)
        """
        conjunction = random.choice(self.CONJUNCTIONS)
        
        if primary_sentiment == 2:  # Преимущественно позитивный
            pos_part = f"{random.choice(self.POSITIVE_STARTERS)}, {random.choice(self.POSITIVE_DETAILS)}"
            neg_part = f"{random.choice(self.NEUTRAL_DETAILS)}"
            text = f"{pos_part}{conjunction}{neg_part}."
            # Может быть классифицирован как нейтральный или позитивный
            actual_class = random.choices([1, 2], weights=[0.3, 0.7])[0]
            
        elif primary_sentiment == 0:  # Преимущественно негативный
            neg_part = f"{random.choice(self.NEGATIVE_STARTERS)}, {random.choice(self.NEGATIVE_DETAILS)}"
            pos_part = f"{random.choice(self.NEUTRAL_DETAILS)}"
            text = f"{neg_part}{conjunction}{pos_part}."
            actual_class = random.choices([0, 1], weights=[0.7, 0.3])[0]
            
        else:  # Нейтральный со смешанными элементами
            if random.random() < 0.5:
                part1 = random.choice(self.POSITIVE_STARTERS)
                part2 = random.choice(self.NEGATIVE_DETAILS)
            else:
                part1 = random.choice(self.NEGATIVE_STARTERS)
                part2 = random.choice(self.POSITIVE_DETAILS)
            text = f"{part1}{conjunction}{part2}."
            actual_class = 1  # Смешанные = нейтральные
            
        return text, actual_class
    
    def _add_noise_to_label(self, true_class: int, noise_rate: float = 0.08) -> int:
        """Добавление шума в метки (имитация ошибок разметки)"""
        if random.random() < noise_rate:
            # Сдвигаем класс на соседний
            if true_class == 0:
                return 1
            elif true_class == 2:
                return 1
            else:
                return random.choice([0, 2])
        return true_class
    
    def _generate_review(self, base_quality: float, category: str) -> Tuple[str, int, int, float]:
        """
        Генерация отзыва с учётом реализма
        
        Returns:
            (текст, оценка, класс_тональности, индекс_репутации)
        """
        aspects = self.ASPECTS.get(category, ["качество", "сервис"])
        aspect = random.choice(aspects)
        
        # Определяем базовую тональность на основе качества + шум
        noise = np.random.normal(0, 0.2)
        adjusted_quality = np.clip(base_quality + noise, 0, 1)
        
        # Вероятности классов зависят от качества
        if adjusted_quality < 0.35:
            primary_class = 0  # негатив
            class_probs = [0.7, 0.25, 0.05]
        elif adjusted_quality < 0.6:
            primary_class = 1  # нейтраль
            class_probs = [0.15, 0.7, 0.15]
        else:
            primary_class = 2  # позитив
            class_probs = [0.05, 0.25, 0.7]
        
        # Выбираем тип отзыва
        review_type = random.choices(
            ["pure", "mixed", "ambiguous"],
            weights=[0.5, 0.35, 0.15]  # 50% чистые, 35% смешанные, 15% неоднозначные
        )[0]
        
        if review_type == "pure":
            # Чистый отзыв одной тональности
            sentiment_class = random.choices([0, 1, 2], weights=class_probs)[0]
            if sentiment_class == 0:
                text = self._generate_pure_negative(aspect)
            elif sentiment_class == 1:
                text = self._generate_pure_neutral(aspect)
            else:
                text = self._generate_pure_positive(aspect)
                
        elif review_type == "mixed":
            # Смешанный отзыв
            text, sentiment_class = self._generate_mixed_review(aspect, primary_class)
            
        else:  # ambiguous
            # Неоднозначный - текст одного класса, но метка может быть соседней
            sentiment_class = random.choices([0, 1, 2], weights=class_probs)[0]
            # Генерируем текст соседнего класса
            neighbor_class = max(0, min(2, sentiment_class + random.choice([-1, 0, 1])))
            if neighbor_class == 0:
                text = self._generate_pure_negative(aspect)
            elif neighbor_class == 1:
                text = self._generate_pure_neutral(aspect)
            else:
                text = self._generate_pure_positive(aspect)
        
        # Добавляем шум в метку (имитация ошибок аннотаторов)
        noisy_class = self._add_noise_to_label(sentiment_class, noise_rate=0.05)
        
        # Оценка коррелирует с тональностью, но не идеально
        if noisy_class == 0:
            rating = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        elif noisy_class == 1:
            rating = random.choices([2, 3, 4], weights=[0.25, 0.5, 0.25])[0]
        else:
            rating = random.choices([3, 4, 5], weights=[0.15, 0.35, 0.5])[0]
        
        # Индекс репутации с шумом
        base_index = (noisy_class - 1) * 0.5  # -0.5, 0, 0.5
        reputation_index = np.clip(base_index + np.random.normal(0, 0.25), -1, 1)
        
        return text, rating, noisy_class, round(reputation_index, 4)
    
    def generate_dataset(self) -> pd.DataFrame:
        """Генерация полного датасета"""
        data = []
        
        for _ in range(self.config["num_samples"]):
            subject_id = random.choice(list(self.subjects.keys()))
            subject = self.subjects[subject_id]
            
            if random.random() < 0.7:
                category = subject["primary_category"]
            else:
                category = random.choice(CATEGORIES)
            
            text, rating, sentiment_class, reputation_index = self._generate_review(
                subject["base_quality"], category
            )
            
            data.append({
                "review_text": text,
                "rating": rating,
                "category": category,
                "subject_id": subject_id,
                "subject_name": subject["name"],
                "sentiment_class": sentiment_class,
                "reputation_index": reputation_index
            })
        
        return pd.DataFrame(data)
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "synthetic_dataset.csv"):
        filepath = DATA_DIR / filename
        df.to_csv(filepath, index=False, encoding="utf-8")
        print(f"Датасет сохранён: {filepath}")
        print(f"   Количество записей: {len(df)}")
        print(f"   Распределение классов:")
        for cls, count in df["sentiment_class"].value_counts().sort_index().items():
            labels = {0: "негативная", 1: "нейтральная", 2: "позитивная"}
            print(f"      {labels[cls]}: {count} ({count/len(df)*100:.1f}%)")
        return filepath


def main():
    generator = ReviewDataGenerator()
    df = generator.generate_dataset()
    generator.save_dataset(df)
    
    print("\nПримеры отзывов:")
    for _, row in df.sample(5).iterrows():
        print(f"\n   [{row['subject_name']}] {row['rating']}")
        print(f"   {row['review_text']}")
        labels = ["негативная", "нейтральная", "позитивная"]
        print(f"   → {labels[row['sentiment_class']]} ({row['reputation_index']:.2f})")


if __name__ == "__main__":
    main()
