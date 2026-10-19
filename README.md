# Методи Data Science — лабораторна №2

Варіант 7. Попередня обробка, візуалізація та дослідження даних.

[Умови завдання](https://github.com/UNI-DS/DS-Lab-2) ·
[Приклади з лекції](https://github.com/mehalyna/UNI-MethDS-Lessons/blob/main/L2_Data_Preprocessing.ipynb)

## Запуск

Python 3.11+. Усі команди виконуються з кореня репозиторію.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python src/task1_data_cleaning.py
python src/task2_outlier_detection.py
python src/task3_scaling.py
python src/task4_feature_engineering.py
python src/task5_pipeline.py
python src/task6_case_study.py
```

У Linux/macOS середовище активується командою `source venv/bin/activate`.

Перевірка та перегляд результатів:

```powershell
python -m pytest tests/ -v
streamlit run app.py
```

## Результати

Результати зберігаються в корені репозиторію. Повторний запуск оновлює відповідні
файли. У `plots_examples/` розміщені приклади графіків із матеріалів лабораторної.

1. **Пропуски та якість даних.** Три PNG, індикатори пропусків у
   `task1_missing_indicators.csv`, порівняння статистик у `task1_imputation_statistics.csv`.
2. **Викиди.** Шість PNG, кількість викидів у `task2_outlier_summary.csv`,
   порівняння R² у `task2_model_comparison.csv`.
3. **Масштабування.** Шість PNG, статистики в `task3_scaling_statistics.csv`,
   оцінки моделей у `task3_model_comparison.csv`.
4. **Нові ознаки.** Шість PNG, каталог `task4_feature_catalog.csv`,
   набір ознак `task4_engineered_data.csv`, порівняння в `task4_feature_impact.csv`
   і `task4_encoding_comparison.csv`. Граф взаємодій додатково будується, якщо встановлено NetworkX.
5. **Pipeline.** П’ять PNG, порівняння конфігурацій у `task5_configuration_comparison.csv`,
   метрики на відкладених даних у `task5_metrics.csv`, навчений об’єкт у `task5_preprocessing_pipeline.pkl`.
6. **Комплексна обробка.** Чотири PNG, звіт `task6_preprocessing_report.json`
   і підготовлений набір `task6_processed_data.csv`.

Streamlit показує збережені результати завдань 1–4. У завданнях 5–6 можна
змінювати параметри й запускати обробку з інтерфейсу.

## Особливості обробки

- Синтетичні дані генеруються з фіксованим `random_state=42`.
- Тип пропусків визначається приблизно за кореляціями. Відсутність кореляції
  не доводить MCAR; MNAR неможливо встановити лише зі спостережених значень.
- У завданнях 2–3 та 5 параметри обробки для оцінювання моделей навчаються
  на тренувальних частинах. Цільова змінна під час оцінювання не масштабується.
- У CV-порівнянні завдання 4 використовуються математичні та взаємодійні ознаки.
  Target encoding і відбір за ціллю до цього порівняння не входять.
- У завданні 3 `linear_regression` оцінюється за R², а KNN — за accuracy.
  Ці метрики не усереднюються між собою. Від’ємний R² є допустимим результатом.
- Завдання 6 видаляє дублікати, обробляє пропуски й викиди, кодує категорії,
  масштабує та відбирає ознаки, вирівнює кількість записів у класах.
- Для завантаження PKL потрібно додати `src` до шляху імпорту. За іншої версії
  scikit-learn слід повторно запустити завдання 5, щоб створити сумісний файл.

```python
import sys
import joblib

sys.path.insert(0, "src")
pipeline = joblib.load("task5_preprocessing_pipeline.pkl")
```
