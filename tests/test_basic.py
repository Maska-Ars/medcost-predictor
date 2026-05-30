import pytest
from unittest.mock import patch

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))\

from app import predict_insurance_cost


class TestPredictionFunction:
    """Тесты для функции предсказания стоимости страховки"""

    def test_prediction_works_with_valid_input(self):
        """Проверяет, что функция предсказания работает на корректном примере и не падает с ошибкой"""
        valid_inputs = {
            'age': 30,
            'sex': 'male',
            'bmi': 25.5,
            'children': 1,
            'smoker': 'no',
            'region': 'southwest'
        }

        try:
            result = predict_insurance_cost(
                age=valid_inputs['age'],
                sex=valid_inputs['sex'],
                bmi=valid_inputs['bmi'],
                children=valid_inputs['children'],
                smoker=valid_inputs['smoker'],
                region=valid_inputs['region']
            )

            assert result is not None, "Функция вернула None"

        except Exception as e:
            pytest.fail(f"Функция предсказания выбросила исключение: {e}")

    def test_prediction_returns_correct_format(self):
        """
        Тест 2: Проверяет, что функция возвращает ответ в правильном формате
        """
        test_cases = [
            {'age': 25, 'sex': 'female', 'bmi': 22.0, 'children': 0, 'smoker': 'no', 'region': 'northeast'},
            {'age': 50, 'sex': 'male', 'bmi': 30.0, 'children': 2, 'smoker': 'yes', 'region': 'southeast'},
            {'age': 35, 'sex': 'female', 'bmi': 28.5, 'children': 1, 'smoker': 'no', 'region': 'northwest'},
        ]

        for case in test_cases:
            result = predict_insurance_cost(**case)

            assert isinstance(result, (int, float)), \
                f"Результат должен быть числом, но получен {type(result)}"

            assert result >= 0, \
                f"Результат должен быть неотрицательным, но получен {result}"


class TestWebInterface:
    """Тест 3: Проверяет, что веб-приложение может запуститься без ошибок"""
    @patch('gradio.Blocks')
    @patch('gradio.Interface')
    def test_app_launch_without_errors(self, mock_interface, mock_blocks):
        """
        Проверяет, что основные компоненты Gradio могут быть созданы без ошибок
        """
        try:
            from app import app

            assert app is not None, "приложение не было создано"

            assert hasattr(app, 'launch'), "app не имеет метода launch"

            with patch.object(app, 'launch') as mock_launch:
                mock_launch.return_value = None
                assert callable(app.launch), "app.launch не является вызываемым"

        except ImportError as e:
            pytest.skip(f"Не удалось импортировать приложение: {e}. Убедитесь, что файл с кодом доступен.")
        except Exception as e:
            pytest.fail(f"Ошибка при проверке веб-приложения: {e}")
