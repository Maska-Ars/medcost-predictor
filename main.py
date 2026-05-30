import gradio as gr
import pandas as pd
import joblib

cat = joblib.load('cat.joblib')


def predict_insurance_cost(age, sex, bmi, children, smoker, region):
    """
    Предсказание стоимости страховки на основе входных данных

    Args:
        age: возраст (целое число)
        sex: 'male' или 'female'
        bmi: индекс массы тела (число с плавающей точкой)
        children: количество детей (целое число)
        smoker: 'yes' или 'no'
        region: 'northeast', 'northwest', 'southeast' или 'southwest'

    Returns:
        Предсказанная стоимость страховки
    """

    input_data = pd.DataFrame({
        'sex': [sex],
        'children': [children],
        'smoker': [smoker],
        'region': [region],
        'age': [age],
        'bmi': [bmi],
    })

    prediction = cat.predict(input_data)

    if isinstance(prediction[0], (int, float)):
        return f"{prediction[0]:,.2f}"
    else:
        return f"{prediction[0]}"


with gr.Blocks(title="Предсказатель стоимости страховки") as demo:
    gr.Markdown("""
    # 🏥 Предсказатель стоимости страховки

    Введите данные пациента ниже, чтобы получить оценку стоимости страховки на основе модели машинного обучения.
    """)

    with gr.Row():
        with gr.Column():
            age = gr.Number(
                label="Возраст (лет)",
                value=30,
                step=1,
                minimum=0,
                maximum=200
            )

            sex = gr.Dropdown(
                choices=['male', 'female'],
                label="Пол",
                value='male'
            )

            bmi = gr.Number(
                label="ИМТ (Индекс массы тела)",
                value=25.0,
                step=0.1,
                minimum=10.0,
                maximum=100.0
            )

            children = gr.Number(
                label="Количество детей",
                value=0,
                step=1,
                minimum=0,
                maximum=100
            )

            smoker = gr.Dropdown(
                choices=['yes', 'no'],
                label="Курильщик",
                value='no'
            )

            region = gr.Dropdown(
                choices=['northeast', 'northwest', 'southeast', 'southwest'],
                label="Регион",
                value='southwest'
            )

            predict_btn = gr.Button("Предсказать стоимость", variant="primary")
            clear_btn = gr.Button("Очистить поля", variant="secondary")

        with gr.Column():
            output = gr.Textbox(
                label="стоимость страховки",
                lines=3,
                interactive=False,
                elem_id="output-box"
            )

            gr.Markdown("""
            ### 📊 О модели
            - **Тип модели**: CatBoostRegressor
            - **Признаки**: Возраст, Пол, ИМТ, Количество детей, Статус курения, Регион
            - **Целевой показатель**: Стоимость страховки

            ### ℹ️ Категории ИМТ
            - Недостаточный вес: < 18.5
            - Нормальный вес: 18.5 - 24.9
            - Избыточный вес: 25 - 29.9
            - Ожирение: ≥ 30
            """)


    def clear_fields():
        return [30, 'male', 25.0, 0, 'no', 'southwest', ""]


    predict_btn.click(
        fn=predict_insurance_cost,
        inputs=[age, sex, bmi, children, smoker, region],
        outputs=output
    )

    clear_btn.click(
        fn=clear_fields,
        outputs=[age, sex, bmi, children, smoker, region, output]
    )

    gr.Markdown("""
    ### 📝 Примеры
    Нажмите на любой пример ниже, чтобы автоматически заполнить поля:
    """)

    gr.Examples(
        examples=[
            [46, 'female', 19.95, 2, 'no', 'northwest'],
            [47, 'female', 24.32, 0, 'no', 'northeast'],
            [52, 'female', 24.86, 0, 'no', 'southeast'],
            [39, 'female', 34.32, 5, 'no', 'southeast'],
            [54, 'female', 21.47, 3, 'no', 'northwest'],
            [30, 'male', 28.5, 1, 'yes', 'southwest'],
            [25, 'male', 22.0, 0, 'no', 'northeast'],
            [60, 'female', 32.0, 2, 'yes', 'northwest']
        ],
        inputs=[age, sex, bmi, children, smoker, region],
        outputs=output,
        fn=predict_insurance_cost,
        cache_examples=False
    )

    gr.HTML("""
    <style>
        #output-box {
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            background-color: #f0f8ff;
            border: 2px solid #4CAF50;
            border-radius: 5px;
            padding: 10px;
        }
        /* Добавьте эти стили для вертикального выравнивания */
        #output-box textarea {
            text-align: center !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            vertical-align: middle !important;
        }
        .gradio-container {
            max-width: 1200px;
            margin: auto;
        }
        .gr-button-primary {
            background-color: #4CAF50 !important;
        }
    </style>
    """)

    gr.Markdown("""
    ---
    ### 💡 Как рассчитывается ИМТ
    ИМТ = вес (кг) / рост² (м²)

    **Пример:** Человек с весом 70 кг и ростом 1.75 м имеет ИМТ = 22.86
    """)


if __name__ == "__main__":
    demo.launch(
        share=False,
        server_port=7860,
        debug=False
    )
