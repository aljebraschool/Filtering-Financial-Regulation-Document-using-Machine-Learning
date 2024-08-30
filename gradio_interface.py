import gradio as gr
import pandas as pd
import requests
import io

# Replace with your actual API endpoint
API_ENDPOINT = "http://127.0.0.1:8000/predict"


def process_single_input(document_id, title, regulator_id, source_language, document_type_id, content):
    # Create a DataFrame with a single row
    df = pd.DataFrame({
        'DocumentID': [document_id],
        'Title': [title],
        'RegulatorId': [regulator_id],
        'SourceLanguage': [source_language],
        'DocumentTypeId': [document_type_id],
        'Content': [content]
    })

    # Convert DataFrame to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    # Send request to API
    files = {'file': ('input.csv', csv_content, 'text/csv')}
    response = requests.post(API_ENDPOINT, files=files)

    if response.status_code == 200:
        result_df = pd.read_csv(io.StringIO(response.text))
        return result_df.to_dict('records')[0]
    else:
        return {"error": f"API request failed with status code {response.status_code}"}


def process_csv(file):
    response = requests.post(API_ENDPOINT, files={'file': file})

    if response.status_code == 200:
        result_df = pd.read_csv(io.StringIO(response.text))
        return result_df.to_dict('records')
    else:
        return [{"error": f"API request failed with status code {response.status_code}"}]


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Document Relevance Predictor")

    with gr.Tab("Single Document"):
        with gr.Row():
            document_id = gr.Textbox(label="Document ID")
            title = gr.Textbox(label="Title")
        with gr.Row():
            regulator_id = gr.Textbox(label="Regulator ID")
            source_language = gr.Textbox(label="Source Language")
            document_type_id = gr.Textbox(label="Document Type ID")
        content = gr.Textbox(label="Content", lines=5)
        single_submit = gr.Button("Predict")
        single_output = gr.JSON(label="Prediction Result")

    with gr.Tab("Batch Processing"):
        file_input = gr.File(label="Upload CSV")
        batch_submit = gr.Button("Process Batch")
        batch_output = gr.JSON(label="Batch Results")

    single_submit.click(
        process_single_input,
        inputs=[document_id, title, regulator_id, source_language, document_type_id, content],
        outputs=single_output
    )

    batch_submit.click(
        process_csv,
        inputs=file_input,
        outputs=batch_output
    )

if __name__ == "__main__":
    demo.launch()