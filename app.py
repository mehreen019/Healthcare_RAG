import gradio as gr
from healthcare_rag import medical_rag

demo = gr.Interface(
    fn=medical_rag,
    inputs=gr.Textbox(label="Medical Question"),
    outputs=gr.Textbox(label="Evidence-Based Answer"),
    title="Healthcare RAG Assistant",
    examples=[
        "How does metformin work in diabetes management?",
        "What are the diagnostic criteria for Parkinson's disease?",
        "what causes migraines?",
    ]
)

demo.launch()