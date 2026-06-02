import gradio as gr

# pyrefly: ignore [missing-import]
from src.inference import (
    generate_base,
    generate_sft,
    generate_dpo,
)

# pyrefly: ignore [missing-import]
from src.benchmark_data import (
    SFT_RESULTS,
    DPO_RESULTS,
)


def compare(prompt, temperature, max_tokens):

    base_output = generate_base(
        prompt,
        temperature,
        max_tokens
    )

    sft_output = generate_sft(
        prompt,
        temperature,
        max_tokens
    )

    dpo_output = generate_dpo(
        prompt,
        temperature,
        max_tokens
    )

    return (
        base_output,
        sft_output,
        dpo_output,
    )


with gr.Blocks(
    title="Alignment Playground"

) as demo:

    # ==========================================================
    # HEADER
    # ==========================================================

    gr.Markdown(
        """
# 🚀 Alignment Playground

Compare Base, SFT, and DPO models side-by-side.

Built on top of the **LLM Alignment System v1.0.0**
"""
    )

    with gr.Tabs():

        # ==========================================================
        # COMPARE TAB
        # ==========================================================

        with gr.Tab("Compare Models"):

            gr.Markdown(
                """
### Models

| Model | Description |
|---------|---------|
| Base | Qwen2.5-0.5B-Instruct |
| SFT | Qwen2.5-0.5B-Instruct + LoRA Adapter |
| DPO | Qwen2.5-0.5B-Instruct + DPO Adapter |
"""
            )

            prompt = gr.Textbox(
                label="Prompt",
                lines=5,
                placeholder="Enter a prompt to compare model responses..."
            )

            with gr.Row():

                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )

                max_tokens = gr.Slider(
                    minimum=20,
                    maximum=200,
                    value=50,
                    step=10,
                    label="Max Tokens"
                )

            compare_btn = gr.Button(
                "Compare Responses",
                variant="primary"
            )

            with gr.Row():

                base_output = gr.Textbox(
                    label="Base Model",
                    lines=18,
                    
                )

                sft_output = gr.Textbox(
                    label="SFT Model",
                    lines=18,
                    
                )

                dpo_output = gr.Textbox(
                    label="DPO Model",
                    lines=18,
                    
                )

            compare_btn.click(
               compare,
               inputs=[
                 prompt,
                 temperature,
                 max_tokens
             ],
               outputs=[
                  base_output,
                  sft_output,
                  dpo_output,
          ]
        )

        # ==========================================================
        # BENCHMARKS
        # ==========================================================

        with gr.Tab("Benchmarks"):

            gr.Markdown("# 📊 Benchmark Dashboard")

            with gr.Row():

                with gr.Column():

                    gr.HTML(
                        f"""
                        <div style="
                            padding:20px;
                            border:1px solid #ddd;
                            border-radius:12px;
                        ">
                            <h2>SFT Results</h2>

                            <h3>Alignment Score</h3>
                            <p>{SFT_RESULTS['alignment']}%</p>

                            <h3>Instruction Following</h3>
                            <p>{SFT_RESULTS['instruction']}%</p>

                            <h3>Format Valid</h3>
                            <p>{SFT_RESULTS['format']}%</p>

                            <h3>Hallucination Rate</h3>
                            <p>{SFT_RESULTS['hallucination']}%</p>
                        </div>
                        """
                    )

                with gr.Column():

                    gr.HTML(
                        f"""
                        <div style="
                            padding:20px;
                            border:1px solid #ddd;
                            border-radius:12px;
                        ">
                            <h2>DPO Results</h2>

                            <h3>Alignment Score</h3>
                            <p>{DPO_RESULTS['alignment']}%</p>

                            <h3>Instruction Following</h3>
                            <p>{DPO_RESULTS['instruction']}%</p>

                            <h3>Format Valid</h3>
                            <p>{DPO_RESULTS['format']}%</p>

                            <h3>Hallucination Rate</h3>
                            <p>{DPO_RESULTS['hallucination']}%</p>
                        </div>
                        """
                    )

            gr.Markdown(
                """
## Research Conclusion

SFT outperformed DPO on this benchmark.

### Key Insights

- Preference dataset quality matters significantly
- Pair construction strongly affects DPO performance
- Hyperparameter tuning remains critical
- Small models are highly sensitive to preference optimization

This benchmark demonstrates that DPO does not automatically improve alignment quality.
"""
            )

        # ==========================================================
        # ARCHITECTURE
        # ==========================================================

        with gr.Tab("Architecture"):

            gr.Markdown("# 🏗️ System Architecture")

            gr.Markdown(
                """
End-to-end workflow used in the project.
"""
            )

            gr.Image(
               "playground/assets/architecture.png",
            show_label=False
      )

        # ==========================================================
        # RESEARCH
        # ==========================================================

        with gr.Tab("Research Findings"):

            gr.Markdown(
                """
# 📖 Research Findings

## Main Observation

SFT achieved higher benchmark performance than DPO.

| Metric | SFT | DPO |
|---------|---------|---------|
| Alignment Score | 57.3% | 52.0% |
| Instruction Following | 55.5% | 51.5% |
| Format Valid | 55.5% | 54.0% |
| Hallucination Rate | 2.0% | 15.0% |

---

## Why Did DPO Underperform?

Potential reasons:

1. Preference pair quality
2. Dataset size limitations
3. Hyperparameter sensitivity
4. Small model scale (0.5B)

---

## Key Takeaway

Preference optimization is highly sensitive to:

- Preference quality
- Dataset design
- Training configuration
- Model capacity

A stronger preference dataset could potentially improve DPO performance.
"""
            )

        # ==========================================================
        # ABOUT
        # ==========================================================

        with gr.Tab("About"):

            gr.Markdown(
                """
# 🤖 LLM Alignment System

## Project Overview

An end-to-end alignment framework implementing:

- Synthetic Dataset Generation
- Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)
- Automated Evaluation
- Benchmark Comparison
- Kaggle Validation

---

## Dataset Statistics

### SFT Dataset

5,500 training examples

### DPO Dataset

3,151 preference pairs

---

## Base Model

Qwen/Qwen2.5-0.5B-Instruct

---

## Alignment Workflow

Base Model  
↓  
SFT Training  
↓  
SFT Evaluation  
↓  
DPO Training  
↓  
DPO Evaluation  
↓  
Benchmark Comparison  
↓  
Alignment Playground

---

## Links

GitHub Repository:

https://github.com/TechySan031/llm-alignment-system

Release:

v1.0.0

Author:

Saniya Mihani
"""
            )

demo.launch(
    share=False
)