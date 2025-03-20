import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Gemma3ForCausalLM
from datasets import load_dataset
import os

class GemmaEvaluator:
    def __init__(
        self,
        model_name="google/gemma-3-1b-it",
        finetuned_model_path=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if finetuned_model_path and os.path.exists(finetuned_model_path):
            print(f"Loading finetuned model from: {finetuned_model_path}")
            self.clear_cuda_memory() # Clear cuda memory before loading finetuned model.
            self.model = Gemma3ForCausalLM.from_pretrained(finetuned_model_path).to(
                self.device
            )
        else:
            if finetuned_model_path:
                print(
                    f"Finetuned model path not found: {finetuned_model_path}. Loading pretrained model."
                )
            self.model = Gemma3ForCausalLM.from_pretrained(model_name).to(
                self.device
            )

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def clear_cuda_memory(self):
        """Clears CUDA memory."""
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def generate_response(self, prompt, max_new_tokens=100):
        try:
            outputs = self.pipeline(
                prompt, max_new_tokens=max_new_tokens, do_sample=True, top_k=50, top_p=0.95
            )
            return outputs[0]["generated_text"]
        except Exception as e:
            print(f"Error during generation: {e}")
            return None

    def evaluate_single_prompt(self, prompt, source=None, score=None):
        generated_response = self.generate_response(prompt)

        if generated_response:
            return {
                "prompt": prompt,
                "generated_response": generated_response,
                "source": source,
                "score": score,
            }
        else:
            return {
                "prompt": prompt,
                "generated_response": "Generation Failed",
                "source": source,
                "score": score,
            }

    def evaluate_dataset(self, dataset_name="mlabonne/FineTome-100k", sample_index=0):
        dataset = load_dataset(dataset_name, split="train")
        sample = dataset[sample_index]
        conversation = sample["conversations"]

        if conversation and isinstance(conversation, list) and len(conversation) > 0:
            prompt_object = conversation[-1]
            if isinstance(prompt_object, dict) and "value" in prompt_object:
                prompt = prompt_object["value"]
            else:
                prompt = "Conversation data format error."
        else:
            prompt = "No conversation data."

        return self.evaluate_single_prompt(prompt, sample["source"], sample["score"])


# Example usage:
if __name__ == "__main__":
    try:
        # Evaluate pretrained model on a single prompt from the dataset
        print("\n\033[94mEvaluating pretrained model:\033[0m")
        pretrained_evaluator = GemmaEvaluator()
        pretrained_result = pretrained_evaluator.evaluate_dataset(sample_index=0)

        print("\n\033[92mPrompt:\033[0m")
        print(pretrained_result["prompt"])
        print("\n\033[91mGenerated Response:\033[0m")
        print(pretrained_result["generated_response"])
        print("\n" + "─" * 50 + "\n")

        # Evaluate finetuned model on a single prompt from the dataset
        finetuned_model_path = "./results/2025-03-20_20-48-27/peft_10/"
        if os.path.exists(finetuned_model_path):
            print("\n\033[94mEvaluating finetuned model:\033[0m")
            finetuned_evaluator = GemmaEvaluator(
                finetuned_model_path=finetuned_model_path
            )
            finetuned_result = finetuned_evaluator.evaluate_dataset(sample_index=0)

            print("\n\033[92mPrompt:\033[0m")
            print(finetuned_result["prompt"])
            print("\n\033[91mGenerated Response:\033[0m")
            print(finetuned_result["generated_response"])
            print("\n" + "─" * 50 + "\n")
        else:
            print(
                f"\nFinetuned model path not found: {finetuned_model_path}. Skipping finetuned model evaluation.\n"
            )

    except Exception as e:
        print(f"\nAn error occurred: {e}\n")