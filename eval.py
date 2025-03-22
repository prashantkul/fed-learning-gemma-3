import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM
from datasets import load_dataset
import os
from peft import PeftModel, PeftConfig  # Import PeftModel and PeftConfig

class ModelEvaluator:
    def __init__(
        self,
        model_name="google/gemma-3-1b-pt",
        finetuned_model_path=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if finetuned_model_path and os.path.exists(finetuned_model_path):
            print(f"Loading finetuned model from: {finetuned_model_path}")
            # Load the PEFT configuration
            peft_config = PeftConfig.from_pretrained(finetuned_model_path)
            # Load the base model
            base_model = Gemma3ForCausalLM.from_pretrained(peft_config.base_model_name_or_path, torch_dtype=torch.bfloat16)
            # Load the PEFT adapter on top of the base model
            self.model = PeftModel.from_pretrained(base_model, finetuned_model_path).to(self.device)
        else:
            if finetuned_model_path:
                print(
                    f"Finetuned model path not found: {finetuned_model_path}. Loading pretrained model."
                )
            self.model = Gemma3ForCausalLM.from_pretrained(self.model_name).to(
                self.device
            )

    def generate_response(self, human_prompt, max_new_tokens=256, temperature=0.7, repetition_penalty=2.0):
        try:
            # Format prompt using Gemma 3's preferred format
            instruction = "You are a helpful assistant."
            full_prompt = f"<start_of_turn>user\n{human_prompt}<end_of_turn>\n<start_of_turn>model\n"
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the model's response part
            if full_prompt in generated_text:
                return generated_text[len(full_prompt):].strip()
            return generated_text
        except Exception as e:
            print(f"Error during generation: {e}")
            return None
            
    def evaluate_single_prompt(self, human_prompt, expected_response, source=None, score=None):

        generated_response = self.generate_response(human_prompt)

        if generated_response:
            return {
                "prompt": human_prompt,
                "generated_response": generated_response,
                "expected_response": expected_response,
                "source": source,
                "score": score,
            }
        else:
            return {
                "prompt": human_prompt,
                "generated_response": "Generation Failed",
                "expected_response": expected_response,
                "source": source,
                "score": score,
            }

    def evaluate_dataset(self, dataset_name="mlabonne/FineTome-100k", sample_index=0):
        dataset = load_dataset(dataset_name, split="train")
        sample = dataset[sample_index]
        conversation = sample["conversations"]

        human_prompt = None
        gpt_response = None

        for turn in conversation:
            if turn["from"] == "human":
                human_prompt = turn["value"]
            elif turn["from"] == "gpt":
                gpt_response = turn["value"]

        if human_prompt and gpt_response:
            return self.evaluate_single_prompt(human_prompt, gpt_response, sample["source"], sample["score"])
        else:
            return {
                "prompt": "Error extracting prompt and expected response.",
                "generated_response": "Evaluation Failed",
                "expected_response": "Evaluation Failed",
                "source": sample["source"],
                "score": sample["score"],
            }

# Example usage:
if __name__ == "__main__":
    try:
        # Load sample to extract prompt.
        dataset = load_dataset("mlabonne/FineTome-100k", split="train")
        sample = dataset[10]  # Change the index to select a different sample
        conversation = sample["conversations"]

        human_prompt = None
        for turn in conversation:
            if turn["from"] == "human":
                human_prompt = turn["value"]
                break

        if human_prompt is None:
            print("Error extracting prompt from dataset.")
        else:

            # Evaluate pretrained model on the 1st prompt from the dataset
            print("\n\033[94mEvaluating pretrained model:\033[0m")
            pretrained_evaluator = ModelEvaluator()
            pretrained_result = pretrained_evaluator.evaluate_single_prompt(human_prompt, sample['conversations'][1]['value'], sample["source"], sample["score"])

            # Define box drawing characters for a neat output
            top_border    = "╔" + "═" * 60 + "╗"
            mid_border    = "╟" + "─" * 60 + "╢"
            bottom_border = "╚" + "═" * 60 + "╝"

            # Display the Prompt in a decorated box with green color
            print("\n" + top_border)
            print("║ \033[92mPrompt:\033[0m")
            print(mid_border)
            print("║ " + pretrained_result["prompt"])
            print(bottom_border)

            # Display the Expected Response in a decorated box with yellow color
            print("\n" + top_border)
            print("║ \033[93mExpected Response:\033[0m")
            print(mid_border)
            print("║ " + pretrained_result["expected_response"])
            print(bottom_border)

            print("================================================================================")
            print("\n\033[91mGenerated Response from pretrained model:\033[0m")
            print(pretrained_result["generated_response"])
            
            print("\n" + "─" * 50 + "\n")

            # Evaluate finetuned model on the 1st prompt from the dataset

            finetuned_model_path = os.path.join(os.getcwd(), "results", "2025-03-22_01-24-59", "peft_10")

            print("Loading fine tuned model from: " + finetuned_model_path)

            if os.path.exists(finetuned_model_path):
                print("\n\033[94mEvaluating finetuned model:\033[0m")
                finetuned_evaluator = ModelEvaluator(
                    finetuned_model_path=finetuned_model_path
                )
                finetuned_result = finetuned_evaluator.evaluate_single_prompt(human_prompt, sample['conversations'][1]['value'], sample["source"], sample["score"])
                
                print("\n\033[91mGenerated Response from finetuned model:\033[0m")
                print(finetuned_result["generated_response"])
               
                print("\n" + "─" * 50 + "\n")
            else:
                print(
                    f"\nFinetuned model path not found: {finetuned_model_path}. Skipping finetuned model evaluation.\n"
                )

    except Exception as e:
        print(f"\nAn error occurred: {e}\n")
