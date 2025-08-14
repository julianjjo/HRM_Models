import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def chat_with_model(model, tokenizer, prompt, device="cpu"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Remove prompt from response if present
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    return response

def main():
    model_path = "output_model"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cargando modelo desde '{model_path}' en dispositivo '{device}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    print("Modelo cargado. Escribe tu prompt y presiona Enter. Escribe 'exit' para salir.")
    while True:
        prompt = input("\nPrompt: ")
        if prompt.strip().lower() == "exit":
            print("Finalizando chat interactivo.")
            break
        response = chat_with_model(model, tokenizer, prompt, device)
        print(f"Respuesta: {response}")

if __name__ == "__main__":
    main()