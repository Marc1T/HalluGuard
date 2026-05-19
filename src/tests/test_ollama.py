import ollama
import time

question = "Qu'est-ce qu'une hallucination dans un LLM ? Réponds en 2 phrases."

print(f"Question : {question}")
print("-" * 60)

start = time.time()
response = ollama.chat(
    model="mistral",
    messages=[{"role": "user", "content": question}]
)
elapsed = time.time() - start

print(f"Réponse  : {response['message']['content']}")
print("-" * 60)
print(f"Temps    : {elapsed:.2f} secondes")
