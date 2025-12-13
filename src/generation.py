import os
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from openai import OpenAI
from jinja2 import Template

GOLDEN_PROMPT = """
You are a financial analyst. Answer the user query based ONLY on the context provided below.
If the answer is not in the context, state 'I don't know'.

Context:
{% for doc in documents %}
Document Source: {{ doc.metadata.source }} (Page {{ doc.metadata.page_label }})
Content:
{{ doc.text }}
---
{% endfor %}

Query: {{ query }}

Answer:
"""

class InferenceEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.template = Template(GOLDEN_PROMPT)
        
        if cfg.model.provider == "local":
            self.client = OpenAI(
                base_url=cfg.model.endpoint,
                api_key="sk-no-key-required"
            )
            self.model_id = cfg.model.name # e.g. "llama-3-8b" (ignored by many local servers but refined)
        elif cfg.model.provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_id = cfg.model.model_id
        else:
            raise ValueError(f"Unknown provider {cfg.model.provider}")

    def generate(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        prompt = self.template.render(query=query, documents=context_docs)
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.cfg.model.temperature,
            max_tokens=self.cfg.model.max_tokens
        )
        
        return response.choices[0].message.content
