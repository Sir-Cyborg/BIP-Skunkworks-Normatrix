"""
evaluator.py - Valutazione LLM tramite Ollama

Chiama un LLM locale (Ollama) per valutare se le evidenze recuperate
soddisfano un requisito normativo.

Il LLM risponde in JSON strutturato:
  {
    "status": "COMPLIANT" | "NON_COMPLIANT" | "PARTIAL",
    "explanation": "motivazione in lingua del documento",
    "confidence": 0.0-1.0
  }

Per passare a un LLM diverso (Azure OpenAI, GPT-4, ecc.)
basta cambiare OLLAMA_BASE_URL e OLLAMA_MODEL nelle variabili d'ambiente.
"""

import os
import json
import re
import requests
from dataclasses import dataclass
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "phi3.5")
LLM_TIMEOUT     = int(os.getenv("LLM_TIMEOUT", 120))   # secondi
MAX_EVIDENCE_CHARS = int(os.getenv("MAX_EVIDENCE_CHARS", 600))  # tronca testi lunghi


# ── Dataclass risultato ───────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    status: str          # "COMPLIANT" | "NON_COMPLIANT" | "PARTIAL"
    explanation: str     # motivazione testuale
    confidence: float    # 0.0 - 1.0
    is_placeholder: bool # False quando il LLM è attivo


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(question: str, evidences: list[dict]) -> str:
    """
    Costruisce il prompt per il LLM.

    Istruzioni chiave:
      - Risposta SOLO in JSON (niente testo libero attorno)
      - Lingua della spiegazione = lingua del requisito
      - Valutazione basata esclusivamente sulle evidenze fornite
    """

    # Formatta le evidenze come testo numerato
    evidence_blocks = []
    for i, e in enumerate(evidences, 1):
        text = e.get("text", "")[:MAX_EVIDENCE_CHARS]
        source = e.get("source", "N/A")
        section = e.get("section_path", "")
        score = e.get("score", 0)
        block = (
            f"[Evidenza {i}]\n"
            f"Fonte: {source}\n"
            f"Sezione: {section}\n"
            f"Rilevanza: {score:.3f}\n"
            f"Testo: {text}"
        )
        evidence_blocks.append(block)

    evidences_text = "\n\n".join(evidence_blocks) if evidence_blocks else "Nessuna evidenza disponibile."

    prompt = f"""Sei un esperto di compliance normativa bancaria e regolamentazione finanziaria.

Il tuo compito è valutare se le evidenze documentali fornite dimostrano che l'organizzazione soddisfa il requisito normativo indicato.

ISTRUZIONI:
- Basa la tua valutazione ESCLUSIVAMENTE sulle evidenze fornite
- Se le evidenze sono insufficienti o assenti, valuta NON_COMPLIANT
- Rispondi nella stessa lingua del REQUISITO
- Rispondi SOLO con il JSON, senza testo aggiuntivo prima o dopo

REQUISITO DA VALUTARE:
{question}

EVIDENZE DOCUMENTALI RECUPERATE:
{evidences_text}

Rispondi ESCLUSIVAMENTE con questo JSON (nessun testo fuori dal JSON):
{{
  "status": "COMPLIANT" o "NON_COMPLIANT" o "PARTIAL",
  "explanation": "motivazione concisa di 2-3 frasi nella lingua del requisito",
  "confidence": <numero tra 0.0 e 1.0>
}}"""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSER (robusto)
# ─────────────────────────────────────────────────────────────────────────────

def parse_llm_response(raw: str) -> dict:
    """
    Estrae il JSON dalla risposta del LLM in modo robusto.

    Il LLM a volte aggiunge testo prima/dopo il JSON nonostante le istruzioni.
    Gestiamo i casi più comuni:
      - JSON puro
      - JSON dentro ```json ... ```
      - JSON preceduto da testo libero
    """
    # Caso 1: rimuovi markdown code block se presente
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    raw = raw.replace("```", "").strip()

    # Caso 2: prova parse diretto
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Caso 3: cerca il primo { ... } nel testo
    match = re.search(r'\{[^{}]*"status"[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Caso 4: cerca qualsiasi oggetto JSON nel testo
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Impossibile estrarre JSON dalla risposta LLM: {raw[:200]}")


def validate_response(data: dict) -> dict:
    """Valida e normalizza i campi del JSON restituito dal LLM."""
    valid_statuses = {"COMPLIANT", "NON_COMPLIANT", "PARTIAL"}

    status = str(data.get("status", "")).upper().strip()
    # Gestisci varianti comuni
    if status in {"COMPLIANT", "CONFORME", "SI", "YES", "TRUE"}:
        status = "COMPLIANT"
    elif status in {"NON_COMPLIANT", "NON COMPLIANT", "NON-COMPLIANT", "NON CONFORME", "NO", "FALSE"}:
        status = "NON_COMPLIANT"
    elif status in {"PARTIAL", "PARZIALE", "PARTIALLY COMPLIANT"}:
        status = "PARTIAL"
    else:
        status = "NON_COMPLIANT"  # fallback sicuro

    explanation = str(data.get("explanation", "Valutazione non disponibile.")).strip()

    try:
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))  # clamp 0-1
    except (TypeError, ValueError):
        confidence = 0.5

    return {"status": status, "explanation": explanation, "confidence": confidence}


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Valuta compliance chiamando Ollama via API HTTP.

    Ollama espone un'API compatibile OpenAI su localhost:11434.
    In Docker, raggiungiamo l'host Windows tramite host.docker.internal.

    Per passare a un altro LLM in futuro:
      - Azure OpenAI: cambia OLLAMA_BASE_URL e aggiungi Authorization header
      - OpenAI:       cambia OLLAMA_BASE_URL = https://api.openai.com
      - LiteLLM:      cambia OLLAMA_BASE_URL = http://litellm:4000
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = LLM_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._ollama_available = None  # cache disponibilità

    def _check_ollama(self) -> bool:
        """Verifica che Ollama sia raggiungibile."""
        if self._ollama_available is not None:
            return self._ollama_available
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            self._ollama_available = r.status_code == 200
        except Exception:
            self._ollama_available = False
        return self._ollama_available

    def _call_ollama(self, prompt: str) -> str:
        """
        Chiama Ollama tramite API /api/generate.
        Usa stream=False per ricevere la risposta completa in una volta.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,   # bassa temperatura = risposte più deterministiche
                "top_p": 0.9,
                "num_predict": 300,   # massimo token output — il JSON è breve
            }
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def evaluate(self, question: str, evidences: list[dict]) -> EvaluationResult:
        """
        Valuta se le evidenze soddisfano il requisito.

        Flusso:
          1. Verifica che Ollama sia disponibile
          2. Costruisce il prompt
          3. Chiama il LLM
          4. Parsa il JSON dalla risposta
          5. Fallback al placeholder se qualcosa va storto

        Args:
            question : domanda requisito normativo
            evidences: lista di dict con 'text', 'score', 'source', 'section_path'

        Returns:
            EvaluationResult con status, explanation, confidence
        """
        if not evidences:
            return EvaluationResult(
                status="NON_COMPLIANT",
                explanation="Nessuna evidenza rilevante trovata nel corpus documentale.",
                confidence=0.0,
                is_placeholder=False,
            )

        # Verifica disponibilità Ollama
        if not self._check_ollama():
            logger.warning(f"Ollama non raggiungibile su {self.base_url} — uso fallback placeholder")
            return self._placeholder_fallback(evidences)

        # Costruisci prompt e chiama LLM
        prompt = build_prompt(question, evidences)
        logger.debug(f"Chiamata LLM | modello={self.model} | domanda={question[:80]}...")

        try:
            raw_response = self._call_ollama(prompt)
            logger.debug(f"Risposta LLM raw: {raw_response[:200]}")

            parsed = parse_llm_response(raw_response)
            validated = validate_response(parsed)

            return EvaluationResult(
                status=validated["status"],
                explanation=validated["explanation"],
                confidence=validated["confidence"],
                is_placeholder=False,
            )

        except requests.exceptions.Timeout:
            logger.error(f"Timeout LLM dopo {self.timeout}s")
            return EvaluationResult(
                status="NON_COMPLIANT",
                explanation=f"Timeout nella valutazione LLM (>{self.timeout}s). Riprovare.",
                confidence=0.0,
                is_placeholder=True,
            )
        except Exception as e:
            logger.error(f"Errore valutazione LLM: {e}")
            return self._placeholder_fallback(evidences)

    def _placeholder_fallback(self, evidences: list[dict]) -> EvaluationResult:
        """
        Fallback basato su score semantico quando il LLM non è disponibile.
        Mantiene il sistema funzionante anche senza Ollama.
        """
        avg_score = sum(e["score"] for e in evidences) / len(evidences)
        top_score = evidences[0]["score"] if evidences else 0.0

        if top_score >= 0.75 and avg_score >= 0.65:
            status, explanation = "COMPLIANT", (
                f"[Fallback — Ollama non disponibile] "
                f"Score semantico elevato (media: {avg_score:.2f}). "
                f"Le evidenze sembrano pertinenti al requisito."
            )
        elif top_score >= 0.60 or avg_score >= 0.55:
            status, explanation = "PARTIAL", (
                f"[Fallback — Ollama non disponibile] "
                f"Score semantico medio (media: {avg_score:.2f}). "
                f"Copertura parziale del requisito."
            )
        else:
            status, explanation = "NON_COMPLIANT", (
                f"[Fallback — Ollama non disponibile] "
                f"Score semantico basso (media: {avg_score:.2f}). "
                f"Evidenze insufficienti per il requisito."
            )

        return EvaluationResult(
            status=status,
            explanation=explanation,
            confidence=round(avg_score, 4),
            is_placeholder=True,
        )


# Singleton globale
evaluator = Evaluator()