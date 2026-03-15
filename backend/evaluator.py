"""
evaluator.py - Valutazione LLM (placeholder)

Per ora la valutazione è simulata:
  - COMPLIANT     se score medio evidenze > soglia
  - NON COMPLIANT altrimenti

Quando integrerete il LLM reale, sostituite il metodo `evaluate`
mantenendo la stessa firma e lo stesso formato di output.
"""

import os
from dataclasses import dataclass

# Soglia di score medio per considerare un requisito COMPLIANT
COMPLIANCE_THRESHOLD = float(os.getenv("COMPLIANCE_THRESHOLD", 0.75))


@dataclass
class EvaluationResult:
    status: str           # "COMPLIANT" | "NON_COMPLIANT" | "PARTIAL"
    explanation: str      # motivazione testuale
    confidence: float     # score medio delle evidenze (0-1)
    is_placeholder: bool  # True finché il LLM reale non è integrato


class Evaluator:
    """
    Valuta se le evidenze recuperate soddisfano un requisito.

    PLACEHOLDER — sostituire il metodo `evaluate` con chiamata LLM reale.
    Firma da mantenere:
        evaluate(question: str, evidences: list[dict]) -> EvaluationResult
    """

    def evaluate(self, question: str, evidences: list[dict]) -> EvaluationResult:
        """
        Valuta compliance basandosi sullo score medio delle evidenze.

        Args:
            question : domanda requisito
            evidences: output del Retriever (lista di dict con 'score', 'text', ecc.)

        Returns:
            EvaluationResult con status, explanation, confidence
        """
        if not evidences:
            return EvaluationResult(
                status="NON_COMPLIANT",
                explanation="Nessuna evidenza rilevante trovata nel corpus documentale.",
                confidence=0.0,
                is_placeholder=True,
            )

        avg_score = sum(e["score"] for e in evidences) / len(evidences)
        top_score = evidences[0]["score"] if evidences else 0.0

        # ── Logica placeholder ────────────────────────────────────────────────
        # Usa score medio + top score per determinare lo stato.
        # Questa logica verrà sostituita dal LLM.

        if top_score >= COMPLIANCE_THRESHOLD and avg_score >= 0.65:
            status = "COMPLIANT"
            explanation = (
                f"Le evidenze recuperate mostrano un'alta rilevanza semantica "
                f"(score medio: {avg_score:.2f}). Il corpus documentale contiene "
                f"contenuti pertinenti al requisito analizzato. "
                f"[Valutazione placeholder — LLM non ancora integrato]"
            )
        elif top_score >= 0.60 or avg_score >= 0.55:
            status = "PARTIAL"
            explanation = (
                f"Le evidenze recuperate mostrano una rilevanza parziale "
                f"(score medio: {avg_score:.2f}). Il corpus contiene riferimenti "
                f"al tema ma potrebbero non coprire completamente il requisito. "
                f"[Valutazione placeholder — LLM non ancora integrato]"
            )
        else:
            status = "NON_COMPLIANT"
            explanation = (
                f"Le evidenze recuperate mostrano bassa rilevanza semantica "
                f"(score medio: {avg_score:.2f}). Il corpus documentale potrebbe "
                f"non contenere documentazione sufficiente per questo requisito. "
                f"[Valutazione placeholder — LLM non ancora integrato]"
            )

        return EvaluationResult(
            status=status,
            explanation=explanation,
            confidence=round(avg_score, 4),
            is_placeholder=True,
        )


# Singleton globale
evaluator = Evaluator()