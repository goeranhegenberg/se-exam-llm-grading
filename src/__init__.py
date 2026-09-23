"""Prompt-basierte Bewertung von Software-Engineering-Klausurantworten.

Pakete:
    config         -- Konfiguration und Geheimnisse laden
    dataset        -- Benchmark laden und gegen das JSON-Schema validieren
    prompt_builder -- Prompt-Templates rendern
    llm_client     -- Modellanbindung (OpenAI) und Mock-Client
    run_grading    -- Bewertungslauf (Runner)
"""
