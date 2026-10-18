import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

from backend.config import settings
from backend.retrieval import retriever
from backend.schemas import (
    CounterEvidenceResponse,
    ExperimentPlanResponse,
    GapAnalysisResponse,
)

logger = logging.getLogger("journallab.rag")

class GroqRAGService:
    """Evidence-grounded RAG service for gap analysis, counter-evidence testing, and experiment planning."""

    def __init__(self) -> None:
        self._client = None

    def get_client(self):
        """Lazy load Groq client when API key is available."""
        api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured. Set GROQ_API_KEY in your .env file or environment variables "
                "to enable AI gap analysis, counter-evidence verification, and experiment planning."
            )
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=api_key)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Groq client: {e}")
        return self._client

    def find_research_gaps(self, topic: str, domain: str = "Auto", top_k: int = 15) -> GapAnalysisResponse:
        """Analyze top retrieved papers to discover grounded research gaps."""
        client = self.get_client()

        search_result = retriever.search(topic, domain=domain, top_k=top_k)
        papers = search_result["results"]

        if not papers:
            raise ValueError(f"No papers were retrieved for topic: '{topic}' in domain: '{domain}'.")

        evidence_blocks = []
        for p in papers:
            evidence_blocks.append(
                f"[arXiv:{p.arxiv_id}]\n"
                f"Title: {p.title}\n"
                f"Domains: {', '.join(p.domain_tags)}\n"
                f"Abstract: {p.abstract}"
            )
        evidence = "\n\n".join(evidence_blocks)

        prompt = f"""
You are JournalLab's Research Gap Analysis Assistant.

RESEARCH TOPIC:
{topic}

RETRIEVED PAPERS:
<evidence>
{evidence}
</evidence>

Analyze the supplied papers and produce these sections:

1. RESEARCH LANDSCAPE & CORE CONTRIBUTIONS
Summarize the main themes, approaches, and findings represented in the evidence.

2. RESEARCH GAPS & OPEN CHALLENGES (EXPLAINED IN VERY SIMPLE AND UNDERSTANDABLE TERMS)
Identify 3-4 promising, evidence-grounded research gap hypotheses directly stemming from this research.
Explain each research gap in very simple, plain, easy-to-understand terms:
- What is missing or unsolved (in 1 simple, plain sentence)
- Why this matters to researchers and practitioners
- Supporting evidence: Exact arXiv IDs cited as [arXiv:ID] from the retrieved literature
- A concrete research question anyone can easily grasp
- A simple, concrete experiment idea to test or validate it

3. REPORTED LIMITATIONS & PRIOR ART
Identify limitations and boundaries explicitly reported in the supplied abstracts.
If a limitation is not stated, do not attribute it to the authors.

4. RESEARCH PRIORITIES & FEASIBILITY
Rank the candidate gaps by potential scientific significance, implementation feasibility, and supporting evidence.
Explain that these rankings are preliminary assessments.

5. REPRODUCIBILITY & VALIDATION CAUTION
State what additional literature review or full-text reading is required before claiming novelty.

STRICT RULES:
- Use only the supplied titles and abstracts for paper-specific claims.
- Cite only the exact arXiv IDs included in the supplied evidence.
- Do not invent experiments, datasets, results, or author conclusions.
- Distinguish reported limitations from your own inferences.
- Do not claim that a topic has never been studied based on this sample.
"""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are JournalLab, a rigorous research-paper analyst. "
                    "Distinguish source-supported findings from hypotheses. "
                    "Never invent citations, results, or claim novelty without evidence."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model=settings.GROQ_RAG_MODEL,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=2500,
        )

        report = response.choices[0].message.content or ""
        valid_ids: Set[str] = {p.arxiv_id for p in papers}
        cited_ids = set(re.findall(r"arXiv:\s*(\d{4}\.\d{4,5})(?:v\d+)?", report, flags=re.IGNORECASE))
        unsupported = cited_ids - valid_ids

        return GapAnalysisResponse(
            topic=topic,
            domain=domain,
            report=report,
            retrieved_papers_count=len(papers),
            cited_paper_ids=sorted(cited_ids & valid_ids),
            unsupported_citation_ids=sorted(unsupported),
            disclaimer="Candidate gaps are inferred from retrieved abstracts and should be validated with full-text literature review.",
        )

    def stress_test_research_gap(
        self,
        research_topic: str,
        candidate_gap: str,
        domain: str = "Auto",
        papers_per_search: int = 10,
    ) -> CounterEvidenceResponse:
        """Search for both supporting evidence and counter-evidence/prior-art to stress-test a proposed gap."""
        client = self.get_client()

        support_query = f"{research_topic}. Research problem and hypothesis: {candidate_gap}"
        counter_query = f"{research_topic}. Existing methods, prior solutions, related work, and evidence addressing or challenging: {candidate_gap}"

        res_sup = retriever.search(support_query, domain=domain, top_k=papers_per_search)
        res_cnt = retriever.search(counter_query, domain=domain, top_k=papers_per_search)

        combined: Dict[str, Dict[str, Any]] = {}
        for route, result_set in [("supporting search", res_sup["results"]), ("counter-evidence search", res_cnt["results"])]:
            for p in result_set:
                if p.arxiv_id not in combined:
                    combined[p.arxiv_id] = {
                        "arxiv_id": p.arxiv_id,
                        "title": p.title,
                        "abstract": p.abstract,
                        "domain_tags": p.domain_tags,
                        "routes": [route],
                    }
                else:
                    combined[p.arxiv_id]["routes"].append(route)

        retrieved_papers = list(combined.values())
        if not retrieved_papers:
            raise ValueError("No literature retrieved for verification.")

        evidence = "\n\n".join(
            f"[arXiv:{p['arxiv_id']}]\n"
            f"Title: {p['title']}\n"
            f"Domains: {', '.join(p['domain_tags'])}\n"
            f"Retrieved via: {', '.join(p['routes'])}\n"
            f"Abstract: {p['abstract']}"
            for p in retrieved_papers
        )

        prompt = f"""
You are JournalLab's Counter-Evidence Research Gap Verifier.

RESEARCH TOPIC:
{research_topic}

CANDIDATE GAP:
{candidate_gap}

RETRIEVED LITERATURE:
<evidence>
{evidence}
</evidence>

Your task is to stress-test the candidate gap, identifying prior art or overlapping solutions.

Produce these sections:
## 1. Candidate Gap
Restate the proposed gap as a precise, testable claim.

## 2. Potential Supporting Evidence
Identify papers whose abstracts support the problem or motivation.

## 3. Potential Counter-Evidence
Identify papers that appear to address, overlap with, or challenge the candidate gap.

## 4. Evidence Comparison
Compare approaches, problem settings, and stated limitations.

## 5. Revised Research Opportunity
Suggest a narrower or more defensible research question that accounts for the counter-evidence.

## 6. Verification Status
Choose one:
- "Potentially promising — requires broader verification"
- "Substantial overlap found — narrow or revise the gap"
- "Insufficient evidence to assess"

RULES:
- Use only the supplied titles and abstracts for paper-specific claims.
- Cite papers using exact [arXiv:ID] identifiers.
- Do not invent claims, findings, or novelty proofs.
"""

        response = client.chat.completions.create(
            model=settings.GROQ_RAG_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a skeptical scientific literature analyst who actively looks for contradictory evidence.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_completion_tokens=2500,
        )

        report = response.choices[0].message.content or ""
        valid_ids = set(combined.keys())
        cited_ids = set(re.findall(r"arXiv:\s*(\d{4}\.\d{4,5})(?:v\d+)?", report, flags=re.IGNORECASE))
        unmatched = cited_ids - valid_ids

        return CounterEvidenceResponse(
            topic=research_topic,
            candidate_gap=candidate_gap,
            report=report,
            unique_papers_examined=len(retrieved_papers),
            unmatched_ids=sorted(unmatched),
        )

    def plan_research_experiment(
        self,
        research_topic: str,
        candidate_gap: str,
        domain: str = "Auto",
        top_k: int = 10,
    ) -> ExperimentPlanResponse:
        """Generate a reproducible, 10-section experimental research plan grounded in retrieved papers."""
        client = self.get_client()

        search_result = retriever.search(research_topic, domain=domain, top_k=top_k)
        papers = search_result["results"]
        if not papers:
            raise ValueError(f"No supporting papers retrieved for topic: '{research_topic}'.")

        evidence = "\n\n".join(
            f"[arXiv:{p.arxiv_id}]\n"
            f"Title: {p.title}\n"
            f"Domains: {', '.join(p.domain_tags)}\n"
            f"Abstract: {p.abstract}"
            for p in papers
        )

        prompt = f"""
You are JournalLab's scientific experiment-planning assistant.

RESEARCH TOPIC:
{research_topic}

CANDIDATE RESEARCH GAP:
{candidate_gap}

RETRIEVED PAPER EVIDENCE:
<evidence>
{evidence}
</evidence>

Create a practical, reproducible experimental research plan using these exact sections:
## 1. Research Problem
## 2. Hypothesis and Research Questions
## 3. Proposed Method
## 4. Datasets and Experimental Split
## 5. Baselines and Comparisons (include a simple baseline)
## 6. Evaluation Metrics
## 7. Ablation Studies
## 8. Reproducibility and Resources
## 9. Risks, Limitations, and Falsification Criteria
## 10. Decision Criteria and Next Steps

SCIENTIFIC RULES:
- Use supplied abstracts only for claims about the retrieved papers.
- Cite supplied papers with their exact [arXiv:ID].
- Label new methods and external datasets as proposals.
- Do not invent reported experimental results.
"""

        response = client.chat.completions.create(
            model=settings.GROQ_RAG_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a rigorous experimental-design assistant. Distinguish evidence, inference, and proposals.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_completion_tokens=3500,
        )

        plan = response.choices[0].message.content or ""
        valid_ids = {p.arxiv_id for p in papers}
        cited_ids = set(re.findall(r"arXiv:\s*(\d{4}\.\d{4,5})(?:v\d+)?", plan, flags=re.IGNORECASE))
        unmatched = cited_ids - valid_ids

        return ExperimentPlanResponse(
            research_topic=research_topic,
            candidate_gap=candidate_gap,
            plan=plan,
            supporting_papers_count=len(papers),
            unmatched_ids=sorted(unmatched),
        )

    def stream_find_research_gaps(self, topic: str, domain: str = "Auto", top_k: int = 15):
        """Stream research gap analysis chunks token-by-token."""
        client = self.get_client()
        search_result = retriever.search(topic, domain=domain, top_k=top_k)
        papers = search_result["results"]
        if not papers:
            yield f"No literature found for topic: '{topic}'."
            return

        evidence_blocks = [
            f"[arXiv:{p.arxiv_id}]\nTitle: {p.title}\nDomains: {', '.join(p.domain_tags)}\nAbstract: {p.abstract}"
            for p in papers
        ]
        evidence = "\n\n".join(evidence_blocks)

        prompt = f"""You are JournalLab's Research Gap Analysis Assistant.

RESEARCH TOPIC:
{topic}

RETRIEVED LITERATURE EVIDENCE:
<evidence>
{evidence}
</evidence>

Produce a rigorous research gap analysis with:
1. EXECUTIVE SUMMARY & RESEARCH LANDSCAPE
2. RESEARCH GAPS & OPEN CHALLENGES (EXPLAINED IN VERY SIMPLE AND UNDERSTANDABLE TERMS - Identify 3-4 grounded candidate gaps with exact [arXiv:ID] citations, plain-language explanation of what is missing, and simple test ideas)
3. REPORTED LIMITATIONS & PRIOR ART BENCHMARKS
4. RESEARCH PRIORITIES & FEASIBILITY (significance, feasibility, expected impact)
5. REPRODUCIBILITY & VALIDATION CAUTION
"""
        stream = client.chat.completions.create(
            model=settings.GROQ_RAG_MODEL,
            messages=[
                {"role": "system", "content": "You are JournalLab, a rigorous research-paper analyst. Never invent citations."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_completion_tokens=2500,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def stream_plan_research_experiment(self, research_topic: str, candidate_gap: str, domain: str = "Auto", top_k: int = 10):
        """Stream 10-section experiment plan chunks token-by-token."""
        client = self.get_client()
        query = f"{research_topic}. {candidate_gap}"
        search_result = retriever.search(query, domain=domain, top_k=top_k)
        papers = search_result["results"]

        evidence = "\n\n".join(
            f"[arXiv:{p.arxiv_id}]\nTitle: {p.title}\nDomains: {', '.join(p.domain_tags)}\nAbstract: {p.abstract}"
            for p in papers
        )

        prompt = f"""You are JournalLab's scientific experiment-planning assistant.

RESEARCH TOPIC:
{research_topic}

CANDIDATE RESEARCH GAP:
{candidate_gap}

RETRIEVED PAPER EVIDENCE:
<evidence>
{evidence}
</evidence>

Generate a complete, reproducible 10-section experimental research plan:
## 1. Research Problem
## 2. Hypothesis and Research Questions
## 3. Proposed Method
## 4. Datasets and Experimental Split
## 5. Baselines and Comparisons (include a simple baseline)
## 6. Evaluation Metrics
## 7. Ablation Studies
## 8. Reproducibility and Resources
## 9. Risks, Limitations, and Falsification Criteria
## 10. Decision Criteria and Next Steps
"""
        stream = client.chat.completions.create(
            model=settings.GROQ_RAG_MODEL,
            messages=[
                {"role": "system", "content": "You are a rigorous experimental-design assistant. Distinguish evidence from proposals."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_completion_tokens=3500,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


rag_service = GroqRAGService()
