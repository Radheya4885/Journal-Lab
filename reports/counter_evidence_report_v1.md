## 1. Candidate Gap
Investigate whether adaptive evidence retrieval combined with claim-level evidence verification improves factual consistency compared with fixed top-k retrieval without verification, specifically in the context of long-document summarization.

## 2. Potential Supporting Evidence
*   **[arXiv:2204.11190]**: This survey explicitly identifies "fact consistency" as a key metric for knowledge-embedded document summarizers, establishing the motivation for improving factual accuracy in this specific domain.
*   **[arXiv:2205.07830]**: FactPEGASUS addresses factuality in abstractive summarization, confirming that hallucination is a recognized problem in summarization tasks and that specific architectural changes (corrector, contrastor) can improve factual behavior.
*   **[arXiv:2407.05131]**: RULE highlights that in RAG systems, "limited retrieved contexts might not cover all necessary information, while excessive retrieval can introduce irrelevant and inaccurate references." This supports the motivation for *adaptive* retrieval (controlling the number of contexts) rather than fixed top-k, particularly to avoid over-reliance on noisy retrieved data.

## 3. Potential Counter-Evidence
*   **[arXiv:2507.04069]**: **AdaPCR** proposes "Adaptive Passage Combination Retrieval" which explicitly "adaptively selects the number of retrieved passages without additional stopping modules." This directly overlaps with the "adaptive evidence retrieval" component of the candidate gap, although it is evaluated on open-domain QA rather than summarization.
*   **[arXiv:2311.09114]**: **Ever** introduces "Real-time Verification and Rectification," which detects and rectifies hallucinations *during* generation. This overlaps with the "claim-level evidence verification" component. While it is not strictly "retrieval-based" in the traditional RAG sense (it uses real-time verification), it challenges the necessity of a separate, post-retrieval verification step by integrating verification into the generation loop.
*   **[arXiv:2410.04657]**: **CFR** (Contrastive Fact-Checking Reranker) improves retrieval for fact-checking by leveraging subquestions and contrastive objectives. This addresses the "retrieval" side of the gap with a sophisticated, non-fixed-top-k approach, suggesting that advanced retrieval alone (without explicit claim-level verification in the generation loop) may already yield significant gains in veracity.
*   **[arXiv:2605.19341]**: **HalluWorld** argues that hallucinations arise from "distinct failure modes" (perceptual, state tracking, causal simulation) rather than a single capability. This challenges the assumption that a single unified pipeline of "adaptive retrieval + verification" will uniformly improve factual consistency across all types of errors in long documents.

## 4. Evidence Comparison
*   **Retrieval Strategy**: The candidate gap proposes "adaptive evidence retrieval." [arXiv:2507.04069] (AdaPCR) already implements adaptive passage selection based on inter-passage dependencies. [arXiv:2407.05131] (RULE) uses calibrated selection of context count. The novelty of "adaptive retrieval" is therefore low, as it is an established technique in QA and medical VQA, though its specific application to *long-document summarization* may be less explored.
*   **Verification Mechanism**: The candidate gap proposes "claim-level evidence verification." [arXiv:2311.09114] (Ever) performs real-time verification/rectification. [arXiv:2410.04657] (CFR) focuses on retrieval quality to support verification. The gap lies in whether *post-retrieval, pre-generation* or *during-generation* claim-level verification adds value *on top of* adaptive retrieval specifically for summarization.
*   **Domain Specificity**: Most counter-evidence (AdaPCR, CFR, Ever) focuses on QA, fact-checking, or general generation. [arXiv:2204.11190] and [arXiv:2205.07830] focus on summarization but do not explicitly combine *adaptive RAG* with *claim-level verification* in a single framework. [arXiv:2407.05131] (RULE) is close but focuses on medical VQA/report generation, not general long-document summarization.
*   **Limitations**: [arXiv:2605.19341] suggests that different hallucination types require different solutions, implying that a single "adaptive retrieval + verification" pipeline might not be a silver bullet. [arXiv:2609.32041] highlights the "statelessness bottleneck," suggesting that without persistent state, even adaptive retrieval may fail to detect contradictions across long documents.

## 5. Revised Research Opportunity
The original gap is too broad because "adaptive retrieval" and "verification" are already well-studied in adjacent domains (QA, fact-checking). A more defensible research question would focus on the **interaction** between these components in the specific context of **long-document summarization**, where the challenge is not just retrieving relevant facts but maintaining **global consistency** and **state tracking** across the entire document.

**Revised Question:**
"Does integrating claim-level verification into an adaptive retrieval pipeline specifically mitigate *cross-document* or *long-range* factual inconsistencies in long-document summarization, compared to adaptive retrieval alone or fixed top-k retrieval with verification?"

This revision:
1.  Acknowledges that adaptive retrieval (AdaPCR) and verification (Ever) exist.
2.  Focuses on the *combination* and its specific benefit for *long-range* consistency, which is a known weakness in summarization (as hinted by [arXiv:2605.19341]'s finding that multi-step state tracking is difficult).
3.  Moves beyond simple "factual consistency" to address the specific failure mode of long documents (state tracking/contradiction detection).

## 6. Verification Status
"Substantial overlap found — narrow or revise the gap"