# Azure OpenAI GPT-4.1 Streamlit Results Analysis

The attached Streamlit runs show that the Azure Container Apps deployment was functional and that Azure OpenAI GPT-4.1 generated grounded answers from the local v5.2 retrieval evidence.

## Overall interpretation

- Retrieval consistently reached the correct evidence pages for the evaluated questions.
- Azure OpenAI was strongest on exact table-value questions: Q3, Q5, and Q9 were exact matches.
- Long-form notes questions often received semantically correct answers but did not always get exact-match credit because the scoring function compares strings strictly.
- Q6 was manually mislabeled before evaluation. The model answer was correct for the intended Q6 source note, while the stored expected answer was from a different note.

## Per-question summary

| Question | Evidence page(s) shown | Model answer quality | Metric caveat |
|---|---:|---|---|
| Q1 | 102 | Correct filing and payment deadlines, including 6 months, 2 months, and 3 months for certain CCPCs | String wording differs from expected answer, so strict contains/value metrics understate performance |
| Q2 | 102 | Correct note on provinces/territories except Alberta/Quebec and 2025 temporary interest relief | Semantically correct but not exact wording |
| Q3 | 101 | Correct threshold: 2,000 | Exact/value hit |
| Q4 | 101 | Correct monthly-instalment notes and methods; includes extra notes (1) and (2) as context | More verbose than expected, value hit succeeds |
| Q5 | 81 | Correct ITC rate: 35% | Exact/value hit |
| Q6 | 81 | Correct R&D ITC note: application/refund/carryforward/carryback, T2 Schedule 31, CRA 12-month filing deadline, Forms T661/T661 Part 2/Schedule 31, R&D pool deduction | Stored expected answer was wrong; automatic score is invalid until Q6 expected answer is corrected |
| Q7 | 80 | Correct 2.80% of Quebec wages and $1.1 billion maximum | Exact match false only because answer is explanatory |
| Q8 | 80 | Correctly lists notes 1-5 for Quebec Compensation Tax | Expected answer only had note 1, so model answered broader question better than expected label |
| Q9 | 78 | Correct Federal Part VI tax rate: 1.25% | Exact/value hit |
| Q10 | 78 | Correct notes for Federal Part VI Tax, including credit union note and holding-company/Part I reduction note | Expected answer omitted note 1, so metrics understate answer quality |

## Report-ready conclusion

The Azure Foundry / Azure OpenAI deployment demonstrates a retrieval-first RAG workflow. The v5.2 retrieval artifact found the relevant tax-fact evidence, while GPT-4.1 synthesized concise answers with cited pages. Exact-value questions were handled especially well. Long-form note questions show that string-based scoring can undercount correctness when the model provides faithful but more complete wording than the expected-answer field. Q6 is the clearest example: the expected answer was manually incorrect, but the model output matched the intended R&D ITC note from page 81.
