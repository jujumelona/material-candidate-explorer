# Genomic AlphaGenome branch

AlphaGenome is an optional evaluation expert for genomic candidate exploration. It is deliberately placed between candidate generation and ranking; it does not generate DNA sequences.

```text
candidate batch -> AlphaGenome -> independent models/literature -> ranked candidates -> next mutation round
```

## Runtime setup

```bash
python -m pip install -e ".[genomics,dev]"
export ALPHAGENOME_API_KEY="..."
```

The key is read only at runtime. It is never written to a candidate record, provenance payload, notebook output, or repository file. Use a secret manager in a shared deployment.

## Candidate contract

`VariantCandidate` requires chromosome, 0-based interval coordinates, variant position, reference bases, and alternate bases. Optional ontology terms are passed to the official client, and `literature_support` is an external evidence prior in `[0, 1]`; it is not an AlphaGenome prediction.

## Results

`AlphaGenomeEvaluation` retains status, expression/splicing/chromatin/contact-map effect summaries, uncertainty, model disagreement, literature support, priority score, output names, and fail-closed errors. `evaluate_many()` evaluates the whole batch and returns a deterministic priority ordering. The adapter stores bounded numeric summaries rather than pretending that heterogeneous output tensors are directly comparable.

## Boundaries

- API outputs are computational predictions, not clinical diagnoses or medical advice.
- The API is subject to Google's current terms, including non-commercial restrictions where applicable.
- Do not commit API keys, personal genomic data, or AlphaGenome server weights.
- A high-ranked candidate still requires independent model checks, genomic controls, functional experiments, and appropriate ethics/privacy review.

The genomic adapter is a separate Python/CLI workflow. It is not included in the MatterGen T4 material notebook.
