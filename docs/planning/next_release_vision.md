# Next release: reliable benchmarks for code similarity

Status: proposal for discussion, 7 September 2026. No feature scope, release date, or version bump is approved. Working target: v0.7.0 once the scope below is agreed and validated.

The proposed direction is to make Matheel a task-based evaluation framework for code similarity, with an approachable CLI and Gradio interface. A user should select a versioned benchmark, select compatible detectors, run it, and obtain comparable metrics with inspectable predictions. Adding more similarity scores or dataset names alone will not achieve that.

The accompanying [review](review_2026_09_07.md) records the baseline, fixes, and validation. Planning issue: [#278](https://github.com/FahadEbrahim/matheel/issues/278). Existing adapter investigation: [#46](https://github.com/FahadEbrahim/matheel/issues/46).

## What exists and what needs to change

Matheel 0.6.0 already has pair and retrieval manifests, six dataset presets, custom pair algorithms, labeled metrics, calibration, grouping-aware split helpers, source fingerprints, a collection-result cache, dataset/model cards, a benchmark-run registry, and report/UI workflows. Preserve those building blocks.

The main design gaps found in the reviewed code are:

- `comparison_suite.py` summarizes a thresholded/top-N collection and sorts runs by maximum and mean raw similarity. These are useful inspection statistics, not detector-quality rankings. The existing scoring documentation already makes that distinction; the product should make it equally clear.
- `leaderboard.py` evaluates labeled data, but task membership and protocols are supplied ad hoc. The runner does not retain its scored rows in the leaderboard artifact. It reloads/adapts datasets per algorithm, then loads them again for cards.
- Algorithm-specific `k` values can be shown under the same `ndcg_at_k` metric label. Aggregate rows and cross-run deltas do not enforce identical task coverage, dataset fingerprints, splits, and metric parameters. A new official benchmark contract must reject such comparisons.
- `DATASET_TASK_TYPES` contains only `plagiarism`; this cannot describe functional similarity or translated code accurately.
- Retrieval materializes every query/document score. Multi-vector pair evaluation also precomputes the entire used-file Cartesian product. Large clone corpora need bounded batches, candidate retrieval, and saved top-k outputs.
- Dense encoders use generic `encode`; query/document roles are currently separated only for multi-vector models. Model loading defaults to mean pooling. A faithful model benchmark needs explicit role/prompt handling and an option to preserve the checkpoint's configured pooling.
- Resampling evaluates already-scored test subsets at a supplied threshold. It does not fit a detector on each training fold. Quantiles of fold metrics are labeled `ci_*`; these should be described as empirical variation until an appropriate uncertainty procedure is specified. They must not be presented as confidence intervals for generalization performance by default.

These are design constraints for the new benchmark interface. The concrete data-corruption and missing-query defects are addressed separately in the review PRs.

## What to take from MTEB

MTEB separates task discovery, curated benchmark membership, model selection, execution, and stored results. It supports selecting benchmarks/tasks and saving predictions with model provenance. This is a useful workflow to adopt. [Selection documentation](https://docs.mteb.org/get_started/usage/selecting_tasks/), [evaluation documentation](https://docs.mteb.org/get_started/usage/running_the_evaluation/).

The inspected MTEB source revision is `146b12817197a4abf3b3b84ca016382fd1b61081`. Its task metadata includes a dataset revision, evaluation splits, main score, languages, license and annotation provenance. Its result schema stores task/dataset/evaluator identity and scores per split/subset, and guards merging results across dataset revisions. [Task metadata](https://github.com/embeddings-benchmark/mteb/blob/146b12817197a4abf3b3b84ca016382fd1b61081/mteb/abstasks/task_metadata.py), [result schema](https://github.com/embeddings-benchmark/mteb/blob/146b12817197a4abf3b3b84ca016382fd1b61081/mteb/results/task_result.py).

MTEB already has `MTEB(Code, v1)`, including code search, code-to-code retrieval, CodeTransOcean, feedback, and text-to-SQL tasks. It is not a complete plagiarism or clone-localization benchmark. Matheel can add those missing task types while offering an optional MTEB bridge for overlapping retrieval tasks. [Benchmark definition](https://github.com/embeddings-benchmark/mteb/blob/146b12817197a4abf3b3b84ca016382fd1b61081/mteb/benchmarks/benchmarks/benchmarks.py).

Recommendation: use a Matheel-native task and detector contract. Keep MTEB optional for interoperability and reference comparisons. Making MTEB mandatory would add dependencies and still leave collection detectors, fragment matching, submission boundaries, and plagiarism provenance to implement.

### Proposed workflow

Illustrative API and CLI only; these do not exist yet:

```python
benchmark = matheel.get_benchmark("Matheel-Functional-v1")
detector = matheel.get_detector("sentence-transformers", model="...", revision="...")
results = matheel.evaluate(detector, benchmark, output_dir="results", resume=True)
```

```text
matheel tasks list --category functional_similarity --language python
matheel benchmark run --suite Matheel-CrossLanguage-v1 --detector configured-model
matheel benchmark compare results/baseline results/candidate
```

The underlying sequence should be:

1. Resolve a task definition and immutable source revision; acquire data only when explicitly requested.
2. Normalize once, validate IDs/labels/metadata, and freeze source/corpus/split hashes.
3. Check detector capabilities; prepare model/index or collection once. Keep gold labels on the evaluator side of new detector APIs.
4. Produce predictions in bounded batches, recording successful, missing, unsupported, and failed items explicitly.
5. Evaluate with the task's fixed protocol and write predictions, per-query/per-pair results, metrics, timings, and provenance.
6. Resume only matching completed work. Compare runs only if their task identities and protocol hashes agree.

A task definition needs `task_id`, task and adapter versions, source and normalized-data hashes, category, languages/directions, granularity, official or derived split provenance, grouping unit, main metric and parameters, negative sampling, self-match exclusion, judgment completeness, and license/access information. Record model revision, pooling, prompts, tokenizer, truncation, precision, dependencies, detector version, seed, hardware, and effective runtime options in each run.

Version suite membership and weighting. First show per-task scores and coverage; aggregate only matching tasks within the same category. Report failures separately rather than silently reducing the denominator. An optional common-task comparison should show the reduced coverage and must not masquerade as the full official suite.

### Protocols and metrics

| Task | Proposed primary measurement | Required controls |
| --- | --- | --- |
| Labeled pair classification | Average precision, with F1 at a development-selected threshold | Report class balance and threshold provenance; preserve official evaluation protocols separately |
| Code retrieval | Task-defined nDCG@10, MRR or MAP@R | Fixed corpus/candidates, exact cutoff, no self-match, deterministic ties and missing-query handling |
| Clone discovery/localization | Recall of reference fragments at a fixed overlap rule | Preserve file/line/token spans; incomplete clone annotations cannot establish unrestricted precision |
| Human plagiarism cases | Case/submission-level precision and recall | Assignment, author/case grouping, base-code policy; preserve the dataset's actual labels |
| Transformation robustness | Change from the original task score, by transformation and severity | Independent source groups, validated variants, frozen generator and seeds |

Implement MAP@R explicitly before claiming POJ-104 compatibility. Its official protocol excludes the query itself and retrieves other solutions to the same problem. Current untruncated MAP is a different metric. [Official POJ-104 evaluator and protocol](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Clone-detection-POJ-104/README.md).

Split by original program/clone family, problem, project, assignment or author as appropriate. Related variants and connected positive pairs must remain in one partition. Preserve official splits for reproduction and offer separately named group-disjoint variants after auditing leakage. Resample queries or independent source groups for uncertainty; freeze development-selected thresholds before evaluating the test partition. Publish the procedure and sampling unit alongside intervals.

## Dataset taxonomy

Use several independent fields instead of one exclusive label:

| Dimension | Examples |
| --- | --- |
| Relation being evaluated | plagiarism, syntactic clone, functional similarity, translation alignment, code search |
| Task | pair classification, retrieval, clone discovery, localization, clustering |
| Languages | per-file language; source→target direction; monolingual or cross-language |
| Granularity | fragment, function, file, multi-file submission, project |
| Origin | observed human examples, curated pairs, mutation-generated, LLM-generated, translated |
| Evidence | human annotation, shared problem, lineage, tests passed, tool-assisted filtering |
| Clone category | T1/T2/T3/T4 only when the source supports the annotation; otherwise unknown |
| Access/provenance | source version, license, source-code licenses, distribution limits, grouping and lineage |

Cross-language is an axis, not another clone type. A corpus containing Java and Python is not automatically a Java→Python benchmark. Solving the same programming problem is useful functional relevance, but it is not evidence of copying. Passing finite tests is supporting evidence, not a proof of program equivalence.

### Candidate additions

The tiers below are recommendations, not implementation commitments. Adapter means local normalization plus a documented acquisition route; it does not mean bundling the corpus with the MIT package. Sizes and language support must be pinned to the exact chosen subset at implementation time.

| Dataset | Category and language scope | Proposed use | Recommendation and constraints |
| --- | --- | --- | --- |
| [POJ-104 / CodeXGLUE](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Clone-detection-POJ-104/README.md) | Human contest solutions, C/C++; same-problem functional retrieval | First MAP@R reference task | First wave. Keep official problem splits and evaluator; exclude self. POJ-104 and OJClone derivatives are not independent evidence. Verify underlying data terms separately from repository code. |
| [Project CodeNet](https://github.com/IBM/Project_CodeNet) | Human contest submissions across 55 languages, including curated C++/Java/Python benchmarks | Larger monolingual tasks; accepted-solution cross-language pilot | First wave as a bounded subset. Use its near-duplicate and identical-problem metadata, accepted status, and problem groups. Full cross-language pairing is a new derived protocol. |
| [XLCoST](https://github.com/reddy-lab-code-research/XLCoST) | Parallel snippets/programs in C, C++, C#, Java, JavaScript, PHP and Python | First explicit directional cross-language retrieval task | First wave, begin Java↔Python and Java↔C++. Keep program and snippet tracks separate; preserve aligned-group splits. Repository license is Apache-2.0; record source-code attribution/terms too. |
| [BigCloneBench v2 / BigCloneEval](https://github.com/clonebench/BigCloneBench) | Java clone fragments from IJaDataset | Classical clone discovery/localization and recall | Separate compatibility suite. Benchmark is CC BY-NC 4.0; source files retain their licenses. Use official fragment matching; do not turn every unannotated pair into a negative. |
| [CodeXGLUE BigCloneBench subset](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Clone-detection-BigCloneBench/README.md) | Derived Java binary pair benchmark | Reproduce published pair-classification experiments | Separate task ID and provenance from full BCB. Official binary F1. Do not count both versions as independent datasets in an aggregate. |
| [SemanticCloneBench](https://harvest.usask.ca/bitstream/10388/13413/1/AL-OMARI-DISSERTATION-2021.pdf) | Curated semantic pairs in C, C#, Java and Python | Semantic challenge set | Second wave. Author dissertation describes 4,000 pairs. Preserve origin/attribution and verify archive terms. Positive pairs alone do not supply a validated negative class. |
| [GPTCloneBench](https://github.com/srlabUsask/GPTCloneBench) | LLM-generated semantic and cross-language pairs in C, C#, Java and Python | Separate synthetic challenge track | Manual/import track pending permitted use. Upstream lists Attribution-NonCommercial-NoDerivatives terms. Preserve generator and validation provenance; do not redistribute adapted data by default. |
| [CLCDSA](https://github.com/Kawser-nerd/CLCDSA) and [C4](https://github.com/Chenning-Tao/C4) | Cross-language clone research artifacts | Reproduction of published language-pair protocols | Second wave investigation. Pin actual subsets, verify grouping and negative construction. CLCDSA has no detected repository license; artifact access/terms remain unresolved. |
| [CodeTransOcean](https://github.com/WeixiangYAN/CodeTransOcean) / [MTEB-CoIR adaptation](https://github.com/embeddings-benchmark/mteb/blob/146b12817197a4abf3b3b84ca016382fd1b61081/mteb/tasks/retrieval/code/code_trans_ocean_contest_retrieval.py) | Translation-derived and cross-framework code retrieval | Interoperability task | Second wave. Give each adaptation a separate identity and inspect actual query/corpus content; do not infer direction from its title. MTEB pins a revision and nDCG@10 for its contest adaptation. |
| [CodeContests](https://github.com/google-deepmind/code_contests) | Correct/incorrect human contest solutions with test cases | Functional hard negatives and variant validation | Later. Overlaps source families with CodeNet; incorrect solutions are not automatically non-clones. Keep functional correctness distinct from syntactic/lineage similarity. |
| [PROGpedia](https://zenodo.org/records/7449056) | Educational submissions to 16 exercises, plus code property graphs | Student-program source corpus and controlled augmentation | Later. The source corpus is not itself a complete plagiarism-pair oracle; freeze chosen versions and inspect labels/access terms before defining a scored task. |
| [CodeSearchNet](https://github.com/github/CodeSearchNet) | Natural-language/code search in six programming languages | Optional MTEB bridge | Adjacent scope. Label NL→code retrieval explicitly; it does not measure code-clone or plagiarism detection. Preserve per-source licensing. |

BCB should not be the headline semantic score: a recent primary study documents substantial problems with using its weak syntactic categories as semantic ground truth. Keep its intended clone-recall use and expose this limitation in task cards. [Krinke and Ragkhitwetsagul, 2025](https://arxiv.org/abs/2505.04311).

GPTCloneBench's “false semantic” category includes Type-1/Type-2 pairs. Those are not generic non-clones; map the labels according to the task's precise definition. [Author paper](https://arxiv.org/abs/2308.13963).

Before promoting any candidate to a stable task, require a pinned sample layout, source/license review, reproducible adapter, label audit, language/group metadata, duplicate/overlap report, baseline reproduction, download-size estimate, and a tiny offline fixture. Keep source license and wrapper-code license separate. No new external dataset has been downloaded or benchmarked during this review.

## Adapters for plagiarism and classical clone detectors

Yes, these are feasible. The current `PairAlgorithm` hook is useful for functions returning pair scores, but collection detectors should run once on a collection and return their native findings. They should not be called separately for every pair, and missing findings must not automatically become numeric zero scores.

| Tool | Integration boundary | Priority / caveat |
| --- | --- | --- |
| [JPlag](https://github.com/jplag/JPlag) | Optional local process; import pairwise CSV and versioned `.jplag` report data | First. Preserve submission boundaries, base code, both-direction coverage and options. Current main requires Java 25 and is GPL-3.0; pin a tested release/runtime. Verify exports are complete despite report limits. |
| [NiCad 7](https://github.com/CordyJ/Open-NiCad) | Optional local process or XML clone-pair/class import | First classical detector. Preserve original spans, file/function/block granularity and normalization config. Requires OpenTxl 11+ and compiler setup; cache extraction separately from detection. |
| [SourcererCC](https://github.com/Mondego/SourcererCC) | Tokenization/index/query runner plus reported block-pair importer | Later for scalability. Map its block IDs back to source spans; isolate its toolchain rather than adding it to the default Python environment. |
| [jscpd](https://github.com/kucherenko/jscpd) | Versioned CLI/report importer | Useful later lightweight baseline with broad syntax support. Syntax support across languages does not establish cross-language semantic matching. |
| [MOSS](https://theory.stanford.edu/~aiken/moss/) | Import an existing report first; opt-in remote submission wrapper later | Separate service track. Requires a user account, uploads code, and enforces 100 submissions/day/user. Its server version is not under our control. Preserve matched lines and directional percentages; never describe its score as proof of plagiarism. |
| Other legacy tools | Versioned output import when a reliable artifact exists | Defer DECKARD/CCFinder-family/Simian/Plaggie runners until licensing, installation, output mapping, and a reproducible test case are checked. |

Use an optional detector package/entry-point registry with declared capabilities: pair scoring, retrieval, collection discovery, span reporting, language support, determinism, batching and local/service execution. Maintain distinct prediction types for scored pairs, ranked candidates, and clone matches. Store native scores and the documented conversion, rather than forcing every detector to emit an artificial probability.

A run must record version/binary hash, argv/options, environment/container digest where available, input ID mapping, runtime, errors, report checksum, completeness/threshold limits, and parsed output version. Separate missing, filtered, unsupported and failed comparisons. Offer candidate-restricted reranking as a distinct task from full-corpus discovery. Compare span detectors using an explicit overlap rule, following established evaluation practice. [BigCloneEval](https://github.com/jeffsvajlenko/BigCloneEval).

Start with report-import fixtures and contract tests, then one real local run per supported pinned tool. Do not bundle tool binaries or send code to an external service as part of normal evaluation. This review assessed interfaces; it did not run JPlag, NiCad or MOSS.

## Synthetic clones and transformations

Integrate this as a generator/validator subsystem feeding a separate robustness suite. Keep originals immutable and record each variant's parent, transformation sequence, tool version, seed, language, hashes, and validation status. Split source groups before generating variants so descendants cannot leak into another partition.

| Approach/tool | Fit | Recommendation |
| --- | --- | --- |
| Matheel-owned deterministic transforms | Formatting/comments, scope-aware local renaming, restricted structural edits | First small feature. Use syntax-aware transforms, documented preconditions, and tiny owned programs with tests. Preserve semantically significant comments/directives. |
| [Mutation and Injection Framework](https://github.com/jeffsvajlenko/MutationInjectionFramework) | Purpose-built clone generation and clone-detector evaluation; fragment coordinates | Best research integration candidate after the native pilot. Requires TXL, astyle and GNU utilities; begin by importing its generated corpus and lineage. |
| [Artifice through OCD](https://github.com/UCL-CREST/OCD) | Java source obfuscation in a published comparison framework | Later reproduction track; validate the legacy toolchain first. OCD also distinguishes bytecode obfuscation and decompilation from source transformation. |
| [javascript-obfuscator](https://github.com/javascript-obfuscator/javascript-obfuscator) | Programmatic/CLI JavaScript source transformations with seeded options | Optional later adapter. Freeze options and runtime; require execution checks for transformed fixtures. |
| [Mossad](https://arxiv.org/abs/2010.01700) | Research on generated plagiarism variants and detector robustness | Research-only candidate. An installable, maintained artifact/API was not verified here; do not promise a working integration. |
| [GPTCloneBench generation](https://github.com/srlabUsask/GPTCloneBench) / other LLM transformations | Semantic rewrites and translation candidates | Later opt-in plugin. Persist outputs and model/prompt provenance; judge independently. A request for “equivalent code” is not a label validator. |
| [SemSeed](https://github.com/sola-st/SemSeed) | Realistic bug seeding | Candidate for functional near-miss negatives, not positive equivalent clones. Its older JavaScript/Python environment belongs in an isolated runner. |

Use validation levels: generated → parsed → compiled (where applicable) → passed tests → reviewed. Compilation alone does not validate behavior; finite tests support a limited claim. Restrict execution to isolated, time-limited runners with no network or secrets. Keep invalid variants and failures in provenance, outside the positive benchmark set. Negative variants require evidence of a behavioral difference, rather than merely a different text or a failed build.

Initially avoid packing/encryption/bytecode-only obfuscation: these change the representation being measured. Keep LLM-generated, mutation-generated and observed-human results in distinct report slices. Do not optimize generation against the same detector and then present the resulting dataset as an unbiased comparison across detectors.

## Proposed release scope and decision gates

| Feature track | Minimum useful deliverable | Recommendation |
| --- | --- | --- |
| Versioned benchmark runner | Task/suite registry, immutable protocol, saved predictions, replay/resume, comparison checks | Required foundation |
| Categorized datasets and cross-language evaluation | New metadata model; POJ-104 MAP@R and XLCoST directional tasks; tiny offline fixtures; bounded CodeNet pilot if validated | Required |
| External detector adapters | Typed result import, one JPlag runner/importer and one NiCad importer/runner with native metadata | Required, extend #46 |
| Controlled clone-generation workflow | Small deterministic transform set, lineage, validation and per-transform report | Proposed fourth feature, keep narrow |
| MTEB bridge, large discovery and extra corpora | Cross-check overlapping retrieval scores; scalable indexing; broader adapters | Follow-up after the contracts stabilize |

Implementation order: define the task/metadata/result contracts, implement one end-to-end reference benchmark, add the two dataset tasks, prove JPlag/NiCad imports, then add the narrow robustness workflow and surface it in Gradio. Preserve old APIs during migration; label the existing collection comparison as exploratory inspection. Do not publish a global “best detector” ranking across incompatible task categories.

Decisions needed before committing feature implementation:

1. Confirm this evaluation-framework direction and the four proposed feature tracks.
2. Choose the smallest release dataset set: recommendation is POJ-104 + XLCoST, retaining existing plagiarism presets; CodeNet is a bounded follow-up if it would delay release readiness.
3. Confirm JPlag and NiCad first, with MOSS import/service support staged separately.
4. Confirm the generator pilot stays deterministic and local, with LLM generation deferred.

Release readiness requires the bug-fix PRs to land, reference-metric parity, no known split/lineage leakage in chosen tasks, reproducible CPU smoke runs, real pinned-tool adapter checks, memory/runtime measurements on a representative corpus, documented license/access routes, UI parity, and green package/CI checks. Set the version and release date only after these gates and scope decisions are settled.
