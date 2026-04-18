Stats 305C Project Instructions

You are helping me with my graduate Stats 305C applied statistics project. I am currently on Milestone 2 (Data and Exploratory Analysis). 

---

## INSTRUCTIONS FOR MILESTONE 2, FOLLOW THESE AS GOLD GUIDELINES

Milestone 2 — Data and Exploratory Analysis (Week 4) This milestone will be completed by each team of two.
Obtain and explore your data:
Data description. Document the provenance, format, and preprocessing steps. Include summary statistics and any data cleaning decisions.
Exploratory analysis. Visualize the data. Describe patterns, anomalies, and distributional properties. What do you observe? What remains unexplained?
Refined model sketch. Based on your exploratory data analysis (EDA), sharpen your description of the model you plan to build. What distributional assumptions seem reasonable? What latent structure might be present
 Deliverable: A 1–2 page PDF with your most important figures and written responses to sections above, and a link to your public GitHub repository. Your repository should have several commits documenting your progress over this two-week period, including the Jupyter notebooks used to perform the analyses.

## MY ORIGINAL PROPOSAL  (PLAN HAS CHANGED SINCE)

#### Interpretable models for pretraining data quality**

**Scientific or applied question:** Modern LLMs are trained on web text filtered by a “quality classifier” that scores each document and discards most. This turns out to be one of the highest-leverage decisions in pretraining: DataComp-LM (DCLM) showed that filter choice moves downstream benchmark scores significantly (Li et al., 2024), and hosts a data filtering competition around it. Interestingly, their best filter is a shallow bag-of-bigrams fastText classifier trained to distinguish high-scoring r/ExplainLikeImFive posts and instruction-formatted data (OpenHermes 2.5) from random web text. This simple classifier beats perplexity filtering from a 154M Transformer, which beats an LLM judge (AskLLM), which beats a PageRank filter, and even beats human judgement of quality. Outside of DCLM, FineWeb-Edu and RefinedWeb use their own filtering practices. All these filters claim to measure "quality," but there is no good intuition behind which filters work and which don't. We propose building an interpretable probabilistic model of qualities that filters select for: **do different quality filters agree on what makes a document high-quality, and where do they systematically diverge?** Stakeholders include the open-source LLM community and downstream users of LLMs (data filtering can have downstream effects, like the ethnic and sexual orientation biases found in the C4 analysis (Dodge et al., 2021)).

**Statistical Framing:** One way to model this is to fit a joint hierarchical Bayesian model of multiple filter scores on the same documents. Each filter's score is decomposed into components from document style, topic, and source domain, plus a filter-specific residual. Posterior covariances between filters at each level answer the scientific question directly, and the residuals show what each filter uniquely rewards. Probabilistic modeling is the right tool here because the question is a variance decomposition across a hierarchy with a heavily imbalanced domain distribution: most domains have very few documents, so partial pooling is needed for reliable domain-level estimates, and posterior uncertainty is needed to tell real filter disagreement apart from noise. Other sources of uncertainty include ambiguity in the topic clustering and filter noise itself.

**Data source:** I will use a **small subset** of DCLM-Pool (mlfoundations/dclm-pool-400m-1x on HuggingFace, 469B tokens total); each doc comes with raw text, source URL, and CC metadata. I’ll directly apply DCLM's released tooling for rule-based filtering and BFF deduplication, but no model-based filtering, then score each document with multiple filters. These filters may include DCLM’s fastText OH-2.5+ELI5 classifier and a perplexity filter from a small released LM (scope permitting, may also include other fastText variants, AskLLM scores, or the FineWebEdu filter). A few known issues: the domain distribution is heavily long-tailed, with most registered domains appearing only a few times, so sample size is a tradeoff between covering rare domains and the cost of LLM-based scoring (if necessary, I could avoid LLM-based scoring). Raw Common Crawl contains substantial spam at the low end even after heuristic filtering, which will likely produce bimodal or heavy-tailed score distributions.

**Related work:** I build on DCLM (Li et al., 2024), whose artifacts I use directly. Penedo et al., 2024 develops FineWeb-Edu, an alternative pipeline trained on Llama-3-70B selecting for educational content. Wettig et al., 2024 (QuRating) rates pretraining documents along interpretable axes (writing style, required expertise, facts, educational value) and shows these correlate with downstream quality, but does so by training new rating models from scratch rather than probabilistically modeling existing SOTA filters. My proposal was also inspired by Dodge et al., 2021 which flagged many phenomena of C4; for instance, from PCA analysis that it removed science, medical, legal, and disproportionately African American and Hispanic content – motivating interpretable analysis of what filters discard beyond empirical performance.

Dodge, J., Sap, M., Marasovic, A., Agnew, W., Ilharco, G., Groeneveld, D., Mitchell, M., & Gardner, M. (2021). Documenting large webtext corpora: A case study on the Colossal Clean Crawled Corpus. _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_, 165–182.

Li, J., Fang, A., Smyrnis, G., Ivgi, M., Jordan, M., Gadre, S., Bansal, H., Guha, E., Keh, S., Oh, K., Thrush, T., Vishniakov, K., Bhatt, P., Jiang, H., Garg, S., Wan, T., Ammanabrolu, P., Barak, L., Bitton, Y., … Schmidt, L. (2024). DataComp-LM: In search of the next generation of training sets for language models. _arXiv preprint arXiv:2406.11794_.

Penedo, G., Kydlíček, H., allal, L. B., Lozhkov, A., Mitchell, M., Raffel, C., Von Werra, L., & Wolf, T. (2024). The FineWeb datasets: Decanting the web for the finest text data at scale. _arXiv preprint arXiv:2406.17557_.

Wettig, A., Gao, T., Zettlemoyer, L., & Chen, D. (2024). QuRating: Selecting high-quality data for training language models. _arXiv preprint arXiv:2402.09739_


## MODIFICATIONS TO PLAN

I'm building a hierarchical Bayesian model that jointly decomposes multiple quality filter scores on web pretraining data. The scientific question: **do different quality filters agree on what makes a document high-quality, and where do they systematically diverge?**

I decompose filter score variance across topic, format, URL domain, and document-level features to understand what each filter actually measures. This extends the WebOrganizer paper (Wettig et al., ICML 2025) which showed quality filters implicitly change domain proportions but didn't model continuous scores, joint filter structure, or within-domain effects.

---

## MY DATA

I'm using the **WebOrganizer Corpus-200B** dataset (`WebOrganizer/Corpus-200B` on HuggingFace). It's a 200B-token cleaned/deduped subset of CommonCrawl from the DCLM 1b-1x pool. It has 9888 shards.

The dataset is already downloaded (or being downloaded) to my local filesystem. It comes with these pre-computed annotations as `.npy` files aligned with the JSONL document shards:

- `scores_dclm-fasttext/` — DCLM quality score per doc (continuous)
- `scores_fineweb-edu/` — FineWeb-Edu score per doc (0-5 scale)
- `domains_topics/` — top-1 topic choice (24 categories) + logits
- `domains_formats/` — top-1 format choice (24 categories) + logits
- `domains_clusters-k24/` — k-means cluster assignments
- `tokens/` — GPT-NeoX token counts per doc
- `documents/` — compressed JSONL shards (`.jsonl.zst`) with raw text + URLs

### Subsampling

I need to work with a **1% shard-level random subsample** (~100 out of 9888 shards). This preserves the within-shard distribution perfectly and is reproducible. When you write the subsampling script:

- Set a fixed random seed and save it
- Randomly select 100 shard indices from `range(9888)` without replacement
- Save the list of sampled shard indices to `data/sampled_shards.json`
- All subsequent scripts should only operate on these sampled shards

---

## WHAT I NEED YOU TO HELP ME BUILD

### Part 1: Feature Extraction Pipeline

I need scripts that extract the following features for all documents in my sampled shards. Prioritize efficiency - bundle extractions that require decompressing JSONL into a single pass per shard. PLEASE test them on single doc or low-volume documents first, before running for super long time! i really want to make sure we iterate on debugging quickly. 

#### 1A. Load existing annotations (just load and concatenate .npy files for sampled shards)

- DCLM-fasttext scores
- FineWeb-Edu scores
- Topic labels (choice) and logits
- Format labels (choice) and logits
- K-means cluster assignments (k=24)
- Token counts

#### 1B. Extract from JSONL (requires decompressing `.jsonl.zst` — do in ONE pass per shard)

**URL domains:**
- Parse `url` field with `urllib.parse.urlparse(url).netloc`
- Strip `www.` prefix
- Build a vocabulary mapping (domain string → integer ID) and save it
- Save integer-encoded domain array per shard

**Heuristic text features** (compute from `text` field, all CPU, fast):

| Feature | How to compute |
|---------|---------------|
| `char_count` | `len(text)` |
| `word_count` | `len(text.split())` |
| `mean_word_length` | mean of `len(w)` for words |
| `frac_alpha` | fraction of chars that are alphabetic |
| `frac_digit` | fraction of chars that are digits |
| `frac_punctuation` | fraction of chars that are punctuation |
| `frac_uppercase` | fraction of alpha chars that are uppercase |
| `frac_lines_terminal_punct` | fraction of lines ending in `.?!"` |
| `frac_lines_bullet` | fraction of lines starting with `-`, `*`, `•`, or `\d+\.` |
| `type_token_ratio` | unique words / total words |
| `stopword_fraction` | stop words / total words (use a standard English stopword list) |
| `ngram_rep_2` | fraction of text covered by repeated 2-grams |
| `ngram_rep_3` | fraction of text covered by repeated 3-grams |
| `ngram_rep_4` | fraction of text covered by repeated 4-grams |
| `num_lines` | number of `\n`-separated lines |
| `mean_line_length` | mean characters per line |

These are the features used by Gopher/MassiveText, C4, and RefinedWeb as heuristic filters. Even though the corpus has already been filtered, there's still variance in these features within the filtered population. I need these as covariates in my Bayesian model to explain what quality classifiers respond to beyond domain membership.

#### 1C. Run additional quality filters

**FastText-based filters (CPU-only, ~2000 docs/sec, embarrassingly parallel):**

Download and run these 4 perplexity-correlations fastText filters from HuggingFace (Thrush et al., ICLR 2025). Each outputs a continuous probability score per document.

- `perplexity-correlations/fasttext-arc-easy-target`
- `perplexity-correlations/fasttext-piqa-target`
- `perplexity-correlations/fasttext-sciq-target`
- `perplexity-correlations/fasttext-lambada-target`

These are benchmark-specific filters. Each was trained to select data that helps a specific downstream benchmark, using correlations across 90 public LLMs. They require no reference corpus.

For each model: load the fastText binary, call `model.predict(text)` on each document's text, extract the probability of the positive class, save as `.npy` per shard. The text needs newlines replaced with spaces before passing to fastText.

**QuRater-1.3B (GPU, A100-40G, moderate cost):**

Download `princeton-nlp/QuRater-1.3B` from HuggingFace. This is a 1.3B parameter model that scores documents on 4 interpretable axes:
- `writing_style`
- `required_expertise`
- `facts_and_trivia`
- `educational_value`

Use the `qurater_annotate.py` script from the QuRating GitHub repo (`princeton-nlp/QuRating`). It takes JSONL input and outputs per-document scores. Save 4 `.npy` arrays per shard. This is the only GPU-intensive annotation. Submit as Slurm jobs, ~15 concurrent A100 GPUs available. Should take a few hours total across my sampled shards.

#### 1D. Compute derived features (no models needed, just arithmetic)

**Within-topic quality percentiles:**
For each quality score (DCLM, FineWeb-Edu, and each of the 4 perplexity-correlation scores), compute each document's percentile rank WITHIN its assigned topic. This means: group documents by their topic label, then within each group, compute the percentile of each document's score.

This captures OLMo 3's key insight that quality is topic-relative. A document about "Adult" content with DCLM score 0.3 might be 90th percentile for that topic.

Also compute within-format percentiles for the same scores.

---

### Part 2: EDA Notebooks (Milestone 2 deliverable)

Build Jupyter notebooks with the following analyses. These serve two purposes: (a) basic sanity checks, and (b) motivating the hierarchical model by showing patterns that simple analysis can't fully explain.

#### Notebook 1: Sanity Checks

1. Histogram of each of the 10 quality scores (2 existing + 4 perplexity-corr + 4 QuRater). Check for NaNs, infinities, clipping, bimodality, heavy tails. Note which are approximately Gaussian vs. heavily skewed — this informs my likelihood choice for the Bayesian model later.
2. Bar charts of topic and format distributions in the subsample. Verify they roughly match WebOrganizer's published proportions (sanity check that subsampling preserved distribution).
3. Histogram of token counts (log-scale).
4. URL domain Zipf plot: log-log plot of URL domain frequency vs. rank. Report how many domains have >100 docs, >1000 docs. This motivates partial pooling.
5. Histograms of all 16 heuristic features. Flag any with near-zero variance or extreme skew.

#### Notebook 2: Bivariate Relationships

6. Pairwise correlation heatmap of all 10 quality scores. Key question: which filters correlate most? Do the 4 perplexity-correlation filters cluster together? Does QuRater-educational_value correlate with FineWeb-Edu?
7. Violin/box plots of each quality score broken out by the 24 topics. Show that topic-level means differ dramatically across filters.
8. Same as 7 but for 24 formats.
9. Scatter plots of quality scores against key heuristic features (doc length, repetition rate, type-token ratio). Do some filters correlate with length and others don't?
10. Scatter plot of raw DCLM score vs. within-topic DCLM percentile, colored by topic. This shows "quality" means different things in different topics. Do for FineWeb-Edu too.

#### Notebook 3: Hierarchy-Specific EDA

11. **Simple variance decomposition (ANOVA-style, NOT Bayesian yet):** For each quality score, compute R² from: (a) topic only, (b) format only, (c) topic + format, (d) topic + format + top-50 URL domains, (e) topic + format + URL domains + heuristic features. Display as a stacked bar chart showing incremental variance explained at each level. This is the frequentist preview of what my Bayesian model will do properly.
12. **Cross-filter correlation at each level:**
    - Compute topic-level means of each score → correlate across filters (do filters agree on which topics are good?)
    - Compute within-topic residuals → correlate across filters (do filters agree on which docs are good WITHIN a topic?)
    - Same for format-level
13. For the top ~50 most frequent URL domains, compute mean quality score per domain per filter. Show as a heatmap.
14. Rank agreement: for each pair of filters, compute Spearman rank correlation (a) globally and (b) within each topic. Show as a heatmap. Does within-topic rank agreement differ from global?

---

### Part 3: Model Sketch (section of Milestone 2 report)

Based on the EDA, I need to write a refined model sketch. The model I'm planning:

**Multivariate Hierarchical Linear Model:**

For each filter k and document i:

y_ik = α_k^(topic)[t_i] + α_k^(format)[f_i] + α_k^(domain)[d_i] + x_i^T β_k + ε_ik

- α_k^(topic) ~ shared prior across topics (partial pooling)
- α_k^(domain) ~ shared prior across URL domains (partial pooling — this is where it matters most, rare domains borrow strength)
- x_i are heuristic features
- ε_ik ~ multivariate normal across filters k (captures cross-filter residual correlations)
- Scores may need transformation (log, logit, etc.) based on EDA marginals

Help me refine this based on what the EDA reveals.

---

## IMPORTANT CONSTRAINTS

- I have ~2 days for all of Milestone 2 (feature extraction + EDA + report)
- I can submit ~15 concurrent single or multi-GPU (A100-40G) Slurm jobs
- I have substantial CPU resources for parallel jobs
- All code should be well-documented and reproducible
- Save random seeds, shard lists, and all intermediate artifacts
- Use standard Python scientific stack: numpy, pandas, matplotlib, seaborn, scipy, sklearn
- For Slurm scripts, use standard `sbatch` with job arrays where possible
- The n-gram repetition features are the most expensive heuristic to compute — optimize them (e.g., use collections.Counter, not naive nested loops)

## IMPLEMENTATION ADVICE

We'll probably have to:
1. Understanding the Corpus-200B file structure on my filesystem 
2. Writing `00_subsample_shards.py`
3. Writing `02_extract_jsonl_features.py` (this is the most complex script — single-pass JSONL decompression + URL parsing + all 16 heuristic features)
4. Then proceed script by script in order

However, I always want to run small debugs first (like doing processing on 1 document) before doing super long jobs, so make sure to debug quickly!!
Always ask me before making major design decisions. Show me the plan before writing long scripts.

## FULL PROJECT INSTRUCTIONS
Milestone 2 — Data and Exploratory Analysis (Week 4) This milestone will be completed by each team of two.
Corresponding to: “Collect data”

Obtain and explore your data:

Data description. Document the provenance, format, and preprocessing steps. Include summary statistics and any data cleaning decisions.

Exploratory analysis. Visualize the data. Describe patterns, anomalies, and distributional properties. What do you observe? What remains unexplained?

Refined model sketch. Based on your EDA, sharpen your description of the model you plan to build. What distributional assumptions seem reasonable? What latent structure might be present?

Deliverable: A 1–2 page report with your most important figures, and a link to your public GitHub repository. Your repository should have several commits documenting your progress over this two-week period, including the Jupyter notebooks used to perform the analyses.

Milestone 3 — Model and Inference (Week 6)
Corresponding to: “Build a model” and “Perform inference” This milestone will be completed by each team of two.

Implement a probabilistic model and fit it to your data:

Model specification. Write out the full generative model (prior and likelihood). Justify your choices.

Inference algorithm. Implement at least one inference method covered in the course (MCMC, VI, EM, etc.). Justify your choice and report convergence diagnostics.

Posterior analysis. Summarize and visualize the posterior. What do the results tell you about your problem?

Baseline comparison. Compare your probabilistic model to a simpler baseline. Does the added complexity help? Be honest.

Deliverable: A 1–2 page report with your most important figures. Your GitHub repository should have several commits documenting your progress, including the code and notebooks used to generate your results.

Milestone 4 — Criticism and Revision (Week 8)
Corresponding to: “Criticize the model” and “Revise” This milestone will be completed by each team of two.

Critically evaluate and improve your model:

Posterior predictive checks. Does the model generate data that looks like your real data? Identify specific failures.

Sensitivity analysis. How sensitive are your conclusions to prior choices or modeling assumptions?

Revision. Make at least one substantive improvement motivated by your criticism — a revised likelihood, a different prior, an alternative algorithm, or a different model family.

Comparison. Report results for both the original and revised models. Which is better, and by what criterion?

Deliverable: A 1–2 page report with your most important figures. Your GitHub repository should have several commits documenting your progress, including the code and notebooks used to generate your results.

Lab Meetings
In the weeks following each milestone submission, the class periods will be devoted to lab meetings — flipped-classroom sessions that simulate the rhythm of a real research group. The class is split into two groups (Monday and Wednesday teams); you will be assigned to one group and should attend only your assigned meeting. The lab meeting format is as follows.

For Milestone 1, each individual will present their pitch with 1 slide, maximum 2 minutes, with 2 minutes for discussion after.

For Milestones 2-4:

Each team presents one slide (≈ 3 minutes) summarizing their milestone: what they did, what they found, and what questions or problems they encountered.

Your peers give feedback (≈ 5 minutes) asking questions, suggesting ideas, and pointing out potential issues.

The teaching staff will facilitate the discussion.

Participation in lab meetings is a major component of your grade. This includes both presenting clearly and engaging substantively with other teams’ work.

Final Report
The final report is due on the same day our final would usually be held. It is an 8-page research paper written in the AISTATS format, as if you were submitting to a machine learning conference. It should synthesize all four milestones into a coherent, self-contained document.

Required sections:

Introduction. Motivate the problem and summarize your findings.

Data. Describe your dataset and key results from your exploratory analysis.

Model. Specify your probabilistic model and justify your choices.

Inference. Describe your inference algorithm and report diagnostics.

Results. Summarize and interpret the posterior. Address your scientific question.

Discussion. Reflect on what worked, what didn’t, and what you would do differently. Be honest about limitations.

Responsible AI Use (if applicable). See AI policy below.

Code must be in a public GitHub repository linked from the report. The commit history should reflect iterative development throughout the quarter — a repository with a single commit made the night before the deadline is a red flag.

Length: 8 pages, excluding references. Appendices are not allowed.

AI Use Policy
The use of LLMs and AI coding assistants (ChatGPT, Claude, GitHub Copilot, etc.) is permitted and in some cases encouraged. These tools can accelerate data processing, help debug code, suggest modeling approaches, and assist with writing. Learning to use AI effectively is itself a valuable skill.

However, the scientific judgment must be yours: choosing a meaningful question, deciding what data is appropriate, evaluating whether a model makes sense, and interpreting what your results actually mean. The lab meetings exist precisely to surface this distinction — shallow AI-generated analysis tends to fall apart under questioning.

Getting Started
A few suggestions for choosing a good problem:

Look at papers you already find interesting. If you are in a research group, talk to your advisor about available datasets and open questions. If not, browse recent issues of journals in a field you care about (neuroscience, economics, epidemiology, ecology) and look for studies that used statistical methods.

Prioritize real data over synthetic. Simulated data can be incredibly useful for validating your methods but it removes the messiness that makes applied statistics interesting and difficult. Start with real data, then use simulated data to isolate and test specific aspects of your model and algorithm.

Scope down early. A narrowly focused analysis of one dataset is better than a broad comparison across many.

The answer can be “the simple method works fine.” Some of the most useful applied statistics papers conclude that a simple model fits the data well and complex extensions are not warranted. That is a valid and valuable conclusion.

If you are stuck, the teaching staff will help you brainstorm during office hours in the first few weeks. Come with a general domain in mind and we will work from there.



TRANSCRIPT OF PROJECT PLANNING Q&A SESSION
===========================================

ROUND 1: Initial framing
-------------------------

USER: I wrote this project report for my graduate applied stats305c class. However, I realized that Organize the Web already implements analysis of FineWebEdu and DCLM filtering, but also leaves the individual document modeling open/not done. Can you explain how I can still make a meaningful project in addition to what Organize the Web did?

CLAUDE's key points in response:

- Your proposal and WebOrganizer are complementary — they overlap on the question but use fundamentally different methods and operate at different levels of granularity.

- What WebOrganizer already covers: It answers "how do filters change the domain mixture?" by looking at marginal proportions of documents in each category before and after filtering (Figure 4), and shows FineWeb-Edu's gains can be ~84% recovered by replicating its implicit Topic x Format mixture with random sampling within domains. That's a descriptive, frequentist accounting at the level of 24 coarse categories.

- What it explicitly leaves open — and where your project fits: WebOrganizer's Table 2 reveals that DCLM-fasttext is NOT well-approximated by domain effects (only 35% of its average gain is recovered), meaning there's substantial within-domain selection happening that their framework doesn't model. The paper acknowledges this directly, noting aspects like misspellings or factual errors that aren't captured by their taxonomy.

- Your hierarchical Bayesian approach fits this gap. Concrete angles:

  1. Continuous score decomposition rather than binary selection. WebOrganizer treats filtering as binary in/out. Your proposal models the continuous filter scores themselves and decomposes their variance across levels of a hierarchy. That tells you not just that Science & Technology gets upsampled, but how much score variance is explained by topic membership versus URL domain versus document-level idiosyncrasies.

  2. Partial pooling on URL domains. WebOrganizer doesn't attempt this at all. They note the Zipfian distribution and treat it as a reason to use coarser categories. Your hierarchical model with partial pooling can actually estimate URL-domain-level effects reliably despite small samples.

  3. Joint posterior covariances between filters. WebOrganizer compares filters side-by-side visually but never formally models the joint structure. Your Bayesian model would produce posterior covariances between filters at each hierarchical level.

- Practical path forward: Reframe your project not as a competitor to WebOrganizer but as building on it. Use their released topic and format annotations as known covariates in your hierarchical model. Your contribution becomes: "WebOrganizer showed that domain mixing explains much of FineWeb-Edu but little of DCLM-fasttext. We use a hierarchical Bayesian variance decomposition to explain where the remaining signal lives — is it URL-domain-specific, or truly document-level? And do the filters' residuals correlate with each other or capture genuinely different notions of quality?"


ROUND 2: Milestone 2 planning + initial questions
--------------------------------------------------

USER: I'm currently on Milestone 2, the visualizing data step. Could you help me plan out what exactly to visualize? We should prioritize BASIC checks here, nothing fancy with modeling yet!! I have only 2 days to do the visualization.

CLAUDE asked the following critical questions before proceeding:

1. How much data have you actually downloaded? The full Corpus-200B is enormous. Are you working with all shards or a subsample?

2. URL domains — your project proposal hinges on partial pooling over URL domains as a key hierarchical level. Extracting URL domains means you'd need to decompress the .jsonl.zst files and parse the URLs. Have you done this yet, or is this still TODO?

3. Which filters are in scope? Just the two provided (DCLM-fasttext + FineWeb-Edu), or are you planning to score documents with additional filters?

4. Subsample strategy — for EDA, do you plan to load all shards' .npy files and just skip the full text, or do you want to work with a random subset of shards where you also read the JSONL?

5. Do you have a project partner?

CLAUDE proposed a rough tiered plan:

- Tier 1 — Sanity / marginals (a few hours): Histograms of each score, domain label frequencies, token length distribution, checking for NaNs/weird values.

- Tier 2 — Bivariate relationships (half a day): Joint distribution of the two scores, score distributions broken out by topic and format, checking whether score distributions are bimodal or heavy-tailed within domains.

- Tier 3 — Hierarchy-specific EDA (the part unique to your project, ~1 day): URL domain extraction, Zipf plot, simple variance decomposition (ANOVA-style, not Bayesian yet), within-domain correlations between the two filters.


ROUND 3: User answers + user's own questions
---------------------------------------------

USER's answers:

1. Already started downloading the 200B tokens, 87G downloaded in 30min. Downloading won't be a problem, but how much data should I actually be working with? The Organize the Web paper used 200B as their analysis. Is that practical?

2. No URL domain parsing has been done yet — it's TODO. Would need to decompress each .jsonl.zst shard, extract the url field, parse the hostname.

3. I want to add new meaningful filters, see below!

4. What do you recommend I do to subsample without changing the distribution? It needs to be done in a reputable way, so it's reproducible and publishable.

5. Ignore this (no partner consideration needed).

USER's own questions:

1. Let's think big picture again. Why specifically do we need to do it by URL? I was also thinking for my project we would find additional properties of "quality" not captured by domain and topic. How will that happen? Is it helpful for me to extract some NLP-based notions of quality from the individual docs? Is that realistic or will it take too long? I want to prepare ALL the data that could even potentially be helpful now, during the data collection phase beyond the simple summary statistics already provided. I want to examine the data thoroughly and extract properties!

2. Also big picture, I was wondering if I could ablate over different training of quality filters and see if that changes anything. But this may be a big step bc then I'd need to train hella quality filters in different ways.

3. What would actually help me get meaningful, publishable results beyond what Organize the Web did? I don't want to just do by URL, that's not that interesting anyways. I was thinking of testing newer filters that came out after DCLM, like Olmo 2 or Olmo 3 filters as well.


ROUND 4: Claude's research + comprehensive response
----------------------------------------------------

CLAUDE searched the web for: OLMo 2/3 pretraining data filtering, new quality classifiers released in 2024-2025, perplexity correlations paper (Thrush et al.), Data Quality Illusion paper, QuRating, NVIDIA quality classifier.

CLAUDE's response to User Q1 (Why URL, and what else?):

- URL domains alone aren't that exciting. The reason URL domains matter for your hierarchical model is as a middle level that demonstrates partial pooling working — but it's a means, not the end.

- What's much more interesting is adding additional quality axes as covariates/responses.

- Very feasible (CPU-only, fast): Perplexity-correlations fastText filters (Thrush et al., ICLR 2025) — released on HuggingFace as fastText binaries targeting specific benchmarks (ARC-easy, LAMBADA, PIQA, etc.). As fast as DCLM-fasttext (~2000 docs/sec on CPU). Gold for your project because they give you benchmark-specific quality scores. Multiple new continuous scores basically for free.

- Feasible (single GPU, a few hours per shard): QuRater (Wettig et al., ICML 2024 — same first author as WebOrganizer!) — a 1.3B model scoring documents on 4 interpretable axes: writing style, required expertise, facts & trivia, educational value. Tailor-made for your project. These interpretable dimensions are exactly the "aspects of quality not captured by domain" that WebOrganizer's Section 5 calls out.

- Probably too expensive for 2 days: Training your own filters / ablating over filter training.

- Strong recommendation: Add the perplexity-correlations fastText filters (trivial compute) and QuRater's 4 axes (moderate compute). This gives you ~8 continuous quality scores per document instead of 2.

CLAUDE's response to User Q2 (Ablating over filter training):

- Strongly advise against this for a Stats 305C project. Training quality filters requires curating reference datasets, training classifiers, and validating them — that's a separate paper. What you CAN do that's almost as informative is compare existing filters that were trained in fundamentally different ways. DCLM was trained on OpenHermes+ELI5. FineWeb-Edu was distilled from Llama-3-70B prompts. QuRater uses pairwise LLM judgments. Perplexity-correlations uses no training data at all. Comparing these diverse approaches is more informative than ablating one approach.

CLAUDE's response to User Q3 (What makes this publishable):

- The "Data Quality Illusion" paper (Nait Saada et al., Oct 2025) is extremely relevant — argues CQF doesn't capture meaningful "quality" and different HQ reference sets align with different benchmarks. Your project can be the statistical machinery that makes this precise. WebOrganizer showed domain effects. Data Quality Illusion showed filters are task-specific. Your hierarchical model would be the first to jointly decompose multiple filter scores and quantify exactly where they agree (topic-level? format-level?) and where they diverge (document-level residuals?), and whether the "interpretable" axes from QuRater explain the residual variance that topic/format don't.

CLAUDE's data size recommendation:

- You do NOT need 200B tokens for a statistical analysis. WebOrganizer used 200B because they trained 1B-param LMs. You're fitting a Bayesian model to filter scores, not training a language model. Recommended a stratified random subsample of ~1-2% of shards (~2-4B tokens, millions of documents). A shard-level subsample is the cleanest approach: randomly select N shards, use all documents in those shards. Preserves within-shard distribution perfectly and is trivially reproducible.


ROUND 5: Claude asks follow-up questions
-----------------------------------------

CLAUDE asked:

1. How many total shards are there in Corpus-200B? → USER answered: 9888 shards

2. GPU availability for scoring: How many concurrent GPU jobs? → USER answered: concurrently like 15, A100s, 40G memory

3. Perplexity-correlations filters: Which benchmarks do you care about? → USER answered: as diverse as possible, probably simplest is English only

4. What's your Bayesian modeling plan at a high level? → USER answered: not sure, can you explain the different ways of modeling possible? I will experiment with multiple.

5. Scope check: continuous scores or binary selection decisions? → USER answered: continuous scores


ROUND 6: User asks for more filters + modeling explanation
-----------------------------------------------------------

USER's questions:

1a. Can you look deeper and broader for more human-interpretable scores or heuristics that I can score the docs on, even simple ones? Would simple ones help?

1b. More easily available filters? Right now only having DCLM and FineWebEdu won't be that informative. What can I even tell about filters with just those two?

2. I'm a little confused about the bayesian hierarchical model. What does it give me? I can see how it gives a correlation matrix between different filters and notions of quality. However, does it require/necessarily use any of the info provided by how good the filter is at all? Is there any way to incorporate that? But this will be very sparse data as I only have a few filters.

CLAUDE's response on simple heuristic features:

- Yes, absolutely, and they're arguably the most important thing to extract. Reason: your hierarchical model wants to decompose filter scores into components. The interesting scientific question is: what do these filters actually respond to? If you can show that 40% of DCLM-fasttext's variance is explained by document length + repetition rate + topic, while only 15% of FineWeb-Edu's variance is, that's a publishable finding. Simple features are COVARIATES in your model, not competing filters.

- Listed ~16 heuristic text features: doc length, mean word length, frac alpha/digit/punct/uppercase, frac lines with terminal punctuation, frac lines with bullets, type-token ratio, stopword fraction, n-gram repetition ratios (2/3/4-gram), num lines, mean line length. These are the exact features used by Gopher/MassiveText, C4, and RefinedWeb filters.

- Why these help the model: In the Bayesian decomposition, these become document-level covariates. Instead of just "topic + format + URL domain + residual," you model the filter score as "topic effect + format effect + URL domain effect + β₁·doc_length + β₂·repetition + ... + residual." The β coefficients tell you what each filter rewards beyond domain membership. E.g., "DCLM-fasttext rewards shorter documents within every domain, but FineWeb-Edu doesn't care about length."

CLAUDE's response on more filters — comprehensive inventory:

- Already in data (free): DCLM-fasttext, FineWeb-Edu
- FastText-based CPU-only: 4 perplexity-correlations filters (ARC-easy, PIQA, SciQ, LAMBADA)
- GPU-based moderate cost: QuRater-1.3B (4 interpretable scores)
- GPU-based cheap: NVIDIA quality-classifier-deberta (3-class)
- Would NOT bother with: OLMo 2/3 doesn't release a standalone quality filter you can apply. Training your own filters. LLM-based scoring (AskLLM) too slow.

- Strong recommendation: Use filters 1-4 above. That gives 10 continuous quality scores from 4 methodologically distinct families: (1) DCLM: bigram classifier trained on curated reference text, (2) FineWeb-Edu: neural classifier distilled from LLM judgments, (3) Perplexity-correlations: statistical approach using no reference data, (4) QuRater: pairwise LLM judgment on 4 interpretable axes. This diversity of construction methods is what makes the comparison scientifically interesting.

CLAUDE's response on the Bayesian model — what it gives you:

- Option A: Multivariate hierarchical linear model (recommended). For each filter k and document i: y_ik = α_k^(topic)[t_i] + α_k^(format)[f_i] + α_k^(domain)[d_i] + x_i^T β_k + ε_ik. Key outputs: (1) Variance decomposition: what fraction of each filter's variance is explained by topic vs. format vs. URL domain vs. heuristics vs. residual, (2) Cross-filter correlation at each level: do topic effects correlate across filters? Do residuals?, (3) β coefficients: what document-level features does each filter reward?

- Option B: Factor model on residuals. After fitting Option A, take residuals and fit a factor model. If strong first principal component → "shared quality beyond domain effects." If not → filters measure fundamentally different things at document level.

- Option C: Latent class model. Model documents as belonging to latent "quality classes" with different score profiles. Less interpretable but could reveal natural clusters.

- Recommendation: start with A, do B as follow-up, consider C only if time.

- Does it incorporate filter performance? Not directly, and that's fine. Not enough filters to do meaningful regression on "filter quality." What you CAN do is interpret results in light of known performance differences. E.g., if WebOrganizer showed FineWeb-Edu's domain effects explain 84% of gains while DCLM's explain only 35%, and your model shows FineWeb-Edu has high variance at topic level but low residual variance while DCLM has the opposite — that's a coherent story connecting your decomposition to downstream performance.


ROUND 7: User asks about OLMo 3 strategy
------------------------------------------

USER: Could you consider adding a "filter" that recreates OLMo 3's strategy? Basically what Olmo 3 did on top of DCLM baseline was: partition corpus into 24 topics using the topic classifier, score quality with their own released quality classifier, bucket docs by within-topic quality percentiles into 5% intervals. However this can get noisy with only 100k docs. To recreate this, probably do like topic classifier + Olmo's quality classifier + maybe drop bottom 40% within each topic, no upsampling. OR could just annotate every doc with topic and quality percentile within topic, without actually selecting. Then compare how DCLM and FineWebEdu selections redistribute topics relative to the full pool!!

CLAUDE researched OLMo 3 extensively and found:

- OLMo 3 / Dolma 3 used: (1) WebOrganizer topic classifier distilled to fastText, (2) fastText quality classifier trained on OpenHermes-2.5, ELI5, UltraChat-200k, WildChat-1M as positives, (3) documents partitioned by topic, then within each topic, quality percentiles computed and divided into vigintile (5%) buckets, (4) quality-aware upsampling: top 5% upsampled ~7x, bottom 40% discarded.

CLAUDE's key recommendation: You don't actually need a separate OLMo 3 classifier at all. What makes OLMo 3's approach interesting is NOT the specific classifier (which is basically a variant of DCLM-fasttext with a slightly expanded positive set — they added UltraChat-200k and WildChat-1M on top of OpenHermes+ELI5), but rather the within-topic quality percentile conditioning strategy. You can compute this from data you already have.

What to compute (zero additional model cost): For each document, you already have its topic label and its DCLM-fasttext score (and FineWeb-Edu score). Compute the within-topic quality percentile for each filter. This gives two new derived features per document.

Why this is scientifically interesting: The within-topic percentile decomposes the raw quality score into a topic-level mean effect and a within-topic residual. Consider a document about "Adult" content that gets a DCLM score of 0.3. Is that "low quality"? Not necessarily — it might be in the 90th percentile for Adult content.

Killer EDA visualization: scatter plot of raw quality score vs. within-topic quality percentile, colored by topic. If all topics have similar score distributions, the dots fall on a single curve. If they don't (they won't), different topics form distinct bands — directly showing why OLMo 3's approach works.

Additional concrete things to compute:
1. Within-topic percentile of DCLM-fasttext for each document
2. Within-topic percentile of FineWeb-Edu for each document
3. Rank correlation between the two within-topic percentiles — tells you whether filters agree on which docs are good within a topic, even if they disagree on which topics are good overall
4. For each topic, what fraction of its documents would survive a top-15% global DCLM threshold vs. a top-15% within-topic threshold? Shows the implicit topic reweighting that global filtering creates.

Extension beyond OLMo 3: also compute within-FORMAT percentiles. OLMo 3 only conditions on topic, but your project has format labels too. You could show whether conditioning on format, or on topic x format jointly, reveals additional structure.

Stats 305C Project Instructions

You are helping me with my graduate Stats 305C applied statistics project. I am currently on Milestone 2 (Data and Exploratory Analysis). 

---

## INSTRUCTIONS FOR MILESTONE 2, FOLLOW THESE AS GOLD GUIDELINES

Milestone 2 — Data and Exploratory Analysis (Week 4) This milestone will be completed by each team of two.
Obtain and explore your data:
Data description. Document the provenance, format, and preprocessing steps. Include summary statistics and any data cleaning decisions.
Exploratory analysis. Visualize the data. Describe patterns, anomalies, and distributional properties. What do you observe? What remains unexplained?
Refined model sketch. Based on your exploratory data analysis (EDA), sharpen your description of the model you plan to build. What distributional assumptions seem reasonable? What latent structure might be present
 Deliverable: A 1–2 page PDF with your most important figures and written responses to sections above, and a link to your public GitHub repository. Your repository should have several commits documenting your progress over this two-week period, including the Jupyter notebooks used to perform the analyses.

## MY ORIGINAL PROPOSAL  (PLAN HAS CHANGED SINCE)

#### Interpretable models for pretraining data quality**

**Scientific or applied question:** Modern LLMs are trained on web text filtered by a “quality classifier” that scores each document and discards most. This turns out to be one of the highest-leverage decisions in pretraining: DataComp-LM (DCLM) showed that filter choice moves downstream benchmark scores significantly (Li et al., 2024), and hosts a data filtering competition around it. Interestingly, their best filter is a shallow bag-of-bigrams fastText classifier trained to distinguish high-scoring r/ExplainLikeImFive posts and instruction-formatted data (OpenHermes 2.5) from random web text. This simple classifier beats perplexity filtering from a 154M Transformer, which beats an LLM judge (AskLLM), which beats a PageRank filter, and even beats human judgement of quality. Outside of DCLM, FineWeb-Edu and RefinedWeb use their own filtering practices. All these filters claim to measure "quality," but there is no good intuition behind which filters work and which don't. We propose building an interpretable probabilistic model of qualities that filters select for: **do different quality filters agree on what makes a document high-quality, and where do they systematically diverge?** Stakeholders include the open-source LLM community and downstream users of LLMs (data filtering can have downstream effects, like the ethnic and sexual orientation biases found in the C4 analysis (Dodge et al., 2021)).

**Statistical Framing:** One way to model this is to fit a joint hierarchical Bayesian model of multiple filter scores on the same documents. Each filter's score is decomposed into components from document style, topic, and source domain, plus a filter-specific residual. Posterior covariances between filters at each level answer the scientific question directly, and the residuals show what each filter uniquely rewards. Probabilistic modeling is the right tool here because the question is a variance decomposition across a hierarchy with a heavily imbalanced domain distribution: most domains have very few documents, so partial pooling is needed for reliable domain-level estimates, and posterior uncertainty is needed to tell real filter disagreement apart from noise. Other sources of uncertainty include ambiguity in the topic clustering and filter noise itself.

**Data source:** I will use a **small subset** of DCLM-Pool (mlfoundations/dclm-pool-400m-1x on HuggingFace, 469B tokens total); each doc comes with raw text, source URL, and CC metadata. I’ll directly apply DCLM's released tooling for rule-based filtering and BFF deduplication, but no model-based filtering, then score each document with multiple filters. These filters may include DCLM’s fastText OH-2.5+ELI5 classifier and a perplexity filter from a small released LM (scope permitting, may also include other fastText variants, AskLLM scores, or the FineWebEdu filter). A few known issues: the domain distribution is heavily long-tailed, with most registered domains appearing only a few times, so sample size is a tradeoff between covering rare domains and the cost of LLM-based scoring (if necessary, I could avoid LLM-based scoring). Raw Common Crawl contains substantial spam at the low end even after heuristic filtering, which will likely produce bimodal or heavy-tailed score distributions.

**Related work:** I build on DCLM (Li et al., 2024), whose artifacts I use directly. Penedo et al., 2024 develops FineWeb-Edu, an alternative pipeline trained on Llama-3-70B selecting for educational content. Wettig et al., 2024 (QuRating) rates pretraining documents along interpretable axes (writing style, required expertise, facts, educational value) and shows these correlate with downstream quality, but does so by training new rating models from scratch rather than probabilistically modeling existing SOTA filters. My proposal was also inspired by Dodge et al., 2021 which flagged many phenomena of C4; for instance, from PCA analysis that it removed science, medical, legal, and disproportionately African American and Hispanic content – motivating interpretable analysis of what filters discard beyond empirical performance.

Dodge, J., Sap, M., Marasovic, A., Agnew, W., Ilharco, G., Groeneveld, D., Mitchell, M., & Gardner, M. (2021). Documenting large webtext corpora: A case study on the Colossal Clean Crawled Corpus. _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_, 165–182.

Li, J., Fang, A., Smyrnis, G., Ivgi, M., Jordan, M., Gadre, S., Bansal, H., Guha, E., Keh, S., Oh, K., Thrush, T., Vishniakov, K., Bhatt, P., Jiang, H., Garg, S., Wan, T., Ammanabrolu, P., Barak, L., Bitton, Y., … Schmidt, L. (2024). DataComp-LM: In search of the next generation of training sets for language models. _arXiv preprint arXiv:2406.11794_.

Penedo, G., Kydlíček, H., allal, L. B., Lozhkov, A., Mitchell, M., Raffel, C., Von Werra, L., & Wolf, T. (2024). The FineWeb datasets: Decanting the web for the finest text data at scale. _arXiv preprint arXiv:2406.17557_.

Wettig, A., Gao, T., Zettlemoyer, L., & Chen, D. (2024). QuRating: Selecting high-quality data for training language models. _arXiv preprint arXiv:2402.09739_


## MODIFICATIONS TO PLAN

I'm building a hierarchical Bayesian model that jointly decomposes multiple quality filter scores on web pretraining data. The scientific question: **do different quality filters agree on what makes a document high-quality, and where do they systematically diverge?**

I decompose filter score variance across topic, format, URL domain, and document-level features to understand what each filter actually measures. This extends the WebOrganizer paper (Wettig et al., ICML 2025) which showed quality filters implicitly change domain proportions but didn't model continuous scores, joint filter structure, or within-domain effects.

---

## MY DATA

I'm using the **WebOrganizer Corpus-200B** dataset (`WebOrganizer/Corpus-200B` on HuggingFace). It's a 200B-token cleaned/deduped subset of CommonCrawl from the DCLM 1b-1x pool. It has 9888 shards.

The dataset is already downloaded (or being downloaded) to my local filesystem. It comes with these pre-computed annotations as `.npy` files aligned with the JSONL document shards:

- `scores_dclm-fasttext/` — DCLM quality score per doc (continuous)
- `scores_fineweb-edu/` — FineWeb-Edu score per doc (0-5 scale)
- `domains_topics/` — top-1 topic choice (24 categories) + logits
- `domains_formats/` — top-1 format choice (24 categories) + logits
- `domains_clusters-k24/` — k-means cluster assignments
- `tokens/` — GPT-NeoX token counts per doc
- `documents/` — compressed JSONL shards (`.jsonl.zst`) with raw text + URLs

### Subsampling

I need to work with a **1% shard-level random subsample** (~100 out of 9888 shards). This preserves the within-shard distribution perfectly and is reproducible. When you write the subsampling script:

- Set a fixed random seed and save it
- Randomly select 100 shard indices from `range(9888)` without replacement
- Save the list of sampled shard indices to `data/sampled_shards.json`
- All subsequent scripts should only operate on these sampled shards

---

## WHAT I NEED YOU TO HELP ME BUILD

### Part 1: Feature Extraction Pipeline

I need scripts that extract the following features for all documents in my sampled shards. Prioritize efficiency - bundle extractions that require decompressing JSONL into a single pass per shard. PLEASE test them on single doc or low-volume documents first, before running for super long time! i really want to make sure we iterate on debugging quickly. 

#### 1A. Load existing annotations (just load and concatenate .npy files for sampled shards)

- DCLM-fasttext scores
- FineWeb-Edu scores
- Topic labels (choice) and logits
- Format labels (choice) and logits
- K-means cluster assignments (k=24)
- Token counts

#### 1B. Extract from JSONL (requires decompressing `.jsonl.zst` — do in ONE pass per shard)

**URL domains:**
- Parse `url` field with `urllib.parse.urlparse(url).netloc`
- Strip `www.` prefix
- Build a vocabulary mapping (domain string → integer ID) and save it
- Save integer-encoded domain array per shard

**Heuristic text features** (compute from `text` field, all CPU, fast):

| Feature | How to compute |
|---------|---------------|
| `char_count` | `len(text)` |
| `word_count` | `len(text.split())` |
| `mean_word_length` | mean of `len(w)` for words |
| `frac_alpha` | fraction of chars that are alphabetic |
| `frac_digit` | fraction of chars that are digits |
| `frac_punctuation` | fraction of chars that are punctuation |
| `frac_uppercase` | fraction of alpha chars that are uppercase |
| `frac_lines_terminal_punct` | fraction of lines ending in `.?!"` |
| `frac_lines_bullet` | fraction of lines starting with `-`, `*`, `•`, or `\d+\.` |
| `type_token_ratio` | unique words / total words |
| `stopword_fraction` | stop words / total words (use a standard English stopword list) |
| `ngram_rep_2` | fraction of text covered by repeated 2-grams |
| `ngram_rep_3` | fraction of text covered by repeated 3-grams |
| `ngram_rep_4` | fraction of text covered by repeated 4-grams |
| `num_lines` | number of `\n`-separated lines |
| `mean_line_length` | mean characters per line |

These are the features used by Gopher/MassiveText, C4, and RefinedWeb as heuristic filters. Even though the corpus has already been filtered, there's still variance in these features within the filtered population. I need these as covariates in my Bayesian model to explain what quality classifiers respond to beyond domain membership.

#### 1C. Run additional quality filters

**FastText-based filters (CPU-only, ~2000 docs/sec, embarrassingly parallel):**

Download and run these 4 perplexity-correlations fastText filters from HuggingFace (Thrush et al., ICLR 2025). Each outputs a continuous probability score per document.

- `perplexity-correlations/fasttext-arc-easy-target`
- `perplexity-correlations/fasttext-piqa-target`
- `perplexity-correlations/fasttext-sciq-target`
- `perplexity-correlations/fasttext-lambada-target`

These are benchmark-specific filters. Each was trained to select data that helps a specific downstream benchmark, using correlations across 90 public LLMs. They require no reference corpus.

For each model: load the fastText binary, call `model.predict(text)` on each document's text, extract the probability of the positive class, save as `.npy` per shard. The text needs newlines replaced with spaces before passing to fastText.

**QuRater-1.3B (GPU, A100-40G, moderate cost):**

Download `princeton-nlp/QuRater-1.3B` from HuggingFace. This is a 1.3B parameter model that scores documents on 4 interpretable axes:
- `writing_style`
- `required_expertise`
- `facts_and_trivia`
- `educational_value`

Use the `qurater_annotate.py` script from the QuRating GitHub repo (`princeton-nlp/QuRating`). It takes JSONL input and outputs per-document scores. Save 4 `.npy` arrays per shard. This is the only GPU-intensive annotation. Submit as Slurm jobs, ~15 concurrent A100 GPUs available. Should take a few hours total across my sampled shards.

#### 1D. Compute derived features (no models needed, just arithmetic)

**Within-topic quality percentiles:**
For each quality score (DCLM, FineWeb-Edu, and each of the 4 perplexity-correlation scores), compute each document's percentile rank WITHIN its assigned topic. This means: group documents by their topic label, then within each group, compute the percentile of each document's score.

This captures OLMo 3's key insight that quality is topic-relative. A document about "Adult" content with DCLM score 0.3 might be 90th percentile for that topic.

Also compute within-format percentiles for the same scores.

---

### Part 2: EDA Notebooks (Milestone 2 deliverable)

Build Jupyter notebooks with the following analyses. These serve two purposes: (a) basic sanity checks, and (b) motivating the hierarchical model by showing patterns that simple analysis can't fully explain.

#### Notebook 1: Sanity Checks

1. Histogram of each of the 10 quality scores (2 existing + 4 perplexity-corr + 4 QuRater). Check for NaNs, infinities, clipping, bimodality, heavy tails. Note which are approximately Gaussian vs. heavily skewed — this informs my likelihood choice for the Bayesian model later.
2. Bar charts of topic and format distributions in the subsample. Verify they roughly match WebOrganizer's published proportions (sanity check that subsampling preserved distribution).
3. Histogram of token counts (log-scale).
4. URL domain Zipf plot: log-log plot of URL domain frequency vs. rank. Report how many domains have >100 docs, >1000 docs. This motivates partial pooling.
5. Histograms of all 16 heuristic features. Flag any with near-zero variance or extreme skew.

#### Notebook 2: Bivariate Relationships

6. Pairwise correlation heatmap of all 10 quality scores. Key question: which filters correlate most? Do the 4 perplexity-correlation filters cluster together? Does QuRater-educational_value correlate with FineWeb-Edu?
7. Violin/box plots of each quality score broken out by the 24 topics. Show that topic-level means differ dramatically across filters.
8. Same as 7 but for 24 formats.
9. Scatter plots of quality scores against key heuristic features (doc length, repetition rate, type-token ratio). Do some filters correlate with length and others don't?
10. Scatter plot of raw DCLM score vs. within-topic DCLM percentile, colored by topic. This shows "quality" means different things in different topics. Do for FineWeb-Edu too.

#### Notebook 3: Hierarchy-Specific EDA

11. **Simple variance decomposition (ANOVA-style, NOT Bayesian yet):** For each quality score, compute R² from: (a) topic only, (b) format only, (c) topic + format, (d) topic + format + top-50 URL domains, (e) topic + format + URL domains + heuristic features. Display as a stacked bar chart showing incremental variance explained at each level. This is the frequentist preview of what my Bayesian model will do properly.
12. **Cross-filter correlation at each level:**
    - Compute topic-level means of each score → correlate across filters (do filters agree on which topics are good?)
    - Compute within-topic residuals → correlate across filters (do filters agree on which docs are good WITHIN a topic?)
    - Same for format-level
13. For the top ~50 most frequent URL domains, compute mean quality score per domain per filter. Show as a heatmap.
14. Rank agreement: for each pair of filters, compute Spearman rank correlation (a) globally and (b) within each topic. Show as a heatmap. Does within-topic rank agreement differ from global?

---

### Part 3: Model Sketch (section of Milestone 2 report)

Based on the EDA, I need to write a refined model sketch. The model I'm planning:

**Multivariate Hierarchical Linear Model:**

For each filter k and document i:

y_ik = α_k^(topic)[t_i] + α_k^(format)[f_i] + α_k^(domain)[d_i] + x_i^T β_k + ε_ik

- α_k^(topic) ~ shared prior across topics (partial pooling)
- α_k^(domain) ~ shared prior across URL domains (partial pooling — this is where it matters most, rare domains borrow strength)
- x_i are heuristic features
- ε_ik ~ multivariate normal across filters k (captures cross-filter residual correlations)
- Scores may need transformation (log, logit, etc.) based on EDA marginals

Help me refine this based on what the EDA reveals.

---

## IMPORTANT CONSTRAINTS

- I have ~2 days for all of Milestone 2 (feature extraction + EDA + report)
- I can submit ~15 concurrent single or multi-GPU (A100-40G) Slurm jobs
- I have substantial CPU resources for parallel jobs
- All code should be well-documented and reproducible
- Save random seeds, shard lists, and all intermediate artifacts
- Use standard Python scientific stack: numpy, pandas, matplotlib, seaborn, scipy, sklearn
- For Slurm scripts, use standard `sbatch` with job arrays where possible
- The n-gram repetition features are the most expensive heuristic to compute — optimize them (e.g., use collections.Counter, not naive nested loops)

## IMPLEMENTATION ADVICE

We'll probably have to:
1. Understanding the Corpus-200B file structure on my filesystem 
2. Writing `00_subsample_shards.py`
3. Writing `02_extract_jsonl_features.py` (this is the most complex script — single-pass JSONL decompression + URL parsing + all 16 heuristic features)
4. Then proceed script by script in order

However, I always want to run small debugs first (like doing processing on 1 document) before doing super long jobs, so make sure to debug quickly!!
Always ask me before making major design decisions. Show me the plan before writing long scripts.

## FULL PROJECT INSTRUCTIONS
Milestone 2 — Data and Exploratory Analysis (Week 4) This milestone will be completed by each team of two.
Corresponding to: “Collect data”

Obtain and explore your data:

Data description. Document the provenance, format, and preprocessing steps. Include summary statistics and any data cleaning decisions.

Exploratory analysis. Visualize the data. Describe patterns, anomalies, and distributional properties. What do you observe? What remains unexplained?

Refined model sketch. Based on your EDA, sharpen your description of the model you plan to build. What distributional assumptions seem reasonable? What latent structure might be present?

Deliverable: A 1–2 page report with your most important figures, and a link to your public GitHub repository. Your repository should have several commits documenting your progress over this two-week period, including the Jupyter notebooks used to perform the analyses.

Milestone 3 — Model and Inference (Week 6)
Corresponding to: “Build a model” and “Perform inference” This milestone will be completed by each team of two.

Implement a probabilistic model and fit it to your data:

Model specification. Write out the full generative model (prior and likelihood). Justify your choices.

Inference algorithm. Implement at least one inference method covered in the course (MCMC, VI, EM, etc.). Justify your choice and report convergence diagnostics.

Posterior analysis. Summarize and visualize the posterior. What do the results tell you about your problem?

Baseline comparison. Compare your probabilistic model to a simpler baseline. Does the added complexity help? Be honest.

Deliverable: A 1–2 page report with your most important figures. Your GitHub repository should have several commits documenting your progress, including the code and notebooks used to generate your results.

Milestone 4 — Criticism and Revision (Week 8)
Corresponding to: “Criticize the model” and “Revise” This milestone will be completed by each team of two.

Critically evaluate and improve your model:

Posterior predictive checks. Does the model generate data that looks like your real data? Identify specific failures.

Sensitivity analysis. How sensitive are your conclusions to prior choices or modeling assumptions?

Revision. Make at least one substantive improvement motivated by your criticism — a revised likelihood, a different prior, an alternative algorithm, or a different model family.

Comparison. Report results for both the original and revised models. Which is better, and by what criterion?

Deliverable: A 1–2 page report with your most important figures. Your GitHub repository should have several commits documenting your progress, including the code and notebooks used to generate your results.

Lab Meetings
In the weeks following each milestone submission, the class periods will be devoted to lab meetings — flipped-classroom sessions that simulate the rhythm of a real research group. The class is split into two groups (Monday and Wednesday teams); you will be assigned to one group and should attend only your assigned meeting. The lab meeting format is as follows.

For Milestone 1, each individual will present their pitch with 1 slide, maximum 2 minutes, with 2 minutes for discussion after.

For Milestones 2-4:

Each team presents one slide (≈ 3 minutes) summarizing their milestone: what they did, what they found, and what questions or problems they encountered.

Your peers give feedback (≈ 5 minutes) asking questions, suggesting ideas, and pointing out potential issues.

The teaching staff will facilitate the discussion.

Participation in lab meetings is a major component of your grade. This includes both presenting clearly and engaging substantively with other teams’ work.

Final Report
The final report is due on the same day our final would usually be held. It is an 8-page research paper written in the AISTATS format, as if you were submitting to a machine learning conference. It should synthesize all four milestones into a coherent, self-contained document.

Required sections:

Introduction. Motivate the problem and summarize your findings.

Data. Describe your dataset and key results from your exploratory analysis.

Model. Specify your probabilistic model and justify your choices.

Inference. Describe your inference algorithm and report diagnostics.

Results. Summarize and interpret the posterior. Address your scientific question.

Discussion. Reflect on what worked, what didn’t, and what you would do differently. Be honest about limitations.

Responsible AI Use (if applicable). See AI policy below.

Code must be in a public GitHub repository linked from the report. The commit history should reflect iterative development throughout the quarter — a repository with a single commit made the night before the deadline is a red flag.

Length: 8 pages, excluding references. Appendices are not allowed.

AI Use Policy
The use of LLMs and AI coding assistants (ChatGPT, Claude, GitHub Copilot, etc.) is permitted and in some cases encouraged. These tools can accelerate data processing, help debug code, suggest modeling approaches, and assist with writing. Learning to use AI effectively is itself a valuable skill.

However, the scientific judgment must be yours: choosing a meaningful question, deciding what data is appropriate, evaluating whether a model makes sense, and interpreting what your results actually mean. The lab meetings exist precisely to surface this distinction — shallow AI-generated analysis tends to fall apart under questioning.

Getting Started
A few suggestions for choosing a good problem:

Look at papers you already find interesting. If you are in a research group, talk to your advisor about available datasets and open questions. If not, browse recent issues of journals in a field you care about (neuroscience, economics, epidemiology, ecology) and look for studies that used statistical methods.

Prioritize real data over synthetic. Simulated data can be incredibly useful for validating your methods but it removes the messiness that makes applied statistics interesting and difficult. Start with real data, then use simulated data to isolate and test specific aspects of your model and algorithm.

Scope down early. A narrowly focused analysis of one dataset is better than a broad comparison across many.

The answer can be “the simple method works fine.” Some of the most useful applied statistics papers conclude that a simple model fits the data well and complex extensions are not warranted. That is a valid and valuable conclusion.

If you are stuck, the teaching staff will help you brainstorm during office hours in the first few weeks. Come with a general domain in mind and we will work from there.



TRANSCRIPT OF PROJECT PLANNING Q&A SESSION
===========================================

ROUND 1: Initial framing
-------------------------

USER: I wrote this project report for my graduate applied stats305c class. However, I realized that Organize the Web already implements analysis of FineWebEdu and DCLM filtering, but also leaves the individual document modeling open/not done. Can you explain how I can still make a meaningful project in addition to what Organize the Web did?

CLAUDE's key points in response:

- Your proposal and WebOrganizer are complementary — they overlap on the question but use fundamentally different methods and operate at different levels of granularity.

- What WebOrganizer already covers: It answers "how do filters change the domain mixture?" by looking at marginal proportions of documents in each category before and after filtering (Figure 4), and shows FineWeb-Edu's gains can be ~84% recovered by replicating its implicit Topic x Format mixture with random sampling within domains. That's a descriptive, frequentist accounting at the level of 24 coarse categories.

- What it explicitly leaves open — and where your project fits: WebOrganizer's Table 2 reveals that DCLM-fasttext is NOT well-approximated by domain effects (only 35% of its average gain is recovered), meaning there's substantial within-domain selection happening that their framework doesn't model. The paper acknowledges this directly, noting aspects like misspellings or factual errors that aren't captured by their taxonomy.

- Your hierarchical Bayesian approach fits this gap. Concrete angles:

  1. Continuous score decomposition rather than binary selection. WebOrganizer treats filtering as binary in/out. Your proposal models the continuous filter scores themselves and decomposes their variance across levels of a hierarchy. That tells you not just that Science & Technology gets upsampled, but how much score variance is explained by topic membership versus URL domain versus document-level idiosyncrasies.

  2. Partial pooling on URL domains. WebOrganizer doesn't attempt this at all. They note the Zipfian distribution and treat it as a reason to use coarser categories. Your hierarchical model with partial pooling can actually estimate URL-domain-level effects reliably despite small samples.

  3. Joint posterior covariances between filters. WebOrganizer compares filters side-by-side visually but never formally models the joint structure. Your Bayesian model would produce posterior covariances between filters at each hierarchical level.

- Practical path forward: Reframe your project not as a competitor to WebOrganizer but as building on it. Use their released topic and format annotations as known covariates in your hierarchical model. Your contribution becomes: "WebOrganizer showed that domain mixing explains much of FineWeb-Edu but little of DCLM-fasttext. We use a hierarchical Bayesian variance decomposition to explain where the remaining signal lives — is it URL-domain-specific, or truly document-level? And do the filters' residuals correlate with each other or capture genuinely different notions of quality?"


ROUND 2: Milestone 2 planning + initial questions
--------------------------------------------------

USER: I'm currently on Milestone 2, the visualizing data step. Could you help me plan out what exactly to visualize? We should prioritize BASIC checks here, nothing fancy with modeling yet!! I have only 2 days to do the visualization.

CLAUDE asked the following critical questions before proceeding:

1. How much data have you actually downloaded? The full Corpus-200B is enormous. Are you working with all shards or a subsample?

2. URL domains — your project proposal hinges on partial pooling over URL domains as a key hierarchical level. Extracting URL domains means you'd need to decompress the .jsonl.zst files and parse the URLs. Have you done this yet, or is this still TODO?

3. Which filters are in scope? Just the two provided (DCLM-fasttext + FineWeb-Edu), or are you planning to score documents with additional filters?

4. Subsample strategy — for EDA, do you plan to load all shards' .npy files and just skip the full text, or do you want to work with a random subset of shards where you also read the JSONL?

5. Do you have a project partner?

CLAUDE proposed a rough tiered plan:

- Tier 1 — Sanity / marginals (a few hours): Histograms of each score, domain label frequencies, token length distribution, checking for NaNs/weird values.

- Tier 2 — Bivariate relationships (half a day): Joint distribution of the two scores, score distributions broken out by topic and format, checking whether score distributions are bimodal or heavy-tailed within domains.

- Tier 3 — Hierarchy-specific EDA (the part unique to your project, ~1 day): URL domain extraction, Zipf plot, simple variance decomposition (ANOVA-style, not Bayesian yet), within-domain correlations between the two filters.


ROUND 3: User answers + user's own questions
---------------------------------------------

USER's answers:

1. Already started downloading the 200B tokens, 87G downloaded in 30min. Downloading won't be a problem, but how much data should I actually be working with? The Organize the Web paper used 200B as their analysis. Is that practical?

2. No URL domain parsing has been done yet — it's TODO. Would need to decompress each .jsonl.zst shard, extract the url field, parse the hostname.

3. I want to add new meaningful filters, see below!

4. What do you recommend I do to subsample without changing the distribution? It needs to be done in a reputable way, so it's reproducible and publishable.

5. Ignore this (no partner consideration needed).

USER's own questions:

1. Let's think big picture again. Why specifically do we need to do it by URL? I was also thinking for my project we would find additional properties of "quality" not captured by domain and topic. How will that happen? Is it helpful for me to extract some NLP-based notions of quality from the individual docs? Is that realistic or will it take too long? I want to prepare ALL the data that could even potentially be helpful now, during the data collection phase beyond the simple summary statistics already provided. I want to examine the data thoroughly and extract properties!

2. Also big picture, I was wondering if I could ablate over different training of quality filters and see if that changes anything. But this may be a big step bc then I'd need to train hella quality filters in different ways.

3. What would actually help me get meaningful, publishable results beyond what Organize the Web did? I don't want to just do by URL, that's not that interesting anyways. I was thinking of testing newer filters that came out after DCLM, like Olmo 2 or Olmo 3 filters as well.


ROUND 4: Claude's research + comprehensive response
----------------------------------------------------

CLAUDE searched the web for: OLMo 2/3 pretraining data filtering, new quality classifiers released in 2024-2025, perplexity correlations paper (Thrush et al.), Data Quality Illusion paper, QuRating, NVIDIA quality classifier.

CLAUDE's response to User Q1 (Why URL, and what else?):

- URL domains alone aren't that exciting. The reason URL domains matter for your hierarchical model is as a middle level that demonstrates partial pooling working — but it's a means, not the end.

- What's much more interesting is adding additional quality axes as covariates/responses.

- Very feasible (CPU-only, fast): Perplexity-correlations fastText filters (Thrush et al., ICLR 2025) — released on HuggingFace as fastText binaries targeting specific benchmarks (ARC-easy, LAMBADA, PIQA, etc.). As fast as DCLM-fasttext (~2000 docs/sec on CPU). Gold for your project because they give you benchmark-specific quality scores. Multiple new continuous scores basically for free.

- Feasible (single GPU, a few hours per shard): QuRater (Wettig et al., ICML 2024 — same first author as WebOrganizer!) — a 1.3B model scoring documents on 4 interpretable axes: writing style, required expertise, facts & trivia, educational value. Tailor-made for your project. These interpretable dimensions are exactly the "aspects of quality not captured by domain" that WebOrganizer's Section 5 calls out.

- Probably too expensive for 2 days: Training your own filters / ablating over filter training.

- Strong recommendation: Add the perplexity-correlations fastText filters (trivial compute) and QuRater's 4 axes (moderate compute). This gives you ~8 continuous quality scores per document instead of 2.

CLAUDE's response to User Q2 (Ablating over filter training):

- Strongly advise against this for a Stats 305C project. Training quality filters requires curating reference datasets, training classifiers, and validating them — that's a separate paper. What you CAN do that's almost as informative is compare existing filters that were trained in fundamentally different ways. DCLM was trained on OpenHermes+ELI5. FineWeb-Edu was distilled from Llama-3-70B prompts. QuRater uses pairwise LLM judgments. Perplexity-correlations uses no training data at all. Comparing these diverse approaches is more informative than ablating one approach.

CLAUDE's response to User Q3 (What makes this publishable):

- The "Data Quality Illusion" paper (Nait Saada et al., Oct 2025) is extremely relevant — argues CQF doesn't capture meaningful "quality" and different HQ reference sets align with different benchmarks. Your project can be the statistical machinery that makes this precise. WebOrganizer showed domain effects. Data Quality Illusion showed filters are task-specific. Your hierarchical model would be the first to jointly decompose multiple filter scores and quantify exactly where they agree (topic-level? format-level?) and where they diverge (document-level residuals?), and whether the "interpretable" axes from QuRater explain the residual variance that topic/format don't.

CLAUDE's data size recommendation:

- You do NOT need 200B tokens for a statistical analysis. WebOrganizer used 200B because they trained 1B-param LMs. You're fitting a Bayesian model to filter scores, not training a language model. Recommended a stratified random subsample of ~1-2% of shards (~2-4B tokens, millions of documents). A shard-level subsample is the cleanest approach: randomly select N shards, use all documents in those shards. Preserves within-shard distribution perfectly and is trivially reproducible.


ROUND 5: Claude asks follow-up questions
-----------------------------------------

CLAUDE asked:

1. How many total shards are there in Corpus-200B? → USER answered: 9888 shards

2. GPU availability for scoring: How many concurrent GPU jobs? → USER answered: concurrently like 15, A100s, 40G memory

3. Perplexity-correlations filters: Which benchmarks do you care about? → USER answered: as diverse as possible, probably simplest is English only

4. What's your Bayesian modeling plan at a high level? → USER answered: not sure, can you explain the different ways of modeling possible? I will experiment with multiple.

5. Scope check: continuous scores or binary selection decisions? → USER answered: continuous scores


ROUND 6: User asks for more filters + modeling explanation
-----------------------------------------------------------

USER's questions:

1a. Can you look deeper and broader for more human-interpretable scores or heuristics that I can score the docs on, even simple ones? Would simple ones help?

1b. More easily available filters? Right now only having DCLM and FineWebEdu won't be that informative. What can I even tell about filters with just those two?

2. I'm a little confused about the bayesian hierarchical model. What does it give me? I can see how it gives a correlation matrix between different filters and notions of quality. However, does it require/necessarily use any of the info provided by how good the filter is at all? Is there any way to incorporate that? But this will be very sparse data as I only have a few filters.

CLAUDE's response on simple heuristic features:

- Yes, absolutely, and they're arguably the most important thing to extract. Reason: your hierarchical model wants to decompose filter scores into components. The interesting scientific question is: what do these filters actually respond to? If you can show that 40% of DCLM-fasttext's variance is explained by document length + repetition rate + topic, while only 15% of FineWeb-Edu's variance is, that's a publishable finding. Simple features are COVARIATES in your model, not competing filters.

- Listed ~16 heuristic text features: doc length, mean word length, frac alpha/digit/punct/uppercase, frac lines with terminal punctuation, frac lines with bullets, type-token ratio, stopword fraction, n-gram repetition ratios (2/3/4-gram), num lines, mean line length. These are the exact features used by Gopher/MassiveText, C4, and RefinedWeb filters.

- Why these help the model: In the Bayesian decomposition, these become document-level covariates. Instead of just "topic + format + URL domain + residual," you model the filter score as "topic effect + format effect + URL domain effect + β₁·doc_length + β₂·repetition + ... + residual." The β coefficients tell you what each filter rewards beyond domain membership. E.g., "DCLM-fasttext rewards shorter documents within every domain, but FineWeb-Edu doesn't care about length."

CLAUDE's response on more filters — comprehensive inventory:

- Already in data (free): DCLM-fasttext, FineWeb-Edu
- FastText-based CPU-only: 4 perplexity-correlations filters (ARC-easy, PIQA, SciQ, LAMBADA)
- GPU-based moderate cost: QuRater-1.3B (4 interpretable scores)
- GPU-based cheap: NVIDIA quality-classifier-deberta (3-class)
- Would NOT bother with: OLMo 2/3 doesn't release a standalone quality filter you can apply. Training your own filters. LLM-based scoring (AskLLM) too slow.

- Strong recommendation: Use filters 1-4 above. That gives 10 continuous quality scores from 4 methodologically distinct families: (1) DCLM: bigram classifier trained on curated reference text, (2) FineWeb-Edu: neural classifier distilled from LLM judgments, (3) Perplexity-correlations: statistical approach using no reference data, (4) QuRater: pairwise LLM judgment on 4 interpretable axes. This diversity of construction methods is what makes the comparison scientifically interesting.

CLAUDE's response on the Bayesian model — what it gives you:

- Option A: Multivariate hierarchical linear model (recommended). For each filter k and document i: y_ik = α_k^(topic)[t_i] + α_k^(format)[f_i] + α_k^(domain)[d_i] + x_i^T β_k + ε_ik. Key outputs: (1) Variance decomposition: what fraction of each filter's variance is explained by topic vs. format vs. URL domain vs. heuristics vs. residual, (2) Cross-filter correlation at each level: do topic effects correlate across filters? Do residuals?, (3) β coefficients: what document-level features does each filter reward?

- Option B: Factor model on residuals. After fitting Option A, take residuals and fit a factor model. If strong first principal component → "shared quality beyond domain effects." If not → filters measure fundamentally different things at document level.

- Option C: Latent class model. Model documents as belonging to latent "quality classes" with different score profiles. Less interpretable but could reveal natural clusters.

- Recommendation: start with A, do B as follow-up, consider C only if time.

- Does it incorporate filter performance? Not directly, and that's fine. Not enough filters to do meaningful regression on "filter quality." What you CAN do is interpret results in light of known performance differences. E.g., if WebOrganizer showed FineWeb-Edu's domain effects explain 84% of gains while DCLM's explain only 35%, and your model shows FineWeb-Edu has high variance at topic level but low residual variance while DCLM has the opposite — that's a coherent story connecting your decomposition to downstream performance.


ROUND 7: User asks about OLMo 3 strategy
------------------------------------------

USER: Could you consider adding a "filter" that recreates OLMo 3's strategy? Basically what Olmo 3 did on top of DCLM baseline was: partition corpus into 24 topics using the topic classifier, score quality with their own released quality classifier, bucket docs by within-topic quality percentiles into 5% intervals. However this can get noisy with only 100k docs. To recreate this, probably do like topic classifier + Olmo's quality classifier + maybe drop bottom 40% within each topic, no upsampling. OR could just annotate every doc with topic and quality percentile within topic, without actually selecting. Then compare how DCLM and FineWebEdu selections redistribute topics relative to the full pool!!

CLAUDE researched OLMo 3 extensively and found:

- OLMo 3 / Dolma 3 used: (1) WebOrganizer topic classifier distilled to fastText, (2) fastText quality classifier trained on OpenHermes-2.5, ELI5, UltraChat-200k, WildChat-1M as positives, (3) documents partitioned by topic, then within each topic, quality percentiles computed and divided into vigintile (5%) buckets, (4) quality-aware upsampling: top 5% upsampled ~7x, bottom 40% discarded.

CLAUDE's key recommendation: You don't actually need a separate OLMo 3 classifier at all. What makes OLMo 3's approach interesting is NOT the specific classifier (which is basically a variant of DCLM-fasttext with a slightly expanded positive set — they added UltraChat-200k and WildChat-1M on top of OpenHermes+ELI5), but rather the within-topic quality percentile conditioning strategy. You can compute this from data you already have.

What to compute (zero additional model cost): For each document, you already have its topic label and its DCLM-fasttext score (and FineWeb-Edu score). Compute the within-topic quality percentile for each filter. This gives two new derived features per document.

Why this is scientifically interesting: The within-topic percentile decomposes the raw quality score into a topic-level mean effect and a within-topic residual. Consider a document about "Adult" content that gets a DCLM score of 0.3. Is that "low quality"? Not necessarily — it might be in the 90th percentile for Adult content.

Killer EDA visualization: scatter plot of raw quality score vs. within-topic quality percentile, colored by topic. If all topics have similar score distributions, the dots fall on a single curve. If they don't (they won't), different topics form distinct bands — directly showing why OLMo 3's approach works.

Additional concrete things to compute:
1. Within-topic percentile of DCLM-fasttext for each document
2. Within-topic percentile of FineWeb-Edu for each document
3. Rank correlation between the two within-topic percentiles — tells you whether filters agree on which docs are good within a topic, even if they disagree on which topics are good overall
4. For each topic, what fraction of its documents would survive a top-15% global DCLM threshold vs. a top-15% within-topic threshold? Shows the implicit topic reweighting that global filtering creates.

Extension beyond OLMo 3: also compute within-FORMAT percentiles. OLMo 3 only conditions on topic, but your project has format labels too. You could show whether conditioning on format, or on topic x format jointly, reveals additional structure.

