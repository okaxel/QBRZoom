# QBRZoom - The Quarterly Business Review Risk Assessment Tool

# 1. QBRZoom Ingestion & Initial Processing

There were 85 messages of 18 projects given. This is a very tiny number therefore I implemented simple for loops to process the given data. As it is not that interesting in the sections above I explain everything from the view of a much larger dataset to provide a meaningful and scalable picture.

**Additional note #01** : In a real-life use case the quality of the given message flows should be improved. E.g. instead of manual collection a built-in solution would be very welcome. Yet a simple ` POP3 ` client can have significant impact on the quality without affecting any user experience.

## 1.1. Data Ingestion Layer

### Event based approach:

- Deploy an event-driven pipeline that watches an email source or a cloud-storage if direct access is not available due to any reasons.
- As soon as a new email text file arrives, fire a lightweight function that publishes a message to a managed queue.

### Micro batches approach

- Orchestrate file-to-queue transfer in micro-batches (e.g., every minute), grouping small attachments to reduce churn.

### Shallow level comparison

| Ingestion Mode | Pros                                  | Cons                              |
| -------------- | ------------------------------------- | --------------------------------- |
| Event-driven   | Near-real-time, cost-efficient at idle | Potential spike under heavy load |
| Micro-batch    | Predictable resource usage            | Slight latency (minutes)          |

## 1.2. Parsing & Metadata Extraction

- Spin up a horizontally scalable Spark or Flink cluster to consume queued file-arrival events.  
- For each email text file:
  - Parse standard headers (Date, From, To, Cc, Subject, Message-ID, Thread-ID).  
  - Decode any inline base64 or quoted-printable attachments.  
  - Assign a unique document ID and thread grouping key.

## 1.3. Text Cleaning & Normalization

- Strip boilerplate (footers, disclaimers) using regex patterns maintained in a reusable library.  
- Normalize encoding (UTF-8), unify line breaks.  
- Optionally remove stop-words and apply lowercasing only if downstream models require it. (For further usability keeping raw email texts in the normalized form can be also useful.)
- Optionally also detect language; route non-English threads through on-the-fly translation.

## 1.4. Feature Engineering & Enrichment

- Chunk large bodies into context windows for parallel NLP.  
- Extract:
  - Named entities (people, projects, dates)  
  - Sentiment scores and emotion flags  
  - Key phrases and action-item patterns  
- Store these features in a columnar store (e.g., Parquet on S3) for high-throughput analytics.

**Additional note #02** : This part is also implemented in the provided code. View the constructor of ` MessageWithFeatures ` class in `qbrz.data ` submodule. Since LLM was implemented aside from entities nothing were used but they can serve as fallback or side sources for an analysis. Entities ware used as part of the privacy protection process.

## 1.5. Indexing & Fast Retrieval

- Index both raw text and extracted metadata into a searchable engine (e.g., Elasticsearch, OpenSearch).  
- Create pre-computed aggregations (per-project risk counts, overdue action items) in a time-series database.
- Create a lessons-learnt database for further use in a potential custom model building process.

## 1.6. Scalability Considerations

- Containerize parsing and feature-extraction services in Kubernetes with HPA (Horizontal Pod Autoscaler).  
- Leverage spot/preemptible nodes for non-latency-critical workloads.  
- Use auto-scaling message queues and cloud storage lifecycle policies to manage retention.
- Monitor pipeline health with end-to-end tracing (Jaeger) and dead-letter queues for failed files.

**Additional note #03** : Some concept from above were implemented in the code. The submodule ` qbrz.data ` is a good place to continue with this topic.

---

## 2 The Analytical Engine (Multi-Step AI Logic)

## 2.1. Attention flags

Selecting the appropriate strategy of attention flags is crucial because it determines the usability of a tool like QBRZoom both from the view of the industry and the characteristics of the tasks and companies.

There are two main approaches of flag selection:

- Fixed flags
- Dynamic flags

### 2.1.1. The fixed flag approach

In case of fixed flags, the use of a tool lik QBRZoom is very limited as the system focuses on errors that are expected by the system's owner or maintainer. This is a good approach if we validate old-school machinery as they always produce the same old errors only the exact time of appearance changes. A good example to this is the problem of physically deployed production chains as their parts always require maintenance only the exact time is a question. An important question though, as any pause costs money because it means loss of production.

Humans mostly adapt to this kind of a system and they learn to "speak" in a way that is "under the radar".

Though this approach has some significant benefits as well:

- It is easy to implement
- It is cheap as it can be managed mostly rule based or with tiny custom models

In case the company maintains an extensive "Lessons Learnt" knowledge base like e.g. NASA does, this approach works very well.

### 2.1.2. The dynamic flag approach

This approach means that flags are grabbed from the textual corpus. At the moment this is possible with any kind of AI models only. An LLM model is not necessarily required but definitely this the fastest solution. However it also has some disadvantages as an LLM even with a well fine grained RAG is quite only generalist for really hard cases.

**Additional note #04** : Dynamic flags are implemented in the submitted code. Consult the submodule ` qbrz.engine ` and especially ` HuggingFaceTool.get_email_risk() ` class member function.

**Additional note #05** : The fact that the smallest model that gave meaningful response to the prompt of the application was  ` meta-llama/Llama-2-7b-chat-hf ` well demonstrates how costly can be a robust and LLM based risk assessment. It is almost infinitely scalable as long as you can pay the bills.

## 2.2. The Engine

**Additional note #06** : An LLM based dynamic flag engine is implemented in the submodule ` qbrz.engine ` .

Potential engine types to consider:

- __Rule based engine__: simple, almost immediately implementable, it has only low effectiveness however as a data visualization tool it makes wonders
- __NLP engine__: almost that simple than rule based, easy-to-implement, in certain situations it is quite effective but overall its effectiveness is low and NLP analyses fail on some points. It is significantly advanced data visualization tool so it can make even larger wonders.
- __Cloud model solutions__: easy-to-implement, seems always promising, performs well till a certain level but high accuracy requires very high level data science skills on-board. It is costly based on usage and on the size of the data science team.
- __Custom model or service of an external provider (non LLM)__: easy-to-implement, may cheap first, but risky based on the life span of the provider and also its effectiveness is a hard-to-prove topic
- __In-house custom model__: limit is the starry sky, in terms of effectiveness, development time and costs as well
- __3rd party LLM based engine__: quite easy to implement on a shallow level and very promising because of immediate "fancy" results. On a longer term a fine-tuned use is a hard task and can cost a lot of money and human resources as well. Hallucination, misunderstandings and generalization are the greatest challenges of LLM based approaches. Also, if provider changes the version, the company may need to develop its entire system from the ground up again.
- __In house hosted 3rd party LLM (open source or closed source)__: on a longer term less costly then the online use, however starting expenses are high. The ultimate advantages are much lower privacy risks, easy to build local RAG and easy-to-manage handling of the conversational history.
- __Custom in-house LLM__: this is way too expensive to consider in most cases.

In case LLM is a must for any reason the use of an online or an in-house model without training is a good starting point. And since we use LLM we can focus prompt engineering.

## 2.3. The Prompt

Not-well-experienced folks are thinking there are only one road to build the best prompts but it is not all true at least based on my experiences. Every LLM powered solutions require different prompt strategies and they react quite differently for the same approach. This is also true if only the version is changed for the same model. It is good to remember especially if a company chains its life to an LLM provider. Though in most cases exactly this is the situation.

### 2.3.1. The prompt scheme I used

``` python

prompt = (f"You are {add_indefinite_article(cls.__role)}. "
           "Given a list of inbound emails, return a JSON object where "
           "each key is the index of the email in the input list, and "
           "each value is an object with:\n"
           "  - risk_score: a number between 0.0 (no risk) and 1.0 (high risk)\n"
           "  - risk_factors: an array of short strings explaining why the email is risky.\n\n"
           f"Input emails:\n{json_dumps(email_texts, indent=2)}\n\n"
           "Output JSON:")

```

**The role** helps the model to figure what our goal is. By changing the role, the results can be significantly changed. Since we expect JSON output, wording does not count that much, but in case a normal textual response would be expected, wording and grammar would significantly changed in some case.

**Output fields**: help the model to create the expected JSON response. However additional parameter ` temperature ` need to be set to zero to expect almost any time real JSON response.

**Input emails**: are sent in the form of a JSON string to help processing.

Overall this is a costly prompt in terms of expected expertise of the models as weaker model cannot respond to this prompt with a meaningful response.

### 2.3.2. Behind the prompt

Aside from the prompt itself, other parameters are also important to consider when using an LLM.

#### Sampling and diversity:

- **temperature**: adjusts randomness.
- **top_k**: limits sampling candidates to the K most probable next tokens.
- **top_p**: aka "nucleus sampling", chooses from the smallest set of tokens whose cumulative probability ≥ p.
- **typical_p**: filters out tokens that deviate too far from the “typical” distribution mass.

#### Length and structure controls:

- **max_length**: maximum number of tokens to generate (including prompt).
- **min_length**: minimum number of tokens before stopping.
- **length_penalty**:  applies a multiplier to beam scores based on length.

#### Repetition and coherence penalties:

- **repetition_penalty**: penalizes tokens that have already appeared.
- **no_repeat_ngram_size**: forbids repeating any n-gram of this size.

#### Search Strategy Parameters:

- **num_beams**: number of beams in beam search.
- **early_stopping**: stop when all beams end.
- **num_return_sequences**: how many distinct outputs to return.
- **do_sample**: whether to use sampling or greedy search.

#### Miscellaneous:

- **seed**: sets the random seed for reproducible sampling.
- **bad_words_ids**: a list of token IDs that the model must avoid entirely.
- **encoder_no_repeat_ngram_size**: applies no-repeat-ngram constraint to the encoded prompt as well.

After reading — or writing down — this list, I always ask myself: ` What is the practical significance of "understanding" in the LLM workflow? `

---

# 3. Cost & Robustness Considerations

Since I decided to use an LLM powered dynamic flag solution to assess risk, I discuss cost and robustness aspects of this approach. However all possible approaches are interesting from the views of these questions (as well).

## 3.1. Cost

As coding design can vary on large scale giving cost assumptions without narrowing it down is far over the limits of this document. Therefore I provide some initial thoughts only.

In case coding capacity is not that expensive for the company, there is a hybrid, multi-level strategy to consider:

- **Lightweight Local Models**: 
   - Use open-source, distilled models (e.g., MiniLM, BART-MNLI) for most zero-shot or simple classification tasks.
   - Run these on on-prem or spot instances at near-zero incremental cost.
- **Mid-Tier Hosted Models**:
  - Reserve cloud LLMs like GPT-3.5-turbo for moderately complex summarization or batch report generation. 
  - Route non time-sensitive, high-volume jobs into off-peak windows (often discounted).
- **Premium LLMs on Demand**:
  - Gate GPT-4 or specialized fine-tuned endpoints only to critical “needs-review” or executive-summary creation.
  - Trigger these sparingly, based on confidence thresholds or manual escalation flags.

With this strategy the company can offer various packages for the clients of QBRZoom for various prices. If QBRZoom is used in-house only, the strategy can help to optimize costs by segmentation of the in-house needs based on importance of departments or time of year.

## 3.2. Robustness

Withe an extensive fallback strategy that covers everything from online LLM to local rule based solution the service uptime can be almost 100 % with high quality outcomes in most cases. Defensive coding and extensive monitoring help at the level to robustness of the code. Monitoring has effects only if there is an automation process or human supervision behind it to use its results for corrections or any other kinds of reactions.

---

# 4. Monitoring & Trust

Maintaining QBRZoom’s reliability isn’t a one-and-done task—it’s an ongoing commitment to monitoring, validation, and feedback loops. Below’s a layered strategy and the key metrics to track:

## 4.1. Continuous Model Performance Validation

- Human-Reviewed Sampling  
  • Randomly sample a fixed % of flagged and non-flagged emails each week.  
  • Have a subject-matter expert (SME) label them “correct” or “incorrect.”  

- Metrics to Track  
  - Precision (flagged & correct / total flagged)  
  - Recall (correct flags / all actual flags)  
  - F1-Score (harmonic mean of precision & recall)  
  - False Positive Rate, False Negative Rate  
  - Calibration Error (are predicted confidences aligned with actual correctness?)

## 4.2. Data and Concept Drift Monitoring

- Input Feature Drift  
  • Monitor statistics (mean, variance, histograms) of key signals: sentiment scores, entity counts, flag rates.  
  • Use divergence metrics (e.g., KL-divergence) to detect when fresh data drifts > threshold.

- Output Stability  
  • Track weekly flag-rate changes per project or per flag type.  
  • Alert if flag rates jump or plummet by X% unexpectedly.

## 4.3. Pipeline Health & Data Quality

- Ingestion Success Rate  
  • % of incoming emails successfully parsed and enriched.  

- Processing Latency  
  • Average and p95 time from file arrival → flagged output.  

- Error & Retry Counts  
  • Number of failures per stage (parsing, enrichment, LLM call).  
  • Ratio of dead-lettered files vs. total files.

## 4.4. Human-in-the-Loop Feedback Integration

- Review Queue Metrics  
  • Volume of “needs_review” items.  
  • SME resolution time (average turnaround).  

- Feedback Utilization  
  • % of SME corrections folded back into the training set.  
  • Improvement in precision/recall post-retraining.

## 4.5. Explainability and Auditability

- Evidence Coverage  
  • % of raised flags with a valid snippet or citation.  

- Explainability Score  
  • SME rating on “transparency” of why a flag was raised (e.g., 1–5 scale).  

- Audit Log Integrity  
  • Ratio of completed logs vs. expected operations.  
  • Frequency of unauthorized access attempts.

## 4.6. Security & Compliance Metrics

- PII Redaction Rate  
  • % of emails containing PII and successfully redacted.  

- Encryption & Decryption Success  
  • Number of HMAC or decryption errors.  

- Access Control Audits  
  • % of API calls with proper RBAC tokens.  
  • Number of role-violation attempts.

## 4.7. Cost versus Impact Tracking

- Cost per Flag  
  • Total compute/API spend divided by # of flags generated.  

- Budget Burn-Down  
  • Monthly AI spend vs. allocated budget.  

- ROI Proxy  
  • Number of high-severity flags leading to prevented escalations or saved hours.

---

# 5. Architectural Risk & Mitigation

In this part I mention only the highest risk factor of the major approaches. There are tons of risks. A serious discussion is far-far behind the limits of this document.

## 5.1. Online LLM approach

Even in a simple design, the biggest architectural risk is depending solely on an external LLM service (e.g., OpenAI’s API) for critical flag detection. If that service experiences downtime, rate limits, cost spikes, or API changes, the entire pipeline grinds to a halt or produces unreliable results.

## 5.2. Local LLM approach

By committing to a fully on-premise, open-source model (e.g., BART-MNLI or a distilled transformer), the single biggest architectural risk shifts to the local model’s ability to maintain accuracy, performance, and scalability over time. Without proper safeguards, model drift, resource constraints, or infra instability can lead to missed flags—or a flood of false positives—undermining trust in QBRZoom.

## 5.3. RUle and simple ML approach

By committing to a hybrid ML / rule-based detection approach, the single biggest architectural risk becomes the dual challenge of **ML model drift** and **rule obsolescence**. If either the classifier’s performance degrades or your handcrafted patterns fall out of sync with evolving email language, QBRZoom will either miss critical flags or overwhelm the Director with false alarms.

**Summary**: layered realization of software like QBRZoom mitigates the greatest technical risks.

---
