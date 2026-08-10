# LogiMind — Usage Playbook

A short guide to asking LogiMind good questions and reading its answers. No code required — this is for anyone using the app, not building it.

Live demo: https://logimind.muhammadawais.dev

## What LogiMind is

LogiMind answers questions about DHL's public operational documents in plain language — rate guides, customs guidelines, packing rules, prohibited/restricted items, incoterms, annual reports, sustainability reports, and DHL's Strategy 2030 documents. It can also look up a (simulated) shipment's tracking status, check a small set of curated compliance rules, and answer a few numeric questions about DHL Group's segment revenue.

Every answer is grounded in something real: a passage from a specific document, a tracking result, a compliance rule, or a number from the dataset. If it doesn't have a real source for something, it says so instead of guessing.

## What it's good at

- **"What does DHL say about X?"** — questions answerable from the ingested documents: prohibited items, packing requirements, incoterms, customs rules, sustainability commitments, company strategy.
- **"Where is my package?"** — give it a tracking number and it returns a status, origin/destination, and estimated delivery. (This is a simulated tracking system for demo purposes, not a live DHL lookup — the same tracking number always returns the same result.)
- **"Can I ship X to Y?"** — a small number of curated compliance rules (lithium batteries, alcohol, perishable food) matched by category and destination country.
- **"What was DHL Group's revenue in [division] in [2023/2024]?"** — a handful of real figures pulled from the 2024 Annual Report.
- **Combined questions** — "Where's my package, and what documentation do I need to ship what's in it to that destination?" LogiMind will look up the tracking first, then use the actual destination it found to check the relevant rule, and combine both into one answer.

## What it won't answer

Anything outside those documents and datasets — general trivia, other companies, or DHL topics the ingested documents don't cover. LogiMind will say plainly that it doesn't have the information, rather than answering from general knowledge. That's deliberate: an answer with no real source behind it is worse than no answer.

## Getting good results

- **Be specific.** "What are the packing requirements for fragile items?" works better than "packing."
- **Give tracking numbers exactly as you have them.** LogiMind extracts them as written.
- **One question at a time reads best**, though a single message can combine a couple of related asks ("where's my package, and can I ship lithium batteries to its destination?").
- **If an answer says it doesn't have the information**, that means the documents genuinely don't cover it — try rephrasing, or ask something the source material is more likely to address.

## Reading a cited answer

Every claim in an answer is tagged with where it came from:

- **`(Document Name, p.N)`** — a passage from an actual DHL PDF, with the page number so you can verify it yourself.
- **A tracking summary** (status, origin, destination, estimated delivery) — from the simulated tracking lookup.
- **`(Compliance reference table)`** — a match against the curated category/destination rule table, not a document passage.
- **`(Structured dataset query)`** — a number pulled directly from the segment revenue dataset, stated exactly as stored (not rounded or recalculated).

If you expand the "Sources" section under an answer, you'll see the actual document excerpts it was grounded in.

## Was the answer helpful?

Each answer has 👍 / 👎 buttons underneath it. Use them — a downvote gets logged and reviewed as a candidate "hard case" for improving the system, alongside the curated test questions LogiMind is already checked against.

## Walkthrough: a real example, end to end

**Question:** *"Where is my package 1234567890, and what compliance rules apply to lithium batteries at that destination?"*

**What happens (in plain terms):**
1. LogiMind recognizes this needs two things: a tracking lookup, then a compliance check — but the compliance check needs to know the *destination*, which it won't know until the tracking lookup runs.
2. It looks up tracking number `1234567890` first. In this simulated system, that number consistently resolves to a shipment in transit from Bonn, DE to Hong Kong, HK.
3. It then checks the compliance table for "lithium batteries" against that destination.
4. Both results are combined into a single answer.

**Answer LogiMind actually gave:**

> Your package **1234567890** is currently **In Transit**, shipped from Bonn, DE to Hong Kong, HK, with an estimated delivery date of **2026-08-11**. Recent tracking events: ...
>
> *(compliance rules for the destination follow, cited as (Compliance reference table))*

Notice what's happening: the second half of the answer depends on the first half's result (the actual destination), computed on the fly — not two unrelated facts stapled together. That's what "one question, one synthesized answer" means in practice.

## Questions this playbook doesn't cover

For how the system is built (architecture, retrieval, evaluation, deployment), see the project [README](../README.md) — that's the engineering-facing document this one deliberately isn't.
