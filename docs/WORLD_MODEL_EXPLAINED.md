# How the world model works (explained simply)

*This is the plain-language version of `DESIGN_NOTE.md` §1.4. Same facts, no jargon.*

## The problem in one picture

A company handles thousands of "cases" — an expense claim, an IT ticket. Each case is a story told one line at a time:

```
Mon 09:49  Declaration SUBMITTED by EMPLOYEE
Mon 11:27  Declaration FINAL_APPROVED by SUPERVISOR
Tue 09:34  Request Payment
Thu 17:31  Payment Handled            ← the end
```

We only ever see the story **so far**. The model's job: after each new line, be ready to answer three questions —
1. **What happens next?** (e.g. "Request Payment", 95 % sure)
2. **When?** (probably in ~1 day, could be 6 hours or a week)
3. **How does the rest of the story go, and when does it end?**

And it should stay honest about how sure it is.

## The idea: a notebook, not a rewind button

Imagine you're watching the case unfold and you're only allowed a **small sticky note** to remember it by. Every time a new event arrives you read the event, look at your note, and write a *new* note that replaces the old one. You never re-read the whole history — the note is all you have.

That sticky note is the model's **state** (128 numbers). The rule for "read event + old note → new note" is a small neural network called a GRU. It's cheap: one update per event, no matter how long the case is. That's what "world model" means here: a compact summary of where the case is, updated as the world produces events.

All three questions are answered by reading the *same* note. That was the central claim to test: one shared summary is enough for several different questions, rather than a separate model per question.

## What goes into an event

Each line is turned into numbers from four ingredients:
- **Which activity** it is (a learned code per activity, like a word embedding).
- **Its parts.** `Declaration APPROVED by SUPERVISOR` is split into *Declaration* / *APPROVED* / *SUPERVISOR*, each with its own code. So if the model later meets `Request For Payment APPROVED by SUPERVISOR` — a label it has never seen — it still recognises *APPROVED* and *SUPERVISOR*. (For the IT-ticket logs the parts are status / sub-status, e.g. *Accepted* / *In Progress*.)
- **Who/what**: role, group, product... Rare values become "unknown".
- **Time**: how long since the previous event, since the case started, what hour and weekday (in local time), how many events so far. All the "how long" numbers are on a log scale because they range from seconds to months.

Facts known at the start of the case (the amount claimed, the department) set the **first** sticky note.

## How the three answers are produced

From the note, three small "heads" read out:
- **Next activity**: a probability for every activity **plus a special "THE END" symbol**. Having "THE END" as just another option is what makes the model say "this case is finished" instead of inventing more steps.
- **Time to next event**: not a single number but a *mixture* of three bumps on the log-time scale (roughly "soon / a day / a week"). That gives a median guess and an 80 % range, e.g. "[6 h, 226 h]".
- **Remaining time to the end**: one bump on the log scale, using a heavy-tailed shape (Laplace) because a few cases drag on for months.

The model is trained on all three at once: total error = next-activity error + ½·time error + ½·remaining-time error. The ½ weights were fixed before any test results were looked at.

## Predicting several steps ahead ("what does the rest look like?")

The model **talks to itself**: it predicts the next activity and its time, pretends that event actually happened, updates its sticky note with it, predicts again, and so on until it predicts "THE END" (or a length cap). Because the pretend-events don't carry real role/group information, those are fed in as "unknown" — and during training we randomly blanked out 15 % of real attributes so the model is used to that. We can run this once greedily (always take the most likely next step) or many times with sampling to get a spread of possible futures.

## Worked example (a real test case from the Domestic Declarations log)

Seen so far:
```
Declaration SUBMITTED by EMPLOYEE
Declaration APPROVED by PRE_APPROVER
Declaration FINAL_APPROVED by SUPERVISOR
```
Model says: next = **Request Payment** (0.95), *Declaration REJECTED by MISSING* (0.03), …; time to next: median 1.3 days, 80 % range 6 h – 9 days; greedy continuation: `Request Payment → Payment Handled → THE END`.

What actually happened:
```
Declaration REJECTED by MISSING → SUBMITTED → APPROVED by PRE_APPROVER → FINAL_APPROVED → Request Payment → Payment Handled
```
So the model was wrong — a rejection came out of nowhere. Nothing in the three lines seen could have told it (the rejection reason isn't in the log), so being 95 % sure was the *right* level of confidence for what it knew; across the whole test set its "95 % sure" predictions are right about 95 % of the time. The time range did contain the true 119 h. And once the model guessed the first step wrong, everything after was off — in the data, 93–95 % of continuations stay wrong after the first wrong step. That is the main weakness of talking-to-yourself prediction: mistakes snowball.

## Decisions I made, and why (short)

| Decision | Why |
|---|---|
| Sticky-note (recurrent) state instead of re-reading the history | Updates in constant time per event; matches "update as new events arrive". |
| "THE END" as an activity | One consistent answer for "is it over?" for next-step and multi-step prediction. |
| Split activity names into parts | Lets a label the model never saw still be half-understood (schema changes). |
| Mixture of bumps for time, on a log scale | Waiting times are extremely spread out and often multi-modal ("same day" vs "next week"). |
| Blank out 15 % of attributes during training | Multi-step prediction has no attributes for imagined events; training must look the same. |
| A case counts as "finished" only if its last line is a real end (Closed / Payment Handled / Rejected) | Many logs contain cases that were still open when the data was exported; teaching the model they "ended" there would be wrong. |
| Fit every vocabulary/scale on training cases only | Otherwise the test set leaks into the model. |
| Compare to a counting model and a tree model on the same splits | So any gain can be attributed to the sticky-note idea, not to preprocessing. |

## What the experiments showed (one line each)

- When train and test come from the same period, the tree model and the sticky-note model are about equally good; counting alone is far behind on the complex logs.
- When we train on the past and test on the future with histories honestly cut off at the cutoff date, the sticky-note models lose the least (IT tickets: +7–8 % error vs +95 % for trees).
- Its confidence is honest (calibration error ≤ 0.02) and its time ranges cover the truth ~80 % of the time, as designed.
- It forgets: three events after a rejection, the note no longer "remembers" it. A look-back (attention) version remembers longer but is not more accurate — see the update below.
- Reusing a model from a *different* log and fine-tuning it ends up about equal to training fresh on the small new data (an earlier "worse" result was an unfair epoch budget); letting the output layer also use the label *parts* helps a little (zero-shot error 5.75 → 5.15).

## Update after the second round of experiments

**Two kinds of sticky note, compared.** Besides the note that gets rewritten each time (GRU), I built a version that keeps *all* the past lines and looks back at them when it needs to (attention, a Transformer). Same inputs, same three readers. Result: in the ordinary tests they tie. When we train only on the past and test on the future *with the training histories honestly cut off at the cutoff date*, the tree model falls apart (error up 95%), both note-keeping models barely change (+7–8%), and the look-back version is the steadier of the two. The look-back model really does remember an early rejection longer (I can measure it), but that memory doesn't make its guesses better here, because what happens next is mostly decided by things outside the log.

**Does the model survive losing its side-information?** I removed every "who/what" attribute at test time. The tree model's error jumped 69%; the note-keeping model's 23%, and it then beats the tree. Training with 15% of attributes randomly blanked is what bought that.

**A better way to read the future.** Instead of always taking the most likely next step (which loops forever on the IT-ticket log: "still in progress, still in progress, …"), sample five possible futures and take the one that is most like the others. That version always ends, and its story matches the truth more often than the tree model's.

**Part 2 uses the very same model.** An agent's log of API calls is just another event log (one call per line, plus whether it errored, plus how the episode ended). So I trained the same sticky-note model on agent traces and used it to choose between candidate actions. It knows *what a good agent does next* and *whether things are going well*, but it can't see inside the arguments (right key vs. wrong key look the same to it). A small text model can. Together they got every held-out mock task right; either alone did not.
