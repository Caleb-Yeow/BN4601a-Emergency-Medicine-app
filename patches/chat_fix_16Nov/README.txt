CHAT FIX (Caleb, 16 Nov) - the chat now answers diagnosis and medication questions
==================================================================================

Based on main 74287be. Two files changed: reasoning/src/prompts.py, reasoning/src/pipeline.py (22 lines added, 7 removed).
test_app.py: 122 PASS, 0 FAIL. Checked with the full AI on my laptop (RTX 4050 6 GB).

HOW TO APPLY (pick one)
  A. With git, in the repo on main:
       git am 0001-Chat-answer-diagnosis-and-medication-questions-Caleb.patch
  B. Without git: copy changed_files/reasoning/src/prompts.py and pipeline.py over the same files.
  Then restart the app (python server.py, or gradio/app.py). No new models or packages.

THE PROBLEM
  Asked "which medications?" or "possible diagnoses?", the chat only summarised the record, and on a follow-up gave
  the SAME summary word for word. Cause: the chat prompt said "Use only information in the record", which MedGemma read
  as "never suggest a diagnosis or a drug" (neither is ever in the record). It also called NEWS2 6 "low".

WHAT CHANGED
  1. prompts.py, VISIT_CHAT_SYSTEM:
     - facts about the patient still only from the record ("not recorded" if missing), but it may now use medical
       knowledge for possible diagnoses and usual treatment;
     - diagnosis questions: top 3 possible diagnoses, each with the supporting findings and the test that confirms it;
     - medication questions: usual first-line ED treatment with adult dose and route, what to check first (allergies,
       weight, kidney function, pregnancy), labelled as a suggestion for the clinician to confirm against local protocol;
     - answer first, don't summarise the record, answer the likely meaning of typos, at most 8 bullets;
     - NEWS2 bands 0-4 low, 5-6 medium, 7+ high.
  2. pipeline.py, _turn(): always uses the current chat prompt, so patients already on the board pick up the change.

RESULTS (full AI)
  Before: P3 "medications we might need" and "I need the medications and possible diagnoses" -> the same record summary
  twice, no drug, no diagnosis.
  After (P8, melena):
    "Possible diagnoses?" -> 1 upper GI bleed, 2 lower GI bleed, 3 haemorrhagic diarrhoea, each with supporting findings;
                             EGD / colonoscopy; IV fluids, pantoprazole, GI referral.
    "Medications with doses?" -> pantoprazole 40 mg IV, why, what to check first, "confirm against local protocol".
  After (P6, right abdominal pain): diagnoses OK (cholecystitis, appendicitis, CT abdomen), but the medication question
  still said "no medications are listed ... none can be suggested" (1 of 4 test questions still fails).

TRIED AND DROPPED
  Chat on plain MedGemma (no task/stage1 adapter): worse. Every answer summarised the record, refused, and two got
  stuck repeating "No follow-up treatment is needed ... monitored closely" until the token limit.

STILL OPEN (not in this patch; suggested next fixes, in order)
  - Copying: the chat keeps every earlier Q and A, and MedGemma repeats its earlier answer to a similar question word
    for word (P6, P7, P5 today). Fix: send the record + only the last 2 Q/A, and regenerate if an answer repeats the
    previous one.
  - The question is buried after the whole record and a "(Time now ... Arrived ...)" prefix. Fix: put "QUESTION: ..."
    first and again at the end; move the time/tests-done note into the record.
  - Wrong facts in the record that the chat repeats: off-topic ED pathway flags and the qSOFA screen ("possible
    stroke" for P5, sepsis for chest pain), AI picture readings stored as fact (template ECG readings on the real test
    set), and two number-checker bugs that drop correct values (B08: ER "heart rate was 30-40" thrown out, baseline
    HR 50 shown; B12: "35-year-old African American female" -> sex blank). Fix: label flags/screens as "rule-based,
    may not apply" and picture readings as "AI reading, not confirmed"; loosen the word check.
  - It accepts nonsense premises ("why my cat blue" -> "the cat is blue because the patient is unconscious").
