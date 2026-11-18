# My test cases and feedback for the ED app (Gradio version)

All patients here are made up. No real patient details.

## Files
- `feedback_form_Caleb.csv`: the group's feedback form, filled in. It covers T01, T02, P01, every button once,
  the "try to break it" list (Gradio guide section 5) and the extra findings below.
- `cases/C01-C06.txt`: six extra notes, each testing something the group pack doesn't cover.
  `C02_followup.txt` is added to C02 afterwards with **Add information**.
- `my_expected_results.csv`: what the app showed for each case (test mode, 9 Nov).

Pictures for C05 and C06 are in `../ED_app_test_pack/pictures/` (CC BY 4.0, see `ATTRIBUTION.txt` there).

| Case | Tests | Picture | What the app showed (test mode) |
|---|---|---|---|
| C01 | temperature in °F, sepsis-like vitals | — | 101.8 °F read as 38.8 °C; NEWS2 15; level 2, urgent |
| C02 | chest pain with normal vitals | — | NEWS2 0 but level 2 (raised by the urgency score); ECG to-do |
| C02 + follow-up | Add "trop 85, ST depression" | — | unchanged: level 2, 22%, no new to-do (test mode) |
| C03 | collapsed, unresponsive, no vitals (safety rule) | — | urgent 95%, level 2 (not 1); resuscitation to-do |
| C04 | impossible HR 1500 | — | HR "not recorded", **no warning shown** |
| C05 | minor ankle injury | `adult_leg_no_fracture.jpg` | level 3, not urgent; **red fracture mark on a normal X-ray** |
| C06 | head injury | `ct_head_bleed_epidural.jpg` | level 2, urgent; bleed inside the skull marked red |

## How these results were made
A script called the same functions the Gradio buttons call, in test mode, on a scratch whiteboard. It wasn't
clicked in a browser. To try a case by hand: start the app (`python app.py --test`), press **Open a text file**,
pick a case, add the picture if there is one, then **Submit**.
