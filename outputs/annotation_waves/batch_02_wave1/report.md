# Wave report

Messages compared (finished and valid in both files): 195.

## Agreement
* span F1 (IoU >= 0.5): 0.7617689015691869  (in A 730, in B 672, matched 534)
* message aspect_status: raw 0.9948717948717949, kappa 0.0, n 195
* category: raw 0.9232209737827716, kappa 0.9110582098707758, n 534
* sentiment: raw 0.9943820224719101, kappa 0.9837327376116978, n 534
* span_type: raw 0.9194756554307116, kappa 0.8237677866977758, n 534
* unresolved: {'NEEDS_REVIEW_rows_a': 2, 'NEEDS_REVIEW_rows_b': 5, 'UNCLEAR_ASPECT_messages_a': 0, 'UNCLEAR_ASPECT_messages_b': 1, 'message_status_disagreements': 1, 'messages_unfinished_or_invalid_in_a': 2, 'messages_unfinished_or_invalid_in_b': 5}

## Adjudication package: 394 items ({'UNMATCHED_SPAN_A_ONLY': 201, 'UNMATCHED_SPAN_B_ONLY': 143, 'CATEGORY_DISAGREEMENT': 42, 'NEEDS_REVIEW': 4, 'SENTIMENT_DISAGREEMENT': 3, 'MESSAGE_STATUS_DISAGREEMENT': 1}) in adjudication_package/

## Measured yield (this wave)
* agreed mentions only: 20 products, 182 reviews, 398 examples, 91 usable series
* all annotated spans counted once (upper bound): 20 products, 195 reviews, 537 examples, 109 usable series
* agreed, without SMALL products (sampling rule pending approval): 15 products, 135 reviews, 288 examples, 66 usable series

## Corpus with the pilot
* agreed-only: 30 products, 221 reviews, 459 examples, 105 usable series
* optimistic: 30 products, 244 reviews, 658 examples, 138 usable series
* agreed-only plus a second wave like this one: 50 products, 403 reviews, 857 examples, 196 usable series
* floors, agreed_only: hard blockers ['15 usable val series (floor 20)', '15 usable test series (floor 20)']; warnings ['about 4 products per evaluation split (< 5)', 'about 321 training examples (< 500)', 'about 68 examples per evaluation split (< 100)', 'about 10 NEGATIVE examples per evaluation split (< 20)']
* floors, optimistic: hard blockers none; warnings ['about 4 products per evaluation split (< 5)', 'about 460 training examples (< 500)', 'about 98 examples per evaluation split (< 100)', 'about 17 NEGATIVE examples per evaluation split (< 20)']
* floors, agreed_only_plus_second_wave: hard blockers none; warnings ['about 19 NEGATIVE examples per evaluation split (< 20)']

## Recommendation
PROCEED with Wave 2 and plan more: even with a second wave of the same size at the measured yield, 0 hard blocker(s) and 1 warning(s) remain (about 19 NEGATIVE examples per evaluation split (< 20)).
