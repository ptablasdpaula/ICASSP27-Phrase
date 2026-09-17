# Findings

The 1 s / 1 octave setting improves fresh-start recovery versus no fading, but
on this phrase it does not reproduce the near-exact result of 1 s / 1000 Hz.
Fresh pitch/onset MAE is 9.737 cents / 27.794 ms, versus 99.728 / 116.178 without
fading and .084 / .025 for 1 s / 1000 Hz. The octave run stops on the common
patience rule after 1009 updates; the linear-Hz run used 3000 updates.

At the saved plateau, pitch error improves from 15.798 to 12.969 cents, while
onset error worsens from 48.300 to 55.507 ms. Joint RMS matched error decreases
from .073773 to .062909, but the unfaded diagonal loss and LSD both worsen.
The octave run stops on patience after 1242 updates. This is not a recovery
of the failed phrase.

The comparison changes the shape and physical width of frequency memory, not
just its unit label. Around 160 Hz, one octave spans to 320 Hz upwards or 80 Hz
downwards; 1000 Hz gives much broader support around the fundamental. At higher
harmonics, one octave covers larger absolute-Hz distances. The result does not
establish that octave coordinates are intrinsically worse: it tests one octave
with one fixed fade shape on one phrase.

Both jobs completed successfully, and best-loss checks passed by rerendering.
All earlier controls are reused without rerunning or altering them. As in the
preceding pilot, source/arithmetic sensitivity and one trajectory per setting
limit generalisation. Raw trajectories and the complete metrics are archived.
