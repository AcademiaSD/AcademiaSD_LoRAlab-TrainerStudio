# Text slider (Krea 2): experimental, parked

**Status: parked. The results are not convincing yet, so this branch is not merged into `main`.**

This branch trains a "dial" LoRA for Krea 2 from two prompt ends (Concept Sliders style):
at a random multiplier `m` the LoRA learns
`base + eta * (m / push) * (pred(caption + positive) - pred(caption + negative))`,
so strength 0 is the base model and the effect grows with the strength.

## What works

- The dial is graded on some subjects: with Ultra mode, 32 practice images and intensity 8,
  the body-weight test moved step by step from +2 to +5.

## What does not

- Krea 2's text signal is weak; it needs intensity (ETA) 8, much more than Z-Image.
- The framing jumps between two options around strength 0, worse with the Turbo 8-step sampler.
- With the user's dataset photos instead of practice images the effect is about half.
- It reproduces what the base model already knows from the prompts; it does not learn from
  the dataset, which is what we wanted.
- Cutting high sigmas and the framing-preservation variants made it worse.

## Where the code is

- `scripts/slider_text.py`: shared logic (settings, cache plan and manifest, PEFT dial, preview strip).
- `scripts/1_pre_cache_krea2.py`, `scripts/2_train_lora_krea2.py`: Krea 2 integration.
- `GUI/slider_text.js`: "Text slider (dial)" option, translated to 7 languages.
