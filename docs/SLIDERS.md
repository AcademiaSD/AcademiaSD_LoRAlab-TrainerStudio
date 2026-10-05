# Slider LoRAs — design

> 🧪 Design document, not implemented yet. It describes the **Slider** LoRA type that every image trainer will share. The working prototype it comes from (FLUX.2 Klein 9B, sad ↔ happy) is in [`experiments/slider`](../experiments/slider/README.md).

A slider is a LoRA whose **strength is a dial along one concept**: hair length, age, eye size, body shape, an expression, the amount of tartar on a set of teeth... In ComfyUI the same LoRA goes from one end to the other just by changing its strength, usually between **−5 and +5**.

## 1. Overview

The training is always the same. What changes is where the data comes from:

```
                 ┌─ A) My dataset: real or already generated images ─┐
LoRA Type        │                                                    ├─→  Slider dataset   ─→  Slider training  ─→  one LoRA with a dial
"Slider"  ───────┤                                                    │     (one format,        (same in every
                 └─ B) Synthetic: the user writes the two ends,      ─┘      reviewable)          image trainer)
                       an edit model builds the pairs
```

- The trainers get a new LoRA type: **Normal / Edit / Slider** (Edit only where the model supports it).
- **A) My dataset**: the user already has the images. Example: a dentist with photos of the same mouth healthy and with tartar, who wants every state in between.
- **B) Synthetic**: the user has no images. They describe the two ends ("very short hair" / "very long hair") and Klein 9B or Qwen-Image 2.1 edits base images towards each end. The result is a normal slider dataset that the user reviews before training, and can reuse for any model.

## 2. Dataset format

Every image is named **`group_position`**:

- **group**: what the images show (the same mouth, the same person, the same scene). Everything before the last `_`.
- **position**: where the image sits on the dial, an integer from **−100 to 100**.
- **`group_anc`**: an *anchor*, an image that must not change (see 2.2).
- **`group.txt`** (optional): one caption per group that describes the scene **without mentioning the concept**, e.g. `close-up photo of a mouth and teeth`. Without it, a neutral caption is used.

```
boca1_0   boca1_25   boca1_50   boca1_75          ← no 100: it is extrapolated (2.1)
boca2_0                          boca2_100        ← only the two ends
boca3_25  boca3_50                                ← only a stretch in the middle
cara1_-100  cara1_0  cara1_100                    ← concept with a neutral state in the middle
fondo1_anc                                        ← anchor
boca1.txt                                         ← caption of group boca1
```

### 2.1 Rules

- Each group needs **at least 2 images at different positions**. Groups do not have to be complete or alike: training only uses the **differences inside each group** (0 → 25, 25 → 75...).
- Positions not in the dataset are reached by following the learned direction. Going from 75 to 100 is a small step (the prototype was usable at 2–3× its training strength), but it only extrapolates what it has seen: if 100 has something no image shows, the slider cannot invent it. The further from the data, the worse it gets.
- The same applies to the negative side: with data only from 0 to 100, negative strengths reverse the effect (remove tartar) by pure extrapolation. It works better with some data on that side.
- The images of a group must be **aligned** (same framing, angle and light). Otherwise the slider also learns the camera change. Anchors help, but do not fix it.
- Pre-Cache checks the dataset and warns about: groups with one image, positions outside −100…100, names it cannot read, a group with images of different sizes, and a dataset with no anchors.

### 2.2 Anchors

Images that are **not** the subject of the concept (scenes, objects, other framings), trained to stay exactly as they are at any strength. They teach the slider not to zoom, reframe or shift the light. They must be different images from the groups: an anchor with the same neutral face as a pair would contradict it. Recommended: about half as many anchors as groups.

## 3. Dial scale

- A field **"Strength at 100%"**, default **5**, sets the scale: position 100 = strength 5, 50 = 2.5, −100 = −5.
- The universal rule is: **a strength change of 5 moves 100 positions**.
- The exported LoRA is already scaled, so the user never sees the internal training strength ("push").

What strength 0 means depends on the model:

| Model type | Strength 0 | Use |
| :--- | :--- | :--- |
| **Edit** (Klein 9B, Qwen-Image 2.1) | exactly the input photo | the dial **modifies a photo you give it**: the dentist puts in a patient's photo and sets the amount of tartar |
| **Text-to-image** (SDXL, Z-Image, Krea 2, Anima, Ideogram 4) | what the model would generate without the slider | the dial **modifies the generation**: same prompt and seed, more or less of the concept |

The same dataset trains both types.

## 4. Training

Internal values:

- `PUSH`: training strength (default 3, as in the prototype).
- `S100`: "Strength at 100%".
- For a position change `Δ` (in positions), the LoRA is trained at multiplier `m = PUSH × Δ / 100`.
- The exported LoRA has its weights multiplied by `PUSH / S100`, so that strength `S100 × Δ / 100` in ComfyUI gives that same multiplier.

### 4.1 Edit models

Each step takes an **ordered pair of the same group**, `p_i → p_j`:

- The image at `p_i` is the input (control) image and the one at `p_j` is the target.
- The multiplier is `m = PUSH × (p_j − p_i) / 100`.
- Both directions come out naturally (0 → 75 and 75 → 0), so the slider learns to add and to remove the concept.
- Anchors: input = target = the anchor, with a random multiplier in ±PUSH.

### 4.2 Text-to-image models

There is no input image, so positions are taken **relative to the centre of each group**, `c = (min + max) / 2`:

- Each image is trained at `m = PUSH × (p − c) / 100`, with the group's caption.
- The two images of a group go in **the same step, with the same noise and timestep**: two forward passes with opposite multipliers and their gradients added together. The noise is identical, so the step only sees the difference between them.
- Anchors: trained at a random ± multiplier towards themselves.

**Why only the concept is learned:** the change a LoRA makes is (almost) proportional to its multiplier. What two images have in common (the person, the style, the light, the "look" of a synthetic dataset) pulls the same way at `+m` and at `−m`, so it cancels out. Only the difference survives, which is the concept. This is also why a dataset generated with Klein can train a slider for SDXL without passing on Klein's look.

### 4.3 Common to all models

- **One sign per step:** the LoRA multiplier is a single value per layer, so a step cannot mix multipliers. Edit models use one pair per micro-step. Text-to-image models use the pair's two passes. The batch size is fixed to 1 in Slider mode, and Gradient Accumulation does the averaging.
- **Low rank:** rank 4 / alpha 4 by default. One direction needs little capacity, and a low rank leaves no room for side effects.
- **"Ultra" (optional, per model):** LoRA only on the blocks that decide composition and text–image fusion, leaving fine detail alone. The dial then holds much higher strengths. Klein 9B is measured (8 double + first 8 single blocks). The other models start without Ultra until a GPU test picks their blocks.
- **Previews:** a strip with the preview image (edit) or prompt + seed (text-to-image) at −S100, −S100/2, 0, +S100/2, +S100, so the user watches the dial form while training.
- **Not live:** "Strength at 100%", Ultra and the slider mode cannot change mid-training, and Resume keeps the checkpoint's values.

## 5. Synthetic mode (pair generator)

A tab in stage 0 (Caption) that writes a slider dataset (section 2):

1. **Subject**: what the images show ("head and shoulders portrait of a person", "full body photo of a person").
2. **The ends**: text for the 100 end and the −100 or 0 end ("very long hair" / "very short hair"). Optional intermediate levels (e.g. 50) with milder wording ("slightly long hair"). Intensity words are less reliable, so they are optional.
3. **Base images**: generated from the subject with varied people, places and light (as `experiments/slider/gen_pairs.py` does), or **the user's own images**, e.g. their character, for a "my character, older/younger" slider.
4. **Edit**: Klein 9B or Qwen-Image 2.1 edits each base towards each end, keeping the same person, clothes, background and framing.
5. **Anchors**: optional scenes without the subject.
6. **Review**: a grid with every group, where the user deletes the pairs where the edit failed. This happens often, and here it costs nothing compared with a failed training.

The output is a normal dataset: it can be edited by hand and trained on any model.

**Limits:** the edit model has to know how to make the change. Age, hair or expression work well. Something like "character height" is hard to show in a single image and needs full-body framings with some reference. The review step is where the user finds out, before spending hours training.

## 6. Model support

| Trainer | Edit slider | Text-to-image slider | Ultra |
| :--- | :---: | :---: | :--- |
| FLUX.2 Klein 9B | ✅ | ✅ | measured (prototype) |
| Qwen-Image 2.1 | ✅ | ✅ | to be measured |
| SDXL, Z-Image, Krea 2, Anima, Ideogram 4 | — | ✅ | to be measured |
| LTX-2.3, MiniMax-H3 (video) | — | — | out of scope for now: different training loop |

## 7. Implementation plan

1. **`scripts/slider_core.py`**, shared by every trainer, with everything that does not depend on the model:
   - Name parsing and dataset validation.
   - Building groups and pairs.
   - Choosing the step's samples and multipliers.
   - Setting the multiplier on PEFT `LoraLayer`s.
   - Scaling the export.
   - The preview strengths.
2. **Klein 9B** with both modes, reproducing test B of the prototype as the check that nothing broke.
3. **A light text-to-image trainer** (SDXL) to validate the mode without edit, the most relevant for 8–12 GB GPUs.
4. **Pair generator** (synthetic mode).
5. **The rest of the image trainers.**
6. **Later:** a text-only mode (the original Concept Sliders method: the frozen model as teacher, no images), for concepts the model understands from text.

Each step goes to `dev` with tests that need no GPU (Flask test client, fake data) and a real training on a GPU before the next one.

## 8. Open questions

- Pairs that span more than 100 positions (−100 → 100) give a multiplier of 2 × PUSH. Should they be used, capped, or only adjacent positions?
- Default `PUSH` (3) and rank (4) for models other than Klein.
- Anchors in synthetic mode: generated by default, or optional?
- Name of the type in the UI: "Slider" with a data-source selector, or two entries ("Slider — my dataset", "Slider — synthetic")?

## Credits

- Training a LoRA in both directions comes from **[Concept Sliders](https://sliders.baulab.info/)** (Gandikota et al., 2023), which also introduced text and image sliders.
- The push strength, the low rank and training only the composition blocks ("ultra") follow what **[Fizgig](https://github.com/shootthesound/Fizgig)** by [@shootthesound](https://github.com/shootthesound) describes for its slider LoRAs.
