# PROMPT 5 — QURE Lab: Mixed-Reality Lab for Meta Quest 3 (replaces Prompts 2 and 3)

> **For the human (read this part, do not paste it):**
> - **Why the old VR went wrong.** Prompts 2 and 3 were written for an early data plan: NoduleMNIST3D voxels, ZNE, QEC, VQE, an AI explainer and a mitigated Bloch arrow. The real project has none of those. Any VR built from them has nothing real to show and mostly fills in fake numbers. This prompt builds against the file the website already uses, `web/qure_bundle.json` (v0.2.1). Every number in VR is then the same number the console shows.
> - **How to use it.** Paste everything below the line as the first message in a fresh Claude Code session at the repo root (preferred). In ChatGPT, also attach `web/qure_bundle.json` and `web/template.html`.
> - **What to check.** The AI stops after each milestone; test on the headset before saying "continue".
> - **Before you start.** Copy `web/qure_bundle.json` (v0.2.1, from the latest Role 1 delivery) into the repo. Delete or archive the old `vr/` folder; the new app starts clean.

---

## 1. Your role and the goal

You are a senior WebXR engineer and spatial-interaction designer. Build **QURE Lab XR**: a mixed-reality lab for the **Meta Quest 3 / 3S browser**. The quantum classifier's real results appear as holograms on the real table and in the room around the user.

The aim is a moment a judge remembers. The user raises a noise lever, and the room fills with 90 real test nodules whose quantum decisions shake and cross a glowing decision wall. Then one switch flips mitigation on and most of them settle back. Everything shown comes from recorded simulation data; nothing is invented.

The website (`web/`, a static console) is the main product. This is a **separate static app in `vr/`** that reads the **same bundle file**, so it attaches to the website with one link and changes nothing else.

## 2. What the project is (so you explain it correctly)

- **Data (Role 2).** CIRDataset: LIDC-IDRI CT patches with expert nodule masks, 597 patients, one nodule each. Labels are radiologist suspicion (rating 1–2 = low, 4–5 = high), not confirmed cancer. Split by patient: 417 train / 90 validation / 90 test. **Test labels are held by Role 2.** Never show or guess them.
- **Inputs.** Four image/mask features per nodule, mapped to four angles θ ∈ (0, π).
- **Circuit (Role 1).** RY(θᵢ) encoding on 4 qubits, then 2 layers of [trainable RY on each qubit + CX chain 0–1, 1–2, 2–3], then a final trainable RY layer: 12 weights, frozen.
  - **Score:** the probability that the 4 measured bits have odd parity. The decision is "high suspicion" if score ≥ threshold (0.39).
  - **The score is not a probability of disease.**
- **Noise.** Five hypothetical simulator settings, N0 (noiseless) to N4 (gate + high readout noise). Each case was run 5 times × 2000 shots per setting.
- **Three results per case and setting:**
  - noiseless reference (exact);
  - noisy execution (5 repeats);
  - mitigated estimate (5 repeats; readout-error mitigation only).
- **Instability rule U** flags a case for review when the raw and mitigated decisions disagree, or the mitigated mean is within 2 standard errors of the threshold. It is a review prompt, not a trust score.
- **Honest result.** On validation, the quantum model (balanced accuracy 0.857) and logistic regression (0.810 at 0.5, 0.845 tuned) are **not distinguishable**. There is no quantum advantage. Mitigation cuts score error but makes results vary more between runs; that cost has to be shown.

## 3. The data contract (the only input)

- **Source.** Load one JSON file: the website's `qure_bundle.json`. Resolve its URL in this order:
  1. `?bundle=<url>` query parameter;
  2. `import.meta.env.VITE_BUNDLE_URL`;
  3. `../qure_bundle.json` (when hosted at `web/vr/`);
  4. `./qure_bundle.json` (dev copy in `vr/public/`).
- **Version check.** Require `bundle_version` to start with `"0.2."`. Otherwise show an in-world error panel naming the expected version.
- **Rules:**
  - Never write to the bundle.
  - Never call a server.
  - Never call an LLM.
  - Never hard-code a result number.

Put these types in `vr/src/data/bundle.ts` and validate the fields you use at load time:

```ts
export type SettingId = 'N0' | 'N1' | 'N2' | 'N3' | 'N4';
export type Vec3 = [number, number, number];
export interface CaseSetting {
  nx: number;            // noisy exact (infinite-shot) score
  n: number[];           // 5 noisy repeat scores
  m: number[];           // 5 mitigated repeat scores (may lie outside [0,1]; never clip)
  eq: number;            // mean of a raw run with the same total shots (6000)
  fn: number; fm: number;// share of repeats whose decision differs from noiseless (noisy / mitigated)
  q: number;             // repeats with negative quasi-probabilities (0..5)
  f: boolean;            // flagged by rule U
  b: Vec3[];             // per-qubit Bloch vectors, gate-noisy state (4)
  b0: Vec3[];            // per-qubit Bloch vectors, noiseless state (4)
}
export interface Case {
  id: string; patient: string; split: 'val' | 'test';
  label: 0 | 1 | null;   // null on test: held by Role 2
  angles: number[];      // 4 encoding angles θ (rad)
  features: number[];    // 4 raw feature values (display only)
  lr: number | null;     // logistic-regression probability (validation only)
  platt: number;         // Platt-calibrated value of the noiseless score (fit on validation)
  abl: number;           // no-CX ablation noiseless score
  nl: number; dec: 0 | 1;// noiseless score and decision
  S: Record<SettingId, CaseSetting>;
}
export interface Gate { name: 'ry' | 'cx' | 'measure'; qubits: number[]; role: 'encoding' | 'trainable' | 'entangling' | 'readout'; param?: string; value?: number; layer?: number }
export interface Bundle {
  bundle_version: string; vr_url: string | null;
  circuit_gates: Gate[]; ablation_circuit_gates: Gate[];
  noise_settings: { id: SettingId; label: string; p1q: number; p2q: number; readout_p10: number; readout_p01: number }[];
  noise_source: string; mitigation: Record<string, string>;
  budget: { shots_per_execution: number; calibration_shots_total: number; mitigated_total_shots: number; equal_budget_raw_shots: number };
  repeats: number; instability_rule: string;
  model: { model_id: string; param_version: string; threshold: number; encoding: string };
  abl_threshold: number;
  summary: any;          // quantum_val[], quantum_reliability_val/test[], classical_val, comparison_val, instability_val[], instability_test_label_free[]
  dataset: { features: string[]; label: string; patients: number; split_counts: Record<string, number>; limitations: string[] };
  linked: { case_id: string; slice_index: number; png: string };   // one real CT slice (data: URL), case LIDC-IDRI-0009
  cases: Case[];         // 90 val + 90 test
}
```

**Data facts you can rely on** (check them in code, and assert them in a unit test):

- `circuit_gates` rebuilds the circuit exactly. Applying the gates to |0000⟩ with a case's `angles` reproduces that case's `nl` to about 1e-4. Use this for the circuit sculpture; do not invent gate values.
- Readout mitigation does not change the quantum state, so **there is no "mitigated Bloch arrow"**. Show noiseless (`b0`) and gate-noisy (`b`) arrows only, and say why.
- Bloch vectors are shorter than 1 even without noise, because the qubits are entangled. Always compare an arrow to its noiseless ghost, never to the sphere's surface.
- At N4 on the test split, 18 of 90 cases have at least one noisy repeat whose decision differs from the noiseless reference, and rule U flags 15. Read these counts from the data; they are listed here only so you can sanity-check your loader.

## 4. The experience (scenes ranked by priority)

Build in this order. Each scene must work alone, so a demo is possible after any milestone.

### Scene A — "The Lab appears" (opening, 20 s)

- **Placement.** Passthrough is on (session mode `immersive-ar`). Use plane detection to find the real table (semantic label `table`, or the largest horizontal plane between 0.6 and 1.1 m). A glass plinth (0.9 × 0.5 m) grows out of it with a soft light-ring.
- **Fallbacks for placement:**
  1. a hit-test reticle the user places with the trigger;
  2. if neither planes nor hit-test are available, a fixed position 0.6 m in front, at 0.9 m height.
- **The CT patch.** The real CT patch (`linked.png`) rises from the plinth as a floating 30 cm image card with its cyan mask outline.
- **Encoding.** Four light-strands peel off the card, one per feature, each labelled with its feature name and value. They twist into the four qubit rails while their angle θ counts up to the case's real value. This shows angle encoding as a physical rotation.
- **Caption:** "One nodule → four numbers → four rotations."

### Scene B — "Circuit sculpture" (core)

- **Layout.** A glass sculpture on the plinth, about 60 cm long: four horizontal qubit rails.
  - Gate blocks are placed from `circuit_gates`, in three visual roles: encoding RY, trainable RY, and CX drawn as vertical light-bridges with a dot on the control and a ring on the target.
  - Gaze or point at a gate to see its role, parameter and frozen value.
- **Pulse.** A pulse of light runs left → right through the sculpture whenever the case or setting changes.
- **Bloch spheres.** Four 9 cm Bloch spheres float at the end of the rails.
  - Arrows: noiseless (`b0`, solid white ghost) and gate-noisy (`b`, dashed amber).
  - Each sphere shows |r| for both arrows.
  - A small "i" panel explains: "Shorter than 1 even without noise because the qubits are entangled. Readout mitigation does not change the state."
- **Measurement columns.** Five thin glass columns, one per repeat, fill to each repeat's noisy score. Next to them, five more fill to the mitigated scores. A horizontal ring marks the threshold, and a white line marks the noiseless score.
  - If a mitigated value is below 0 or above 1, the column overflows past the glass in red and keeps its true value; this shows that values are not clipped.
- **Ablation toggle.** Removes the CX bridges, rebuilds the sculpture from `ablation_circuit_gates`, and shows the ablation score `abl` with its threshold.

### Scene C — "The Decision Wall" (THE wow moment; build it beautifully)

- **The wall.** A huge translucent vertical membrane, 3 m wide × 2 m tall, at the threshold. It is drawn as a shimmering wall about 1.5 m in front of the user and curves around them in a 140° arc.
- **The orbs.** All 90 cases of the chosen split are glowing orbs arranged in that arc.
  - Each orb's depth from the wall is proportional to (score − threshold), with "high" in front and "low" behind, using one fixed scale (state it on a legend).
  - Height spreads the cases evenly so none overlap.
  - On validation, each orb has a thin ring coloured by its reference label. On test, the ring is dashed grey with "label held by Role 2".
- **The 5 noisy repeats** of each case are tiny sparks orbiting their orb at their own recorded scores.
- **The noise lever.** A physical lever on the plinth with 5 detents (N0…N4), a haptic click on each detent, and a big readout "N3 · gate + readout (moderate)". It can never rest between detents.
  - On each change, orbs and sparks move to the new recorded positions in ≤ 300 ms. No numbers are shown during the motion: values change only at detents. (The rule: playback of precomputed settings, never interpolation.)
- **A decision flip.** When a repeat lands on the other side of the wall from the noiseless decision, that spark punches through the membrane with a ripple and a short click sound.
- **Rule U.** Orbs flagged by U get an amber halo and a small "U".
- **The mitigation switch.** Big and glowing. ON replaces the noisy sparks with the mitigated sparks. Most return to their side of the wall; a few stay, and those must stay visible.
  - A **counter panel** reads straight from the data: "decision flips: 16.9% → 3.3% of repeats" and "U flags: 15 / 90".
  - A **cost bar** shows "shots: 2000 → 6000" and the run-to-run spread getting larger (`summary.quantum_reliability_*`). Mitigation must never look free.
- **Inspecting a case.** Point at any orb and pull the trigger: the orb flies to the plinth, and Scenes A and B reload for that case. "Back to wall" returns it.
- **Split switch.** Validation / Test, read from `cases[].split`.

### Scene D — "Verdict podium" (honesty, short)

- **The bars.** Three free-standing 3D bars rise from the floor beside the plinth:
  - quantum (noiseless);
  - logistic regression at 0.5;
  - logistic regression, val-tuned.

  Bar height is validation balanced accuracy. Each bar carries its 95% CI as a glowing whisker.
- **Plaque.** It reads `summary.comparison_val`: "difference +0.011, CI −0.022 to 0.050: not distinguishable. No quantum-advantage claim."
- **Disclaimer.** "Research prototype. Simulated quantum execution. Not a medical device." It appears on this plaque and in small text on the plinth in every scene.

### Scene E — "Guided tour" (judge mode, 90 seconds)

- **Steps.** Eight steps with captions, plus optional `speechSynthesis` voice behind a speaker toggle. Captions are always on.
  1. the lab appears;
  2. one nodule becomes four rotations;
  3. the circuit;
  4. noiseless vs noisy Bloch arrows;
  5. the decision wall at N0;
  6. lever to N4 (flips appear);
  7. mitigation ON (most return; cost bar grows);
  8. the verdict podium.
- **Controls.** Step forward with the A button or a pinch on "Next". The tour sets state; it never moves the user's head or camera.

## 5. Interaction and comfort rules (mandatory)

- **Inputs.** Controllers and hand tracking both work: ray + trigger, or pinch. Every button is at least 4 cm and lights up on hover. Grab the plinth's edge with grip/pinch to move or rotate the whole lab; snap rotation in 15° steps.
- **No locomotion and no camera motion.** Everything happens within reach or within view from where the user stands. The wall arc sits at a fixed distance.
- **Placement and text.** Content sits 0.5–3 m away, between waist and eye height. Text is never smaller than 1.2 cm per metre of viewing distance. Use high contrast, and never colour alone: noisy is dashed, mitigated is solid, flags carry a letter.
- **Desktop and projector mode (required).** Without WebXR, the same scene renders with OrbitControls and a bottom bar of HTML buttons: setting N0–N4, mitigation, split, tour next. Judges can watch on a laptop even if the headset fails.
- **Casting.** Document it in the README: Quest → phone app, or `oculus.com/casting` in a laptop browser.

## 6. Performance budget (Quest 3; the frame rate is how the app will be judged)

Meta's guidance: hold at least **72 fps (13.7 ms per frame)**, with **90 fps (11.1 ms)** recommended on Quest 3; budget **< 200 draw calls and < 1.5 M triangles** per frame on Quest 3 (< 100 and < 750 k on Quest 2). Treat these as ceilings, not targets.

| Item | Budget |
| --- | --- |
| Frame rate | request 90 (`frameRate: 'high'`); must never drop below 72 in any scene |
| Draw calls | ≤ 120 in the worst scene (measure; log it) |
| Triangles | ≤ 300 k visible |
| Orbs and sparks | 2 `InstancedMesh` total (90 orbs; 90 × 5 sparks), updated with matrices, not React re-renders |
| Text | ≤ 40 drei `<Text>` instances visible; reuse one font; no `<Html>` in XR |
| Materials | `MeshBasicMaterial` / `MeshLambertMaterial`; one shared glass material; no real-time shadows; no post-processing |
| Membrane | one plane with a small custom shader (fresnel + ripple uniforms from a ring buffer of ≤ 16 hits) |
| Framebuffer | `frameBufferScaling` default; drop to 0.9 only if measured below 72 fps; `foveation: 1` |
| Per-frame JS | ≤ 2 ms; precompute every position per (split, setting, mitigation) once at load |
| Assets | procedural geometry only; no downloaded models, HDRIs or textures except `linked.png` from the bundle |

Measure on the device with the OVR Metrics Tool performance HUD (or `r3f-perf` in desktop dev only). Put the measured fps and draw calls for each scene in `vr/README.md`.

## 7. Stack and setup

- **Stack.** Vite + React 19 + TypeScript + three + @react-three/fiber 9 + @react-three/drei 10 + **@react-three/xr 6** + zustand 5. Check the installed versions' types before coding; if an API below differs, adapt it and log the change in `vr/DECISIONS.md`.
- **The XR store:**
  ```ts
  createXRStore({
    emulate: 'metaQuest3', frameRate: 'high', foveation: 1,
    planeDetection: true, hitTest: true, anchors: true, handTracking: true,
    offerSession: false,
  })
  ```
  - Enter with `store.enterAR()` when `immersive-ar` is supported, else `store.enterVR()` (a dark lab-room backdrop replaces passthrough), else desktop mode.
  - Use `<XR store>`, `useXRPlanes('table')`, `XRHitTest` / `useXRHitTest`, `useXRAnchor`, `XROrigin`, and pointer events (`onClick`, `onPointerDown/Move/Up` with `setPointerCapture`) on meshes.
- **Scripts:**
  - `npm run sync` copies `../web/qure_bundle.json` to `vr/public/qure_bundle.json`.
  - `npm run dev` runs Vite with `@vitejs/plugin-basic-ssl` and `server.host = true`. Open `https://<laptop-LAN-IP>:5173` in the Quest browser and accept the certificate. WebXR needs HTTPS.
  - `npm run build` builds with `base: './'`, so `vr/dist` works under any subfolder.
- **Headset emulation on laptops.** The store's `emulate` option, or Meta's Immersive Web Emulator extension.

## 8. Integration with the website (zero-friction contract)

1. **Write only inside `vr/`.** Do not edit `web/`, `src/`, `results/` or the bundle.
2. **One state model, mirrored in the URL:** `?case=<id>&n=<N0..N4>&mit=<0|1>&split=<val|test>&scene=<lab|wall|verdict>`. Read it on load; write it back with `history.replaceState`. The website already links here as `vr/?case=<id>&n=<setting>&split=<split>`.
3. **Back link.** A "Back to console" button links to `../index.html#cases` (or `VITE_WEB_URL`).
4. **Deploy layout.** A static host with HTTPS (GitHub Pages, Vercel or Netlify):
   ```
   site/index.html             ← web/index.html (the console)
   site/qure_bundle.json       ← web/qure_bundle.json
   site/vr/…                   ← vr/dist/*
   ```
   Then rebuild the console with its links switched on: `python tools/build_web_bundle.py --vr-url vr/`. This adds an "Open 3D lab" button and a per-case "Open this case in the 3D lab" link. The claude.ai published console stays as the shareable 2D view; WebXR runs from the HTTPS host.
5. **Re-running the experiment** only regenerates `qure_bundle.json`; VR picks it up with no code change.

## 9. Code layout (`vr/src/`)

```
main.tsx  App.tsx
data/bundle.ts (types + loader + validation)   data/derive.ts (precomputed positions per split/setting/mit)
state/store.ts (zustand: caseId, setting, mitigation, split, scene, tourStep)   state/url.ts
xr/xrStore.ts  xr/Placement.tsx (planes → hit-test → fixed)  xr/haptics.ts  xr/DesktopBar.tsx
scenes/LabAppears.tsx  scenes/CircuitSculpture.tsx  scenes/DecisionWall.tsx  scenes/Verdict.tsx  tour/Tour.tsx
three/BlochSphere.tsx  three/RepeatColumns.tsx  three/Membrane.tsx (shader)  three/OrbField.tsx (instanced)
ui3d/Button3D.tsx  ui3d/Lever3D.tsx  ui3d/Switch3D.tsx  ui3d/Panel3D.tsx
audio/sfx.ts (WebAudio-generated clicks; no files)
tests/derive.test.ts (vitest: gates reproduce nl; flip counts match summary; no value clipped)
```

## 10. Milestones (stop after each; give exact headset and desktop test steps)

| # | Done when |
| --- | --- |
| M1 | Scaffold, HTTPS dev, bundle loads and validates; desktop mode shows the plinth; Enter MR on Quest shows passthrough with the plinth placed on the real table (or a fallback) |
| M2 | Scene A + B: the CT card, encoding strands, a circuit sculpture built from `circuit_gates`, Bloch spheres and repeat columns for any case; vitest passes |
| M3 | Scene C: the wall, 90 orbs + 450 sparks instanced, lever with detents and haptics, mitigation switch, flips punching through, U halos, counters and cost bar from data; ≥ 72 fps measured |
| M4 | Scene D + tour + URL state + back link + desktop bar; deep link from the console opens the right case |
| M5 | Performance pass (log fps and draw calls per scene), README (devices, HTTPS, emulator, casting, controls, measured performance, known limits), build deployed under `site/vr/` |

## 11. Do NOT

- Do not invent data: no fake voxels, ZNE, QEC, molecules, AI explanations or mitigated Bloch arrows. This project has none of these.
- Do not show or infer test labels.
- Do not clip mitigated values.
- Do not interpolate between noise settings, and do not let the lever rest between detents.
- Do not call U, Bloch length or glow a "trust", "confidence" or "accuracy" meter.
- Do not use smooth locomotion, move the camera, or auto-rotate the world in XR.
- Do not use drei `<Html>` in XR, download external assets, or call any server or LLM.
- Do not claim quantum advantage, diagnosis, clinical validity, or that VR is proven better.
- Do not edit anything outside `vr/`.
