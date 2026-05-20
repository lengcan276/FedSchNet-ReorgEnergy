# Reference TODO Audit — `paper/references.bib`

**Date:** 2026-05-11
**Scope:** Read-only audit of `paper/references.bib` entries marked `note = {TODO: ...}`.  No changes were made to `references.bib`, `manuscript.md` or `SI.md`.
**Constraint:** This audit was performed without internet access; conclusions about DOIs are limited to what can be derived from the local codebase, the bib file itself, and conservative pattern-matching against known publisher DOI formats.  No DOI was filled in from training-data memory.

---

## Summary table

| # | Bib key | Currently filled DOI / URL | Status of the value | Safe to commit as-is? |
|---|---|---|---|---|
| 1 | `Sasabe2011_OLED` | `10.1021/cm1024309` | Pattern-plausible for *Chem. Mater.* but specific suffix not verified | **No — keep TODO** |
| 2 | `AtahanEvrenk2019_RE` | `10.1021/acs.jpca.9b02733` | Two competing candidate DOIs surfaced during drafting; only publisher metadata can resolve | **No — keep TODO** |
| 3 | `Nelsen1987` | `10.1021/ja00237a007` | Pattern-plausible for vintage *JACS* but the *attribution* to the "four-point Nelsen" method requires chemistry-side verification | **No — keep TODO** |
| 4 | `EfficientKAN2024` | URL only; no DOI | Acceptable as a software citation; version pinning recommended | **Conditional — see §4 below for recommended form** |
| 5 | `RDKit2023` | URL only; no DOI; bib year `2023` does not match installed version `2024.09.6` | Acceptable as a software citation; version + year mismatch must be fixed | **Conditional — see §5 below for recommended form** |

Entry counts: **3** entries with publisher DOIs that cannot be confirmed without internet access (priorities 1–3); **2** software citations whose form can be tightened locally (priorities 4–5).

---

## 1. `Sasabe2011_OLED`

```
@article{Sasabe2011_OLED,
  title   = {Multifunctional Materials in High-Performance {OLED}s: Challenges for Solid-State Lighting},
  author  = {Sasabe, Hisahiro and Kido, Junji},
  journal = {Chemistry of Materials},
  year    = {2011},
  volume  = {23},
  number  = {3},
  pages   = {621--630},
  doi     = {10.1021/cm1024309},
  note    = {TODO: confirm DOI and full title against publisher metadata}
}
```

**What I am reasonably confident about**

- The author pair *Sasabe, H. & Kido, J.* is consistent with a known OLED-review collaboration; Junji Kido is a well-known OLED researcher at Yamagata University and Hisahiro Sasabe is a frequent co-author.
- *Chem. Mater.* Vol 23 is the 2011 volume; the page range 621–630 is plausible for a Issue 3 paper.
- The DOI prefix `10.1021/cm…` is the ACS *Chemistry of Materials* DOI prefix and is consistent with this venue.

**What I cannot confirm from this offline session**

- The exact DOI suffix `cm1024309`.  ACS DOIs of this era follow the pattern `cm<6-7 digits>` and small errors in the digit string would silently resolve to a different paper or to nothing.
- The exact title.  The version *"Multifunctional Materials in High-Performance OLEDs: Challenges for Solid-State Lighting"* is a plausible Sasabe & Kido OLED-review title but a competing form may exist.

**Recommended verification source**

1. ACS *Chemistry of Materials* journal page: `https://pubs.acs.org/journal/cmatex`, search by author "Sasabe Kido" within the 2011 volume.
2. Crossref REST API: `https://api.crossref.org/works?query.author=Sasabe+Kido&filter=from-pub-date:2011-01-01,until-pub-date:2011-12-31,container-title:Chemistry%20of%20Materials`.
3. Google Scholar search for the title.
4. The author's institutional page (Sasabe — Yamagata University, or Kido — Yamagata University).

**Decision**

Keep `note = {TODO}`.  Do **not** commit the DOI from the bib as final.  An author with publisher-website access can resolve in ~2 minutes.

---

## 2. `AtahanEvrenk2019_RE`

```
@article{AtahanEvrenk2019_RE,
  title   = {Prediction of Intramolecular Reorganization Energy Using Machine Learning},
  author  = {Atahan-Evrenk, S{\"u}le and Atalay, F. Bet{\"u}l},
  journal = {The Journal of Physical Chemistry A},
  year    = {2019},
  volume  = {123},
  pages   = {7855--7863},
  doi     = {10.1021/acs.jpca.9b02733},
  note    = {TODO: confirm DOI against publisher metadata. Provides both the prediction methodology and the n = 5,876 hole-reorganization-energy dataset used as Client D in this manuscript.}
}
```

**What I am reasonably confident about**

- The author pair *Atahan-Evrenk, S. & Atalay, F. B.* is consistent with the well-cited Bahçeşehir University paper on ML prediction of reorganization energies.
- The journal (*J. Phys. Chem. A*, Vol 123, 2019) is correct.
- The page range 7855–7863 is consistent with an Issue 36 (September 2019) paper.

**What I cannot confirm from this offline session**

- The exact DOI.  During earlier drafting two competing candidate DOIs surfaced:
  - `10.1021/acs.jpca.9b02733` (currently in the bib),
  - `10.1021/acs.jpca.9b00873` (suggested by a previous draft assistant pass).
- I have *no offline mechanism* to determine which is correct — both follow the canonical ACS J. Phys. Chem. A 2019 DOI pattern `acs.jpca.9b<5 digits>`.

**Special concern for this manuscript**

This paper is also the source of **Client D** (n = 5,876 hole-reorganization-energy dataset).  An incorrect DOI here would silently mis-attribute the dataset citation in the Methods and SI sections — a higher-impact error than a misroute citation in the related-work paragraph.

**Recommended verification source**

1. ACS *J. Phys. Chem. A* page for Vol 123, Issue 36 (September 12, 2019).
2. Crossref title search: `https://api.crossref.org/works?query.title=Prediction+Intramolecular+Reorganization+Energy+Machine+Learning&filter=container-title:The%20Journal%20of%20Physical%20Chemistry%20A`.
3. The author's ORCID record: Süle Atahan-Evrenk has an ORCID profile that lists publications with DOIs.
4. Web of Science / Scopus.

**Decision**

Keep `note = {TODO}`.  This is the highest-priority TODO for the manuscript because the entry serves as both a methodology citation [5] and a dataset citation [12].

---

## 3. `Nelsen1987`

```
@article{Nelsen1987,
  title   = {Estimation of Inner Shell {M}arcus Terms for Amino Nitrogen Compounds by Molecular Orbital Calculations},
  author  = {Nelsen, Stephen F. and Blackstock, S. C. and Kim, Y.},
  journal = {Journal of the American Chemical Society},
  year    = {1987},
  volume  = {109},
  pages   = {677--682},
  doi     = {10.1021/ja00237a007},
  note    = {TODO: verify DOI. Source of the four-point Nelsen geometry treatment described in Section 2.1.}
}
```

**What I am reasonably confident about**

- The author triplet *Nelsen, S. F., Blackstock, S. C., Kim, Y.* is consistent with Stephen F. Nelsen's University of Wisconsin–Madison group; Silas C. Blackstock and Younwha Kim co-authored several mid-1980s amine reorganization-energy papers.
- *JACS* Vol 109 (1987), pages 677–682 are consistent with a JACS Issue 3 (February 1987) paper.
- The DOI prefix `10.1021/ja…` is the ACS *JACS* prefix and `ja00237a007` follows the vintage JACS DOI pattern `ja00<5 digits>a0NN`.

**What I cannot confirm from this offline session**

- The exact DOI suffix `00237a007` — this is a numeric/check-digit pattern that requires publisher cross-check.
- More importantly, the *attribution* of the "four-point Nelsen" method to *this specific* 1987 paper.  Nelsen has multiple seminal papers on intramolecular reorganization energy in 1986–1996, and the canonical "four-point method" is often cited to:
  - this 1987 JACS paper, *or*
  - a 1996 *J. Phys. Chem.* paper by Nelsen and co-workers that consolidates the four-point procedure.
- Whether the Methods text in §2.1 should cite the 1987 origin or the 1996 consolidation is a chemistry-domain convention that an author with access to the relevant reorganization-energy review (e.g., Coropceanu *et al.* 2007 *Chem. Rev.* [2]) can settle in a few minutes by looking at how [2] cites Nelsen.

**Recommended verification source**

1. Cross-reference with Coropceanu *et al.* 2007 *Chem. Rev.* [2] — its reference list cites the Nelsen four-point paper and will pin the canonical year/DOI.
2. JACS legacy DOI search at `https://pubs.acs.org/journal/jacsat`.
3. Crossref search: `https://api.crossref.org/works?query.author=Nelsen+Blackstock+Kim&filter=container-title:Journal%20of%20the%20American%20Chemical%20Society`.

**Decision**

Keep `note = {TODO}`.  Additionally, **flag the attribution question** to the chemistry-side co-author: confirm whether the 1987 JACS paper or a later (1996) Nelsen paper is the canonical citation for the "four-point method".

---

## 4. `EfficientKAN2024`

```
@misc{EfficientKAN2024,
  title   = {An Efficient Implementation of {K}olmogorov--{A}rnold Networks},
  author  = {Blealtan},
  year    = {2024},
  note    = {\url{https://github.com/Blealtan/efficient-kan}. TODO: pin to a Zenodo / arXiv companion (or a tagged release commit) before submission. The library is GitHub-hosted and has no DOI at writing time.}
}
```

**What I verified locally from the codebase**

- The installed library (used to produce all reported KAN-head results in the manuscript) is `efficient-kan` version `0.1.0`.
- The author identifier in the package metadata is `Blealtan Cao` with contact email `blealtan@outlook.com`; the license is MIT.
- The library is loaded from `/vol1/home/lengcan/cleng/miniconda3/envs/H-CAAN/lib/python3.10/site-packages/efficient_kan/__init__.py` and exports the `KAN` and `KANLinear` classes used in `src/models.py`.

**What I cannot confirm from this offline session**

- Whether a Zenodo deposit exists for `efficient-kan`.  Many GitHub repos do, but I have no way to query Zenodo offline.
- Whether the author has published a CITATION.cff or BibTeX recommendation in the repository README.

**Recommended verification source (and the recommended bib form if no DOI is found)**

1. Visit `https://github.com/Blealtan/efficient-kan` and look for:
   - a CITATION.cff file at the repository root;
   - a "Cite this repository" widget on the right-hand sidebar (GitHub auto-renders this if a citation file is present);
   - a Zenodo / Software Heritage badge in the README.
2. Search `https://zenodo.org` for `efficient-kan` to see whether the author or a third party has minted a DOI.
3. If neither exists, software-citation form (this is acceptable for SCI submission):

```
@misc{EfficientKAN2024,
  author       = {Cao, Blealtan and {the efficient-kan contributors}},
  title        = {efficient-kan: An Efficient Pure-{PyTorch} Implementation of Kolmogorov--Arnold Networks (v0.1.0)},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Blealtan/efficient-kan}},
  version      = {0.1.0},
  urldate      = {2026-05-11},
  note         = {MIT licence; no Zenodo DOI at writing time}
}
```

**Decision**

This entry is **safe to commit as a versioned software citation** even if no DOI is ever minted.  The minimal acceptable patch (compatible with the journal's software-citation policy for ACS / RSC / Nature) is to:

1. Replace `author = {Blealtan}` with `author = {Cao, Blealtan and {the efficient-kan contributors}}` (full author identifier from pip metadata).
2. Add `version = {0.1.0}`.
3. Add `urldate = {2026-05-11}` (the date the URL was last accessed for the present writing pass).
4. Retain the `\url{...}` field.
5. Demote the TODO to a soft note: "Zenodo DOI not available at writing time; cite as software."

I will **not** apply this patch in the present pass — final decision rests with the lead author.

---

## 5. `RDKit2023`

```
@misc{RDKit2023,
  title   = {{RDKit}: Open-Source Cheminformatics},
  author  = {Landrum, Greg and others},
  year    = {2023},
  note    = {\url{https://www.rdkit.org}. TODO: pin to a specific release version with a Zenodo DOI before submission. The current run uses the conda-forge build present in the H-CAAN environment.}
}
```

**What I verified locally from the codebase**

- The installed RDKit version (used to produce Morgan fingerprints, Murcko scaffolds and 2D depictions for Figures 1, 2 and 6) is `rdkit 2024.09.6`.
- The bib year `2023` is therefore **inconsistent with the actual software used**.
- The conda-forge build is recorded in the H-CAAN environment used by both the figure-generation pipeline and the data-preprocessing pipeline.

**What I cannot confirm from this offline session**

- The version-specific Zenodo DOI for `rdkit 2024.09.6`.  RDKit maintains a *concept DOI* on Zenodo (covering all releases under one citation) and per-version DOIs are minted at every release.  The exact DOI for 2024.09.6 is published on the RDKit Zenodo page but I cannot resolve it without internet access.

**Recommended verification source**

1. Visit `https://zenodo.org/search?q=rdkit` and locate the specific deposit for release `Release_2024_09_6`.
2. Alternatively, visit `https://github.com/rdkit/rdkit/releases/tag/Release_2024_09_6` — the release page contains the Zenodo DOI badge.
3. As a fallback, use the canonical concept-DOI form (covers any release under one citation) — the form is published at the top of the RDKit GitHub README.

**Recommended bib form (lead author to confirm DOI)**

```
@misc{RDKit2024,
  author       = {Landrum, Greg and {the RDKit contributors}},
  title        = {{RDKit}: Open-source cheminformatics (release 2024.09.6)},
  year         = {2024},
  howpublished = {\url{https://www.rdkit.org}},
  version      = {2024.09.6},
  urldate      = {2026-05-11},
  doi          = {TODO: lookup version-specific Zenodo DOI for Release_2024_09_6},
  note         = {Used via the conda-forge build in the H-CAAN environment.}
}
```

**Additional issue: bib key consistency**

The bib key `RDKit2023` no longer matches the installed software year (`2024.09.6` = 2024).  If the lead author chooses to change the key to `RDKit2024`, the inline citation `[18]` in `manuscript.md` and the references-list entry must be updated in parallel to keep the cross-reference valid.

**Decision**

Two patches are recommended, both pending lead-author approval:

1. Update bib year from `2023` to `2024` and add `version = {2024.09.6}` to match the software actually used. This is a factual correction, not a metadata guess.
2. Look up the version-specific Zenodo DOI at Zenodo / the GitHub release page.

I will **not** apply these patches in the present pass.

---

## Cross-cutting recommendations

1. **DOI verification batch** — for entries 1, 2, and 3 the same workflow applies: open the publisher page, copy the canonical DOI string, compare. Estimated total time: under 10 minutes with internet access.  This is a strict prerequisite for submission and should be done by the lead author or a librarian.

2. **Software citation policy alignment** — confirm the target journal's software-citation requirements before applying the patches to entries 4 and 5.  *J. Chem. Inf. Model.* and *J. Chem. Theory Comput.* both accept GitHub-URL + version + access-date software citations; *Nature*-family journals prefer Zenodo / Software Heritage DOIs where available.

3. **Authorship corrections** — for entry 4, the more correct author field is `Cao, Blealtan and {the efficient-kan contributors}` (from the pip metadata) rather than the bare GitHub handle `Blealtan`.  For entry 5, the canonical RDKit citation form uses `Landrum, Greg and {the RDKit contributors}` (matching the form used by the official RDKit README).

4. **Year–key consistency** — the `RDKit2023` key embeds a year that no longer matches the installed software (2024.09.6).  Renaming to `RDKit2024` requires a coordinated update of the inline citation marker `[18]` in `manuscript.md`.

5. **No DOI was invented in this audit.** All entries flagged as uncertain remain `note = {TODO: ...}` until publisher-side verification.

---

## What this audit did NOT do

- Did not modify `paper/references.bib`.
- Did not modify `paper/manuscript.md`.
- Did not modify `paper/SI.md`.
- Did not modify `paper/submission_checklist.md`.
- Did not run any training, network, or commit operations.
- Did not contact publisher APIs (no internet access available in this session).

This audit is a recommendation document only.  Apply patches in a follow-up turn after the lead author confirms verified DOIs and the preferred software-citation form.

*End of audit.*
