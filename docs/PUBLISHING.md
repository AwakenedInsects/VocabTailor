# Publishing Walkthrough

Detailed steps for publishing VocabTailor to GitHub, Hugging Face Hub, and PyPI. For a short checklist, see [RELEASE.md](RELEASE.md).

---

## Before you start

1. **Version** — Bump `version` in `pyproject.toml` and `src/vocab_tailor/version.py` (e.g. `0.1.0`). Commit.
2. **Tests** — From repo root: `pip install -e .` then run tests, e.g. `python tests/test_vocab_tailor.py` and (if applicable) `python tests/validate_profiling_pipeline.py`.
3. **Changelog** — Update CHANGELOG.md or release notes if you keep them.

---

## 1. GitHub

1. **Create the repo** (if it doesn’t exist): e.g. `your-org/vocab-tailor`. Do not initialize with a README if you already have one locally.
2. **Add remote and push** (from your local repo root):
   ```bash
   git remote add origin https://github.com/your-org/vocab-tailor.git
   git branch -M main
   git push -u origin main
   ```
3. **Tag and push the tag**:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
4. **Optional:** On GitHub, go to **Releases → Draft a new release**, choose tag `v0.1.0`, add release notes, and publish.

---

## 2. Hugging Face Hub

1. **Create the Hub repo:** On Hugging Face, go to **New → Model** (or create an organization first). Name it e.g. `vocab-tailor` (e.g. `your-org/vocab-tailor`).
2. **Add README:** In the Hub repo, add a README (copy from this repo or paste content). Set visibility (public/private).
3. **Optional — Sync from GitHub:** In the repo **Settings**, enable **Sync with GitHub** and connect the GitHub repo so the Hub mirrors it.
4. **URLs in package:** In `pyproject.toml`, set `[project.urls]` (and Homepage/Documentation if desired) to your Hub and GitHub URLs. Commit and push (and re-tag if you want the tag to include this change).

---

## 3. PyPI

1. **Accounts:** Create an account at [pypi.org](https://pypi.org) (and at [test.pypi.org](https://test.pypi.org) for dry runs).
2. **Build** (from repo root):
   ```bash
   pip install build twine
   python -m build
   ```
   This produces `dist/*.whl` and `dist/*.tar.gz`.
3. **Test PyPI (recommended):**
   ```bash
   twine upload --repository testpypi dist/*
   ```
   Use your Test PyPI credentials. Then try: `pip install -i https://test.pypi.org/simple/ vocab-tailor`.
4. **Real PyPI:**
   ```bash
   twine upload dist/*
   ```
   Use your PyPI credentials. After that, `pip install vocab-tailor` will work.

---

## 4. Dataset on Hub (optional)

Only if you publish benchmark or demo data: **New → Dataset** on the Hub, upload files, and document the format in the dataset README.

---

## URLs to update

After creating GitHub and Hub repos, set in `pyproject.toml` under `[project.urls]`:

- **Homepage** — Hugging Face repo (e.g. `https://huggingface.co/your-org/vocab-tailor`).
- **Repository** — GitHub repo (e.g. `https://github.com/your-org/vocab-tailor`).
- **Documentation** — Hub or GitHub README link.
