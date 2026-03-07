# Release checklist

When cutting a new release (e.g. v0.1.0). For a detailed first-time walkthrough, see [PUBLISHING.md](PUBLISHING.md).

1. **Version** — Bump `version` in `pyproject.toml` and `src/vocab_tailor/version.py`. Commit.
2. **Tests** — From repo root: `pip install -e .` then `python tests/test_vocab_tailor.py`.
3. **Changelog** — Update CHANGELOG.md or release notes if you keep them.

---

## 1. GitHub

- Create repo (if not exists): e.g. `your-org/vocab-tailor`.
- Push code. Create a **tag** (e.g. `git tag v0.1.0 && git push origin v0.1.0`).
- Optionally create a **GitHub Release** from the tag with notes.

---

## 2. Hugging Face Hub

- **New → Model** (or Organization first). Name e.g. `vocab-tailor`.
- Add a README (copy from this repo or link to GitHub). Set visibility.
- **Optional:** Use Hub’s “Sync with GitHub” so the Hub repo mirrors the GitHub repo.
- In `pyproject.toml` set `Homepage` (and optionally `Documentation`) to your Hub repo URL, e.g. `https://huggingface.co/your-org/vocab-tailor`.

---

## 3. PyPI (optional)

```bash
pip install build twine
python -m build
twine upload dist/*
```

- Prefer testing with Test PyPI first: `twine upload --repository testpypi dist/*`, then `pip install -i https://test.pypi.org/simple/ vocab-tailor`.
- After upload, users can `pip install vocab-tailor`.

---

## 4. Dataset on Hub (optional)

- Only if you publish benchmark/demo data. **New → Dataset**, upload files, document format in the dataset README.

---

## URLs to update

After creating both repos, set in `pyproject.toml` under `[project.urls]`:

- **Homepage** — Hugging Face repo (e.g. `https://huggingface.co/your-org/vocab-tailor`).
- **Repository** — GitHub repo (e.g. `https://github.com/your-org/vocab-tailor`).
- **Documentation** — Hub or GitHub README link.
