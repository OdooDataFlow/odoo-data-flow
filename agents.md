# Agent Instructions for Project: odoo-data-flow

This document guides the AI agent in developing the `odoo-data-flow` project.

## 🎯 Project Context & Goal

This is an **Odoo module**. All code, dependencies, and architectural decisions must be compatible with the Odoo version specified in the Git branch.

-   **Odoo Version Detection:** Before starting, determine the Odoo version by inspecting the Git branch name. Use this command to extract it:
    ```sh
    git rev-parse --abbrev-ref HEAD | sed -n 's/^\([0-9]\{1,2\}\.0\).*/\1/p'
    ```
-   **Strategic Goal:** All implemented functionality must align with the ODF strategic blueprint.

---

## 🛠️ Environment Setup

Jules must set up its environment using **Nox**.

1.  Ensure `nox` is installed: `pip install nox`.
2.  Run the initial setup session to create the virtual environment and install all dependencies: `nox -s setup`.

---

## 📝 Development & Quality Rules

For every task, you **MUST** adhere to the following rules without exception.

#### 1. Testing and Validation

All code execution, testing, and checks **must** be performed within the Nox-managed environment.

-   **Primary Command:** `nox` (runs the full suite)
-   **Quick Checks:** For faster iteration, run a subset: `nox -s pre-commit mypy tests`

#### 2. Code Quality & Formatting

All generated Python code must be 100% compliant with the project's pre-commit configuration.

-   **Linter/Formatter:** `ruff`
-   **Import Sorting:** `isort`
-   **Line Length:** Maximum **88 characters**.

#### 3. Type Safety

Code must be fully type-hinted and pass `mypy` static analysis with zero errors.

-   **Typing Style:** Use lowercase standard types (e.g., `list`, `dict`).

#### 4. Documentation

All functions, methods, and classes require **Google-style docstrings**.

-   **Format:** Start with a one-line summary ending in a period, followed by a blank line.
-   **Content:** Clearly document `Args:` and `Returns:`.
