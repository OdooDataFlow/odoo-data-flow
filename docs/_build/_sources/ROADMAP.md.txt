# Project Roadmap

This document outlines the planned future enhancements and major refactoring efforts for the `odoo-data-flow` library. Contributions are welcome!

## Planned Features & Enhancements

### 1. Modernize Post-Import Workflows

- **Current Status:** The library includes a legacy `InvoiceWorkflowV9` class designed specifically for Odoo version 9. This class uses outdated API calls (e.g., `exec_workflow`) and will not work on modern Odoo versions.
- **Goal:** Refactor the workflow system to support recent Odoo versions (16.0, 17.0, 18.0+).
- **Tasks:**
  - Create a new `InvoiceWorkflowV18` (or similar) class that uses the modern Odoo API for validating and paying invoices (e.g., calling button actions like `action_post`).
  - Update the `workflow_runner.py` and the `__main__.py` CLI to allow users to specify which workflow version they want to run (e.g., `odoo-data-flow workflow invoice-v18`).
  - Consider creating a base `Workflow` class that new, custom workflows can inherit from to promote a consistent structure.

### 2. Add Support for More Data Formats

- **Goal:** Expand the `Processor` to natively handle other common data formats beyond CSV and XML.
- **Potential Formats:**
  - JSONL (JSON Lines)
  - Direct database connections (e.g., PostgreSQL, MySQL)

### 3. Enhance Test Coverage

- **Goal:** Increase unit and integration test coverage to improve reliability.
- **Tasks:**
  - Add specific tests for each `mapper` function to validate its behavior with edge cases.
  - Add unit and integration tests for the workflow_runner and the legacy InvoiceWorkflowV9 module.
  - Create integration tests that run against a live, containerized Odoo instance as part of the CI/CD pipeline.
