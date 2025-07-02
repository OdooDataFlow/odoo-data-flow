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
  - Add E2E test which perform an actual import /export.
  * **On Pull Requests to `main` / After Merging:** Run the slow E2E integration tests. This ensures that only fully validated code gets into your main branch, without slowing down the development process for every small change.

Advanced Pre-flight Check: Smart Field Verification
---------------------------------------------------

*   **Status:** Planned

*   **Priority:** Medium

*   **Complexity:** High


### Description

This feature will expand on the basic --verify-fields check to create a much more intelligent "pre-flight" validation system. Before starting a large import, the tool would not only check if the target fields exist but would also inspect their properties to prevent a wider range of common import errors.

The goal is to fail fast with clear, human-readable error messages, saving developers from discovering these issues halfway through a long import process.

### Key Validations to Implement

The "Smart Field Verification" would check for the following common error scenarios:

1.  **Importing into Read-Only Fields:**

    *   **Check:** The tool would verify if any field in the import file is marked as readonly=True or compute=True in Odoo.

    *   **Error Message:** Error: You are trying to import data into the field 'price\_total', which is a computed (read-only) field. You should import the source fields (e.g., 'price\_unit', 'product\_uom\_qty') instead.

2.  **Data Type Mismatches:**

    *   **Check:** The tool would perform a basic check to see if the data in a column is compatible with the target field's type (ttype in Odoo).

    *   **Examples:**

        *   It would warn if a column being mapped to a Many2one or Many2many field does not have the required /id suffix.

        *   It would warn if a column being mapped to an Integer or Float field contains non-numeric characters.

    *   **Error Message:** Warning: The column 'partner\_id' appears to be a relational field but is missing the '/id' suffix. The correct header is 'partner\_id/id'.

3.  **Selection Field Value Check:**

    *   **Check:** For Selection fields, the tool could fetch the list of valid keys from Odoo and check if the values in the source CSV are all valid.

    *   **Error Message:** Error: The value 'Shipped' in column 'delivery\_status' is not a valid key for the selection field. Valid keys are: 'draft', 'in\_progress', 'done'.


### Implementation Plan

This would be implemented via a new, more powerful command-line flag, for example, --validate-mapping. When this flag is used, the run\_import orchestrator would:

1.  Connect to the Odoo instance.

2.  Fetch the complete field definitions for the target model from ir.model.fields, including name, ttype, readonly, and compute.

3.  For Selection fields, it would perform an additional query to get the valid keys.

4.  It would then iterate through the header of the source CSV file and perform the series of checks described above.

5.  If any check fails, it would abort the import immediately with a clear, actionable error message.
