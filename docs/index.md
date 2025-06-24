```{include} ../README.md
---
end-before: <!-- github-only -->
---
```

# Odoo Data Flow

**A robust, declarative library for managing complex data imports and exports with Odoo.**

Odoo Data Flow is a powerful and flexible Python library designed to simplify the import and export of data to and from Odoo. It allows you to define data mappings and transformations in a declarative way, making complex data operations manageable and repeatable.
You can easily manage complex transformations, relationships, and validations, making your data integration tasks simpler and more reliable.

This library is the successor to the `odoo-csv-import-export` library, refactored for modern development practices and enhanced clarity.

```{mermaid}
graph TD
    subgraph External Data
        A[CSV / XLSX File]
    end

    subgraph odoo-data-flow
        B{Model Definition in Python}
        C["@field Decorators"]
        D[Transformation & Validation Logic]
    end
    
    subgraph Odoo
        E[Odoo Database]
    end

    A --> B
    B -- Defines --> C
    C -- Applies --> D
    B -- Orchestrates --> E

    style B fill:#777,stroke:#333,stroke-width:2px,color:#fff
```

## Key Features


* **Declarative Python Configuration**: Define your entire data flow using clear and readable Python objects. This "configuration-as-code" approach allows for powerful, dynamic, and easily debugged setups. Making complex data operations manageable and repeatable.
* **Multiple Data Sources**: Natively supports CSV, JSON, and XML files. Easily extendable to support other sources like databases or APIs.
* **Built-in Data Transformation:** Clean, modify, and format data on the fly using simple `lambda` functions or your own custom python code.
* **Relational Field Handling:** Easily import and export `Many2one`, `One2many`, and `Many2many` relationships.
* **Data Validation:** Ensure data integrity before it even reaches Odoo.
* **Extensible and Customizable:** Write your own custom methods to handle unique or complex data-processing requirements.
* **Support for CSV and Excel:** Works seamlessly with the most common file formats for business data exchange.
* **Robust Error Handling**: Provides clear logging and error reports to help you debug your data flows quickly.


## Getting Started

Ready to simplify your Odoo data integrations?

| Step                        | Description                                          |
| --------------------------- | ---------------------------------------------------- |
| 🚀 **[Quickstart](./quickstart.md)** | Your first end-to-end example. Go from file to Odoo in minutes. |
| ⚙️ **[Installation](./installation.md)** | How to install the library in your project. |
| 🧠 **[Core Concepts](./core_concepts.md)** | Understand the key ideas behind the library. |


[license]: license
[contributor guide]: contributing
[command-line reference]: usage


```{toctree}
---
hidden:
maxdepth: 1
---

installation
quickstart
core_concepts
guides/index
faq
reference
contributing
Code of Conduct <codeofconduct>
License <license>
Changelog <https://github.com/OdooDataFlow/odoo-data-flow/releases>
```
